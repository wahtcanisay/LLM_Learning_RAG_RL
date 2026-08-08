"""Unified document embedding artifact (spec §5.1, P0-8).

``documents.jsonl -> build_document_embeddings()`` produces a single frozen
artifact consumed by exactly three consumers:

    document_embeddings.npy            (float32, row = frozen document order)
    document_embedding_metadata.jsonl  (``{"doc_id": ...}`` in the same order)
    document_embedding_report.json     (manifest/documents hash + model + params
                                        + aggregation rule + artifact hashes)

Consumers — Dense index, Similarity kNN edges and Graph passage prior — only
LOAD this artifact; none re-encodes the documents. Every consumer report
records the same ``embedding_report_sha256`` and ``embeddings_sha256`` and
fails closed on any mismatch.

Full-coverage rule (spec §5.1): the model's own tokenizer produces untruncated
token IDs; the sequence is split into windows whose length, after adding the
model's special tokens, never exceeds ``max_seq_length``. Every original token
position is covered by at least one window (proved with a boolean position
mask, not by token-value sets). Each window is embedded independently and L2
normalized, then the window vectors are mean-averaged and L2-normalized once
more. ``truncated_token_count`` is derived from the position-coverage mask and
must be 0; a non-zero value fails the build.
"""
import json
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import numpy as np

from medical_graphrag.data.io import sha256_file, write_json, write_jsonl
from medical_graphrag.retrieval.bm25 import validate_frozen_dataset

AGGREGATION_RULE = "mean_window_then_l2"
TOKEN_WINDOW_ENCODING = "token_ids_direct_v1"
_NORM_TOLERANCE = 1e-3


def _window_ranges(
    n: int,
    *,
    max_window_len: int,
    overlap_tokens: int,
) -> list[tuple[int, int]]:
    """Index ranges [start, end) covering every position of a length-n token seq.

    Raises on invalid parameters. Stride = max_window_len - overlap_tokens.
    """
    if max_window_len <= 0 or overlap_tokens < 0 or overlap_tokens >= max_window_len:
        raise ValueError("require max_window_len > overlap_tokens >= 0")
    if n == 0:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + max_window_len, n)
        ranges.append((start, end))
        if end == n:
            break
        start += max_window_len - overlap_tokens
    return ranges


def _coverage(
    n: int,
    ranges: list[tuple[int, int]],
) -> tuple[bool, int]:
    """Return (full_coverage, truncated_count) from a boolean position mask.

    full_coverage is True iff every original position in [0, n) lies in at
    least one range. truncated_count = number of uncovered positions.
    """
    if n == 0:
        return True, 0
    covered = [False] * n
    for start, end in ranges:
        for i in range(start, end):
            if 0 <= i < n:
                covered[i] = True
    truncated = sum(1 for flag in covered if not flag)
    return truncated == 0, truncated


def _num_special_tokens(tokenizer) -> int:
    """Number of special tokens a window adds (e.g. [CLS]+[SEP] = 2)."""
    if hasattr(tokenizer, "num_special_tokens_to_add"):
        return int(tokenizer.num_special_tokens_to_add(pair=False))
    full = tokenizer.encode("x", add_special_tokens=True)
    plain = tokenizer.encode("x", add_special_tokens=False)
    return len(full) - len(plain)


def _special_token_affixes(tokenizer) -> tuple[list[int], list[int]]:
    """Infer single-sequence special-token prefix/suffix without decoding IDs."""
    plain = list(tokenizer.encode("x", add_special_tokens=False))
    full = list(tokenizer.encode("x", add_special_tokens=True))
    for start in range(len(full) - len(plain) + 1):
        if full[start:start + len(plain)] == plain:
            prefix = full[:start]
            suffix = full[start + len(plain):]
            if len(prefix) + len(suffix) == _num_special_tokens(tokenizer):
                return prefix, suffix
    raise ValueError("cannot infer tokenizer special-token layout")


def _encode_token_windows(
    embedder,
    windows: list[list[int]],
    *,
    batch_size: int,
) -> np.ndarray:
    """Encode the exact token-ID windows without decode/re-tokenize drift.

    Tests may inject ``encode_token_windows`` on a lightweight embedder.  A
    real SentenceTransformer is evaluated through its public ``forward`` path
    after the tokenizer adds special tokens and pads the already-frozen token
    IDs.  This preserves the token windows proved by ``_coverage``.
    """
    injected = getattr(embedder, "encode_token_windows", None)
    if callable(injected):
        return np.asarray(
            injected(
                windows,
                batch_size=batch_size,
                normalize_embeddings=True,
            ),
            dtype="float32",
        )

    import torch
    import torch.nn.functional as functional

    tokenizer = embedder.tokenizer
    embedder.eval()
    special_prefix, special_suffix = _special_token_affixes(tokenizer)
    encoded_batches: list[np.ndarray] = []
    for start in range(0, len(windows), batch_size):
        prepared = [
            {
                "input_ids": special_prefix + list(token_ids) + special_suffix,
                "attention_mask": [
                    1
                ] * (len(special_prefix) + len(token_ids) + len(special_suffix)),
            }
            for token_ids in windows[start:start + batch_size]
        ]
        features = tokenizer.pad(prepared, padding=True, return_tensors="pt")
        device = getattr(embedder, "device", None)
        if device is not None:
            features = {name: value.to(device) for name, value in features.items()}
        with torch.inference_mode():
            sentence_embeddings = embedder.forward(dict(features))["sentence_embedding"]
            sentence_embeddings = functional.normalize(
                sentence_embeddings, p=2, dim=1
            )
        encoded_batches.append(
            sentence_embeddings.detach().cpu().numpy().astype("float32", copy=False)
        )
    return np.vstack(encoded_batches)


def embed_documents_full(
    texts: list[str],
    embedder,
    *,
    overlap_tokens: int = 32,
    batch_size: int = 64,
    normalize: bool = True,
) -> tuple[np.ndarray, list[int], int]:
    """Full-coverage document embeddings (mean-window-then-L2).

    Returns ``(embeddings, window_counts, truncated_token_count)``. Raises if a
    text is empty (no token windows) or any original position is uncovered.
    """
    tokenizer = embedder.tokenizer
    max_seq_length = int(embedder.get_max_seq_length() or 512)
    special = _num_special_tokens(tokenizer)
    max_window_len = max_seq_length - special
    if max_window_len <= 0:
        raise ValueError("embedder max_seq_length too small for its special tokens")
    if overlap_tokens >= max_window_len:
        raise ValueError("overlap_tokens must be < effective window tokens")

    all_windows: list[list[int]] = []
    doc_ranges: list[tuple[int, int]] = []
    window_counts: list[int] = []
    total_truncated = 0
    for text in texts:
        token_ids = list(tokenizer.encode(text, add_special_tokens=False))
        ranges = _window_ranges(
            len(token_ids),
            max_window_len=max_window_len,
            overlap_tokens=overlap_tokens,
        )
        if not ranges:
            raise ValueError("empty text produces no token windows")
        full, truncated = _coverage(len(token_ids), ranges)
        if not full:
            raise ValueError("token-window construction left uncovered positions")
        total_truncated += truncated
        windows = [token_ids[start:end] for start, end in ranges]
        start = len(all_windows)
        all_windows.extend(windows)
        doc_ranges.append((start, len(all_windows)))
        window_counts.append(len(windows))

    if not all_windows:
        raise ValueError("no token windows to embed")
    window_embeddings = _encode_token_windows(
        embedder,
        all_windows,
        batch_size=batch_size,
    )
    if window_embeddings.shape[0] != len(all_windows):
        raise ValueError("window embedding count does not match window count")
    doc_embeddings: list[np.ndarray] = []
    for start, end in doc_ranges:
        mean_vec = window_embeddings[start:end].mean(axis=0)
        if normalize:
            norm = float(np.linalg.norm(mean_vec))
            if norm > 0:
                mean_vec = mean_vec / norm
        doc_embeddings.append(np.asarray(mean_vec, dtype=np.float32))
    return np.vstack(doc_embeddings), window_counts, int(total_truncated)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    k = int((len(sorted_values) - 1) * q)
    return float(sorted_values[k])


def _model_identity(embedder, model_name: str) -> dict[str, Any]:
    max_seq_length = int(embedder.get_max_seq_length() or 512)
    try:
        st_version = distribution_version("sentence-transformers")
    except Exception:
        st_version = "unknown"
    return {
        "embedding_model": model_name,
        "sentence_transformers_version": st_version,
        "max_seq_length": max_seq_length,
    }


def build_document_embeddings(
    dataset_dir: Path,
    output_dir: Path,
    *,
    model_name: str,
    overlap_tokens: int = 32,
    batch_size: int = 64,
    embedder: Any | None = None,
) -> dict[str, Any]:
    """Build the frozen document embedding artifact from documents.jsonl.

    If ``embedder`` is not injected, a SentenceTransformer is loaded from
    ``model_name`` (integration/smoke only). Returns the report dict.
    """
    from medical_graphrag.retrieval.dense import _load_embedder

    manifest = validate_frozen_dataset(dataset_dir)
    rows = [json.loads(line) for line in
            (dataset_dir / "documents.jsonl").open(encoding="utf-8") if line.strip()]
    if len(rows) != manifest["counts"]["documents"]:
        raise ValueError("document count does not match manifest")
    doc_ids = [str(row["doc_id"]) for row in rows]
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("duplicate doc_id in documents.jsonl")
    texts = [str(row["content"]) for row in rows]

    if embedder is None:
        embedder = _load_embedder(model_name)
    embeddings, window_counts, truncated = embed_documents_full(
        texts,
        embedder,
        overlap_tokens=overlap_tokens,
        batch_size=batch_size,
    )
    if truncated != 0:
        raise ValueError("document embedding truncated tokens != 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "document_embeddings.npy"
    np.save(embeddings_path, embeddings)
    metadata_path = output_dir / "document_embedding_metadata.jsonl"
    write_jsonl(metadata_path, [{"doc_id": doc_id} for doc_id in doc_ids])
    report_path = output_dir / "document_embedding_report.json"

    window_counts_sorted = sorted(window_counts)
    report = {
        "graph_schema_version": 2,
        "retrieval_unit": "document",
        "source_artifact": "documents.jsonl",
        "source_artifact_sha256": manifest["artifact_hashes"]["documents.jsonl"],
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "aggregation_rule": AGGREGATION_RULE,
        "token_window_encoding": TOKEN_WINDOW_ENCODING,
        "overlap_tokens": overlap_tokens,
        **{k: v for k, v in _model_identity(embedder, model_name).items()},
        "dim": int(embeddings.shape[1]),
        "document_count": len(doc_ids),
        "window_coverage": {
            "window_count_min": min(window_counts_sorted),
            "window_count_mean": sum(window_counts) / max(len(window_counts), 1),
            "window_count_p95": _percentile(window_counts_sorted, 0.95),
            "window_count_max": max(window_counts_sorted),
            "truncated_token_count": int(truncated),
        },
        "embeddings_sha256": sha256_file(embeddings_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
    write_json(report_path, report)
    return report


def load_document_embedding_artifact(
    dataset_dir: Path,
    artifact_dir: Path,
    *,
    expected_model_name: str | None = None,
    expected_overlap_tokens: int | None = None,
) -> tuple[np.ndarray, list[str], dict[str, Any], str]:
    """Load and fully validate the frozen document embedding artifact.

    The dataset binding, metadata hash and row order, embedding hash/shape,
    aggregation contract and optional requested build parameters must all
    agree.  Consumers use this single loader so Dense, Graph and Similarity
    cannot silently interpret the same matrix with different document IDs.
    """
    manifest = validate_frozen_dataset(dataset_dir)
    report_path = artifact_dir / "document_embedding_report.json"
    embeddings_path = artifact_dir / "document_embeddings.npy"
    metadata_path = artifact_dir / "document_embedding_metadata.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    if report.get("retrieval_unit") != "document":
        raise ValueError("embedding artifact is not a document retrieval unit")
    if report.get("source_artifact") != "documents.jsonl":
        raise ValueError("embedding artifact source must be documents.jsonl")
    if report.get("aggregation_rule") != AGGREGATION_RULE:
        raise ValueError("embedding artifact aggregation rule mismatch")
    if report.get("token_window_encoding") != TOKEN_WINDOW_ENCODING:
        raise ValueError("embedding artifact token-window encoding mismatch")
    if report.get("dataset_manifest_sha256") != sha256_file(
        dataset_dir / "manifest.json"
    ):
        raise ValueError("embedding artifact does not match dataset manifest")
    if report.get("source_artifact_sha256") != manifest["artifact_hashes"][
        "documents.jsonl"
    ]:
        raise ValueError("embedding artifact does not match documents.jsonl")
    if expected_model_name is not None and report.get("embedding_model") != expected_model_name:
        raise ValueError("embedding artifact model does not match requested model")
    if (
        expected_overlap_tokens is not None
        and report.get("overlap_tokens") != expected_overlap_tokens
    ):
        raise ValueError("embedding artifact overlap does not match requested overlap")
    if report.get("embeddings_sha256") != sha256_file(embeddings_path):
        raise ValueError("embedding artifact embeddings SHA-256 mismatch")
    if report.get("metadata_sha256") != sha256_file(metadata_path):
        raise ValueError("embedding artifact metadata SHA-256 mismatch")

    document_rows = [
        json.loads(line)
        for line in (dataset_dir / "documents.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    expected_doc_ids = [str(row["doc_id"]) for row in document_rows]
    metadata_rows = [
        json.loads(line)
        for line in metadata_path.open(encoding="utf-8")
        if line.strip()
    ]
    doc_ids = [str(row["doc_id"]) for row in metadata_rows]
    if doc_ids != expected_doc_ids:
        raise ValueError("embedding metadata doc_id order does not match documents.jsonl")
    if report.get("document_count") != len(doc_ids):
        raise ValueError("embedding artifact document count mismatch")

    embeddings = np.load(embeddings_path, allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(doc_ids):
        raise ValueError("embedding matrix shape does not match metadata")
    if embeddings.shape[1] != report.get("dim"):
        raise ValueError("embedding matrix dimension does not match report")
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding matrix must contain only finite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=_NORM_TOLERANCE):
        raise ValueError("document embeddings must be L2-normalized")
    coverage = report.get("window_coverage", {})
    if coverage.get("truncated_token_count") != 0:
        raise ValueError("embedding artifact contains truncated document tokens")
    return embeddings, doc_ids, report, sha256_file(report_path)


def ensure_document_embeddings(
    dataset_dir: Path,
    artifact_dir: Path,
    *,
    model_name: str,
    overlap_tokens: int = 32,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Idempotently produce the frozen document embedding artifact.

    If a valid artifact already exists (manifest hash, documents hash and
    embeddings file hash all match), it is reused WITHOUT re-encoding
    (P0-8: consumers only load). Otherwise it is rebuilt deterministically.
    """
    report_path = artifact_dir / "document_embedding_report.json"
    if report_path.exists():
        _, _, existing, _ = load_document_embedding_artifact(
            dataset_dir,
            artifact_dir,
            expected_model_name=model_name,
            expected_overlap_tokens=overlap_tokens,
        )
        return existing
    partial_paths = (
        artifact_dir / "document_embeddings.npy",
        artifact_dir / "document_embedding_metadata.jsonl",
    )
    if any(path.exists() for path in partial_paths):
        raise ValueError("partial document embedding artifact exists without report")
    return build_document_embeddings(
        dataset_dir,
        artifact_dir,
        model_name=model_name,
        overlap_tokens=overlap_tokens,
        batch_size=batch_size,
    )
