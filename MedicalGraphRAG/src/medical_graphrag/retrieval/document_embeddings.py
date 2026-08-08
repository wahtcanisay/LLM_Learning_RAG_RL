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
    full = tokenizer.encode("x", add_special_tokens=True)
    plain = tokenizer.encode("x", add_special_tokens=False)
    return len(full) - len(plain)


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
    # 所有窗口一次批量编码(避免逐文档小批量调用带来的巨大开销)
    window_embeddings = np.asarray(
        embedder.encode(
            [tokenizer.decode(w, skip_special_tokens=True) for w in all_windows],
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ),
        dtype="float32",
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


def _model_identity(embedder) -> dict[str, Any]:
    max_seq_length = int(embedder.get_max_seq_length() or 512)
    try:
        st_version = distribution_version("sentence-transformers")
    except Exception:
        st_version = "unknown"
    return {
        "embedding_model": getattr(embedder, "_model_name", ""),
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
        "overlap_tokens": overlap_tokens,
        **{k: v for k, v in _model_identity(embedder).items()},
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
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            manifest = validate_frozen_dataset(dataset_dir)
            if (
                existing.get("dataset_manifest_sha256")
                == sha256_file(dataset_dir / "manifest.json")
                and existing.get("source_artifact_sha256")
                == manifest["artifact_hashes"]["documents.jsonl"]
                and existing.get("embeddings_sha256")
                == sha256_file(artifact_dir / "document_embeddings.npy")
            ):
                return existing
        except Exception:
            pass
    return build_document_embeddings(
        dataset_dir,
        artifact_dir,
        model_name=model_name,
        overlap_tokens=overlap_tokens,
        batch_size=batch_size,
    )
