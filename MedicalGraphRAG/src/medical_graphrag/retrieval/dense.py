"""Dense bi-encoder retrieval helpers (embedding + FAISS index).

`sentence-transformers`, `faiss`, and `numpy` are heavy and only guaranteed in
the `llm-pytorch` container, so they are imported lazily inside the functions
that need them. Importing this module stays side-effect free for the local CLI
and unit tests.

The frozen chunks are embedded with `all-mpnet-base-v2` (the same model LinearRAG
uses and the one used to tokenize the frozen data), stored as float32 vectors,
indexed with FAISS `IndexFlatIP` under L2-normalized vectors so that inner
product equals cosine similarity.
"""
import json
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file, write_json, write_jsonl
from medical_graphrag.retrieval.bm25 import validate_frozen_dataset

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_TEXT_MODE = "abstract_only"
INDEX_TYPE = "IndexFlatIP"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def export_chunk_metadata(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Write `chunk_metadata.jsonl` in frozen-chunk order plus a small export report.

    The metadata rows keep the same order as the frozen chunks, which is the order
    the embedding matrix rows use. Returns the export report with dataset hash
    bindings so every later stage can verify nothing changed under it.
    """
    manifest = validate_frozen_dataset(dataset_dir)
    chunks = _read_jsonl(dataset_dir / "chunks.jsonl")
    if len(chunks) != manifest["counts"]["chunks"]:
        raise ValueError("chunk count does not match manifest")
    metadata = [
        {
            "chunk_id": str(row["chunk_id"]),
            "doc_id": str(row["doc_id"]),
            "order": int(row["order"]),
            "title": str(row["title"]),
            "source": str(row["source"]),
        }
        for row in chunks
    ]
    metadata_path = output_dir / "chunk_metadata.jsonl"
    write_jsonl(metadata_path, metadata)
    report = {
        "text_mode": DEFAULT_TEXT_MODE,
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "dataset_artifact_hashes": manifest["artifact_hashes"],
        "chunk_count": len(metadata),
        "metadata_sha256": sha256_file(metadata_path),
    }
    write_json(output_dir / "export_report.json", report)
    return report


def _load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def build_dense_document_index(
    dataset_dir: Path,
    output_dir: Path,
    *,
    document_embeddings_dir: Path,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Dense index over full documents, CONSUMING the frozen embedding artifact.

    Loads ``document_embeddings.npy`` / ``document_embedding_metadata.jsonl`` /
    ``document_embedding_report.json`` produced by ``build_document_embeddings``
    (P0-8) — it never re-encodes the documents. Builds FAISS IndexFlatIP and
    records the same embedding-report/embeddings hashes as the similarity-edge
    and graph-prior consumers.
    """
    import faiss
    import numpy as np

    from medical_graphrag.retrieval.document_embeddings import _percentile

    manifest = validate_frozen_dataset(dataset_dir)
    artifact_report = json.loads(
        (document_embeddings_dir / "document_embedding_report.json").read_text(
            encoding="utf-8"
        )
    )
    if artifact_report.get("dataset_manifest_sha256") != sha256_file(
        dataset_dir / "manifest.json"
    ):
        raise ValueError("embedding artifact does not match dataset manifest")
    if artifact_report.get("source_artifact_sha256") != manifest["artifact_hashes"][
        "documents.jsonl"
    ]:
        raise ValueError("embedding artifact does not match documents.jsonl")
    embeddings_path = document_embeddings_dir / "document_embeddings.npy"
    if artifact_report.get("embeddings_sha256") != sha256_file(embeddings_path):
        raise ValueError("embedding artifact embeddings SHA-256 mismatch")
    embeddings = np.load(embeddings_path)
    metadata_rows = _read_jsonl(
        document_embeddings_dir / "document_embedding_metadata.jsonl"
    )
    doc_ids = [str(row["doc_id"]) for row in metadata_rows]
    if len(doc_ids) != artifact_report["document_count"] or len(set(doc_ids)) != len(
        doc_ids
    ):
        raise ValueError("embedding metadata count/duplicates mismatch")
    if embeddings.shape[0] != len(doc_ids):
        raise ValueError("embedding row count does not match metadata")

    dim = int(embeddings.shape[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    index_path = output_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    metadata_path = output_dir / "document_metadata.jsonl"
    write_jsonl(metadata_path, [{"doc_id": doc_id} for doc_id in doc_ids])

    window_counts = artifact_report.get("window_coverage", {})
    report = {
        "retrieval_unit": "document",
        "source_artifact": "documents.jsonl",
        "source_artifact_sha256": artifact_report["source_artifact_sha256"],
        "embedding_model": artifact_report["embedding_model"],
        "dim": dim,
        "index_type": INDEX_TYPE,
        "document_count": len(doc_ids),
        "embedding_report_sha256": sha256_file(
            document_embeddings_dir / "document_embedding_report.json"
        ),
        "embedding_embeddings_sha256": artifact_report["embeddings_sha256"],
        "index_sha256": sha256_file(index_path),
        "metadata_sha256": sha256_file(metadata_path),
        "window_coverage": window_counts,
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "dataset_artifact_hashes": manifest["artifact_hashes"],
    }
    write_json(output_dir / "index_build.json", report)
    return report


def build_dense_index(
    dataset_dir: Path,
    output_dir: Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    normalize: bool = True,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Embed chunk `content` and write embeddings + FAISS IndexFlatIP + index report.

    Writes ``chunk_embeddings.npy`` (float32, row = frozen chunk order),
    ``index.faiss``, ``chunk_metadata.jsonl``, ``export_report.json`` and
    ``index_build.json`` with every artifact SHA-256.
    """
    import faiss
    import numpy as np

    export_report = export_chunk_metadata(dataset_dir, output_dir)
    chunks = _read_jsonl(dataset_dir / "chunks.jsonl")
    if len(chunks) != export_report["chunk_count"]:
        raise ValueError("chunk count changed between export and index build")
    contents = [str(row["content"]) for row in chunks]
    embedder = _load_embedder(model_name)
    embeddings = np.asarray(
        embedder.encode(
            contents,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=False,
        ),
        dtype="float32",
    )
    if embeddings.shape[0] != len(chunks):
        raise ValueError("embedding count does not match chunk count")
    dim = int(embeddings.shape[1])
    embeddings_path = output_dir / "chunk_embeddings.npy"
    np.save(embeddings_path, embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    index_path = output_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    report = {
        **export_report,
        "embedding_model": model_name,
        "dim": dim,
        "normalized": normalize,
        "index_type": INDEX_TYPE,
        "embeddings_sha256": sha256_file(embeddings_path),
        "index_sha256": sha256_file(index_path),
    }
    write_json(output_dir / "index_build.json", report)
    return report
