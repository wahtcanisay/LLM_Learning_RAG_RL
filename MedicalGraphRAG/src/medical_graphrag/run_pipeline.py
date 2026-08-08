"""End-to-end retrieval pipelines: one call per retriever over a frozen dataset.

This is the unified evaluate interface — the successor to the per-retriever
scripts under ``scripts/`` (build_* / search_* / rerank_candidates). Each
``run_*`` function drives the full  build → search → evaluate  chain for one
retriever and one frozen dataset, in-process, preserving the same audit chain
(hash-bound reports consumed by ``evaluate_*_run``).

The dataset argument is a name under ``data/processed/`` (e.g. ``hotpotqa_v1``).
``root`` defaults to the project root (parents[1] of this file); override for
tests.
"""
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file
from medical_graphrag.evaluation.bm25 import (
    evaluate_bm25_document_run,
    evaluate_bm25_run,
)
from medical_graphrag.evaluation.dense import (
    evaluate_dense_document_run,
    evaluate_dense_run,
)
from medical_graphrag.evaluation.graph import evaluate_graph_run, write_graph_pair_cases
from medical_graphrag.evaluation.hybrid import (
    evaluate_hybrid_document_run,
    evaluate_hybrid_run,
)
from medical_graphrag.evaluation.reranker import evaluate_reranker_run
from medical_graphrag.retrieval.bm25 import (
    export_document_collection,
    export_pyserini_collection,
    validate_frozen_dataset,
)
from medical_graphrag.retrieval.dense import (
    DEFAULT_EMBEDDING_MODEL as DENSE_DEFAULT_EMBEDDING_MODEL,
    build_dense_document_index,
    build_dense_index,
)
from medical_graphrag.retrieval.document_embeddings import ensure_document_embeddings
from medical_graphrag.retrieval.graph import (
    DEFAULT_EMBEDDING_MODEL as GRAPH_DEFAULT_EMBEDDING_MODEL,
)
from medical_graphrag.retrieval.graph import (
    DEFAULT_NER_MODEL,
    GraphBuildConfig,
    GraphConfig,
    build_graph_index,
)
from medical_graphrag.retrieval.rerank import run_rerank
from medical_graphrag.retrieval.rerank_document import run_rerank_document
from medical_graphrag.retrieval.reranker import DEFAULT_RERANKER_MODEL
from medical_graphrag.retrieval.search_bm25 import (
    build_lucene_document_index,
    build_lucene_index,
    run_search as run_bm25_search,
)
from medical_graphrag.retrieval.search_dense import run_search as run_dense_search
from medical_graphrag.retrieval.search_document import (
    run_bm25_document_search,
    run_dense_document_search,
)
from medical_graphrag.retrieval.search_graph import run_search as run_graph_search

ROOT = Path(__file__).resolve().parents[2]
DOCKER_IMAGE_DEFAULT = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel"


def _dataset_dir(dataset: str, root: Path = ROOT) -> Path:
    return root / "data" / "processed" / dataset


def _build_context(
    *,
    git_commit: str,
    docker_image: str,
    command: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "git_commit": git_commit,
        "host_platform": platform.platform(),
        "host_python_version": platform.python_version(),
        "docker_image": docker_image,
        "evaluation_command": command,
    }
    if extra:
        context.update(extra)
    return context


def run_bm25(
    dataset: str,
    *,
    git_commit: str,
    docker_image: str = DOCKER_IMAGE_DEFAULT,
    root: Path = ROOT,
    top_k: int = 100,
    k1: float = 0.9,
    b: float = 0.4,
    threads: int = 8,
) -> dict[str, Any]:
    """Build the Lucene index, search, and evaluate BM25 on ``dataset``."""
    dataset_dir = _dataset_dir(dataset, root)
    stage = dataset_dir.parent / f"../outputs/{dataset}/bm25"
    # outputs/<dataset>/bm25
    out_dir = root / "outputs" / dataset / "bm25"
    idx_dir = root / "indexes" / dataset / "bm25"
    exp_dir = root / "experiments" / dataset / "bm25"

    export_pyserini_collection(dataset_dir, out_dir)
    build_lucene_index(
        collection=out_dir / "collection",
        index=idx_dir,
        report=out_dir / "index_build.json",
        export_report=out_dir / "export_report.json",
        threads=threads,
    )
    run_bm25_search(
        index=idx_dir,
        index_report=out_dir / "index_build.json",
        questions=dataset_dir / "questions.jsonl",
        metadata=out_dir / "chunk_metadata.jsonl",
        output=out_dir / "raw_rankings.jsonl",
        report=out_dir / "search_run.json",
        top_k=top_k,
        k1=k1,
        b=b,
    )
    context = _build_context(
        git_commit=git_commit,
        docker_image=docker_image,
        command=["cli", "run", "bm25", "--dataset", dataset],
        extra={
            "index": json.loads((out_dir / "index_build.json").read_text(encoding="utf-8")),
            "search": json.loads((out_dir / "search_run.json").read_text(encoding="utf-8")),
            "index_report_sha256": sha256_file(out_dir / "index_build.json"),
            "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        },
    )
    return evaluate_bm25_run(
        dataset_dir,
        out_dir / "chunk_metadata.jsonl",
        out_dir / "raw_rankings.jsonl",
        exp_dir,
        run_context=context,
    )


def run_dense(
    dataset: str,
    *,
    git_commit: str,
    docker_image: str = DOCKER_IMAGE_DEFAULT,
    root: Path = ROOT,
    top_k: int = 100,
    embedding_model: str = DENSE_DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Build the FAISS index, search, and evaluate Dense on ``dataset``."""
    dataset_dir = _dataset_dir(dataset, root)
    out_dir = root / "outputs" / dataset / "dense_abstract_only"
    exp_dir = root / "experiments" / dataset / "dense_abstract_only"

    build_dense_index(
        dataset_dir,
        out_dir,
        model_name=embedding_model,
        batch_size=batch_size,
    )
    run_dense_search(
        index=out_dir / "index.faiss",
        embeddings=out_dir / "chunk_embeddings.npy",
        index_report=out_dir / "index_build.json",
        questions=dataset_dir / "questions.jsonl",
        metadata=out_dir / "chunk_metadata.jsonl",
        output=out_dir / "raw_rankings.jsonl",
        report=out_dir / "search_run.json",
        top_k=top_k,
        embedding_model=embedding_model,
    )
    context = _build_context(
        git_commit=git_commit,
        docker_image=docker_image,
        command=["cli", "run", "dense", "--dataset", dataset],
        extra={
            "index": json.loads((out_dir / "index_build.json").read_text(encoding="utf-8")),
            "search": json.loads((out_dir / "search_run.json").read_text(encoding="utf-8")),
            "index_report_sha256": sha256_file(out_dir / "index_build.json"),
            "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        },
    )
    return evaluate_dense_run(
        dataset_dir,
        out_dir / "chunk_metadata.jsonl",
        out_dir / "raw_rankings.jsonl",
        exp_dir,
        run_context=context,
    )


def run_graph(
    dataset: str,
    *,
    git_commit: str,
    docker_image: str = DOCKER_IMAGE_DEFAULT,
    root: Path = ROOT,
    top_k: int = 100,
    config: GraphConfig | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Build the graph index, search, and evaluate Graph on ``dataset``."""
    dataset_dir = _dataset_dir(dataset, root)
    idx_dir = root / "indexes" / dataset / "graph_abstract_only"
    out_dir = root / "outputs" / dataset / "graph_abstract_only"
    exp_dir = root / "experiments" / dataset / "graph_abstract_only"

    build_graph_index(dataset_dir, idx_dir, config=config, batch_size=batch_size)
    run_graph_search(
        index=idx_dir,
        index_report=idx_dir / "graph_build.json",
        questions=dataset_dir / "questions.jsonl",
        chunks=dataset_dir / "chunks.jsonl",
        output=out_dir / "raw_rankings.jsonl",
        report=out_dir / "search_run.json",
        top_k=top_k,
        ner_model=(config.ner_model if config else DEFAULT_NER_MODEL),
        embedding_model=(config.embedding_model if config else GRAPH_DEFAULT_EMBEDDING_MODEL),
    )
    context = _build_context(
        git_commit=git_commit,
        docker_image=docker_image,
        command=["cli", "run", "graph", "--dataset", dataset],
        extra={
            "index": json.loads((idx_dir / "graph_build.json").read_text(encoding="utf-8")),
            "search": json.loads((out_dir / "search_run.json").read_text(encoding="utf-8")),
            "index_report_sha256": sha256_file(idx_dir / "graph_build.json"),
            "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        },
    )
    return evaluate_graph_run(
        dataset_dir,
        out_dir / "raw_rankings.jsonl",
        exp_dir,
        run_context=context,
    )


def run_hybrid(
    dataset: str,
    *,
    git_commit: str,
    docker_image: str = DOCKER_IMAGE_DEFAULT,
    root: Path = ROOT,
    top_k: int = 100,
    rrf_k: int = 60,
) -> dict[str, Any]:
    """Run BM25 + Dense, then fuse with RRF and evaluate Hybrid on ``dataset``."""
    run_bm25(dataset, git_commit=git_commit, docker_image=docker_image, root=root, top_k=top_k)
    run_dense(dataset, git_commit=git_commit, docker_image=docker_image, root=root, top_k=top_k)

    dataset_dir = _dataset_dir(dataset, root)
    bm25_dir = root / "outputs" / dataset / "bm25"
    dense_dir = root / "outputs" / dataset / "dense_abstract_only"
    exp_dir = root / "experiments" / dataset / "hybrid_rrf"
    context = _build_context(
        git_commit=git_commit,
        docker_image=docker_image,
        command=["cli", "run", "hybrid", "--dataset", dataset],
        extra={
            "bm25": {
                "index": json.loads((bm25_dir / "index_build.json").read_text(encoding="utf-8")),
                "search": json.loads((bm25_dir / "search_run.json").read_text(encoding="utf-8")),
                "index_report_sha256": sha256_file(bm25_dir / "index_build.json"),
            },
            "dense": {
                "index": json.loads(
                    (dense_dir / "index_build.json").read_text(encoding="utf-8")
                ),
                "search": json.loads(
                    (dense_dir / "search_run.json").read_text(encoding="utf-8")
                ),
                "index_report_sha256": sha256_file(dense_dir / "index_build.json"),
            },
        },
    )
    return evaluate_hybrid_run(
        dataset_dir,
        bm25_dir / "raw_rankings.jsonl",
        dense_dir / "raw_rankings.jsonl",
        dense_dir / "chunk_metadata.jsonl",
        exp_dir,
        k=rrf_k,
        run_context=context,
    )


def run_reranker(
    dataset: str,
    *,
    git_commit: str,
    docker_image: str = DOCKER_IMAGE_DEFAULT,
    root: Path = ROOT,
    top_k: int = 100,
    top_n: int = 50,
    model: str = DEFAULT_RERANKER_MODEL,
) -> dict[str, Any]:
    """Run BM25 + Dense + Graph, rerank the union with Qwen3-Reranker, evaluate."""
    run_bm25(dataset, git_commit=git_commit, docker_image=docker_image, root=root, top_k=top_k)
    run_dense(dataset, git_commit=git_commit, docker_image=docker_image, root=root, top_k=top_k)
    run_graph(dataset, git_commit=git_commit, docker_image=docker_image, root=root, top_k=top_k)

    dataset_dir = _dataset_dir(dataset, root)
    bm25_dir = root / "outputs" / dataset / "bm25"
    dense_dir = root / "outputs" / dataset / "dense_abstract_only"
    graph_out = root / "outputs" / dataset / "graph_abstract_only"
    graph_idx = root / "indexes" / dataset / "graph_abstract_only"
    rr_dir = root / "outputs" / dataset / "reranker_qwen"
    exp_dir = root / "experiments" / dataset / "reranker_qwen"

    rerank_report = run_rerank(
        bm25_rankings=bm25_dir / "raw_rankings.jsonl",
        dense_rankings=dense_dir / "raw_rankings.jsonl",
        graph_rankings=graph_out / "raw_rankings.jsonl",
        bm25_search_report=bm25_dir / "search_run.json",
        dense_search_report=dense_dir / "search_run.json",
        graph_search_report=graph_out / "search_run.json",
        questions=dataset_dir / "questions.jsonl",
        documents=dataset_dir / "documents.jsonl",
        chunks=dataset_dir / "chunks.jsonl",
        dataset_manifest=dataset_dir / "manifest.json",
        output=rr_dir / "reranked.jsonl",
        report=rr_dir / "rerank_report.json",
        top_n=top_n,
        model=model,
    )
    context = _build_context(
        git_commit=git_commit,
        docker_image=docker_image,
        command=["cli", "run", "reranker", "--dataset", dataset],
        extra={
            "reranker": rerank_report,
            "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        },
    )
    return evaluate_reranker_run(
        dataset_dir,
        rr_dir / "reranked.jsonl",
        exp_dir,
        run_context=context,
    )


def _document_embedding_model() -> str:
    """Single embedding model shared by every document-level consumer (P0-8)."""
    return DENSE_DEFAULT_EMBEDDING_MODEL


def _document_artifact_dir(dataset: str, root: Path) -> Path:
    return root / "outputs" / dataset / "document_embeddings_v1"


def run_bm25_document(
    dataset: str,
    *,
    git_commit: str,
    docker_image: str = DOCKER_IMAGE_DEFAULT,
    root: Path = ROOT,
    top_k: int = 100,
    k1: float = 0.9,
    b: float = 0.4,
    threads: int = 8,
) -> dict[str, Any]:
    """BM25-document: export full documents to Lucene, search, evaluate."""
    dataset_dir = _dataset_dir(dataset, root)
    out_dir = root / "outputs" / dataset / "bm25_document_v1"
    idx_dir = root / "indexes" / dataset / "bm25_document_v1"
    exp_dir = root / "experiments" / dataset / "bm25_document_v1"
    export_document_collection(dataset_dir, out_dir)
    build_lucene_document_index(
        collection=out_dir / "collection",
        index=idx_dir,
        report=out_dir / "index_build.json",
        export_report=out_dir / "export_report.json",
        threads=threads,
    )
    run_bm25_document_search(
        index=idx_dir,
        metadata=out_dir / "document_metadata.jsonl",
        index_report=out_dir / "index_build.json",
        questions=dataset_dir / "questions.jsonl",
        output=out_dir / "raw_rankings.jsonl",
        report=out_dir / "search_run.json",
        top_k=top_k,
        k1=k1,
        b=b,
    )
    context = _build_context(
        git_commit=git_commit,
        docker_image=docker_image,
        command=["cli", "run", "bm25-document", "--dataset", dataset],
        extra={
            "index": json.loads((out_dir / "index_build.json").read_text(encoding="utf-8")),
            "search": json.loads((out_dir / "search_run.json").read_text(encoding="utf-8")),
            "index_report_sha256": sha256_file(out_dir / "index_build.json"),
            "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        },
    )
    return evaluate_bm25_document_run(
        dataset_dir, out_dir / "raw_rankings.jsonl", exp_dir, run_context=context
    )


def run_dense_document(
    dataset: str,
    *,
    git_commit: str,
    docker_image: str = DOCKER_IMAGE_DEFAULT,
    root: Path = ROOT,
    top_k: int = 100,
    batch_size: int = 64,
    overlap_tokens: int = 32,
) -> dict[str, Any]:
    """Dense-document: consume the frozen embedding artifact, FAISS, search, evaluate."""
    dataset_dir = _dataset_dir(dataset, root)
    artifact_dir = _document_artifact_dir(dataset, root)
    out_dir = root / "outputs" / dataset / "dense_document_v1"
    exp_dir = root / "experiments" / dataset / "dense_document_v1"
    model = _document_embedding_model()
    ensure_document_embeddings(
        dataset_dir, artifact_dir, model_name=model,
        overlap_tokens=overlap_tokens, batch_size=batch_size,
    )
    build_dense_document_index(
        dataset_dir, out_dir, document_embeddings_dir=artifact_dir,
        batch_size=batch_size,
    )
    run_dense_document_search(
        index=out_dir / "index.faiss",
        metadata=out_dir / "document_metadata.jsonl",
        index_report=out_dir / "index_build.json",
        questions=dataset_dir / "questions.jsonl",
        output=out_dir / "raw_rankings.jsonl",
        report=out_dir / "search_run.json",
        top_k=top_k,
        embedding_model=model,
    )
    context = _build_context(
        git_commit=git_commit,
        docker_image=docker_image,
        command=["cli", "run", "dense-document", "--dataset", dataset],
        extra={
            "index": json.loads((out_dir / "index_build.json").read_text(encoding="utf-8")),
            "search": json.loads((out_dir / "search_run.json").read_text(encoding="utf-8")),
            "index_report_sha256": sha256_file(out_dir / "index_build.json"),
            "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        },
    )
    return evaluate_dense_document_run(
        dataset_dir, out_dir / "raw_rankings.jsonl", exp_dir, run_context=context
    )


def run_graph_document(
    dataset: str,
    *,
    git_commit: str,
    docker_image: str = DOCKER_IMAGE_DEFAULT,
    root: Path = ROOT,
    profile: str = "similarity",
    top_k: int = 100,
    batch_size: int = 64,
    overlap_tokens: int = 32,
) -> dict[str, Any]:
    """Graph-document: EP (no passage-passage edges) or Similarity (kNN soft edges)."""
    if profile not in {"ep", "similarity"}:
        raise ValueError("graph-document profile must be 'ep' or 'similarity'")
    dataset_dir = _dataset_dir(dataset, root)
    artifact_dir = _document_artifact_dir(dataset, root)
    name = f"graph_document_{profile}_v1"
    idx_dir = root / "indexes" / dataset / name
    out_dir = root / "outputs" / dataset / name
    exp_dir = root / "experiments" / dataset / name
    model = _document_embedding_model()
    ensure_document_embeddings(
        dataset_dir, artifact_dir, model_name=model,
        overlap_tokens=overlap_tokens, batch_size=batch_size,
    )
    build_config = GraphBuildConfig(
        retrieval_unit="document",
        passage_edge_mode="similarity" if profile == "similarity" else "none",
        embedding_model=model,
        ner_model=DEFAULT_NER_MODEL,
    )
    build_graph_index(
        dataset_dir, idx_dir, build_config=build_config, batch_size=batch_size,
        document_embeddings_dir=artifact_dir,
    )
    run_graph_search(
        index=idx_dir,
        index_report=idx_dir / "graph_build.json",
        questions=dataset_dir / "questions.jsonl",
        chunks=dataset_dir / "chunks.jsonl",
        output=out_dir / "raw_rankings.jsonl",
        report=out_dir / "search_run.json",
        top_k=top_k,
        ner_model=DEFAULT_NER_MODEL,
        embedding_model=model,
    )
    context = _build_context(
        git_commit=git_commit,
        docker_image=docker_image,
        command=["cli", "run", "graph-document", "--dataset", dataset, "--profile", profile],
        extra={
            "index": json.loads((idx_dir / "graph_build.json").read_text(encoding="utf-8")),
            "search": json.loads((out_dir / "search_run.json").read_text(encoding="utf-8")),
            "index_report_sha256": sha256_file(idx_dir / "graph_build.json"),
            "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        },
    )
    return evaluate_graph_run(
        dataset_dir, out_dir / "raw_rankings.jsonl", exp_dir, run_context=context
    )


def run_hybrid_document(
    dataset: str,
    *,
    git_commit: str,
    docker_image: str = DOCKER_IMAGE_DEFAULT,
    root: Path = ROOT,
    top_k: int = 100,
    rrf_k: int = 60,
) -> dict[str, Any]:
    """Hybrid-document: RRF-fuse BM25-document + Dense-document, evaluate."""
    run_bm25_document(dataset, git_commit=git_commit, docker_image=docker_image,
                      root=root, top_k=top_k)
    run_dense_document(dataset, git_commit=git_commit, docker_image=docker_image,
                       root=root, top_k=top_k)
    dataset_dir = _dataset_dir(dataset, root)
    bm25_dir = root / "outputs" / dataset / "bm25_document_v1"
    dense_dir = root / "outputs" / dataset / "dense_document_v1"
    exp_dir = root / "experiments" / dataset / "hybrid_document_v1"
    context = _build_context(
        git_commit=git_commit,
        docker_image=docker_image,
        command=["cli", "run", "hybrid-document", "--dataset", dataset],
        extra={
            "bm25": {
                "index": json.loads((bm25_dir / "index_build.json").read_text(encoding="utf-8")),
                "search": json.loads((bm25_dir / "search_run.json").read_text(encoding="utf-8")),
                "index_report_sha256": sha256_file(bm25_dir / "index_build.json"),
            },
            "dense": {
                "index": json.loads((dense_dir / "index_build.json").read_text(encoding="utf-8")),
                "search": json.loads((dense_dir / "search_run.json").read_text(encoding="utf-8")),
                "index_report_sha256": sha256_file(dense_dir / "index_build.json"),
            },
            "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        },
    )
    return evaluate_hybrid_document_run(
        dataset_dir,
        bm25_dir / "raw_rankings.jsonl",
        dense_dir / "raw_rankings.jsonl",
        exp_dir,
        k=rrf_k,
        run_context=context,
    )


def run_reranker_document(
    dataset: str,
    *,
    git_commit: str,
    docker_image: str = DOCKER_IMAGE_DEFAULT,
    root: Path = ROOT,
    top_k: int = 100,
    top_n: int = 50,
    sources: str = "bdg",
    model: str = DEFAULT_RERANKER_MODEL,
) -> dict[str, Any]:
    """Document-level Hybrid2: rerank a candidate-union with Qwen3-Reranker.

    ``sources`` selects the candidate sources: ``bd`` (BM25-doc + Dense-doc),
    ``bdg`` (adds Graph-Similarity-doc — tests whether the soft-edge graph adds
    recall a reranker can rescue), ``bde`` (adds Graph-EP-doc). Candidates are
    already document-level, so no chunk→doc collapse is needed.
    """
    if sources not in {"bd", "bdg", "bde"}:
        raise ValueError("reranker-document sources must be one of bd/bdg/bde")
    base = root / "outputs" / dataset

    def _ready(name: str) -> bool:
        return (base / name / "raw_rankings.jsonl").exists() and (
            base / name / "search_run.json"
        ).exists()

    if not _ready("bm25_document_v1"):
        run_bm25_document(dataset, git_commit=git_commit, docker_image=docker_image,
                          root=root, top_k=top_k)
    if not _ready("dense_document_v1"):
        run_dense_document(dataset, git_commit=git_commit, docker_image=docker_image,
                           root=root, top_k=top_k)
    if sources in {"bdg", "bde"}:
        profile = "similarity" if sources == "bdg" else "ep"
        graph_name = f"graph_document_{profile}_v1"
        if not _ready(graph_name):
            run_graph_document(dataset, git_commit=git_commit, docker_image=docker_image,
                               root=root, profile=profile, top_k=top_k)

    dataset_dir = _dataset_dir(dataset, root)
    name = f"reranker_document_{sources}_v1"
    rr_dir = root / "outputs" / dataset / name
    exp_dir = root / "experiments" / dataset / name
    sources_paths = {
        "bm25": base / "bm25_document_v1" / "raw_rankings.jsonl",
        "dense": base / "dense_document_v1" / "raw_rankings.jsonl",
    }
    source_reports = {
        "bm25": base / "bm25_document_v1" / "search_run.json",
        "dense": base / "dense_document_v1" / "search_run.json",
    }
    if sources in {"bdg", "bde"}:
        profile = "similarity" if sources == "bdg" else "ep"
        sources_paths["graph"] = base / f"graph_document_{profile}_v1" / "raw_rankings.jsonl"
        source_reports["graph"] = base / f"graph_document_{profile}_v1" / "search_run.json"

    rerank_report = run_rerank_document(
        sources=sources_paths,
        source_reports=source_reports,
        questions=dataset_dir / "questions.jsonl",
        documents=dataset_dir / "documents.jsonl",
        dataset_manifest=dataset_dir / "manifest.json",
        output=rr_dir / "reranked.jsonl",
        report=rr_dir / "rerank_report.json",
        top_n=top_n,
        model=model,
    )
    context = _build_context(
        git_commit=git_commit,
        docker_image=docker_image,
        command=["cli", "run", "reranker-document", "--dataset", dataset,
                 "--sources", sources],
        extra={
            "reranker": rerank_report,
            "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        },
    )
    return evaluate_reranker_run(
        dataset_dir, rr_dir / "reranked.jsonl", exp_dir, run_context=context
    )


def run_graph_pair_cases(
    dataset: str,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Write the Graph-EP vs Graph-Sim paired case comparison (P1-5)."""
    dataset_dir = _dataset_dir(dataset, root)
    ep_rankings = root / "outputs" / dataset / "graph_document_ep_v1" / "raw_rankings.jsonl"
    sim_rankings = root / "outputs" / dataset / "graph_document_similarity_v1" / "raw_rankings.jsonl"
    output = root / "experiments" / dataset / "graph_ep_vs_sim_v1" / "paired_cases.jsonl"
    return write_graph_pair_cases(dataset_dir, ep_rankings, sim_rankings, output)


RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "bm25": run_bm25,
    "dense": run_dense,
    "graph": run_graph,
    "hybrid": run_hybrid,
    "reranker": run_reranker,
    "bm25-document": run_bm25_document,
    "dense-document": run_dense_document,
    "graph-document": run_graph_document,
    "hybrid-document": run_hybrid_document,
    "reranker-document": run_reranker_document,
}
