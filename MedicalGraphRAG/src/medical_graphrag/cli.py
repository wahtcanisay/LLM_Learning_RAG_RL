import argparse
import json
import platform
import sys
import urllib.request
from pathlib import Path

from medical_graphrag.data.benchmark import audit_benchmark, build_benchmark
from medical_graphrag.data.io import sha256_file
from medical_graphrag.evaluation.bm25 import evaluate_bm25_run
from medical_graphrag.evaluation.dense import evaluate_dense_run
from medical_graphrag.evaluation.graph import evaluate_graph_run
from medical_graphrag.evaluation.hybrid import evaluate_hybrid_run
from medical_graphrag.evaluation.reranker import evaluate_reranker_run
from medical_graphrag.retrieval.bm25 import (
    export_pyserini_collection,
    validate_frozen_dataset,
)
from medical_graphrag.run_pipeline import RUNNERS


PQA_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
GROUND_TRUTH_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"
)


def fetch_pubmedqa(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, url in (("ori_pqal.json", PQA_URL), ("test_ground_truth.json", GROUND_TRUTH_URL)):
        destination = output_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medical-graphrag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch-pubmedqa")
    fetch.add_argument("--output-dir", type=Path, default=Path("data/raw/pubmedqa"))

    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--pubmedqa-dir", type=Path, required=True)
    audit.add_argument("--medrag-pubmed-dir", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--pubmedqa-dir", type=Path, required=True)
    build.add_argument("--medrag-pubmed-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)

    export = subparsers.add_parser("export-pyserini")
    export.add_argument("--dataset-dir", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate-bm25")
    evaluate.add_argument("--dataset-dir", type=Path, required=True)
    evaluate.add_argument("--metadata", type=Path, required=True)
    evaluate.add_argument("--rankings", type=Path, required=True)
    evaluate.add_argument("--index-report", type=Path, required=True)
    evaluate.add_argument("--search-report", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--git-commit", required=True)
    evaluate.add_argument("--docker-image", required=True)

    evaluate_dense = subparsers.add_parser("evaluate-dense")
    evaluate_dense.add_argument("--dataset-dir", type=Path, required=True)
    evaluate_dense.add_argument("--metadata", type=Path, required=True)
    evaluate_dense.add_argument("--rankings", type=Path, required=True)
    evaluate_dense.add_argument("--index-report", type=Path, required=True)
    evaluate_dense.add_argument("--search-report", type=Path, required=True)
    evaluate_dense.add_argument("--output-dir", type=Path, required=True)
    evaluate_dense.add_argument("--git-commit", required=True)
    evaluate_dense.add_argument("--docker-image", required=True)

    evaluate_hybrid = subparsers.add_parser("evaluate-hybrid")
    evaluate_hybrid.add_argument("--dataset-dir", type=Path, required=True)
    evaluate_hybrid.add_argument("--bm25-rankings", type=Path, required=True)
    evaluate_hybrid.add_argument("--dense-rankings", type=Path, required=True)
    evaluate_hybrid.add_argument("--metadata", type=Path, required=True)
    evaluate_hybrid.add_argument("--bm25-index-report", type=Path, required=True)
    evaluate_hybrid.add_argument("--bm25-search-report", type=Path, required=True)
    evaluate_hybrid.add_argument("--dense-index-report", type=Path, required=True)
    evaluate_hybrid.add_argument("--dense-search-report", type=Path, required=True)
    evaluate_hybrid.add_argument("--output-dir", type=Path, required=True)
    evaluate_hybrid.add_argument("--git-commit", required=True)
    evaluate_hybrid.add_argument("--docker-image", required=True)
    evaluate_hybrid.add_argument("--rrf-k", type=int, default=60)

    evaluate_graph = subparsers.add_parser("evaluate-graph")
    evaluate_graph.add_argument("--dataset-dir", type=Path, required=True)
    evaluate_graph.add_argument("--rankings", type=Path, required=True)
    evaluate_graph.add_argument("--index-report", type=Path, required=True)
    evaluate_graph.add_argument("--search-report", type=Path, required=True)
    evaluate_graph.add_argument("--output-dir", type=Path, required=True)
    evaluate_graph.add_argument("--git-commit", required=True)
    evaluate_graph.add_argument("--docker-image", required=True)

    evaluate_reranker = subparsers.add_parser("evaluate-reranker")
    evaluate_reranker.add_argument("--dataset-dir", type=Path, required=True)
    evaluate_reranker.add_argument("--rankings", type=Path, required=True)
    evaluate_reranker.add_argument("--reranker-report", type=Path, required=True)
    evaluate_reranker.add_argument("--output-dir", type=Path, required=True)
    evaluate_reranker.add_argument("--git-commit", required=True)
    evaluate_reranker.add_argument("--docker-image", required=True)

    run = subparsers.add_parser(
        "run",
        help="End-to-end pipeline: build → search → evaluate for one retriever.",
    )
    run.add_argument("retriever", choices=sorted(RUNNERS))
    run.add_argument("--dataset", required=True, help="frozen dataset name under data/processed/")
    run.add_argument("--git-commit", required=True)
    run.add_argument("--docker-image", default="pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel")
    run.add_argument("--top-k", type=int, default=100)
    run.add_argument("--rrf-k", type=int, default=60)
    run.add_argument("--top-n", type=int, default=50)
    run.add_argument("--profile", choices=["ep", "similarity"], default="similarity",
                     help="graph-document edge strategy (ep = entity edges only, "
                          "similarity = similarity soft edges)")
    run.add_argument("--sources", choices=["bd", "bdg", "bde"], default="bdg",
                     help="reranker-document candidate sources "
                          "(bd = bm25+dense, bdg = + graph-similarity, bde = + graph-ep)")

    graph_pairs = subparsers.add_parser(
        "graph-pairs",
        help="Write the Graph-EP vs Graph-Sim paired case comparison (P1-5).",
    )
    graph_pairs.add_argument("--dataset", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "fetch-pubmedqa":
        fetch_pubmedqa(args.output_dir)
        return 0
    if args.command == "audit":
        audit_benchmark(args.config, args.pubmedqa_dir, args.medrag_pubmed_dir, args.output_dir)
        return 0
    if args.command == "build":
        build_benchmark(args.config, args.pubmedqa_dir, args.medrag_pubmed_dir, args.output_dir)
        return 0
    if args.command == "export-pyserini":
        export_pyserini_collection(args.dataset_dir, args.output_dir)
        return 0
    if args.command == "evaluate-bm25":
        return _evaluate_command(args, evaluate_bm25_run)
    if args.command == "evaluate-dense":
        return _evaluate_command(args, evaluate_dense_run)
    if args.command == "evaluate-hybrid":
        return _evaluate_hybrid_command(args)
    if args.command == "evaluate-graph":
        return _evaluate_graph_command(args)
    if args.command == "evaluate-reranker":
        return _evaluate_reranker_command(args)
    if args.command == "run":
        return _run_command(args)
    if args.command == "graph-pairs":
        from medical_graphrag.run_pipeline import run_graph_pair_cases

        run_graph_pair_cases(args.dataset)
        return 0
    return 2


def _run_command(args: argparse.Namespace) -> int:
    """Wiring for `run`: dispatch to the pipeline runner for one retriever.

    Each runner accepts only the pipeline parameters it uses; build the kwargs
    per-retriever so an unused flag (e.g. ``rrf_k`` for bm25) never leaks.
    """
    runner = RUNNERS[args.retriever]
    kwargs: dict[str, object] = {
        "git_commit": args.git_commit,
        "docker_image": args.docker_image,
    }
    if args.retriever in ("bm25", "dense", "graph", "hybrid", "reranker",
                          "bm25-document", "dense-document", "hybrid-document",
                          "graph-document", "reranker-document"):
        kwargs["top_k"] = args.top_k
    if args.retriever in ("hybrid", "hybrid-document"):
        kwargs["rrf_k"] = args.rrf_k
    if args.retriever in ("reranker", "reranker-document"):
        kwargs["top_n"] = args.top_n
    if args.retriever == "graph-document":
        kwargs["profile"] = args.profile
    if args.retriever == "reranker-document":
        kwargs["sources"] = args.sources
    runner(args.dataset, **kwargs)
    return 0


def _evaluate_reranker_command(args: argparse.Namespace) -> int:
    """Wiring for evaluate-reranker: bind the reranker run report + dataset."""
    reranker_report = json.loads(args.reranker_report.read_text(encoding="utf-8"))
    evaluate_reranker_run(
        args.dataset_dir,
        args.rankings,
        args.output_dir,
        run_context={
            "git_commit": args.git_commit,
            "host_platform": platform.platform(),
            "host_python_version": platform.python_version(),
            "docker_image": args.docker_image,
            "evaluation_command": sys.argv,
            "reranker": reranker_report,
            "dataset_manifest_sha256": sha256_file(
                args.dataset_dir / "manifest.json"
            ),
        },
    )
    return 0


def _evaluate_graph_command(args: argparse.Namespace) -> int:
    """Wiring for evaluate-graph: bind the graph index report and rankings."""
    index_report = json.loads(args.index_report.read_text(encoding="utf-8"))
    search_report = json.loads(args.search_report.read_text(encoding="utf-8"))
    dataset_manifest = validate_frozen_dataset(args.dataset_dir)
    evaluate_graph_run(
        args.dataset_dir,
        args.rankings,
        args.output_dir,
        run_context={
            "git_commit": args.git_commit,
            "host_platform": platform.platform(),
            "host_python_version": platform.python_version(),
            "docker_image": args.docker_image,
            "evaluation_command": sys.argv,
            "index": index_report,
            "search": search_report,
            "index_report_sha256": sha256_file(args.index_report),
            "dataset_manifest_sha256": sha256_file(args.dataset_dir / "manifest.json"),
            "dataset_artifact_hashes": dataset_manifest["artifact_hashes"],
        },
    )
    return 0


def _evaluate_command(args: argparse.Namespace, evaluator) -> int:
    """Shared wiring for evaluate-bm25 / evaluate-dense."""
    index_report = json.loads(args.index_report.read_text(encoding="utf-8"))
    search_report = json.loads(args.search_report.read_text(encoding="utf-8"))
    dataset_manifest = validate_frozen_dataset(args.dataset_dir)
    evaluator(
        args.dataset_dir,
        args.metadata,
        args.rankings,
        args.output_dir,
        run_context={
            "git_commit": args.git_commit,
            "host_platform": platform.platform(),
            "host_python_version": platform.python_version(),
            "docker_image": args.docker_image,
            "evaluation_command": sys.argv,
            "index": index_report,
            "search": search_report,
            "index_report_sha256": sha256_file(args.index_report),
            "dataset_manifest_sha256": sha256_file(
                args.dataset_dir / "manifest.json"
            ),
            "dataset_artifact_hashes": dataset_manifest["artifact_hashes"],
        },
    )
    return 0


def _evaluate_hybrid_command(args: argparse.Namespace) -> int:
    """Wiring for evaluate-hybrid: bind both source retrievers' audit reports."""
    evaluate_hybrid_run(
        args.dataset_dir,
        args.bm25_rankings,
        args.dense_rankings,
        args.metadata,
        args.output_dir,
        k=args.rrf_k,
        run_context={
            "git_commit": args.git_commit,
            "host_platform": platform.platform(),
            "host_python_version": platform.python_version(),
            "docker_image": args.docker_image,
            "evaluation_command": sys.argv,
            "bm25": {
                "index": json.loads(
                    args.bm25_index_report.read_text(encoding="utf-8")
                ),
                "search": json.loads(
                    args.bm25_search_report.read_text(encoding="utf-8")
                ),
                "index_report_sha256": sha256_file(args.bm25_index_report),
            },
            "dense": {
                "index": json.loads(
                    args.dense_index_report.read_text(encoding="utf-8")
                ),
                "search": json.loads(
                    args.dense_search_report.read_text(encoding="utf-8")
                ),
                "index_report_sha256": sha256_file(args.dense_index_report),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
