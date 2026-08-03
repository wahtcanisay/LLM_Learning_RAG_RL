import argparse
import json
import platform
import sys
import urllib.request
from pathlib import Path

from medical_graphrag.data.benchmark import audit_benchmark, build_benchmark
from medical_graphrag.data.io import sha256_file
from medical_graphrag.evaluation.bm25 import evaluate_bm25_run
from medical_graphrag.retrieval.bm25 import (
    export_pyserini_collection,
    validate_frozen_dataset,
)


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
        index_report = json.loads(args.index_report.read_text(encoding="utf-8"))
        search_report = json.loads(args.search_report.read_text(encoding="utf-8"))
        dataset_manifest = validate_frozen_dataset(args.dataset_dir)
        evaluate_bm25_run(
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
                "dataset_manifest_sha256": sha256_file(
                    args.dataset_dir / "manifest.json"
                ),
                "dataset_artifact_hashes": dataset_manifest["artifact_hashes"],
            },
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
