import argparse
import urllib.request
from pathlib import Path

from medical_graphrag.data.benchmark import audit_benchmark, build_benchmark


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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
