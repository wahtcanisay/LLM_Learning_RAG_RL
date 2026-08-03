import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from importlib.metadata import version as distribution_version
from pathlib import Path

from medical_graphrag.data.io import sha256_directory, sha256_file


def package_version(
    package: str,
    resolver: Callable[[str], str] = distribution_version,
) -> str:
    return resolver(package)


def build_command(collection: Path, index: Path, threads: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(collection),
        "--index",
        str(index),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--stemmer",
        "porter",
    ]


def validate_export(collection: Path, report_path: Path) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("text_mode") != "abstract_only":
        raise ValueError("export must use abstract_only text mode")
    collection_file = collection / "chunks.jsonl"
    actual = sha256_file(collection_file)
    if actual != report.get("collection_sha256"):
        raise ValueError("collection SHA-256 mismatch")
    count = sum(1 for line in collection_file.open(encoding="utf-8") if line.strip())
    if count != report.get("chunk_count"):
        raise ValueError("collection count does not match export report")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    export_report = validate_export(args.collection, args.export_report)
    command = build_command(args.collection, args.index, args.threads)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started
    size = sum(path.stat().st_size for path in args.index.rglob("*") if path.is_file())

    java_version = subprocess.run(
        ["java", "-version"],
        text=True,
        capture_output=True,
        check=True,
    ).stderr.splitlines()[0]
    report = {
        "elapsed_seconds": elapsed,
        "index_bytes": size,
        "pyserini_version": package_version("pyserini"),
        "python_version": platform.python_version(),
        "java_version": java_version,
        "threads": args.threads,
        "stemmer": "porter",
        "command": command,
        "document_count": export_report["chunk_count"],
        "text_mode": export_report["text_mode"],
        "collection_sha256": export_report["collection_sha256"],
        "metadata_sha256": export_report["metadata_sha256"],
        "dataset_manifest_sha256": export_report["dataset_manifest_sha256"],
        "dataset_artifact_hashes": export_report["dataset_artifact_hashes"],
        "export_report_sha256": sha256_file(args.export_report),
        "index_sha256": sha256_directory(args.index),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
