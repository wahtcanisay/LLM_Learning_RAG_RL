import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    command = build_command(args.collection, args.index, args.threads)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started
    size = sum(path.stat().st_size for path in args.index.rglob("*") if path.is_file())

    import pyserini

    java_version = subprocess.run(
        ["java", "-version"],
        text=True,
        capture_output=True,
        check=True,
    ).stderr.splitlines()[0]
    report = {
        "elapsed_seconds": elapsed,
        "index_bytes": size,
        "pyserini_version": pyserini.__version__,
        "python_version": platform.python_version(),
        "java_version": java_version,
        "threads": args.threads,
        "stemmer": "porter",
        "command": command,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
