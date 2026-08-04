"""Robustly download the sentence-transformers/all-mpnet-base-v2 model files.

The HF CDN connection on this network is unreliable and breaks mid-download, so
a one-shot `snapshot_download` frequently aborts with IncompleteRead. This script
downloads each file with a resume-capable, retrying loop (Range header + 1 MB
chunks) until every file is complete, then writes the plain model directory that
the Dense pipeline reads via `--embedding-model <path>`.

Run inside the `llm-pytorch` container. The target directory is gitignored
(`MedicalGraphRAG/models/`) so the ~430 MB weights are never committed.
"""
import argparse
import time
from pathlib import Path

import requests

REPO = "sentence-transformers/all-mpnet-base-v2"
SOURCES = (
    f"https://hf-mirror.com/{REPO}/resolve/main",
    f"https://huggingface.co/{REPO}/resolve/main",
)

FILES = (
    "config.json",
    "sentence_bert_config.json",
    "modules.json",
    "pytorch_model.bin",
    "1_Pooling/config.json",
    "tokenizer_config.json",
    "vocab.txt",
    "tokenizer.json",
    "special_tokens_map.json",
)

CHUNK = 1024 * 1024


def download_file(session: requests.Session, url: str, dest: Path) -> bool:
    """Download one file from one source. Returns True when the stream completed."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with session.get(url, headers=headers, stream=True, timeout=(30, 300)) as resp:
        if existing and resp.status_code == 416:
            # Already downloaded completely; finish without rewriting.
            tmp.replace(dest)
            return True
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {resp.status_code} for {url}")
        mode = "ab" if (existing and resp.status_code == 206) else "wb"
        with tmp.open(mode) as handle:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                if chunk:
                    handle.write(chunk)
        # Stream completed without exception => this chunk of the file is complete.
        tmp.replace(dest)
        return True


def download_with_retries(
    session: requests.Session,
    sources: tuple[str, ...],
    name: str,
    dest: Path,
) -> None:
    """Try every source with an infinite retry loop until the file is complete."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    while True:
        for base in sources:
            try:
                download_file(session, f"{base}/{name}", dest)
                return
            except (requests.RequestException, OSError, RuntimeError) as exc:
                print(
                    f"  retry {name} via {base}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-minutes", type=float, default=30.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + args.timeout_minutes * 60
    session = requests.Session()
    for name in FILES:
        dest = args.output_dir / name
        if dest.exists():
            print(f"skip   {name} (present)", flush=True)
            continue
        print(f"start  {name}", flush=True)
        while time.monotonic() < deadline:
            download_with_retries(session, SOURCES, name, dest)
            if dest.exists():
                break
        if not dest.exists():
            print(f"FAILED {name} within deadline", flush=True)
            return 1
        print(f"done   {name} ({dest.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
