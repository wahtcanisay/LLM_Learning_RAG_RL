"""Embed frozen chunks and build the FAISS IndexFlatIP index inside the container.

Run inside the `llm-pytorch` container. Standalone script: adds `src/` to
``sys.path`` so it works even when the package is not pip-installed in the
container. Writes ``chunk_embeddings.npy``, ``index.faiss``,
``chunk_metadata.jsonl``, ``export_report.json`` and ``index_build.json`` under
the output directory.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from medical_graphrag.retrieval.dense import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    build_dense_index,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_dense_index(
        args.dataset_dir,
        args.output_dir,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
