"""Build the LinearGraphRetriever graph index inside the container.

Run with the project venv python: `/opt/venv/bin/python`. Standalone script:
adds `src/` to ``sys.path``. Runs medical NER (BC5CDR) over the frozen chunks,
builds the Entity<->Sentence bridge and Entity-Passage edges, and persists the
igraph plus stores plus a hashed ``graph_build.json`` report.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from medical_graphrag.retrieval.graph import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NER_MODEL,
    GraphConfig,
    build_graph_index,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ner-model", default=DEFAULT_NER_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--passage-ratio", type=float, default=1.5)
    parser.add_argument("--passage-node-weight", type=float, default=0.5)
    parser.add_argument("--iteration-threshold", type=float, default=0.5)
    parser.add_argument("--top-k-sentence", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = GraphConfig(
        damping=args.damping,
        passage_ratio=args.passage_ratio,
        passage_node_weight=args.passage_node_weight,
        iteration_threshold=args.iteration_threshold,
        top_k_sentence=args.top_k_sentence,
        max_iterations=args.max_iterations,
        embedding_model=args.embedding_model,
        ner_model=args.ner_model,
    )
    build_graph_index(args.dataset_dir, args.output_dir, config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
