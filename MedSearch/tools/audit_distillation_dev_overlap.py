"""Queue possible train/dev paraphrases in the frozen distillation pools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from MedSearch.tools.audit_v2_lexical import find_nearest_pairs
from MedSearch.tools.check_distillation_inputs import _records


ROOT = Path(__file__).resolve().parents[2]
POOLS = ROOT / "MedSearch/data/distillation/pools_v2_20260926"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", type=Path, default=POOLS)
    args = parser.parse_args()
    pools = args.pools
    candidate_paths = {"rl_train": pools / "rl_train.parquet", "cold_start": pools / "cold_start.jsonl"}
    dev_path = pools / "dev.parquet"
    candidates = [row for name, path in candidate_paths.items() for row in _records(name, path)]
    dev = list(_records("dev", dev_path))
    pairs = find_nearest_pairs([row["question"] for row in candidates], [row["question"] for row in dev], 0.8)
    output = pools / "dev_lexical_neighbors.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            left, right = candidates[pair["candidate_index"]], dev[pair["match_index"]]
            handle.write(json.dumps({
                "train_pool": left["pool"], "train_id": left["id"], "train_question": left["question"],
                "dev_id": right["id"], "dev_question": right["question"], "cosine": pair["cosine"],
                "decision": "requires_pair_review",
            }, ensure_ascii=False) + "\n")
    with output.open("rb") as handle:
        output_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {"method": "TF-IDF char_wb 3-5 grams, train-to-dev Top-1", "threshold": 0.8,
              "train_rows": len(candidates), "dev_rows": len(dev), "queued_pairs": len(pairs),
              "queue_sha256": output_hash, "scope": "near-duplicate screening, not semantic proof"}
    (pools / "dev_lexical_neighbors.manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
