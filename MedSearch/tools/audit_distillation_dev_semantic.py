"""Queue fixed-dev semantic near neighbors for the frozen training pools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from MedSearch.tools.audit_v2_semantic import _embeddings
from MedSearch.tools.check_distillation_inputs import _records


ROOT = Path(__file__).resolve().parents[2]
POOLS = ROOT / "MedSearch/data/distillation/pools_v2_20260926"
MODEL = ROOT / "MedicalGraphRAG/models/all-mpnet-base-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", type=Path, default=POOLS)
    args = parser.parse_args()
    pools = args.pools
    train = [row for name, path in (("rl_train", pools / "rl_train.parquet"), ("cold_start", pools / "cold_start.jsonl")) for row in _records(name, path)]
    dev = list(_records("dev", pools / "dev.parquet"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True).eval().to(device)
    train_vectors = _embeddings(model, tokenizer, train, device, 32).to(device)
    dev_vectors = _embeddings(model, tokenizer, dev, device, 32).to(device)
    output = pools / "dev_semantic_neighbors.jsonl"
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(train), 128):
            scores, indices = (train_vectors[start:start + 128] @ dev_vectors.T).max(dim=1)
            for offset, score in enumerate(scores.tolist()):
                if score < 0.92:
                    continue
                left, right = train[start + offset], dev[indices[offset].item()]
                handle.write(json.dumps({
                    "train_pool": left["pool"], "train_id": left["id"], "train_question": left["question"],
                    "dev_id": right["id"], "dev_question": right["question"],
                    "cosine": round(score, 6), "decision": "requires_pair_review",
                }, ensure_ascii=False) + "\n")
                count += 1
    with output.open("rb") as handle:
        output_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {"method": "local all-mpnet-base-v2 mean pooling; train-to-dev Top-1", "threshold": 0.92,
              "train_rows": len(train), "dev_rows": len(dev), "queued_pairs": count,
              "model_sha256": hashlib.sha256((MODEL / "pytorch_model.bin").read_bytes()).hexdigest(),
              "queue_sha256": output_hash, "scope": "near-duplicate screening, not semantic proof"}
    (pools / "dev_semantic_neighbors.manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
