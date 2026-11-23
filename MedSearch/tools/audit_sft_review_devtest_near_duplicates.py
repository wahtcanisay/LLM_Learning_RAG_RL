"""Screen reviewed SFT priorities for near duplicates against frozen dev/test."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from MedSearch.tools.audit_v2_semantic import _embeddings
from MedSearch.tools.check_distillation_inputs import _records


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "MedSearch/data/distillation/sft_environment_review_20260929_full"
ACCEPTED = ROOT / "MedSearch/data/distillation/teacher_cold1000_20260929_answered/accepted.jsonl"
DEV = ROOT / "MedSearch/data/distillation/pools_v2_20260926_devclean/dev.parquet"
TEST = ROOT / "MIRAGE/benchmark.json"
MODEL = ROOT / "MedicalGraphRAG/models/all-mpnet-base-v2"
THRESHOLD = 0.92


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    accepted = {row["question_id"]: row for row in read_jsonl(ACCEPTED)}
    reviews = [
        row
        for batch in range(1, 5)
        for row in read_jsonl(REVIEW / f"review_batch_{batch:02d}.jsonl")
    ]
    pilot = read_jsonl(ROOT / "MedSearch/data/distillation/sft_environment_review_20260929_pilot/review.jsonl")
    reviews += pilot
    candidates = []
    for review in reviews:
        if review["sft_recommendation"] != "priority":
            continue
        row = accepted[review["question_id"]]
        candidates.append({"id": review["question_id"], "question": row["question"], "pool": "sft_priority"})
    if len({row["id"] for row in candidates}) != len(candidates):
        raise ValueError("priority question IDs are not unique")

    history = list(_records("frozen_dev", DEV)) + list(_records("frozen_final_test", TEST))
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True).eval().to(device)
    candidate_vectors = _embeddings(model, tokenizer, candidates, device, 32)
    history_vectors = _embeddings(model, tokenizer, history, device, 32)
    del model

    matches = []
    if candidates:
        scores, indices = (candidate_vectors @ history_vectors.T).max(dim=1)
        for i, score in enumerate(scores.tolist()):
            if score >= THRESHOLD:
                match = history[indices[i].item()]
                matches.append({
                    "kind": "frozen_boundary",
                    "candidate_id": candidates[i]["id"],
                    "candidate_question": candidates[i]["question"],
                    "match_id": match["id"],
                    "match_pool": match["pool"],
                    "match_question": match["question"],
                    "cosine": round(score, 6),
                    "decision": "requires_manual_pair_review",
                })
        pair_scores = candidate_vectors @ candidate_vectors.T
        pair_scores.fill_diagonal_(-float("inf"))
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                score = float(pair_scores[i, j])
                if score >= THRESHOLD:
                    matches.append({
                        "kind": "internal_sft_priority",
                        "candidate_id": candidates[i]["id"],
                        "candidate_question": candidates[i]["question"],
                        "match_id": candidates[j]["id"],
                        "match_question": candidates[j]["question"],
                        "cosine": round(score, 6),
                        "decision": "requires_manual_pair_review",
                    })

    output = REVIEW / "semantic_duplicate_queue.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in matches), encoding="utf-8")
    report = {
        "status": "screen_complete_manual_queue_review_pending",
        "candidate_count": len(candidates),
        "candidate_id_sha256": hashlib.sha256("\n".join(sorted(row["id"] for row in candidates)).encode()).hexdigest(),
        "accepted_sha256": sha256(ACCEPTED),
        "review_sha256": {f"batch_{batch:02d}": sha256(REVIEW / f"review_batch_{batch:02d}.jsonl") for batch in range(1, 5)},
        "pilot_review_sha256": sha256(REVIEW.parent / "sft_environment_review_20260929_pilot/review.jsonl"),
        "history_count": len(history),
        "history_sha256": {"dev": sha256(DEV), "final_test": sha256(TEST)},
        "model_sha256": sha256(MODEL / "pytorch_model.bin"),
        "tokenizer_sha256": sha256(MODEL / "tokenizer.json"),
        "method": "local all-mpnet-base-v2, mean pooling, 384-token truncation, cosine Top-1; screening only",
        "threshold": THRESHOLD,
        "matches": len(matches),
        "match_counts": dict(Counter(row["kind"] for row in matches)),
        "queue_sha256": sha256(output),
        "queue_path": str(output.relative_to(ROOT)),
        "training_ready": False,
    }
    (REVIEW / "semantic_duplicate_queue.manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
