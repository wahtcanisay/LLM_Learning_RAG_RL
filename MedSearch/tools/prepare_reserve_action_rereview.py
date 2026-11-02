"""Prepare a blind trace re-review for the eight promising reserve rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "MedSearch/data/distillation/sft_environment_review_20260929_full"
ANSWERED = ROOT / "MedSearch/data/distillation/teacher_cold1000_20260929_answered/accepted.jsonl"
OUT = REVIEW / "reserve_action_rereview"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite re-review package: {OUT}")
    review_path = REVIEW / "review.jsonl"
    reviews = read_jsonl(review_path)
    selected = [row for row in reviews if (
        row.get("sft_recommendation") == "reserve"
        and row.get("question_decision") == "keep"
        and row.get("support_label") == "sufficient"
        and row.get("teacher_answer_relation") == "equivalent"
    )]
    if len(selected) != 8 or len({row["question_id"] for row in reviews}) != len(reviews):
        raise ValueError(f"expected eight unique promising reserve rows, found {len(selected)}")

    traces = read_jsonl(ANSWERED)
    trace_by_id = {row["question_id"]: row for row in traces}
    if len(trace_by_id) != len(traces) or set(trace_by_id) != {row["question_id"] for row in reviews}:
        raise ValueError("accepted trajectories do not match the full review ledger IDs")

    output_rows = []
    for review in sorted(selected, key=lambda row: row["question_id"]):
        trace = trace_by_id[review["question_id"]]
        if not trace.get("answered") or not trace.get("final_answer") or not trace.get("segments"):
            raise ValueError(f"missing legal-answer trace: {review['question_id']}")
        if trace["segments"][-1].get("kind") != "answer":
            raise ValueError(f"last trace segment is not an answer: {review['question_id']}")
        response_ids, response_mask = trace.get("response_ids", []), trace.get("response_mask", [])
        output_rows.append({
            "trajectory_id": trace["trajectory_id"],
            "question_id": trace["question_id"],
            "question_sha256": hashlib.sha256(trace["question"].encode("utf-8")).hexdigest(),
            "question": trace["question"],
            "references": trace["references"],
            "final_answer": trace["final_answer"],
            "search_count": trace["search_count"],
            "searches": trace["searches"],
            "segments": trace["segments"],
            "response": trace["response"],
            "response_ids_sha256": hashlib.sha256(json.dumps(response_ids, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "response_mask_sha256": hashlib.sha256(json.dumps(response_mask, separators=(",", ":")).encode("utf-8")).hexdigest(),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "reserve_evidence_sufficient_blind_trace_rereview.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "independent_trace_rereview_input_only",
        "training_ready": False,
        "review_rows": len(output_rows),
        "first_pass_review_decisions_removed": True,
        "verifier_fields_removed": True,
        "selection_rule": "reserve + question keep + visible support sufficient + teacher answer equivalent",
        "review_sha256": sha256(review_path),
        "accepted_trajectories_sha256": sha256(ANSWERED),
        "output_sha256": sha256(output),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"review_rows": len(output_rows), "output_sha256": manifest["output_sha256"], "training_ready": False}))


if __name__ == "__main__":
    main()
