"""Derive the user-selected answer-only cold-start set from frozen teacher attempts."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def select_answered(record):
    """Keep a complete legal final answer regardless of verifier or earlier actions."""
    if len(record["response_ids"]) != len(record["response_mask"]):
        raise ValueError(f"token/mask mismatch: {record['question_id']}")
    if not (
        record.get("answered")
        and record.get("final_answer")
        and record.get("segments")
        and record["segments"][-1]["kind"] == "answer"
    ):
        return None
    selected = dict(record)
    selected["prior_filter_status"] = record["status"]
    selected["prior_filter_reasons"] = record["stop_reason"]
    selected["status"] = "accepted"
    selected["stop_reason"] = []
    selected["selection_rule"] = "legal_final_answer"
    return selected


def build_answered_set(source_dir, output_dir):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    attempts_path = source_dir / "teacher_attempts.jsonl"
    if source_manifest["status"] != "complete":
        raise ValueError("source teacher export is incomplete")
    attempts_hash = sha256(attempts_path)
    if attempts_hash != source_manifest["output_sha256"]["teacher_attempts.jsonl"]:
        raise ValueError("source teacher attempts hash mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("output directory must be empty")

    selected_path = output_dir / "accepted.jsonl"
    excluded_path = output_dir / "excluded_no_answer.jsonl"
    counts = {"total": 0, "accepted": 0, "excluded_no_answer": 0}
    with attempts_path.open(encoding="utf-8") as attempts, selected_path.open("w", encoding="utf-8") as selected_handle, excluded_path.open("w", encoding="utf-8") as excluded_handle:
        for line in attempts:
            if not line.strip():
                continue
            record = json.loads(line)
            selected = select_answered(record)
            counts["total"] += 1
            if selected is None:
                excluded_handle.write(line)
                counts["excluded_no_answer"] += 1
            else:
                selected_handle.write(json.dumps(selected, ensure_ascii=False) + "\n")
                counts["accepted"] += 1
    if counts["total"] != source_manifest["processed"]:
        raise ValueError("source attempt count differs from manifest")
    manifest = {
        "status": "complete",
        "selection_rule": "answered=true, nonempty final_answer, last segment is legal answer",
        "verifier_result_is_not_a_filter": True,
        "earlier_invalid_actions_and_no_search_are_allowed": True,
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_attempts_sha256": attempts_hash,
        "source_model_revision": source_manifest["config"]["model_revision"],
        "counts": counts,
        "output_sha256": {
            selected_path.name: sha256(selected_path),
            excluded_path.name: sha256(excluded_path),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build_answered_set(args.source_dir, args.output_dir)["counts"]))


if __name__ == "__main__":
    main()
