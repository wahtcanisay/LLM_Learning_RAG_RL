"""Flag V2 semantic near duplicates with the local all-mpnet-base-v2 model."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoModel, AutoTokenizer

from MedSearch.tools.check_distillation_inputs import _records


ROOT = Path(__file__).resolve().parents[2]


def find_semantic_matches(candidates, history, candidate_ids, history_ids, threshold):
    """Return unique top-neighbor pairs at or above a cosine threshold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates, history = candidates.to(device), history.to(device)
    matches, seen_internal = [], set()
    for start in range(0, len(candidate_ids), 128):
        stop = min(start + 128, len(candidate_ids))
        query = candidates[start:stop]
        history_scores, history_indices = (query @ history.T).max(dim=1)
        internal = query @ candidates.T
        internal[torch.arange(stop - start, device=device), torch.arange(start, stop, device=device)] = -float("inf")
        internal_scores, internal_indices = internal.max(dim=1)
        for offset, index in enumerate(range(start, stop)):
            if history_scores[offset] >= threshold:
                matches.append({
                    "kind": "history", "candidate": candidate_ids[index],
                    "match": history_ids[int(history_indices[offset])],
                    "cosine": round(float(history_scores[offset]), 6),
                })
            if internal_scores[offset] >= threshold:
                other = int(internal_indices[offset])
                pair = tuple(sorted((index, other)))
                if pair not in seen_internal:
                    seen_internal.add(pair)
                    matches.append({
                        "kind": "internal", "candidate": candidate_ids[index],
                        "match": candidate_ids[other],
                        "cosine": round(float(internal_scores[offset]), 6),
                    })
    return matches


def _embeddings(model, tokenizer, records, device, batch_size):
    vectors = []
    for start in range(0, len(records), batch_size):
        questions = [row["question"] for row in records[start:start + batch_size]]
        batch = tokenizer(questions, padding=True, truncation=True, max_length=384, return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu())
        if (start + batch_size) % 5000 < batch_size:
            print(f"encoded {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    return torch.cat(vectors)


def _sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--model", type=Path,
        default=ROOT / "MedicalGraphRAG/models/all-mpnet-base-v2",
    )
    parser.add_argument(
        "--candidate", type=Path,
        default=ROOT / "MedSearch/data/medical_verifiable_v2_20260919/candidate_keep.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "MedSearch/data/medical_verifiable_v2_20260919/semantic_neighbors.jsonl",
    )
    parser.add_argument("--batch1-history", type=Path)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1 or args.batch_size <= 0:
        parser.error("threshold must be in (0,1] and batch-size must be positive")
    boundaries = {
        "rl_train": ROOT / "MedSearch/data/rl/train.parquet",
        "rl_dev": ROOT / "MedSearch/data/rl/dev.parquet",
        "final_test": ROOT / "MIRAGE/benchmark.json",
    }
    if args.batch1_history is not None:
        boundaries["batch1"] = args.batch1_history
    for path in (args.candidate, args.model / "pytorch_model.bin", *boundaries.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    candidates = list(_records("v2", args.candidate))
    references = [row for name, path in boundaries.items() for row in _records(name, path)]
    candidate_ids = [row["id"] for row in candidates]
    history_ids = [f"{row['pool']}:{row['row']}:{row['id'] or ''}" for row in references]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).eval().to(device)
    candidate_vectors = _embeddings(model, tokenizer, candidates, device, args.batch_size)
    history_vectors = _embeddings(model, tokenizer, references, device, args.batch_size)
    del model
    matches = find_semantic_matches(candidate_vectors, history_vectors, candidate_ids, history_ids, args.threshold)
    candidate_by_id = {row["id"]: row for row in candidates}
    history_by_id = dict(zip(history_ids, references, strict=True))
    for match in matches:
        other = history_by_id[match["match"]] if match["kind"] == "history" else candidate_by_id[match["match"]]
        match["candidate_question"] = candidate_by_id[match["candidate"]]["question"]
        match["matched_question"] = other["question"]
        match["matched_pool"] = other["pool"]
        match["matched_path"] = other["path"]
        match["matched_row"] = other["row"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for match in matches:
            handle.write(json.dumps(match, ensure_ascii=False) + "\n")
    summary = {
        "status": "review_queue_not_automatic_exclusion",
        "candidate_rows": len(candidates),
        "reference_rows": len(references),
        "threshold": args.threshold,
        "model_path": str(args.model),
        "model_sha256": _sha256(args.model / "pytorch_model.bin"),
        "candidate_sha256": _sha256(args.candidate),
        "boundary_sha256": {name: _sha256(path) for name, path in boundaries.items()},
        "output_sha256": _sha256(args.output),
        "device": str(device),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "match_counts": dict(Counter(match["kind"] for match in matches)),
        "medical_review": "not_done",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
