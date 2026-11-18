"""Queue high character-overlap questions for V2 duplicate review."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from MedSearch.tools.check_distillation_inputs import _records


ROOT = Path(__file__).resolve().parents[2]


def find_nearest_pairs(candidates: list[str], history: list[str], threshold: float, same_pool: bool = False) -> list[dict]:
    if not candidates or not history:
        return []
    if same_pool and candidates != history:
        raise ValueError("same_pool requires identical question lists")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), dtype=np.float32)
    vectors = vectorizer.fit_transform(candidates + ([] if same_pool else history))
    candidate_vectors = vectors[:len(candidates)]
    history_vectors = candidate_vectors if same_pool else vectors[len(candidates):]
    matches, seen = [], set()
    for start in range(0, len(candidates), 64):
        stop = min(start + 64, len(candidates))
        scores = (candidate_vectors[start:stop] @ history_vectors.T).toarray()
        for offset, index in enumerate(range(start, stop)):
            if same_pool:
                scores[offset, index] = -1
            nearest = int(scores[offset].argmax())
            similarity = float(scores[offset, nearest])
            if similarity < threshold:
                continue
            pair = tuple(sorted((index, nearest))) if same_pool else (index, nearest)
            if pair in seen:
                continue
            seen.add(pair)
            matches.append({"candidate_index": pair[0], "match_index": pair[1], "cosine": round(similarity, 6)})
    return matches


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--candidate", type=Path, default=ROOT / "MedSearch/data/medical_verifiable_v2_20260919/candidate_keep.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch1-history", type=Path)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("threshold must be in (0,1]")
    candidate_path = args.candidate
    boundaries = {
        "rl_train": ROOT / "MedSearch/data/rl/train.parquet",
        "rl_dev": ROOT / "MedSearch/data/rl/dev.parquet",
        "final_test": ROOT / "MIRAGE/benchmark.json",
    }
    if args.batch1_history is not None:
        boundaries["batch1"] = args.batch1_history
    missing = [str(path) for path in (candidate_path, *boundaries.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    candidates = list(_records("v2", candidate_path))
    history = [row for name, path in boundaries.items() for row in _records(name, path)]
    matches = []
    for item in find_nearest_pairs([row["question"] for row in candidates], [row["question"] for row in history], args.threshold):
        candidate, other = candidates[item["candidate_index"]], history[item["match_index"]]
        matches.append({
            "kind": "history", "candidate": candidate["id"], "match": other["id"],
            "matched_pool": other["pool"], "matched_row": other["row"],
            "candidate_question": candidate["question"], "matched_question": other["question"],
            "cosine": item["cosine"],
        })
    for item in find_nearest_pairs([row["question"] for row in candidates], [row["question"] for row in candidates], args.threshold, same_pool=True):
        candidate, other = candidates[item["candidate_index"]], candidates[item["match_index"]]
        matches.append({
            "kind": "internal", "candidate": candidate["id"], "match": other["id"],
            "matched_pool": "v2", "matched_row": other["row"],
            "candidate_question": candidate["question"], "matched_question": other["question"],
            "cosine": item["cosine"],
        })
    output = args.output or candidate_path.with_name("lexical_neighbors.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in matches:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "status": "review_queue_not_automatic_exclusion",
        "method": "TF-IDF char_wb 3-5 grams; Top-1 cosine",
        "threshold": args.threshold,
        "candidate_sha256": _sha256(candidate_path),
        "boundary_sha256": {name: _sha256(path) for name, path in boundaries.items()},
        "output_sha256": _sha256(output),
        "match_counts": dict(Counter(row["kind"] for row in matches)),
        "manual_review": "not_done",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
