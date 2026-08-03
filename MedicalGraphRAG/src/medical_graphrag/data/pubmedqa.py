import json
from pathlib import Path

from medical_graphrag.data.schemas import PubMedQARecord


def load_pubmedqa(pqal_path: Path, ground_truth_path: Path) -> list[PubMedQARecord]:
    pqal = json.loads(pqal_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    missing = sorted(set(ground_truth) - set(pqal), key=int)
    if missing:
        raise ValueError(f"official test PMIDs missing from PQA-L: {missing[:5]}")

    records: list[PubMedQARecord] = []
    for pmid in sorted(pqal, key=int):
        raw = pqal[pmid]
        answer = ground_truth.get(pmid, raw.get("final_decision"))
        if pmid in ground_truth and answer != raw.get("final_decision"):
            raise ValueError(f"ground-truth label mismatch for PMID {pmid}")
        records.append(
            PubMedQARecord(
                pmid=pmid,
                question=str(raw.get("QUESTION", "")).strip(),
                contexts=tuple(str(value).strip() for value in raw.get("CONTEXTS", [])),
                answer=str(answer).strip().lower(),
                long_answer=str(raw.get("LONG_ANSWER", "")).strip(),
                year=str(raw["YEAR"]).strip() if raw.get("YEAR") is not None else None,
                split="test" if pmid in ground_truth else "dev",
            )
        )
    return records
