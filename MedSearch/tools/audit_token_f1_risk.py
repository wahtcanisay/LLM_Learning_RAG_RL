"""Flag gold answers where token-overlap F1 can hide semantic contradictions.

This is a conservative audit queue, not a medical correctness verifier.
"""

import argparse
import re
import string
from collections import Counter
from pathlib import Path


OPPOSITES = {
    "direction": (
        ("left", "right"),
        ("upper", "lower"),
        ("anterior", "posterior"),
        ("proximal", "distal"),
        ("ipsilateral", "contralateral"),
        ("unilateral", "bilateral"),
    ),
    "polarity": (
        ("positive", "negative"),
        ("present", "absent"),
        ("increased", "decreased"),
        ("increase", "decrease"),
        ("high", "low"),
        ("benign", "malignant"),
        ("more", "less"),
        ("most", "least"),
    ),
}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(left: str, right: str) -> float:
    left_tokens = normalize_answer(left).split()
    right_tokens = normalize_answer(right).split()
    if not left_tokens or not right_tokens:
        return float(left_tokens == right_tokens)
    overlap = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def counterfactuals(answer: str) -> list[tuple[str, str]]:
    variants = []
    for category, pairs in OPPOSITES.items():
        for left, right in pairs:
            for source, target in ((left, right), (right, left)):
                if re.search(rf"\b{re.escape(source)}\b", answer, flags=re.IGNORECASE):
                    changed = re.sub(
                        rf"\b{re.escape(source)}\b", target, answer, count=1, flags=re.IGNORECASE
                    )
                    variants.append((category, changed))
    if re.search(r"\d", answer):
        variants.append(("numeric", ""))
    if re.search(r"\b(no|not|without|except|never)\b", answer, flags=re.IGNORECASE):
        variants.append(("negation", ""))
    return variants


def count_high_f1(values) -> int:
    count = 0
    for value in values:
        try:
            count += float(value) >= 0.5
        except (TypeError, ValueError):
            pass
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    import pandas as pd

    data = pd.read_parquet(args.input)
    rows = []
    for _, row in data.iterrows():
        answers = row["golden_answers"]
        answer = str(answers[0] if not isinstance(answers, str) else answers)
        for category, counterfactual in counterfactuals(answer):
            rows.append(
                {
                    "id": row["id"],
                    "category": category,
                    "gold_answer": answer,
                    "counterfactual": counterfactual,
                    "counterfactual_f1": (
                        token_f1(answer, counterfactual) if counterfactual else ""
                    ),
                    "question": row["question"],
                }
            )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result["category"].value_counts().to_string())
    misleading_count = count_high_f1(result["counterfactual_f1"])
    print(f"flagged_rows={len(result)} high_f1_counterfactuals={misleading_count}")


if __name__ == "__main__":
    main()
