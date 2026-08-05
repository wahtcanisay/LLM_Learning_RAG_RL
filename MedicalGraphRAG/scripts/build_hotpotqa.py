"""Build the `hotpotqa_v1` frozen multi-hop retrieval dataset from HotpotQA.

Run inside the `llm-pytorch` container (needs pandas + pyarrow for parquet).
Reads the downloaded raw `distractor/validation` split and writes a frozen
dataset under `--output-dir` in the same contract as the other benchmarks:

- documents.jsonl: one document per Wikipedia paragraph (title-unique).
  content = the paragraph's sentences joined and Unicode-normalized
  (NFKC + whitespace collapse), so identical paragraphs appearing under
  repeated titles across questions collapse to one document.
- chunks.jsonl: one chunk per document (content = same normalized text).
- questions.jsonl: the 7,405 validation questions (split="test" for eval).
- qrels.tsv: multi-gold rows, one per supporting-fact paragraph per query.
  HotpotQA supporting_facts is sentence-level; we promote each distinct
  supporting-fact *title* to a gold document (the paragraph containing it).
- manifest.json: counts + artifact SHA-256.

Source: `hotpotqa/hotpot_qa` on Hugging Face, `distractor` config,
`validation` split (7,405 questions; each with 10 paragraphs = 2+ gold
supporting facts + distractors). This yields a self-contained closed corpus
(no external Wikipedia dump needed) with genuine multi-hop gold, which is
exactly the setting where graph retrieval should add value.

Usage:
    python build_hotpotqa.py --raw parquet PATH --output-dir data/processed/hotpotqa_v1
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from medical_graphrag.data.io import sha256_file, write_json, write_jsonl, write_qrels  # noqa: E402


_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFKC + whitespace collapse; removes HotpotQA's Unicode artifacts so
    identical paragraphs under a repeated title map to one document."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def iter_paragraphs(df: pd.DataFrame):
    """Yield (title, normalized_sentences_joined) for every context paragraph."""
    for row in df["context"]:
        titles = row["title"]
        sentences = row["sentences"]
        for title, sent_list in zip(titles, sentences):
            yield str(title), normalize_text(" ".join(sent_list))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.raw)

    # 1) Documents: title-unique, content = normalized paragraph text.
    paragraphs: dict[str, str] = {}
    for title, content in iter_paragraphs(df):
        if not content:
            continue
        paragraphs[title] = content  # first occurrence wins (identical after norm)

    documents = [
        {
            "doc_id": title,
            "title": title,
            "content": content,
            "source": "hotpotqa",
            "year": None,
        }
        for title, content in paragraphs.items()
    ]
    chunks = [
        {
            "chunk_id": title,
            "doc_id": title,
            "order": 0,
            "title": title,
            "content": content,
            "source": "hotpotqa",
        }
        for title, content in paragraphs.items()
    ]

    # 2) Questions + qrels: every distinct supporting-fact title is gold.
    questions: list[dict[str, object]] = []
    qrels: list[tuple[str, str, int]] = []
    seen_gold: set[str] = set()  # titles dropped because no content / not in corpus
    missing: set[str] = set()
    for row in df.to_dict("records"):
        qid = str(row["id"])
        question = str(row["question"])
        answer = str(row.get("answer") or "")
        sf_titles = list(row["supporting_facts"]["title"])
        gold_docs = set()
        for title in sf_titles:
            title = str(title)
            if title in paragraphs:
                gold_docs.add(title)
            else:
                missing.add(title)
        if not gold_docs:
            seen_gold.add(qid)
            continue
        questions.append(
            {
                "query_id": qid,
                "question": question,
                "answer": answer,
                "long_answer": "",
                "split": "test",
            }
        )
        for doc_id in sorted(gold_docs):
            qrels.append((qid, doc_id, 1))

    # 3) Freeze + hash.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "documents.jsonl", documents)
    write_jsonl(args.output_dir / "chunks.jsonl", chunks)
    write_jsonl(args.output_dir / "questions.jsonl", questions)
    write_qrels(args.output_dir / "qrels.tsv", qrels)

    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    artifact_hashes = {name: sha256_file(args.output_dir / name) for name in names}
    counts = {
        "questions": len(questions),
        "documents": len(documents),
        "chunks": len(chunks),
        "qrels": len(qrels),
    }
    write_json(
        args.output_dir / "manifest.json",
        {
            "counts": counts,
            "artifact_hashes": artifact_hashes,
            "meta": {
                "source": "hotpotqa/hotpot_qa distractor validation",
                "raw_sha256": sha256_file(args.raw),
                "dropped_empty_gold_questions": len(seen_gold),
                "missing_gold_titles": sorted(missing)[:50],
            },
        },
    )
    print(json.dumps(counts))
    print(f"gold docs per query: mean {len(qrels)/max(len(questions),1):.2f}")
    print(f"dropped questions (no gold in corpus): {len(seen_gold)}")
    if missing:
        print(f"missing gold titles (not in corpus): {len(missing)} e.g. {sorted(missing)[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
