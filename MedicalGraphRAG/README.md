# MedicalGraphRAG

Standalone implementation for reproducible medical retrieval experiments.

The sibling `MedRAG/` and `LinearRAG/` directories are read-only references and
data sources. This project owns all new cleaning, retrieval, evaluation, tests,
and experiment records.

The first milestone is `pubmedqa_hard_v1`: 1,000 PubMedQA gold documents plus
4,000 deterministic MedRAG PubMed distractors with document-level qrels.

## Scope

- Primary retrieval text: `abstract_only` (chunk `content`).
- Leakage comparison only: `title_abstract`; PubMedQA questions often reproduce
  article titles, so this result must never be mixed into the primary table.
- Hard metrics: Recall@1/5/10, MRR@10, and binary nDCG@10.
- The sibling `MedRAG/` and `LinearRAG/` repositories remain unmodified.

## Setup and tests

```powershell
python -m pip install -e ".[dev]"
python -m pytest -v
python -m medical_graphrag.cli --help
```

## Fetch PubMedQA

```powershell
python -m medical_graphrag.cli fetch-pubmedqa --output-dir data/raw/pubmedqa
```

## Run the 20-question audit gate

```powershell
python -m medical_graphrag.cli audit `
  --config configs/pubmedqa_hard_v1.json `
  --pubmedqa-dir data/raw/pubmedqa `
  --medrag-pubmed-dir ../MedRAG/corpus/pubmed/chunk `
  --output-dir data/processed/pubmedqa_hard_v1
```

Do not run the full build unless `audit_20.json` reports `passed: true` for all
20 items.

## Build deterministic artifacts

```powershell
python -m medical_graphrag.cli build `
  --config configs/pubmedqa_hard_v1.json `
  --pubmedqa-dir data/raw/pubmedqa `
  --medrag-pubmed-dir ../MedRAG/corpus/pubmed/chunk `
  --output-dir data/processed/pubmedqa_hard_v1
```

The build writes `questions.jsonl`, `documents.jsonl`, `chunks.jsonl`,
`qrels.tsv`, and `manifest.json`. Large generated files remain ignored; the
config, manifest, and audit evidence are versioned.

No retrieval score is reported until a retriever has produced rankings against
the generated qrels.

## Verified audit evidence (2026-08-03)

- Command: the 20-question `audit` command shown above; the isolated worktree
  used the absolute path to the same local MedRAG PubMed directory.
- PubMedQA PQA-L: 1,000 records,
  SHA-256 `8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d`.
- Official test ground truth: 500 records,
  SHA-256 `939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51`.
- MedRAG PubMed source: 1,166 JSONL shards, 1,165 non-empty; 40 shards were
  deterministically selected by the configured seed.
- Result: 20/20 items passed, 0 empty contexts, 0 exact duplicate gold/distractor
  documents, 75 gold chunks in the audited items.
- Clean cached-tokenizer rerun elapsed time: 15.08 seconds on this machine.
- `audit_20.json` SHA-256:
  `6dca9821b4342ea39982111f71c3df550395d489b994c7b842a9523115f24c56`.

These are data-integrity results, not retrieval metrics. Recall, MRR, nDCG, QA
accuracy, latency, and GPU memory remain unreported until their scripts run.
