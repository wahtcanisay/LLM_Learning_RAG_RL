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
