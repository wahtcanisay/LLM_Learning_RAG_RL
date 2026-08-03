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
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')"
python -m pytest -v
python -m medical_graphrag.cli --help
```

The audit/build commands load the tokenizer from the local cache only. Cache it
once during setup so later data builds do not perform hidden network checks.

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
  documents, 75 gold chunks in the audited items. Every audit item includes its
  question, full contexts, and generated gold chunk text for human inspection.
- Clean cached-tokenizer rerun elapsed time: 8.60 seconds on this machine.
- `audit_20.json` SHA-256:
  `229fb01f097aa50471fc5f0b0664c0f774e65767c6f5f7ce26d92c587d33e734`.

These are data-integrity results, not retrieval metrics. Recall, MRR, nDCG, QA
accuracy, latency, and GPU memory remain unreported until their scripts run.

## Verified full build evidence (2026-08-03)

- Build time: 13.18 seconds on this machine with the tokenizer already cached.
- Output counts: 1,000 questions, 5,000 documents, 7,562 chunks, 1,000 qrels.
- Split counts: 500 dev and 500 official test questions.
- Source counts: 1,000 PubMedQA gold documents and 4,000 MedRAG PubMed distractors.
- All query, document, and chunk identifiers are unique; every qrel resolves to
  its expected `PMID:<query_id>` document; all artifact hashes match the manifest.
- `manifest.json` SHA-256:
  `cf9b75917bb6c73ff5e5d1862293e31caf86ec5d93c05c24f40760c83b727baa`.

The generated JSONL/TSV artifacts remain local and ignored. Only the compact
manifest is versioned. Retrieval metrics are still pending the BM25 run.

## BM25 hard retrieval baseline

The primary BM25 run indexes only chunk `content` (`abstract_only`). It retrieves
Top-100 chunks with Pyserini 0.22.1 using explicit `k1=0.9` and `b=0.4`, then
collapses chunks into unique documents with `max(chunk_score)`. The 500-question
dev split is for debugging; the 500-question official test split is the primary
report. Never merge them into one 1,000-question headline metric.

Run the export from `MedicalGraphRAG/`:

```powershell
python -m medical_graphrag.cli export-pyserini `
  --dataset-dir data/processed/pubmedqa_hard_v1 `
  --output-dir outputs/pubmedqa_hard_v1/bm25_abstract_only
```

Build the Lucene index in the existing `llm-pytorch` container. The repository
is mounted from `/mnt/d/code_list` to `/workspace/code_list`:

```powershell
docker exec llm-pytorch python "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/scripts/build_pyserini_index.py" `
  --collection "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/outputs/pubmedqa_hard_v1/bm25_abstract_only/collection" `
  --index "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/indexes/pubmedqa_hard_v1/bm25_abstract_only" `
  --report "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/outputs/pubmedqa_hard_v1/bm25_abstract_only/index_build.json" `
  --threads 8
```

Search all questions and retain the raw Top-100 chunk hits:

```powershell
docker exec llm-pytorch python "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/scripts/search_pyserini_bm25.py" `
  --index "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/indexes/pubmedqa_hard_v1/bm25_abstract_only" `
  --questions "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/data/processed/pubmedqa_hard_v1/questions.jsonl" `
  --metadata "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/outputs/pubmedqa_hard_v1/bm25_abstract_only/chunk_metadata.jsonl" `
  --output "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/outputs/pubmedqa_hard_v1/bm25_abstract_only/raw_rankings.jsonl" `
  --report "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/outputs/pubmedqa_hard_v1/bm25_abstract_only/search_run.json" `
  --top-k 100 --k1 0.9 --b 0.4
```

Evaluate document rankings and write the compact experiment record:

```powershell
$commit = git rev-parse HEAD
python -m medical_graphrag.cli evaluate-bm25 `
  --dataset-dir data/processed/pubmedqa_hard_v1 `
  --metadata outputs/pubmedqa_hard_v1/bm25_abstract_only/chunk_metadata.jsonl `
  --rankings outputs/pubmedqa_hard_v1/bm25_abstract_only/raw_rankings.jsonl `
  --index-report outputs/pubmedqa_hard_v1/bm25_abstract_only/index_build.json `
  --search-report outputs/pubmedqa_hard_v1/bm25_abstract_only/search_run.json `
  --output-dir experiments/pubmedqa_hard_v1/bm25_abstract_only `
  --git-commit $commit `
  --docker-image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
```

The tracked result contains Recall@1/5/10, MRR@10, binary nDCG@10,
mean/P50/P95 search latency, index time and index size. BM25/Lucene runs on CPU,
so GPU peak memory is recorded as not applicable rather than estimated.
