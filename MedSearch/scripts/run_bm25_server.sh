#!/usr/bin/env bash
set -euo pipefail

# Select the retrieval corpus without changing the AgentLoop.
# Usage: MEDSEARCH_DATASET=base|medrag bash MedSearch/scripts/run_bm25_server.sh
dataset="${MEDSEARCH_DATASET:-base}"
topk="${MEDSEARCH_TOPK:-5}"

case "$dataset" in
  base)
    corpus_path="${MEDSEARCH_BASE_CORPUS:-MedSearch/data/retrieval/medrag_textbooks.jsonl}"
    index_path="${MEDSEARCH_BASE_INDEX:-MedSearch/data/retrieval/medrag_textbooks/bm25}"
    ;;
  medrag)
    corpus_path="${MEDSEARCH_MEDRAG_CORPUS:-MedSearch/data/retrieval/medrag_textbooks.jsonl}"
    index_path="${MEDSEARCH_MEDRAG_INDEX:-MedSearch/data/retrieval/medrag_textbooks/bm25}"
    ;;
  *)
    echo "MEDSEARCH_DATASET must be base or medrag; got: $dataset" >&2
    exit 2
    ;;
esac

test -f "$corpus_path" || {
  echo "Corpus not found: $corpus_path" >&2
  exit 1
}
test -d "$index_path" || {
  echo "BM25 index not found: $index_path" >&2
  exit 1
}

echo "dataset=$dataset"
echo "corpus=$corpus_path"
echo "index=$index_path"

python Search-R1/search_r1/search/retrieval_server.py \
  --index_path "$index_path" \
  --corpus_path "$corpus_path" \
  --topk "$topk" \
  --retriever_name bm25
