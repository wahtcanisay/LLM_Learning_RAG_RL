# MedRAG Stage 1 Study Plan

> **For agentic workers:** This is a study plan, not an implementation authorization. Execute one checked learning task per session.

**Goal:** Within 10–14 study sessions, understand and reproduce the MedRAG baseline retrieval pipeline on a small medical corpus without downloading the full PubMed/MedCorp collection.

**Architecture:** Read the public toolkit from the top-level `MedRAG` entry point into `RetrievalSystem`, then into corpus loading, chunk/index construction, retrieval/fusion, prompt construction, generation, and evaluation. Keep the official code intact first; add a small experimental adapter only after the official minimal path is understood.

**Tech Stack:** Python, PyTorch, Transformers, Sentence-Transformers, Pyserini/BM25, FAISS, JSONL, Hugging Face datasets/models, Docker/WSL2.

---

### Task 1: Build the code map (tomorrow)

**Files:**
- Read: `D:\code_list\some tricks\LLMLeanring\MedRAG\README.md`
- Read: `D:\code_list\some tricks\LLMLeanring\MedRAG\src\medrag.py:43-255`
- Read: `D:\code_list\some tricks\LLMLeanring\MedRAG\src\utils.py:11-27,129-321`
- Read: `D:\code_list\some tricks\LLMLeanring\MedRAG\src\template.py`

- [ ] Explain the call chain: `MedRAG.__init__` → `RetrievalSystem` → `Retriever` → `retrieve`/`merge` → context truncation → `generate`.
- [ ] Mark the code path for `rag=False` (No-RAG baseline) and `rag=True` (retrieval baseline).
- [ ] Identify the constructor side effect that can clone a corpus or build an index (`utils.py:136-197`).
- [ ] Do not download a large corpus or model during this task.

### Task 2: Learn the data contract

**Files:**
- Read: `D:\code_list\some tricks\LLMLeanring\MedRAG\src\data\textbooks.py:17-31`
- Read: `D:\code_list\some tricks\LLMLeanring\MedRAG\src\data\pubmed.py:16-59`

- [ ] Explain the JSONL fields `id`, `title`, `content`, and `contents`.
- [ ] Explain `chunk_size=1000` and `chunk_overlap=200` in the textbook preprocessing path.
- [ ] Create a three-document toy JSONL fixture for later retrieval tests.

### Task 3: Run a no-download smoke test

- [ ] Verify imports and report exact failures without changing pinned dependencies blindly.
- [ ] Use the toy JSONL fixture to inspect the expected document shape.
- [ ] Record the command, environment, output, and unresolved dependency issues in `STUDY_PROGRESS.md`.

### Task 4: Understand and run BM25

**Files:**
- Read: `D:\code_list\some tricks\LLMLeanring\MedRAG\src\utils.py:149-157,204-233`

- [ ] Explain the role of the Lucene/Pyserini index and Java dependency.
- [ ] Run BM25 on the smallest available corpus subset.
- [ ] Record top-k documents, scores, latency, and one success/one failure case.

### Task 5: Understand Dense Retrieval and FAISS

**Files:**
- Read: `D:\code_list\some tricks\LLMLeanring\MedRAG\src\utils.py:63-126,158-202`

- [ ] Explain embedding generation, `.npy` shards, `metadatas.jsonl`, and `faiss.index`.
- [ ] Explain why SPECTER uses L2 while other configured dense retrievers use inner product.
- [ ] Run a small dense retrieval index and record index time, index size, top-k scores, and GPU memory.

### Task 6: Understand Hybrid/RRF fusion

**Files:**
- Read: `D:\code_list\some tricks\LLMLeanring\MedRAG\src\utils.py:235-321`

- [ ] Explain `RRF-2`, `RRF-4`, `rrf_k`, rank merging, and deduplication by document id.
- [ ] Compare BM25, Dense, and Hybrid on the same questions.

### Task 7: Understand prompt construction and generation

**Files:**
- Read: `D:\code_list\some tricks\LLMLeanring\MedRAG\src\medrag.py:136-180,182-255`
- Read: `D:\code_list\some tricks\LLMLeanring\MedRAG\src\template.py`

- [ ] Explain how retrieved snippets become the `context` prompt.
- [ ] Explain context truncation and the difference between retrieval success and answer correctness.
- [ ] Run the smallest local generation/no-RAG or predetermined-snippet example that the environment supports.

### Task 8: Implement the unified Retriever learning adapter

- [ ] Define a minimal `search(query, top_k)` interface in a separate experiment file; do not refactor official files yet.
- [ ] Wrap BM25 and Dense with the same output shape.
- [ ] Add a simple evaluator for Recall@k and MRR on a tiny labeled set.

### Task 9: Complete the Stage 1 comparison

- [ ] Run No-RAG, BM25, Dense, Hybrid, and (if dependencies allow) Reranker baselines.
- [ ] Record Recall@5, Recall@10, MRR, QA accuracy, latency, and peak memory.
- [ ] Preserve failures and explain whether the bottleneck is data, retrieval, generation, or evaluation.
- [ ] Update `STUDY_PROGRESS.md` and write a Stage 1 summary before starting R2RAG.

