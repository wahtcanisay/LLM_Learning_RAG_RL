# R2RAG Core Code Annotations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add detailed Chinese learning comments to the four R2RAG files scheduled for the next study session without changing runtime behavior.

**Architecture:** Comments follow the request through the common RAG interface, the LLM router, the query-complexity classifier, and the single-pass VanillaRAG path. Verification checks UTF-8 readability, Python syntax, Git diff scope, and corpus ignore rules.

**Tech Stack:** Python 3.11+, async generators, Pydantic, OpenAI-compatible clients, Git.

---

### Task 1: Annotate the common interface

**Files:**
- Modify: `R2RAG/src/systems/rag_interface.py`

- [ ] Add a file-level call-chain explanation.
- [ ] Explain request/response models and the meaning of streaming state fields.
- [ ] Explain how `evaluate()` consumes `run_streaming()` and extracts citations/contexts.
- [ ] Explain why `run_streaming()` returns a generator factory instead of an active generator.

### Task 2: Annotate routing and classification

**Files:**
- Modify: `R2RAG/src/systems/rag_router/rag_router_llm.py`
- Modify: `R2RAG/src/tools/classifiers/llm_query_complexity.py`

- [ ] Explain the simple/complex strategy objects created by `RAGRouterLLM`.
- [ ] Explain routing without merging both branches.
- [ ] Explain lazy vLLM client initialization and the classifier prompt semantics.
- [ ] Explain deterministic label parsing and `PredictionResult` fields.

### Task 3: Annotate the single-pass RAG path

**Files:**
- Modify: `R2RAG/src/systems/vanilla_agent/vanilla_rag.py`

- [ ] Group constructor parameters by model, retrieval, and external-service configuration.
- [ ] Explain model selection precedence in `get_active_models()`.
- [ ] Explain query variants, search, reranking, truncation, prompt construction, and streaming generation.
- [ ] Explain intermediate reasoning events, final answer chunks, citations, metadata, and terminal events.

### Task 4: Record and verify

**Files:**
- Modify: `STUDY_PROGRESS.md`

- [ ] Record the annotated files, exact verification command, result boundary, and next reading order.
- [ ] Read all four files as UTF-8 and confirm learning markers exist.
- [ ] Compile all four files without importing project dependencies; expected output: `R2RAG_CORE_COMMENTS_SYNTAX_OK`.
- [ ] Inspect `git diff --check` and `git diff --stat`; expected: no whitespace errors and only scoped files/docs changed.
- [ ] Confirm downloaded corpora and generated data remain ignored.

### Task 5: Publish

**Files:**
- Stage only the four annotated files, progress/design/plan documents, and no data directories.

- [ ] Commit with message `docs: annotate R2RAG core learning path`.
- [ ] Push the current `main` branch to the configured `origin`.
- [ ] Verify local `HEAD` and `origin/main` identify the same commit.
