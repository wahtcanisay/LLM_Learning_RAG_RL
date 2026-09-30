# Stage-B Protocol Evaluation Implementation Plan

> **For agentic workers:** Execute inline with the project teacher/student checkpoints below. Do not delegate or implement the student functions on the student's behalf.

**Goal:** Compare the Stage-A and Stage-B adapters on the same frozen 12-item protocol dev using real local BM25 search and a maximum of two search turns.

**Architecture:** Add one standalone evaluator beside `evaluate_stage_a.py`. Pure parsing and metric functions remain independently testable; the CLI owns QLoRA model loading, Search-R1-style rolling context, local BM25 retrieval, JSONL trajectories, and aggregate metrics. Reuse `format_stage_b_prompt`, the existing model-loading functions, the existing Lucene index, and the frozen protocol JSON; do not import veRL or start an HTTP server for 12 development examples.

**Tech Stack:** Python, pytest, Transformers/PEFT, PyTorch, Pyserini LuceneSearcher, JSON/JSONL.

---

## File map

- Create `MedSearch/sft/evaluate_stage_b.py`: evaluator CLI, pure action parser, BM25 adapter, rollout loop, metrics, output files.
- Create `MedSearch/docs/tests/test_stage_b_evaluator.py`: teacher-owned RED contracts for parsing, evidence hits, duplicate queries, forced stop, and aggregate rates.
- Modify `MedSearch/README.md`: exact Stage-A/Stage-B commands and interpretation boundary.
- Modify `STUDY_PROGRESS.md`: real metrics only after both variants finish.

### Task 1: Freeze the student action-parser contract

**Files:**
- Create: `MedSearch/docs/tests/test_stage_b_evaluator.py`
- Create: `MedSearch/sft/evaluate_stage_b.py`

- [ ] **Step 1: Write failing parser tests**

Tests must call `parse_stage_b_action(text)` and require a dictionary with `action`, `content`, `valid_think`, `malformed_action`, and `both_search_and_answer`. Cover exactly these cases:

```python
def test_parse_valid_search_action():
    result = parse_stage_b_action("<think>Need evidence.</think>\n<search>asthma eosinophilia</search>")
    assert result == {
        "action": "search",
        "content": "asthma eosinophilia",
        "valid_think": True,
        "malformed_action": False,
        "both_search_and_answer": False,
    }


def test_parse_empty_search_is_malformed():
    result = parse_stage_b_action("<think>Need evidence.</think><search> </search>")
    assert result["action"] is None
    assert result["malformed_action"] is True


def test_parse_search_and_answer_uses_first_action_but_flags_both():
    result = parse_stage_b_action(
        "<think>Need evidence.</think><search>query</search><answer>B</answer>"
    )
    assert result["action"] == "search"
    assert result["both_search_and_answer"] is True
    assert result["malformed_action"] is True
```

- [ ] **Step 2: Verify RED**

Run:

```cmd
wsl docker exec -w "/workspace/code_list/some tricks/LLMLeanring" llm-pytorch pytest -q MedSearch/docs/tests/test_stage_b_evaluator.py
```

Expected: collection/import failure because `evaluate_stage_b.py` and `parse_stage_b_action` do not exist.

- [ ] **Step 3: Add the student skeleton**

```python
def parse_stage_b_action(text: str) -> dict:
    """Parse the first Search-R1 action and report strict protocol defects.

    Student contract:
    - Accept only string input.
    - Inspect the first complete `<search>...</search>` or `<answer>...</answer>` action.
    - Require non-empty action content.
    - `valid_think` means exactly one non-empty `<think>...</think>` occurs before that action.
    - Flag output containing both complete search and answer tags.
    - Do not execute retrieval, normalize answer letters, or modify the text here.
    """
    raise NotImplementedError("student task: implement Stage-B action parsing")
```

- [ ] **Step 4: Student implements the parser and runs the focused tests**

Expected: parser tests pass before any GPU work.

### Task 2: Freeze evidence and aggregate-metric contracts

**Files:**
- Modify: `MedSearch/docs/tests/test_stage_b_evaluator.py`
- Modify: `MedSearch/sft/evaluate_stage_b.py`

- [ ] **Step 1: Add RED tests for `evidence_hit` and `summarize_stage_b_records`**

```python
def test_evidence_hit_matches_document_ids():
    docs = [{"document_id": "doc-a"}, {"document_id": "doc-b"}]
    assert evidence_hit(docs, ["doc-b", "doc-c"]) is True
    assert evidence_hit(docs, ["doc-c"]) is False


def test_summary_uses_trajectory_level_denominators():
    records = [
        {
            "valid_think": True,
            "valid_search": True,
            "nonempty_query": True,
            "malformed_action": False,
            "both_search_and_answer": False,
            "evidence_hit": True,
            "duplicate_query": False,
            "closed_loop_completion": True,
            "valid_answer": True,
            "answer_correct": True,
            "forced_stop": False,
        },
        {
            "valid_think": False,
            "valid_search": False,
            "nonempty_query": False,
            "malformed_action": True,
            "both_search_and_answer": True,
            "evidence_hit": False,
            "duplicate_query": True,
            "closed_loop_completion": False,
            "valid_answer": False,
            "answer_correct": False,
            "forced_stop": True,
        },
    ]
    metrics = summarize_stage_b_records(records)
    assert metrics["examples"] == 2
    assert metrics["valid_search_rate"] == 0.5
    assert metrics["evidence_hit_at_k"] == 0.5
    assert metrics["mcq_accuracy"] == 0.5
    assert metrics["forced_stop_rate"] == 0.5
```

- [ ] **Step 2: Verify RED, then add student skeletons**

```python
def evidence_hit(documents: list[dict], supporting_document_ids: list[str]) -> bool:
    """Return whether any retrieved `document_id` belongs to the frozen gold set."""
    raise NotImplementedError("student task: implement Evidence Hit@k")


def summarize_stage_b_records(records: list[dict]) -> dict:
    """Average fixed boolean trajectory fields over examples, never over turns."""
    raise NotImplementedError("student task: implement Stage-B metric aggregation")
```

- [ ] **Step 3: Student implements both pure functions and makes focused tests green**

No model, tokenizer, Lucene index, HTTP server, Dataset, Trainer, or W&B call belongs in these functions.

### Task 3: Teacher-owned local BM25 and observation formatting

**Files:**
- Modify: `MedSearch/sft/evaluate_stage_b.py`
- Modify: `MedSearch/docs/tests/test_stage_b_evaluator.py`

- [ ] **Step 1: Test observation formatting**

Require retrieved records to retain `document_id`, `title`, `content`, and `score`; prompt text must use Search-R1's `Doc N(Title: ...) ...` format.

- [ ] **Step 2: Implement `LocalBM25` with the existing Lucene index**

Use `pyserini.search.lucene.LuceneSearcher` and the corpus's numeric `id` field. Build a one-time byte-offset list for the JSONL corpus, seek directly to hit lines, and never load the 746 MB corpus into a Python list. `search(query, top_k)` returns ordered dictionaries with real `document_id` and Lucene score.

- [ ] **Step 3: Verify against one frozen query**

Run a CPU-only test that retrieves at least one document and confirms every returned `document_id` exists in the corpus. Do not judge model quality in this test.

### Task 4: Teacher skeleton for the two-turn environment loop

**Files:**
- Modify: `MedSearch/sft/evaluate_stage_b.py`
- Modify: `MedSearch/docs/tests/test_stage_b_evaluator.py`

- [ ] **Step 1: Test the loop with deterministic fake generation and fake retrieval**

The fake sequence must perform search → information → answer and prove that only the first complete action is appended, environment information is inserted exactly once, and an answer ends the trajectory.

- [ ] **Step 2: Implement `evaluate_one` around injected callables**

Signature:

```python
def evaluate_one(
    sample: dict,
    generate_action,
    retriever,
    *,
    top_k: int,
    max_search_turns: int,
) -> dict:
```

Rules:

- Render `format_stage_b_prompt(sample)` once.
- At each active turn, generate one action, parse it, and retain the raw output.
- For search, retrieve real documents, calculate per-turn evidence hit, append only the parsed action prefix plus `<information>...</information>`, and continue.
- After `max_search_turns`, allow one final generation but execute no further search.
- `forced_stop=True` only when no valid answer exists after the final opportunity.
- A completed answer is correct only when its stripped uppercase content is exactly one of A/B/C/D and equals `gold_answer`.
- Emit full turn records for later human review; never discard malformed outputs.

- [ ] **Step 3: Make loop tests green**

No GPU is used until all pure and fake-loop tests pass.

### Task 5: CLI, model variants, and real paired run

**Files:**
- Modify: `MedSearch/sft/evaluate_stage_b.py`
- Modify: `MedSearch/README.md`

- [ ] **Step 1: Add CLI arguments**

Required arguments: `--config`, `--data`, `--variant`, `--adapter-path`, `--output-dir`. Defaults: `--top-k 5`, `--max-search-turns 2`, `--max-new-tokens 256`, `--temperature 0`, `--seed 20260902`.

- [ ] **Step 2: Reuse Stage-A evaluator loading behavior**

Load the same local Qwen2.5-3B base in 4-bit, attach either `outputs/stage_a/adapter` or `outputs/stage_b/formal/adapter` inference-only, use the same tokenizer and generation arguments, and release the first model before loading the second.

- [ ] **Step 3: Write outputs**

Each variant writes `metrics.json`, `trajectories.jsonl`, and `review.md`. Metadata includes config/data/adapter paths, top-k, turn limit, generation settings, seed, Git commit, protocol SHA-256, and corpus/index paths.

- [ ] **Step 4: Run Stage-A then Stage-B**

Use identical protocol data, BM25 index, top-k, maximum turns, tokenizer, seed, and generation arguments. Do not run both 3B models concurrently on the single GPU.

### Task 6: Acceptance and interpretation

**Files:**
- Modify: `STUDY_PROGRESS.md`
- Modify: `MedSearch/README.md`

- [ ] **Step 1: Run the full CPU regression**

```cmd
wsl bash MedSearch/docs/tests/run_sft_checks.sh
```

Expected: all previous 67 tests plus the new evaluator tests pass.

- [ ] **Step 2: Compare paired metrics**

Report Stage-A and Stage-B side by side for valid think/search, non-empty query, malformed/both-action, Evidence Hit@5, duplicate query, closed-loop completion, valid answer, MCQ accuracy, and forced stop.

- [ ] **Step 3: Preserve the decision boundary**

Stage-B passes only if protocol behavior improves over Stage-A without unacceptable degradation on frozen Stage-A dev and medical human review. Twelve development examples support recipe decisions, not final benchmark claims.
