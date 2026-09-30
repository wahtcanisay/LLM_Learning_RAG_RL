# Stage-A Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Stage-A evaluator that measures held-out assistant-token loss, exports stratified medical-dialogue generations, and audits `<think>`/`<answer>` tag stability without touching the frozen final test sets.

**Architecture:** One small module reuses the frozen Stage-A JSONL, Qwen chat template, existing encoder, and existing QLoRA configuration. Pure helper functions own stratified sampling and tag parsing so CPU tests cover the evaluation contract; the CLI owns model loading, loss inference, generation, and JSON/JSONL/Markdown artifacts.

**Tech Stack:** Python standard library, PyTorch, Transformers, PEFT, existing MedSearch SFT helpers, pytest.

---

### Task 1: Freeze sampling and tag-audit behavior

**Files:**
- Create: `MedSearch/docs/tests/test_stage_a_evaluate.py`
- Create: `MedSearch/sft/evaluate_stage_a.py`

- [ ] **Step 1: Write failing tests**

```python
def test_stratified_sample_is_deterministic_and_balanced():
    selected = stratified_sample(SAMPLES, per_group=2, seed=7)
    assert [sample["id"] for sample in selected] == [sample["id"] for sample in stratified_sample(SAMPLES, 2, 7)]
    assert Counter(sample["mixture_group"] for sample in selected) == {"single": 2, "reasoning": 2}

def test_analyze_tags_distinguishes_valid_missing_and_malformed():
    assert analyze_tags("<think>x</think>\n<answer>y</answer>")["strict_think_answer"] is True
    assert analyze_tags("plain answer")["has_think"] is False
    assert analyze_tags("<think>x")["malformed_think"] is True

def test_build_prompt_removes_only_final_assistant_turn():
    prompt, reference = split_prompt_and_reference(SAMPLE)
    assert prompt == SAMPLE["messages"][:-1]
    assert reference == SAMPLE["messages"][-1]["content"]
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest --rootdir=MedSearch MedSearch/docs/tests/test_stage_a_evaluate.py -q`
Expected: collection fails because `sft.evaluate_stage_a` does not exist.

- [ ] **Step 3: Add minimal pure helpers**

```python
GROUPS = ("single", "multi", "reasoning", "general")

def stratified_sample(samples, per_group, seed): ...
def split_prompt_and_reference(sample): ...
def analyze_tags(text): ...
def summarize_tag_records(records): ...
```

`analyze_tags` records exact tag counts, correct ordering, non-empty content, strict full-output `<think>...<answer>...` structure, and malformed tags. `summarize_tag_records` reports missing expected tags for `reasoning` and unexpected tags for all other groups.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest --rootdir=MedSearch MedSearch/docs/tests/test_stage_a_evaluate.py -q`
Expected: all helper tests pass.

### Task 2: Add loss aggregation and review artifacts

**Files:**
- Modify: `MedSearch/docs/tests/test_stage_a_evaluate.py`
- Modify: `MedSearch/sft/evaluate_stage_a.py`

- [ ] **Step 1: Write failing tests**

```python
def test_aggregate_nll_is_token_weighted_and_grouped():
    result = aggregate_nll([("single", 4.0, 2), ("single", 3.0, 1)])
    assert result["overall"]["nll"] == pytest.approx(7 / 3)
    assert result["single"]["supervised_tokens"] == 3

def test_write_review_markdown_keeps_generated_answer_and_blank_rubric(tmp_path):
    write_review_markdown(tmp_path / "review.md", [RECORD], metadata={"variant": "base"})
    text = (tmp_path / "review.md").read_text(encoding="utf-8")
    assert RECORD["generated"] in text
    assert "医学正确性（0/1/2）：" in text
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest --rootdir=MedSearch MedSearch/docs/tests/test_stage_a_evaluate.py -q`
Expected: missing aggregation and writer functions fail.

- [ ] **Step 3: Add minimal aggregation and writers**

```python
def aggregate_nll(rows): ...
def write_json(path, value): ...
def write_jsonl(path, records): ...
def write_review_markdown(path, records, metadata): ...
```

The Markdown rubric contains medical correctness, relevance/completeness, hallucination, safety risk, tag judgment, and notes. Metrics use summed negative log-likelihood divided by supervised assistant-token count, both overall and by mixture group.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest --rootdir=MedSearch MedSearch/docs/tests/test_stage_a_evaluate.py -q`
Expected: all evaluator tests pass.

### Task 3: Add the local Base/adapter evaluation CLI

**Files:**
- Modify: `MedSearch/docs/tests/test_stage_a_evaluate.py`
- Modify: `MedSearch/sft/evaluate_stage_a.py`
- Modify: `MedSearch/README.md`

- [ ] **Step 1: Write failing CLI-validation tests**

```python
def test_validate_generation_repeats_requires_sampling():
    with pytest.raises(ValueError, match="temperature"):
        validate_generation_settings(repeats=3, temperature=0.0)
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest --rootdir=MedSearch MedSearch/docs/tests/test_stage_a_evaluate.py -q`
Expected: missing validation helper fails.

- [ ] **Step 3: Implement the minimal CLI**

```text
python -m MedSearch.sft.evaluate_stage_a --config MedSearch/configs/stage_a.yaml \
  --data MedSearch/data/processed/sft_stage_a/stage_a_dev.jsonl \
  --variant stage-a-final --adapter-path MedSearch/outputs/stage_a/adapter \
  --output-dir MedSearch/outputs/eval/stage-a-final --per-group 8
```

The CLI loads the same local base/tokenizer, optionally attaches one inference-only adapter, computes loss on all dev examples, generates a deterministic stratified sample, and writes `metrics.json`, `generations.jsonl`, and `review.md`. `--repeats > 1` requires nonzero temperature and enables stochastic prompt-level tag stability checks. No external judge, RAG, final MedQA/MedMCQA test, or new dependency is added.

- [ ] **Step 4: Document Base and adapter commands**

Document two commands differing only in `--variant`, `--adapter-path`, and output directory. State that both runs must keep seed, sample count, and generation settings identical.

- [ ] **Step 5: Run full verification**

Run: `python -m pytest --rootdir=MedSearch MedSearch/docs/tests/test_stage_a_evaluate.py -q`
Run: `python -m compileall -q MedSearch/sft/evaluate_stage_a.py MedSearch/docs/tests/test_stage_a_evaluate.py`
Run: `git diff --check -- MedSearch/sft/evaluate_stage_a.py MedSearch/docs/tests/test_stage_a_evaluate.py MedSearch/README.md docs/superpowers/plans/2026-09-01-stage-a-evaluation.md`
Expected: tests pass and both static checks exit 0.

