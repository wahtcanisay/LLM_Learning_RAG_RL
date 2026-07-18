# Integrate MedSearch-R1 Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Mini-WebSeer with the approved MedSearch-R1 project across the repository's formal learning-route documents and push the documentation update to GitHub.

**Architecture:** Keep the existing six-stage order and preserve stages 1–5. Replace only stage 6 with MedSearch-R1, explicitly connecting MedRAG retrieval, MedicalGPT domain SFT, Search-R1 agentic RL, and LA-CDM-inspired hypothesis, confidence, and cost control. Record the decision in the progress file without changing the current Stage 1 daily task.

**Tech Stack:** Markdown, UTF-8 text, Git

---

### Task 1: Update the authoritative route

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Replace the route entry**

Change `阶段 6：Mini-WebSeer-3B` to `阶段 6：MedSearch-R1 医学证据搜索 Agent` while leaving stages 1–5 unchanged.

- [ ] **Step 2: Replace the complete Stage 6 section**

Define the project goal, its relationship to MedRAG/MedicalGPT/Search-R1/LA-CDM, the Search/Inspect/Submit tools, the 32GB starting configuration, dataset boundaries, required baselines, metrics, failure analysis, and completion criteria from the approved design.

- [ ] **Step 3: Update cross-references**

Replace statements that prepare for or warn against skipping directly to WebSeer with equivalent MedSearch-R1 language.

### Task 2: Synchronize the secondary instruction file

**Files:**
- Modify: `LLM_Study_Agent_Instructions.txt`

- [ ] **Step 1: Replace its Mini-WebSeer stage**

Use the same MedSearch-R1 scope and safety boundary as `AGENTS.md`, adapted to the shorter structure of this file.

- [ ] **Step 2: Update skip-stage guidance**

Require completion of MedRAG, MedicalGPT, and Search-R1 prerequisites before MedSearch-R1.

### Task 3: Record the approved route decision

**Files:**
- Modify: `STUDY_PROGRESS.md`

- [ ] **Step 1: Add a route-decision record**

Record the 2026-07-18 decision to replace Mini-WebSeer with MedSearch-R1 and link the approved design document.

- [ ] **Step 2: Preserve the current task**

Confirm that the current stage remains Stage 1 MedRAG and that the existing daily Retriever-reading task is not replaced.

### Task 4: Verify, commit, and push

**Files:**
- Verify: `AGENTS.md`
- Verify: `LLM_Study_Agent_Instructions.txt`
- Verify: `STUDY_PROGRESS.md`
- Verify: `docs/superpowers/specs/2026-07-18-medsearch-r1-design.md`
- Verify: `docs/superpowers/plans/2026-07-18-integrate-medsearch-r1-route.md`

- [ ] **Step 1: Search for stale route references**

Run:

```powershell
rg -n "Mini-WebSeer|阶段 6：MedSearch-R1|MedSearch-R1" AGENTS.md LLM_Study_Agent_Instructions.txt STUDY_PROGRESS.md docs/superpowers
```

Expected: no `Mini-WebSeer` reference remains in the formal route files; MedSearch-R1 appears in both instruction files, the progress record, and the approved design.

- [ ] **Step 2: Validate whitespace and scope**

Run targeted `git diff --check` only on the documentation files because `MedRAG/src/utils.py` contains a pre-existing user modification that must remain untouched.

- [ ] **Step 3: Commit only route documentation**

Stage `AGENTS.md`, `LLM_Study_Agent_Instructions.txt`, `STUDY_PROGRESS.md`, and this plan. Confirm `MedRAG/src/utils.py` is not staged. Commit with message `docs: integrate MedSearch-R1 into learning route`.

- [ ] **Step 4: Push and verify remote**

Push the current `main` branch to `origin`, then compare local `HEAD` with `refs/remotes/origin/main`. Expected: both commit IDs are identical.
