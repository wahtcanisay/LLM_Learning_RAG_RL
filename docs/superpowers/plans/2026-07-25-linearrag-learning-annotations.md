# LinearRAG Learning Annotations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add detailed Chinese learning comments to all seven LinearRAG Python files, expose the true execution flow in `run.py`, explain GraphRAG terminology relative to prior MedRAG knowledge, and preserve executable behavior.

**Architecture:** Treat `run.py` as the reading map, `src/LinearRAG.py` as the core algorithm, and the remaining modules as supporting contracts. Add only comments and docstrings to Python files, then prove semantic equivalence by comparing ASTs after removing docstrings.

**Tech Stack:** Python 3, AST, SentenceTransformers, spaCy/SciSpaCy, NumPy, PyTorch sparse tensors, igraph, pandas/Parquet.

---

### Task 1: Establish the behavior-preservation baseline

**Files:**
- Inspect: `LinearRAG/run.py`
- Inspect: `LinearRAG/src/config.py`
- Inspect: `LinearRAG/src/embedding_store.py`
- Inspect: `LinearRAG/src/ner.py`
- Inspect: `LinearRAG/src/LinearRAG.py`
- Inspect: `LinearRAG/src/evaluate.py`
- Inspect: `LinearRAG/src/utils.py`

- [ ] **Step 1: Confirm the worktree contains no unrelated changes**

Run:

```powershell
git status --short
```

Expected: no output before annotation work begins.

Use commit `0a9cfb5` as the immutable pre-annotation source reference for the final AST comparison. This commit contains the approved design and terminology baseline but no LinearRAG Python annotations.

- [ ] **Step 2: Compile the official baseline without importing dependencies**

Run:

```powershell
python -m py_compile LinearRAG/run.py LinearRAG/src/config.py LinearRAG/src/embedding_store.py LinearRAG/src/ner.py LinearRAG/src/LinearRAG.py LinearRAG/src/evaluate.py LinearRAG/src/utils.py
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Record the baseline symbol map**

Run:

```powershell
rg -n "^(class |def |    def |if __name__)" LinearRAG
```

Expected: the output includes `main`, `LinearRAG.index`, `LinearRAG.retrieve`, `LinearRAG.run_ppr`, `SpacyNER`, `EmbeddingStore`, and `Evaluator`.

### Task 2: Turn `run.py` and configuration into the reading map

**Files:**
- Modify: `LinearRAG/run.py:1-75`
- Modify: `LinearRAG/src/config.py:1-31`

- [ ] **Step 1: Add a file-level execution map to `run.py`**

Add a module docstring containing this exact logical order:

```text
参数解析
→ 加载 Embedding、问题、Passage、日志和 LLM
→ 创建 LinearRAGConfig
→ LinearRAG.index() 离线构图
→ LinearRAG.qa()：retrieve() 在线检索 + LLM 生成
→ 保存 predictions.json
→ Evaluator.evaluate()
```

The docstring must explicitly state that `index()` may reuse cached Parquet/JSON files and that this file remains behaviorally unchanged.

- [ ] **Step 2: Annotate `load_dataset()`**

Explain:

```python
passages = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
```

as a stable numeric order prefix used later by `add_adjacent_passage_edges()` to reconstruct neighboring Passage edges. Clarify that the prefix becomes part of the embedded and hashed text.

- [ ] **Step 3: Number the five phases inside `main()`**

Insert block comments for:

```text
① Parse experiment arguments
② Load resources and data
③ Build configuration and initialize LinearRAG
④ Offline index, then online retrieve-and-generate
⑤ Persist predictions and evaluate
```

Also label `CUDA_VISIBLE_DEVICES="4"` as an official multi-GPU environment assumption that is documented but not changed in this task.

- [ ] **Step 4: Group and explain configuration fields**

In `src/config.py`, explain these groups:

```text
Resource identity: dataset_name, embedding_model, llm_model, spacy_model
Persistence/batching: working_dir, batch_size, max_workers
Retrieval output: retrieval_top_k
Entity propagation: max_iterations, top_k_sentence, iteration_threshold
PPR injection: passage_ratio, passage_node_weight, damping
Implementation choice: use_vectorized_retrieval
Optional attribute fallback: enable_hybrid_attribute_fallback, attribute_keyword_boost, attribute_query_keywords
```

State that “vectorized retrieval” here means sparse-matrix graph propagation, not MedRAG Dense Retrieval.

- [ ] **Step 5: Compile the two files**

Run:

```powershell
python -m py_compile LinearRAG/run.py LinearRAG/src/config.py
```

Expected: exit code 0.

- [ ] **Step 6: Commit the entry-map annotations**

```powershell
git add LinearRAG/run.py LinearRAG/src/config.py
git commit -m "docs: map LinearRAG execution flow"
```

### Task 3: Explain persistence, hashing, and entity extraction

**Files:**
- Modify: `LinearRAG/src/embedding_store.py:1-80`
- Modify: `LinearRAG/src/ner.py:1-49`

- [ ] **Step 1: Document `EmbeddingStore` as an incremental text-to-vector cache**

The class docstring must define:

```text
hash_ids: ordered stable IDs
texts: text aligned with hash_ids
embeddings: vector aligned with hash_ids
hash_id_to_idx: ID → vector row
hash_id_to_text: ID → original text
text_to_hash_id: original text → ID
```

Explain that namespaces (`passage`, `entity`, `sentence`) prevent equal text from being treated as the same object type.

- [ ] **Step 2: Explain insertion and persistence**

Annotate `insert_text()` and `_upsert()` to show:

```text
text → namespaced MD5 → remove existing IDs → encode only missing text
→ append aligned lists → rebuild lookup maps → save Parquet
```

At `normalize_embeddings=True`, explain that dot product then corresponds to cosine similarity for normalized vectors.

- [ ] **Step 3: Document `SpacyNER` data contracts**

The class and method comments must distinguish:

```text
passage_hash_id_to_entities:
    Passage hash ID → unique entity mentions in that Passage

sentence_to_entities:
    sentence text → entity mentions occurring in that sentence
```

Explain NER, the filtering of `ORDINAL`/`CARDINAL`, and why sentence text later acts as a semantic bridge.

- [ ] **Step 4: Explain question-side NER**

At `question_ner()`, explain that lowercasing produces normalized query entity strings, but final seed matching is semantic Embedding matching rather than exact string equality.

- [ ] **Step 5: Mark but do not fix the batch-size boundary**

Add a `学习注意` comment explaining:

```python
batch_size = len(passage_list) // max_workers
```

can become zero when passages are fewer than workers. State that behavior is intentionally unchanged because this task is annotation-only.

- [ ] **Step 6: Compile and commit**

Run:

```powershell
python -m py_compile LinearRAG/src/embedding_store.py LinearRAG/src/ner.py
git add LinearRAG/src/embedding_store.py LinearRAG/src/ner.py
git commit -m "docs: explain LinearRAG text stores and NER"
```

Expected: compilation succeeds and the commit contains only comments/docstrings.

### Task 4: Annotate offline relation-free graph construction

**Files:**
- Modify: `LinearRAG/src/LinearRAG.py:18-52`
- Modify: `LinearRAG/src/LinearRAG.py:555-672`

- [ ] **Step 1: Add the core class mental model**

The `LinearRAG` class docstring must define:

```text
Offline:
Passage → NER → Entity/Sentence mappings → embeddings
→ Entity–Passage edges + adjacent Passage edges → igraph

Online:
Question → seed entities → semantic sentence bridges → active entities
→ Passage priors → personalized PageRank → Top-k passages
```

State that relation-free means no typed relation extraction by an LLM.

- [ ] **Step 2: Explain initialization and cached resources**

Annotate `__init__()`, `load_embedding_store()`, and `load_existing_data()` to explain:

- separate stores for Passage, Entity, and Sentence;
- an undirected igraph;
- NER JSON reuse and detection of new Passage IDs;
- BFS-style or vectorized propagation selection.

- [ ] **Step 3: Number the stages of `index()`**

Use comments for these exact stages:

```text
① Persist/encode Passage
② Reuse NER cache and process only new Passage
③ Materialize Entity/Sentence sets and bidirectional mappings
④ Persist/encode Sentence and Entity
⑤ Convert text mappings to hash-ID mappings
⑥ Build Entity–Passage and adjacent Passage edges
⑦ Add graph nodes/edges and save GraphML
```

- [ ] **Step 4: Explain why Sentence is a bridge but not an igraph vertex**

At `extract_nodes_and_edges()` and `add_nodes()`, explicitly state:

- Sentence Embeddings and Entity–Sentence mappings participate in online propagation;
- `add_nodes()` merges only Entity and Passage stores;
- therefore the current implementation does not add Sentence as a formal igraph vertex.

- [ ] **Step 5: Explain the two formal graph-edge families**

At `add_entity_to_passage_edges()`:

```text
edge weight = occurrences of this entity in the Passage
              / total occurrences of all extracted entities in the Passage
```

At `add_adjacent_passage_edges()`:

```text
numeric prefixes created in load_dataset()
→ sort passages
→ connect consecutive passages with weight 1.0
```

- [ ] **Step 6: Compile and commit**

Run:

```powershell
python -m py_compile LinearRAG/src/LinearRAG.py
git add LinearRAG/src/LinearRAG.py
git commit -m "docs: explain LinearRAG offline graph construction"
```

Expected: compilation succeeds.

### Task 5: Annotate online two-stage graph retrieval

**Files:**
- Modify: `LinearRAG/src/LinearRAG.py:53-554`

- [ ] **Step 1: Explain `retrieve()` as the online dispatcher**

Document:

```text
Question embedding + question NER
→ seed entities found?
   yes: graph_search_with_seed_entities()
   no: dense_passage_retrieval()
→ Top-k Passage text and scores
```

Define Dense fallback and distinguish it from graph retrieval.

- [ ] **Step 2: Explain seed entity alignment**

At `get_seed_entities()`, describe:

```text
Question entity text
→ normalized Embedding
→ similarity against all corpus Entity Embeddings
→ best corpus Entity per question entity
```

Define seed entity and note that the current code always chooses one best corpus entity per recognized question entity without an explicit minimum similarity threshold.

- [ ] **Step 3: Explain BFS-style Entity → Sentence → Entity propagation**

At `calculate_entity_scores()`, define:

- active entity;
- tier/hop;
- sentence deduplication;
- per-entity Top-k sentence selection;
- multiplication of incoming entity score and question–sentence similarity;
- threshold pruning;
- `max_iterations` stopping.

Clarify that “BFS-style” describes layer-by-layer control flow, not a call to a standard BFS library.

- [ ] **Step 4: Explain the vectorized propagation path**

At `_precompute_sparse_matrices()` and `calculate_entity_scores_vectorized()`, define:

- sparse adjacency matrix;
- COO indices and values;
- Entity-to-Sentence and Sentence-to-Entity matrix shapes;
- sparse matrix multiplication;
- masks for used sentences and threshold pruning;
- why “vectorized retrieval” is graph propagation acceleration rather than Dense Retrieval.

- [ ] **Step 5: Explain Passage prior scoring**

At `calculate_passage_scores()`, document:

```text
passage_score
  = passage_ratio × normalized dense similarity
  + log(1 + entity occurrence bonus)
  + optional attribute keyword bonus

PPR reset weight
  = passage_score × passage_node_weight
```

Explain tier-based decay, occurrence counts, Min-Max normalization, and the optional attribute fallback.

- [ ] **Step 6: Explain Personalized PageRank**

At `run_ppr()`, define:

- personalization/reset vector;
- damping factor;
- edge weights;
- why only Passage-node scores are retained after PageRank;
- the difference between this graph score and MedRAG’s BM25/Dense/RRF scores.

- [ ] **Step 7: Explain QA generation**

At `qa()`, show:

```text
retrieved Passage text → prompt context → parallel LLM inference
→ parse text after "Answer:" → pred_answer
```

State that retrieval success and answer correctness are still separate evaluation targets.

- [ ] **Step 8: Compile and commit**

Run:

```powershell
python -m py_compile LinearRAG/src/LinearRAG.py
git add LinearRAG/src/LinearRAG.py
git commit -m "docs: explain LinearRAG online graph retrieval"
```

Expected: compilation succeeds.

### Task 6: Explain generation utilities and evaluation boundaries

**Files:**
- Modify: `LinearRAG/src/utils.py:1-77`
- Modify: `LinearRAG/src/evaluate.py:1-96`

- [ ] **Step 1: Annotate utility contracts**

Explain:

- `compute_mdhash_id()` and namespace prefixes;
- the OpenAI-compatible client, `temperature=0`, and why no API is called during annotation verification;
- normalization steps in `normalize_answer()`;
- file plus console logging in `setup_logging()`;
- Min-Max normalization and its equal-value fallback.

- [ ] **Step 2: Annotate evaluation metrics**

At `Evaluator`, distinguish:

```text
LLM Accuracy:
model judges semantic correctness against gold answer

Contain Accuracy:
normalized gold answer must occur inside normalized prediction
```

State that neither metric measures retrieval Recall@k, MRR, or nDCG.

- [ ] **Step 3: Explain concurrent evaluation and output files**

Document the mapping:

```text
prediction sample → evaluate_sig_sample()
→ llm_accuracy + contain_accuracy
→ write per-sample fields back to predictions.json
→ aggregate evaluation_results.json
```

- [ ] **Step 4: Compile and commit**

Run:

```powershell
python -m py_compile LinearRAG/src/utils.py LinearRAG/src/evaluate.py
git add LinearRAG/src/utils.py LinearRAG/src/evaluate.py
git commit -m "docs: explain LinearRAG utilities and evaluation"
```

Expected: compilation succeeds.

### Task 7: Verify comment-only semantics and prepare today’s study checkpoint

**Files:**
- Modify: `STUDY_PROGRESS.md`
- Verify: all seven LinearRAG Python files

- [ ] **Step 1: Run syntax verification**

Run:

```powershell
python -m py_compile LinearRAG/run.py LinearRAG/src/config.py LinearRAG/src/embedding_store.py LinearRAG/src/ner.py LinearRAG/src/LinearRAG.py LinearRAG/src/evaluate.py LinearRAG/src/utils.py
```

Expected: exit code 0.

- [ ] **Step 2: Compare executable ASTs with `HEAD`**

Run this temporary in-memory Python check:

1. reads each working-tree Python file;
2. reads the same file from `git show 0a9cfb5:<path>`;
3. parses both with `ast.parse`;
4. removes leading string-expression docstrings from modules, classes, and functions;
5. compares `ast.dump(..., include_attributes=False)`.

```powershell
@'
import ast
import pathlib
import subprocess

paths = [
    "LinearRAG/run.py",
    "LinearRAG/src/config.py",
    "LinearRAG/src/embedding_store.py",
    "LinearRAG/src/ner.py",
    "LinearRAG/src/LinearRAG.py",
    "LinearRAG/src/evaluate.py",
    "LinearRAG/src/utils.py",
]

class RemoveDocstrings(ast.NodeTransformer):
    def _strip(self, node):
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

for path in paths:
    before = subprocess.check_output(
        ["git", "show", f"0a9cfb5:{path}"],
        text=True,
        encoding="utf-8",
    )
    after = pathlib.Path(path).read_text(encoding="utf-8")
    before_tree = RemoveDocstrings().visit(ast.parse(before))
    after_tree = RemoveDocstrings().visit(ast.parse(after))
    ast.fix_missing_locations(before_tree)
    ast.fix_missing_locations(after_tree)
    assert ast.dump(before_tree, include_attributes=False) == ast.dump(
        after_tree,
        include_attributes=False,
    ), f"EXECUTABLE_AST_CHANGED: {path}"
    print(f"COMMENT_ONLY_AST_OK: {path}")
'@ | python -
```

Expected:

```text
COMMENT_ONLY_AST_OK: LinearRAG/run.py
COMMENT_ONLY_AST_OK: LinearRAG/src/config.py
COMMENT_ONLY_AST_OK: LinearRAG/src/embedding_store.py
COMMENT_ONLY_AST_OK: LinearRAG/src/ner.py
COMMENT_ONLY_AST_OK: LinearRAG/src/LinearRAG.py
COMMENT_ONLY_AST_OK: LinearRAG/src/evaluate.py
COMMENT_ONLY_AST_OK: LinearRAG/src/utils.py
```

- [ ] **Step 3: Check whitespace and inspect the final diff**

Run:

```powershell
git diff --check
git diff --stat
git diff -- LinearRAG
```

Expected: no whitespace errors; Python diffs contain only comments/docstrings.

- [ ] **Step 4: Update `STUDY_PROGRESS.md` with facts only**

Record:

- current stage: Stage 3 LinearRAG;
- completed work: seven-file annotation and entry call map;
- verification commands and their actual outputs;
- no datasets/models downloaded, no API call, no GPU use, no retrieval/QA metrics;
- today’s sole task: read `run.py → LinearRAG.index()`;
- completion criteria and five code-based questions.

- [ ] **Step 5: Commit the verified annotation set**

```powershell
git add LinearRAG STUDY_PROGRESS.md
git commit -m "docs: annotate LinearRAG learning path"
```

- [ ] **Step 6: Ask the learner these five questions**

```text
1. 为什么 LinearRAG 被称为 relation-free？它没有抽取什么，又用什么连接语料？
2. Sentence 为什么被称为语义桥？它是否真的被 add_nodes() 加成 igraph 顶点？
3. Seed Entity 如何从问题实体得到？这一步与 MedRAG Dense Passage Retrieval 有什么不同？
4. calculate_entity_scores() 中 threshold、top_k_sentence、max_iterations 分别控制什么？
5. PPR 的 reset vector 由哪些分数组成？最终为什么只排序 Passage 节点？
```
