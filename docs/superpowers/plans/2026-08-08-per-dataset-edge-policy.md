# Per-Dataset Document Edge Policy 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `MedicalGraphRAG/` 内实现单数据集 document/chunk 两种 retrieval unit 的独立建库——摘要型数据以完整 document 为 Passage 节点并构建 kNN Similarity 软边,长文本数据以 chunk 为 Passage 节点并构建同文档 Adjacent 边(权 1.0)——通过文档级公平基线对比,产出 Phase 1 诚实检索指标。

**Architecture:** 规格 `docs/superpowers/specs/2026-08-08-per-dataset-document-edge-policy-design.md` 是唯一依据。新增 `data/retrieval_passages.py`(loader + window 覆盖)与 `retrieval/graph_edges.py`(纯函数边构建);把离线边策略参数拆为 `GraphBuildConfig`,与在线 PPR 的 `GraphConfig` 分离;`build_graph_index` / `LinearGraphRetriever` / `search_graph` / `evaluation.graph` 按 `retrieval_unit` 分支 document/chunk。**禁止合并大库、禁止跨数据集建边、禁止用 test split 调参。**

**Tech Stack:** Python 3.12 / igraph / sentence-transformers(all-mpnet-base-v2)/ faiss(IndexFlatIP)/ spacy(BC5CDR)/ Pyserini(Lucene BM25)/ numpy。运行环境 WSL2 容器 `llm-pytorch`,`/opt/venv/bin/python`。

**文件结构(本计划要新建/修改):**

```text
MedicalGraphRAG/src/medical_graphrag/
  data/retrieval_passages.py       # 新建: RetrievalPassage + load_retrieval_passages
  retrieval/graph_edges.py         # 新建: build_similarity_edges / build_adjacent_edges(纯函数)
  retrieval/graph.py               # 改: GraphBuildConfig + build_graph_index 重构 + LinearGraphRetriever 读 unit
  retrieval/dense.py               # 改: embed_documents_full(window 全覆盖) + build_dense_document_index
  retrieval/bm25.py                # 改: export_document_collection(文档级 Lucene)
  retrieval/search_graph.py        # 改: search_one/run_search 按 retrieval_unit 分支
  evaluation/graph.py              # 改: document unit 不做 collapse
  run_pipeline.py                  # 改: run_graph_document / run_dense_document / run_bm25_document / run_hybrid_document
  cli.py                           # 改: run 子命令支持 profile 参数
  configs/                         # 新建: pubmedqa_hard_v1_document.json
  tests/test_retrieval_passages.py # 新建
  tests/test_graph_edges.py        # 新建
  tests/test_window_embedding.py   # 新建
  tests/test_graph_document.py     # 新建
```

**约定:**
- 产物目录命名(规格 §10):`indexes/<ds>/graph_document_ep_v1/`、`indexes/<ds>/graph_document_similarity_v1/`、`experiments/<ds>/{bm25,dense,hybrid,graph_document_ep_v1,graph_document_similarity_v1}/`。历史 chunk-level 目录(`graph_abstract_only` 等)**不删不改不重命名**。
- 提交信息按项目风格:`feat:`/`test:`/`refactor:`/`docs:` + 中文描述,末尾追加 `Co-Authored-By: Claude <noreply@anthropic.com>`。
- 每次提交前运行 `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/ -q`,必须全绿;新增文件只依赖 stdlib + 轻量 mock,不在 host 拉模型。

---

## Phase 0: 基线确认

### Task 0: 确认历史 chunk-level 结果与测试基线

**Files:** 无(只读)

- [ ] **Step 1: 跑现有测试,确认全绿**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/ -q`
Expected: 全部 PASS(或已知失败集不变)。

- [ ] **Step 2: 记录 pubmedqa graph_abstract_only 历史指标(作对照,不覆盖)**

```bash
cat experiments/pubmedqa_hard_v1/graph_abstract_only/metrics.json
```

Expected: 记录 test 的 recall@10/mrr@10/ndcg@10 到本计划的备注,供 Phase 1 完成后做"不跨 retrieval unit 对比"的说明。

- [ ] **Step 3: Commit 基线快照**

```bash
git add docs/superpowers/plans/2026-08-08-per-dataset-edge-policy.md
git commit -m "docs: per-dataset edge policy 实施计划(Phase 0 基线)"
```

---

## Phase 1: PubMedQA document-level 闭环

### Task 1: RetrievalPassage loader

**Files:**
- Create: `MedicalGraphRAG/src/medical_graphrag/data/retrieval_passages.py`
- Test: `MedicalGraphRAG/tests/test_retrieval_passages.py`

- [ ] **Step 1: 写失败测试**

```python
import json
from pathlib import Path

import pytest

from medical_graphrag.data.retrieval_passages import (
    RetrievalPassage,
    load_retrieval_passages,
)


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "documents.jsonl").write_text(
        "\n".join([
            json.dumps({"doc_id": "PMID:1", "title": "t1",
                        "content": "full abstract one", "source": "pubmedqa", "year": "1992"}),
            json.dumps({"doc_id": "PMID:2", "title": "t2",
                        "content": "full abstract two", "source": "pubmedqa", "year": "1993"}),
        ]) + "\n",
        encoding="utf-8",
    )
    (dataset / "chunks.jsonl").write_text(
        "\n".join([
            json.dumps({"chunk_id": "PMID:1#0", "doc_id": "PMID:1", "order": 0,
                        "title": "t1", "content": "full abstract one", "source": "pubmedqa"}),
            json.dumps({"chunk_id": "PMID:1#1", "doc_id": "PMID:1", "order": 1,
                        "title": "t1", "content": "more text", "source": "pubmedqa"}),
        ]) + "\n",
        encoding="utf-8",
    )
    for name in ("questions.jsonl", "qrels.tsv"):
        (dataset / name).write_text("", encoding="utf-8")
    (dataset / "manifest.json").write_text(
        json.dumps({
            "counts": {"questions": 0, "documents": 2, "chunks": 2, "qrels": 0},
            "artifact_hashes": {name: _sha(dataset / name)
                                for name in ("questions.jsonl", "documents.jsonl",
                                             "chunks.jsonl", "qrels.tsv")},
        }),
        encoding="utf-8",
    )
    return dataset


def test_load_documents_uses_full_content_as_single_passage(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    passages = load_retrieval_passages(dataset, "document")
    assert len(passages) == 2
    assert passages[0].passage_id == "PMID:1"
    assert passages[0].doc_id == "PMID:1"
    assert passages[0].content == "full abstract one"
    assert passages[0].order is None
    assert passages[1].passage_id == "PMID:2"


def test_load_chunks_keeps_doc_and_order(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    passages = load_retrieval_passages(dataset, "chunk")
    assert [p.passage_id for p in passages] == ["PMID:1#0", "PMID:1#1"]
    assert [p.order for p in passages] == [0, 1]
    assert passages[1].doc_id == "PMID:1"


def test_document_loader_rejects_missing_or_duplicate_doc(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    # 破坏哈希,使 manifest 校验失败
    (dataset / "documents.jsonl").write_text(
        '{"doc_id":"PMID:1","title":"t","content":"c","source":"s","year":"y"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        load_retrieval_passages(dataset, "document")


def test_chunk_loader_rejects_missing_order_field(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    row = {"chunk_id": "x#0", "doc_id": "x", "title": "t",
           "content": "c", "source": "s"}  # 缺 order
    (dataset / "chunks.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="order"):
        load_retrieval_passages(dataset, "chunk")
```

- [ ] **Step 2: 运行确认失败**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_retrieval_passages.py -v`
Expected: FAIL("ModuleNotFoundError: medical_graphrag.data.retrieval_passages")

- [ ] **Step 3: 实现 loader**

```python
"""Unified retrieval-passage loading for document / chunk retrieval units.

document unit → one Passage per frozen ``documents.jsonl`` row, passage_id = doc_id.
chunk unit    → one Passage per frozen ``chunks.jsonl`` row, doc/order retained
                (the only inputs allowed to build Adjacent edges).
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from medical_graphrag.data.io import sha256_file
from medical_graphrag.retrieval.bm25 import validate_frozen_dataset


@dataclass(frozen=True)
class RetrievalPassage:
    passage_id: str
    doc_id: str
    order: int | None
    title: str
    content: str
    source: str


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path.name} must contain JSON objects")
                rows.append(value)
    return rows


def load_retrieval_passages(
    dataset_dir: Path,
    retrieval_unit: Literal["document", "chunk"],
) -> list[RetrievalPassage]:
    validate_frozen_dataset(dataset_dir)
    if retrieval_unit == "document":
        rows = _read_jsonl(dataset_dir / "documents.jsonl")
        seen: set[str] = set()
        passages: list[RetrievalPassage] = []
        for row in rows:
            doc_id = str(row["doc_id"]).strip()
            content = str(row["content"]).strip()
            if not doc_id or doc_id in seen:
                raise ValueError(f"doc_id must be non-empty and unique: {doc_id!r}")
            if not content:
                raise ValueError(f"document {doc_id} content must not be empty")
            seen.add(doc_id)
            passages.append(RetrievalPassage(
                passage_id=doc_id,
                doc_id=doc_id,
                order=None,
                title=str(row.get("title", "")),
                content=content,
                source=str(row.get("source", "")),
            ))
        return passages
    rows = _read_jsonl(dataset_dir / "chunks.jsonl")
    passages = []
    seen_chunks: set[str] = set()
    for row in rows:
        chunk_id = str(row["chunk_id"]).strip()
        doc_id = str(row["doc_id"]).strip()
        order = row.get("order")
        if order is None:
            raise ValueError(f"chunk {chunk_id} missing order")
        if not chunk_id or chunk_id in seen_chunks:
            raise ValueError(f"chunk_id must be non-empty and unique: {chunk_id!r}")
        if not doc_id or not str(row.get("content", "")).strip():
            raise ValueError(f"chunk {chunk_id} must have doc_id and content")
        seen_chunks.add(chunk_id)
        passages.append(RetrievalPassage(
            passage_id=chunk_id,
            doc_id=doc_id,
            order=int(order),
            title=str(row.get("title", "")),
            content=str(row["content"]),
            source=str(row.get("source", "")),
        ))
    return passages
```

- [ ] **Step 4: 运行确认通过**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_retrieval_passages.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/medical_graphrag/data/retrieval_passages.py tests/test_retrieval_passages.py
git commit -m "feat: RetrievalPassage document/chunk loader

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 2: document 全窗口覆盖 embedding

**Files:**
- Modify: `MedicalGraphRAG/src/medical_graphrag/retrieval/dense.py`
- Test: `MedicalGraphRAG/tests/test_window_embedding.py`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pytest

from medical_graphrag.retrieval.dense import _token_windows, _num_special_tokens


class _MockTokenizer:
    def encode(self, text, add_special_tokens=True):
        # 简单字符 tokenizer: 每个字符一个 token; special token = "s" 前缀
        toks = list(text)
        if add_special_tokens:
            toks = ["[CLS]"] + toks + ["[SEP]"]
        return toks


class _MockEmbedder:
    def __init__(self):
        self.tokenizer = _MockTokenizer()

    def get_max_seq_length(self):
        return 8  # 最多 8 个(含 special)token

    def encode(self, texts, normalize_embeddings=True, batch_size=64,
               show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            # 确定性: 向量 = one-hot of 字符数, L2 归一化
            v = np.zeros(6, dtype=np.float32)
            for ch in t:
                v[ord(ch) % 6] += 1.0
            if normalize_embeddings:
                v = v / (np.linalg.norm(v) + 1e-12)
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)


def test_token_windows_cover_all_tokens():
    windows = _token_windows([1, 2, 3, 4, 5, 6, 7, 8, 9], max_window_len=4, overlap_tokens=1)
    flat = [tok for w in windows for tok in w]
    assert set(flat) == {1, 2, 3, 4, 5, 6, 7, 8, 9}  # 全覆盖
    assert windows == [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9]]


def test_token_windows_reject_bad_params():
    with pytest.raises(ValueError):
        _token_windows([1, 2], max_window_len=0, overlap_tokens=0)
    with pytest.raises(ValueError):
        _token_windows([1, 2], max_window_len=3, overlap_tokens=3)


def test_num_special_tokens_mock():
    assert _num_special_tokens(_MockTokenizer()) == 2  # [CLS], [SEP]
```

- [ ] **Step 2: 运行确认失败**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_window_embedding.py -v`
Expected: FAIL("ImportError: cannot import name '_token_windows'")

- [ ] **Step 3: 实现窗口函数**

```python
def _token_windows(
    token_ids: list[int],
    *,
    max_window_len: int,
    overlap_tokens: int,
) -> list[list[int]]:
    """Split token ids into overlapping windows covering every token at least once.

    max_window_len is the max CONTENT tokens per window such that the model's
    special tokens still fit inside its max_seq_length.
    """
    if max_window_len <= 0 or overlap_tokens < 0 or overlap_tokens >= max_window_len:
        raise ValueError("require max_window_len > overlap_tokens >= 0")
    stride = max_window_len - overlap_tokens
    windows: list[list[int]] = []
    start = 0
    n = len(token_ids)
    while start < n:
        windows.append(token_ids[start:start + max_window_len])
        if start + max_window_len >= n:
            break
        start += stride
    return windows


def _num_special_tokens(tokenizer) -> int:
    """Number of special tokens the model prepends/appends (e.g. [CLS]+[SEP] = 2)."""
    full = tokenizer.encode("x", add_special_tokens=True)
    plain = tokenizer.encode("x", add_special_tokens=False)
    return len(full) - len(plain)
```

- [ ] **Step 4: 写 embed_documents_full 失败测试**

```python
from medical_graphrag.retrieval.dense import embed_documents_full


def test_embed_documents_full_mean_then_l2(tmp_path):
    texts = ["abc" + "d" * 40]  # 超过 mock max_seq_length(8)→ 多窗口
    embeddings, window_counts, truncated = embed_documents_full(
        texts, _MockEmbedder(), overlap_tokens=2
    )
    assert truncated == 0
    assert embeddings.shape[1] == 6
    assert window_counts[0] >= 2
    # 单文档 embedding 是单位向量
    assert np.isclose(np.linalg.norm(embeddings[0]), 1.0, atol=1e-5)


def test_embed_documents_full_hand_computed_mean():
    # "ab": 一个窗口 → embedding == 单窗口向量
    embedder = _MockEmbedder()
    embeddings, counts, truncated = embed_documents_full(["ab"], embedder, overlap_tokens=0)
    expected = embedder.encode("ab", normalize_embeddings=True)[0]
    assert np.allclose(embeddings[0], expected, atol=1e-5)
```

- [ ] **Step 5: 实现 embed_documents_full**

```python
def embed_documents_full(
    texts: list[str],
    embedder,
    *,
    overlap_tokens: int = 32,
    batch_size: int = 64,
    normalize: bool = True,
) -> tuple[np.ndarray, list[int], int]:
    """Full-coverage document embeddings (mean-window-then-L2).

    Tokenize without truncation → split into windows that fit max_seq_length
    after special tokens → embed each window (L2-normalized) → mean → L2.
    Returns (embeddings, window_counts, truncated_token_count).
    truncated_token_count is 0 by construction; a non-zero value must fail the build.
    """
    import numpy as np

    tokenizer = embedder.tokenizer
    max_seq_length = int(embedder.get_max_seq_length() or 512)
    special = _num_special_tokens(tokenizer)
    max_window_len = max_seq_length - special
    if max_window_len <= 0:
        raise ValueError("embedder max_seq_length too small for its special tokens")

    all_embeddings: list[np.ndarray] = []
    window_counts: list[int] = []
    total_truncated = 0
    for text in texts:
        token_ids = list(tokenizer.encode(text, add_special_tokens=False))
        windows = _token_windows(token_ids, max_window_len=max_window_len,
                                 overlap_tokens=overlap_tokens)
        # 每个 window 解码回文本,独立编码(不截断:window 已受 max_window_len 约束)
        window_embeddings = embedder.encode(
            [tokenizer.decode(w, skip_special_tokens=True) for w in windows],
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        if len(windows) == 0:
            window_embeddings = np.zeros((1, 0), dtype=np.float32)
        mean_vec = np.asarray(window_embeddings, dtype=np.float32).mean(axis=0)
        if normalize:
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
        all_embeddings.append(np.asarray(mean_vec, dtype=np.float32))
        window_counts.append(len(windows))
        total_truncated += max(len(token_ids) - sum(len(w) for w in windows), 0)
    return np.vstack(all_embeddings), window_counts, int(total_truncated)
```

- [ ] **Step 6: 运行全部窗口测试确认通过**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_window_embedding.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add src/medical_graphrag/retrieval/dense.py tests/test_window_embedding.py
git commit -m "feat: document 全窗口覆盖 embedding(mean-window-then-L2)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 3: Similarity 边构建(纯函数)

**Files:**
- Create: `MedicalGraphRAG/src/medical_graphrag/retrieval/graph_edges.py`
- Test: `MedicalGraphRAG/tests/test_graph_edges.py`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pytest

from medical_graphrag.retrieval.graph_edges import build_similarity_edges


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def test_similarity_edges_knn_threshold_union_dedup():
    # 3 个文档: d0 与 d1 相似, d2 与二者都远
    emb = np.array([
        _unit([1.0, 0.0]),
        _unit([0.95, 0.312]),   # cos(d0,d1) ≈ 0.95
        _unit([0.0, 1.0]),
    ], dtype=np.float32)
    edges = build_similarity_edges(["d0", "d1", "d2"], emb, k=5, min_cosine=0.90)
    assert ("d0", "d1") in edges or ("d1", "d0") in edges
    for a, b, w in edges:
        assert a != b                     # 无自环
        assert 0.90 <= w <= 1.0 + 1e-6    # 权值域
    assert not any("d2" in (a, b) for a, b, _ in edges)  # d2 无合格邻居


def test_similarity_edges_union_knn_keeps_pair_when_one_direction_in_topk():
    # d1 的 top-1 是 d2,但 d2 的 top-1 是 d3; union 下 (d1,d2) 仍保留
    emb = np.array([
        _unit([1.0, 0.0, 0.0]),   # d0
        _unit([0.0, 1.0, 0.0]),   # d1
        _unit([0.0, 0.99, 0.1]),  # d2 ~ d1
        _unit([0.0, 0.1, 0.99]),  # d3 ~ d2
    ], dtype=np.float32)
    edges = build_similarity_edges(["d0", "d1", "d2", "d3"], emb, k=1, min_cosine=0.5)
    pairs = {(a, b) for a, b, _ in edges}
    assert ("d1", "d2") in pairs or ("d2", "d1") in pairs


def test_similarity_edges_no_duplicate_pairs_and_all_isolated_reports(tmp_path):
    emb = np.eye(4, dtype=np.float32)  # 全部正交
    edges = build_similarity_edges(["a", "b", "c", "d"], emb, k=2, min_cosine=0.99)
    assert edges == []
```

- [ ] **Step 2: 运行确认失败**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_graph_edges.py -v`
Expected: FAIL("ModuleNotFoundError: graph_edges")

- [ ] **Step 3: 实现 build_similarity_edges**

```python
"""Passage–Passage edge builders (pure functions, no graph side effects).

Similarity and Adjacent are mutually exclusive: build_graph_index must call
exactly one of them per index (spec §6.3).
"""
import faiss
import numpy as np


def build_similarity_edges(
    passage_ids: list[str],
    embeddings: np.ndarray,
    *,
    k: int = 5,
    min_cosine: float = 0.50,
    scale: float = 1.0,
) -> list[tuple[str, str, float]]:
    """kNN similarity soft edges over L2-normalized passage embeddings.

    - request k+1 nearest neighbours, drop self;
    - drop candidates below min_cosine;
    - union-kNN: keep undirected pair (min_id, max_id) if EITHER direction is in
      its top-k;
    - dedup identical pairs; edge weight = dot product × scale (default 1.0).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not (0 <= min_cosine <= 1):
        raise ValueError("min_cosine must be in [0, 1]")
    if scale <= 0:
        raise ValueError("similarity_edge_scale must be > 0")
    n = len(passage_ids)
    if n == 0:
        return []
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)
    k_eff = min(k + 1, n)
    scores, idx = index.search(embeddings, k_eff)
    edges: dict[tuple[str, str], float] = {}
    for i in range(n):
        for pos in range(k_eff):
            j = int(idx[i][pos])
            if j == i:
                continue
            s = float(scores[i][pos])
            if s < min_cosine:
                continue
            a, b = (passage_ids[i], passage_ids[j])
            lo, hi = (a, b) if a <= b else (b, a)
            if lo == hi:
                continue
            edges[(lo, hi)] = min(max(s, min_cosine), 1.0) * scale
    return [(a, b, w) for (a, b), w in sorted(edges.items())]
```

- [ ] **Step 4: 运行确认通过**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_graph_edges.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/medical_graphrag/retrieval/graph_edges.py tests/test_graph_edges.py
git commit -m "feat: similarity kNN 软边构建(union-knn/阈值/去重)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 4: Adjacent 边构建(纯函数)

**Files:**
- Modify: `MedicalGraphRAG/src/medical_graphrag/retrieval/graph_edges.py`
- Test: `MedicalGraphRAG/tests/test_graph_edges.py`

- [ ] **Step 1: 追加失败测试**

```python
from medical_graphrag.data.retrieval_passages import RetrievalPassage
from medical_graphrag.retrieval.graph_edges import build_adjacent_edges


def _p(passage_id, doc_id, order):
    return RetrievalPassage(passage_id=passage_id, doc_id=doc_id, order=order,
                            title="t", content="c", source="s")


def test_adjacent_edges_two_docs_three_chunks_each():
    passages = [_p("A0", "A", 0), _p("A1", "A", 1), _p("A2", "A", 2),
                _p("B0", "B", 0), _p("B1", "B", 1), _p("B2", "B", 2)]
    edges, gaps = build_adjacent_edges(passages)
    assert len(edges) == 4  # (A0,A1)(A1,A2)(B0,B1)(B1,B2)
    assert all(w == 1.0 for _, _, w in edges)
    assert not any(a.startswith("A") and b.startswith("B") for a, b, _ in edges)
    assert gaps == []


def test_adjacent_edges_order_gap_never_crosses():
    passages = [_p("A0", "A", 0), _p("A1", "A", 2), _p("A2", "A", 3)]
    edges, gaps = build_adjacent_edges(passages)
    assert edges == [("A1", "A2", 1.0)]  # A0→A1 有 gap(order 1 缺失),不连
    assert gaps == [("A", 0, 2)]


def test_adjacent_edges_duplicate_order_fails():
    passages = [_p("A0", "A", 0), _p("A1", "A", 0)]
    with pytest.raises(ValueError, match="unique"):
        build_adjacent_edges(passages)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_graph_edges.py -v`
Expected: FAIL("ImportError: cannot import name 'build_adjacent_edges'")

- [ ] **Step 3: 实现 build_adjacent_edges**

```python
def build_adjacent_edges(
    passages: list["RetrievalPassage"],
) -> tuple[list[tuple[str, str, float]], list[tuple[str, int, int]]]:
    """Same-document adjacent edges, weight strictly 1.0.

    Group by doc_id, sort by numeric order, only connect next.order == order+1.
    Order gaps are recorded and never crossed; duplicate/negative/unparseable
    order raises. Returns (edges, gaps) where gaps is [(doc_id, from_order, to_order)].
    """
    from collections import defaultdict

    by_doc: dict[str, list[RetrievalPassage]] = defaultdict(list)
    for p in passages:
        by_doc[p.doc_id].append(p)

    edges: list[tuple[str, str, float]] = []
    gaps: list[tuple[str, int, int]] = []
    for doc_id in sorted(by_doc):
        group = sorted(by_doc[doc_id], key=lambda p: p.order)
        orders = [p.order for p in group]
        if any(o is None or o < 0 for o in orders):
            raise ValueError(f"doc {doc_id} has negative/missing order")
        if len(set(orders)) != len(orders):
            raise ValueError(f"doc {doc_id} has duplicate order values")
        for prev, cur in zip(group, group[1:]):
            if cur.order == prev.order + 1:
                a, b = prev.passage_id, cur.passage_id
                lo, hi = (a, b) if a <= b else (b, a)
                edges.append((lo, hi, 1.0))
            else:
                gaps.append((doc_id, prev.order, cur.order))
    edges.sort(key=lambda e: (e[0], e[1]))
    return edges, gaps
```

- [ ] **Step 4: 运行确认通过**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_graph_edges.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/medical_graphrag/retrieval/graph_edges.py tests/test_graph_edges.py
git commit -m "feat: adjacent 同文档相邻边(权1.0,跨 gap 禁止)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 5: GraphBuildConfig 与配置校验

**Files:**
- Modify: `MedicalGraphRAG/src/medical_graphrag/retrieval/graph.py`
- Test: `MedicalGraphRAG/tests/test_graph.py`(追加)

- [ ] **Step 1: 写失败测试(追加到 test_graph.py)**

```python
import pytest

from medical_graphrag.retrieval.graph import GraphBuildConfig


def test_graph_build_config_defaults():
    c = GraphBuildConfig(retrieval_unit="document", passage_edge_mode="similarity")
    assert c.similarity_k == 5
    assert c.similarity_min_cosine == 0.50
    assert c.similarity_edge_scale == 1.0
    assert c.window_overlap_tokens == 32


@pytest.mark.parametrize("unit,mode", [
    ("document", "adjacent"),   # document + adjacent 非法
    ("chunk", "similarity"),    # chunk + similarity 在 v1 非法
])
def test_graph_build_config_illegal_combinations(unit, mode):
    with pytest.raises(ValueError):
        GraphBuildConfig(retrieval_unit=unit, passage_edge_mode=mode)


@pytest.mark.parametrize("kwargs", [
    {"similarity_k": 0},
    {"similarity_min_cosine": 1.5},
    {"similarity_edge_scale": 0.0},
])
def test_graph_build_config_bad_ranges(kwargs):
    with pytest.raises(ValueError):
        GraphBuildConfig(retrieval_unit="document", passage_edge_mode="similarity", **kwargs)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_graph.py -v`
Expected: FAIL("ImportError: cannot import name 'GraphBuildConfig'")

- [ ] **Step 3: 实现 GraphBuildConfig**

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class GraphBuildConfig:
    """Offline graph-construction parameters (edge strategy), kept separate from
    the online PPR GraphConfig (spec §9)."""
    retrieval_unit: Literal["document", "chunk"]
    passage_edge_mode: Literal["none", "similarity", "adjacent"]
    embedding_model: str
    ner_model: str
    similarity_k: int = 5
    similarity_min_cosine: float = 0.50
    similarity_edge_scale: float = 1.0
    window_overlap_tokens: int = 32

    def __post_init__(self) -> None:
        if self.retrieval_unit == "document" and self.passage_edge_mode == "adjacent":
            raise ValueError("document + adjacent is illegal")
        if self.retrieval_unit == "chunk" and self.passage_edge_mode == "similarity":
            raise ValueError("chunk + similarity is illegal in v1")
        if self.similarity_k < 1:
            raise ValueError("similarity_k must be >= 1")
        if not (0 <= self.similarity_min_cosine <= 1):
            raise ValueError("similarity_min_cosine must be in [0, 1]")
        if self.similarity_edge_scale <= 0:
            raise ValueError("similarity_edge_scale must be > 0")
        if self.window_overlap_tokens < 0:
            raise ValueError("window_overlap_tokens must be >= 0")
```

- [ ] **Step 4: 运行确认通过**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_graph.py -v`
Expected: 新 3 组测试通过,原有测试仍通过

- [ ] **Step 5: Commit**

```bash
git add src/medical_graphrag/retrieval/graph.py tests/test_graph.py
git commit -m "feat: GraphBuildConfig 离线边策略配置与校验

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 6: build_graph_index 重构(多策略 + 报告)

**Files:**
- Modify: `MedicalGraphRAG/src/medical_graphrag/retrieval/graph.py`
- Test: `MedicalGraphRAG/tests/test_graph_document.py`

**背景:** `build_graph_index` 当前硬编码读 `chunks.jsonl`、只建 Entity–Passage 边(`graph.py:130-283`)。本任务改为接收 `GraphBuildConfig`,经 `load_retrieval_passages` 取 passage,按 `passage_edge_mode` 调 `graph_edges` 建 Passage–Passage 边,并在报告中写入 `retrieval_unit` / `passage_edge_mode` / `source_artifact` / 诊断统计。**旧的 chunk-level 入口保留**(通过默认 `GraphBuildConfig(retrieval_unit="chunk", passage_edge_mode="none")` 兼容历史调用与测试)。

- [ ] **Step 1: 写 document+similarity 构建失败测试**

```python
import hashlib
import json
from pathlib import Path

import pytest

from medical_graphrag.retrieval.graph import GraphBuildConfig, build_graph_index


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pubmedqa_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    docs = [
        {"doc_id": "PMID:1", "title": "vaccine storage", "content": (
            "To assess quality of storage of vaccines in the community. "
            "Vaccines were exposed to subzero temperatures in three fridges."),
         "source": "pubmedqa", "year": "1992"},
        {"doc_id": "PMID:2", "title": "first names", "content": (
            "To assess the acceptability to patients of the use of patients' first names. "
            "Most patients liked being called by their first names."),
         "source": "pubmedqa", "year": "1990"},
    ]
    (dataset / "documents.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
    # 每篇文档 2 个 chunk(模拟多 chunk 文档)
    chunks = [
        {"chunk_id": "PMID:1#0", "doc_id": "PMID:1", "order": 0, "title": "t",
         "content": "To assess quality of storage of vaccines.", "source": "pubmedqa"},
        {"chunk_id": "PMID:1#1", "doc_id": "PMID:1", "order": 1, "title": "t",
         "content": "Vaccines exposed to subzero temperatures.", "source": "pubmedqa"},
        {"chunk_id": "PMID:2#0", "doc_id": "PMID:2", "order": 0, "title": "t",
         "content": "Acceptability of first names to patients.", "source": "pubmedqa"},
        {"chunk_id": "PMID:2#1", "doc_id": "PMID:2", "order": 1, "title": "t",
         "content": "Most patients liked first names.", "source": "pubmedqa"},
    ]
    (dataset / "chunks.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    (dataset / "questions.jsonl").write_text(
        '{"query_id":"q1","question":"Do vaccines need cold storage?","answer":"yes","long_answer":"x","split":"dev"}\n',
        encoding="utf-8")
    (dataset / "qrels.tsv").write_text("query_id\tdoc_id\trelevance\nq1\tPMID:1\t1\n",
                                       encoding="utf-8")
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    (dataset / "manifest.json").write_text(json.dumps({
        "counts": {"questions": 1, "documents": 2, "chunks": 4, "qrels": 1},
        "artifact_hashes": {n: _sha(dataset / n) for n in names},
    }), encoding="utf-8")
    return dataset


def test_build_document_similarity_index(tmp_path: Path) -> None:
    dataset = _write_pubmedqa_dataset(tmp_path)
    index_dir = tmp_path / "graph"
    config = GraphBuildConfig(
        retrieval_unit="document",
        passage_edge_mode="similarity",
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        ner_model="en_ner_bc5cdr_md",
        similarity_k=5,
        similarity_min_cosine=0.0,  # 测试环境宽松阈值,保证出边
    )
    report = build_graph_index(dataset, index_dir, build_config=config)
    assert report["retrieval_unit"] == "document"
    assert report["passage_edge_mode"] == "similarity"
    assert report["passage_count"] == 2          # 2 篇完整摘要
    assert report["passage_ids"] == ["PMID:1", "PMID:2"]
    assert report["source_artifact"] == "documents.jsonl"
    assert report["adjacent_passage_edges"] is False
    assert report["edge_count_by_type"]["similarity"] >= 0
    assert report["entity_passage_edge_count"] >= 1
    assert (index_dir / "graph.graphml").exists()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_graph_document.py -v`
Expected: FAIL("TypeError: build_graph_index() got an unexpected keyword argument 'build_config'")

- [ ] **Step 3: 重构 build_graph_index 签名与报告字段**

修改 `graph.py`:

```python
def build_graph_index(
    dataset_dir: Path,
    output_dir: Path,
    *,
    build_config: GraphBuildConfig | None = None,
    config: GraphConfig | None = None,   # 兼容历史调用;新代码用 search_config 语义
    batch_size: int = 64,
) -> dict[str, Any]:
    import faiss
    import igraph as ig
    import numpy as np

    from medical_graphrag.data.retrieval_passages import load_retrieval_passages
    from medical_graphrag.retrieval.dense import embed_documents_full
    from medical_graphrag.retrieval.graph_edges import (
        build_adjacent_edges,
        build_similarity_edges,
    )

    if build_config is None:
        build_config = GraphBuildConfig(
            retrieval_unit="chunk",
            passage_edge_mode="none",
            embedding_model=config.embedding_model if config else DEFAULT_EMBEDDING_MODEL,
            ner_model=config.ner_model if config else DEFAULT_NER_MODEL,
        )
    if config is None:
        config = GraphConfig(
            ner_model=build_config.ner_model,
            embedding_model=build_config.embedding_model,
        )
    manifest = validate_frozen_dataset(dataset_dir)
    passages = load_retrieval_passages(dataset_dir, build_config.retrieval_unit)
    if build_config.retrieval_unit == "document":
        if len(passages) != manifest["counts"]["documents"]:
            raise ValueError("passage count does not match documents count")
        source_artifact = "documents.jsonl"
    else:
        if len(passages) != manifest["counts"]["chunks"]:
            raise ValueError("passage count does not match chunks count")
        source_artifact = "chunks.jsonl"
    passage_ids = [p.passage_id for p in passages]
    texts = [p.content for p in passages]

    nlp = _load_nlp(build_config.ner_model)
    passage_entities = extract_entities(nlp, texts, batch_size=batch_size)
    sentences = split_sentences(nlp, texts)
    del nlp

    embedder = _load_embedder(build_config.embedding_model)
    if build_config.retrieval_unit == "document":
        passage_embeddings, window_counts, truncated = embed_documents_full(
            texts, embedder, overlap_tokens=build_config.window_overlap_tokens,
            batch_size=batch_size,
        )
        if truncated != 0:
            raise ValueError("document embedding truncated tokens != 0")
    else:
        passage_embeddings = np.asarray(
            embedder.encode(texts, normalize_embeddings=True, batch_size=batch_size,
                            show_progress_bar=False),
            dtype="float32",
        )
        window_counts = [1] * len(texts)

    # (以下句子桥/实体边/节点构建沿用现有实现,passage 由 texts/passage_ids 驱动)
    # ... 现有 entity_to_sentences / sentence_embeddings / entity_embeddings ...
```

> 注意:重构时把现有函数体中所有 `passage_ids` / `texts` 的数据来源改为上面的 `passages` 变量;句子桥、Entity–Passage 边、实体 embedding 逻辑保持不变。随后追加 Passage–Passage 边与报告字段。

- [ ] **Step 4: 追加 Passage–Passage 边与报告字段**

在边构建处插入:

```python
    edge_list = []
    edge_weights = []
    # Entity–Passage 边(现有)
    for passage_id, entity_scores in edges.items():
        for entity, weight in entity_scores.items():
            edge_list.append((passage_index[passage_id], entity_index[entity]))
            edge_weights.append(weight)
    entity_passage_edge_count = len(edge_list)

    # Passage–Passage 边(按 mode)
    passage_passage_edge_count = 0
    edge_count_by_type = {"entity_passage": entity_passage_edge_count,
                          "similarity": 0, "adjacent": 0}
    if build_config.passage_edge_mode == "similarity":
        sim_edges = build_similarity_edges(
            passage_ids, passage_embeddings,
            k=build_config.similarity_k,
            min_cosine=build_config.similarity_min_cosine,
            scale=build_config.similarity_edge_scale,
        )
        for a, b, w in sim_edges:
            edge_list.append((passage_index[a], passage_index[b]))
            edge_weights.append(w)
        passage_passage_edge_count = len(sim_edges)
        edge_count_by_type["similarity"] = len(sim_edges)
    elif build_config.passage_edge_mode == "adjacent":
        adj_edges, gaps = build_adjacent_edges(passages)
        for a, b, w in adj_edges:
            edge_list.append((passage_index[a], passage_index[b]))
            edge_weights.append(w)
        passage_passage_edge_count = len(adj_edges)
        edge_count_by_type["adjacent"] = len(adj_edges)
```

在边构建之后追加诊断计算(规格 §7.3 / §8):

```python
    # §7.3/§8 诊断:similarity 度/孤立/权重分布;adjacent expected vs actual
    passage_passage_diagnostics: dict[str, float] = {}
    if build_config.passage_edge_mode == "similarity":
        index_of = {pid: i for i, pid in enumerate(passage_ids)}
        degree = [0] * len(passage_ids)
        for a, b, _ in sim_edges:
            degree[index_of[a]] += 1
            degree[index_of[b]] += 1
        weights = [w for _, _, w in sim_edges]
        isolated = [pid for pid, d in zip(passage_ids, degree) if d == 0]
        passage_passage_diagnostics = {
            "edge_count": float(len(sim_edges)),
            "isolated_count": float(len(isolated)),
            "isolated_rate": float(len(isolated)) / max(len(passage_ids), 1),
            "degree_min": _percentile(sorted(degree), 0.0),
            "degree_mean": sum(degree) / max(len(degree), 1),
            "degree_p50": _percentile(sorted(degree), 0.50),
            "degree_p95": _percentile(sorted(degree), 0.95),
            "degree_p99": _percentile(sorted(degree), 0.99),
            "degree_max": _percentile(sorted(degree), 1.0),
            "weight_min": _percentile(sorted(weights), 0.0),
            "weight_mean": sum(weights) / max(len(weights), 1),
            "weight_p50": _percentile(sorted(weights), 0.50),
            "weight_p95": _percentile(sorted(weights), 0.95),
            "weight_max": _percentile(sorted(weights), 1.0),
            "exact_content_duplicate_pairs": float(
                _count_exact_content_duplicates(passages)
            ),
        }
    elif build_config.passage_edge_mode == "adjacent":
        from collections import defaultdict
        doc_counts: dict[str, int] = defaultdict(int)
        for p in passages:
            doc_counts[p.doc_id] += 1
        expected_adjacent = sum(max(c - 1, 0) for c in doc_counts.values())
        if expected_adjacent != len(adj_edges):
            raise ValueError(
                f"adjacent expected {expected_adjacent} != actual {len(adj_edges)}"
            )
        passage_passage_diagnostics = {
            "expected_edge_count": float(expected_adjacent),
            "actual_edge_count": float(len(adj_edges)),
        }
```

同时把 `passage_passage_diagnostics` 与两类边权重统计并入报告:

报告字段追加:

```python
    report = {
        "graph_schema_version": 2,
        "graph_profile": ("abstract_similarity_v1" if build_config.retrieval_unit == "document"
                          else "linearrag_adjacent_v1" if build_config.passage_edge_mode == "adjacent"
                          else "chunk_entity_only_v1"),
        "retrieval_unit": build_config.retrieval_unit,
        "passage_edge_mode": build_config.passage_edge_mode,
        "source_artifact": source_artifact,
        "source_artifact_sha256": manifest["artifact_hashes"][source_artifact],
        "text_mode": TEXT_MODE,
        "ner_model": build_config.ner_model,
        "embedding_model": build_config.embedding_model,
        "entity_count": len(all_entities),
        "passage_count": len(passage_ids),
        "sentence_count": len(sentence_ids),
        "edge_count": len(edge_list),
        "entity_passage_edge_count": entity_passage_edge_count,
        "passage_passage_edge_count": passage_passage_edge_count,
        "edge_count_by_type": edge_count_by_type,
        "edge_weight_stats_by_type": {
            "entity_passage": _weight_stats(edge_weights[:entity_passage_edge_count]),
            "similarity": _weight_stats(edge_weights[entity_passage_edge_count:])
            if build_config.passage_edge_mode == "similarity"
            else {"count": 0},
            "adjacent": _weight_stats(edge_weights[entity_passage_edge_count:])
            if build_config.passage_edge_mode == "adjacent"
            else {"count": 0},
        },
        "passage_passage_diagnostics": passage_passage_diagnostics,
        "adjacent_passage_edges": build_config.passage_edge_mode == "adjacent",
        "window_coverage": {
            "window_count_min": min(window_counts),
            "window_count_mean": sum(window_counts) / max(len(window_counts), 1),
            "window_count_p95": _percentile(sorted(window_counts), 0.95),
            "truncated_token_count": 0,
        },
        "build_config": {
            **{
                "retrieval_unit": build_config.retrieval_unit,
                "passage_edge_mode": build_config.passage_edge_mode,
                "similarity_k": build_config.similarity_k,
                "similarity_min_cosine": build_config.similarity_min_cosine,
                "similarity_edge_scale": build_config.similarity_edge_scale,
                "window_overlap_tokens": build_config.window_overlap_tokens,
            },
            "config_sha256": _hash_dict(build_config),
        },
        # 现有 hash 绑定字段(graph_sha256 / embeddings_sha256 / ...)保持不变
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "dataset_artifact_hashes": manifest["artifact_hashes"],
        "config": {
            "damping": config.damping,
            "passage_ratio": config.passage_ratio,
            "passage_node_weight": config.passage_node_weight,
            "iteration_threshold": config.iteration_threshold,
            "top_k_sentence": config.top_k_sentence,
            "max_iterations": config.max_iterations,
        },
    }
```

新增三个小工具函数到 `graph.py`:

```python
from collections.abc import Sequence

def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    k = int((len(sorted_values) - 1) * q)
    return float(sorted_values[k])


def _count_exact_content_duplicates(passages: Sequence[object]) -> int:
    from collections import Counter
    counts = Counter(p.content for p in passages)
    return sum(c * (c - 1) // 2 for c in counts.values() if c > 1)


def _weight_stats(weights: Sequence[float]) -> dict[str, float]:
    if not weights:
        return {"count": 0}
    return {
        "count": float(len(weights)),
        "sum": float(sum(weights)),
        "min": float(min(weights)),
        "mean": float(sum(weights) / len(weights)),
        "p50": _percentile(sorted(weights), 0.50),
        "p95": _percentile(sorted(weights), 0.95),
        "max": float(max(weights)),
    }


def _hash_dict(value: object) -> str:
    import hashlib
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
```

- [ ] **Step 5: 运行 document 测试 + 原有测试**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_graph_document.py tests/test_graph.py -v`
Expected: 新测试通过,原 `test_graph_build_and_search_smoke`(chunk+none 兼容路径)仍通过

- [ ] **Step 6: Commit**

```bash
git add src/medical_graphrag/retrieval/graph.py tests/test_graph_document.py
git commit -m "refactor: build_graph_index 支持 retrieval_unit + edge_mode 报告

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 7: LinearGraphRetriever / search 契约按 unit 分支

**Files:**
- Modify: `MedicalGraphRAG/src/medical_graphrag/retrieval/graph.py`
- Modify: `MedicalGraphRAG/src/medical_graphrag/retrieval/search_graph.py`
- Test: `MedicalGraphRAG/tests/test_graph_document.py`(追加)

- [ ] **Step 1: 追加失败测试(document 搜索直接出 doc_id)**

```python
from medical_graphrag.retrieval.graph import LinearGraphRetriever


def test_document_retriever_returns_doc_ids(tmp_path: Path) -> None:
    dataset = _write_pubmedqa_dataset(tmp_path)
    index_dir = tmp_path / "graph"
    config = GraphBuildConfig(
        retrieval_unit="document", passage_edge_mode="similarity",
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        ner_model="en_ner_bc5cdr_md", similarity_k=5, similarity_min_cosine=0.0,
    )
    build_graph_index(dataset, index_dir, build_config=config)
    retriever = LinearGraphRetriever(index_dir)
    assert retriever.retrieval_unit == "document"
    ids, scores = retriever.search("Do vaccines need cold storage?", top_k=2)
    assert set(ids) <= {"PMID:1", "PMID:2"}       # 直接是 doc_id,不是 chunk_id
    assert len(scores) == len(ids)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_graph_document.py -v`
Expected: FAIL("AttributeError: 'LinearGraphRetriever' object has no attribute 'retrieval_unit'")

- [ ] **Step 3: LinearGraphRetriever 读 retrieval_unit**

在 `LinearGraphRetriever.__init__` 中,读取报告并设置:

```python
        self.retrieval_unit = str(self.report.get("retrieval_unit", "chunk"))
        self.passage_edge_mode = str(self.report.get("passage_edge_mode", "none"))
```

(现有 `self.passage_ids = names[entity_count:]` 对 document 模式自动等于 doc_ids,无需改。)

- [ ] **Step 4: search_one / run_search 分支**

在 `search_graph.py` 中,`search_one` 增加 `retrieval_unit` 参数:

```python
def search_one(
    retriever: LinearGraphRetriever,
    question: Mapping[str, object],
    chunk_to_doc: Mapping[str, str] | None,
    top_k: int,
    clock: Any = time.perf_counter,
) -> dict[str, object]:
    started = clock()
    passage_ids, scores = retriever.search(str(question["question"]), top_k=top_k)
    latency_ms = round((clock() - started) * 1000, 6)
    if retriever.retrieval_unit == "document":
        hits = [
            {"doc_id": doc_id, "rank": rank, "score": float(score)}
            for rank, (doc_id, score) in enumerate(zip(passage_ids, scores), start=1)
        ]
    else:
        assert chunk_to_doc is not None
        hits = [
            {"chunk_id": chunk_id, "doc_id": chunk_to_doc[chunk_id],
             "chunk_rank": rank, "score": float(score)}
            for rank, (chunk_id, score) in enumerate(zip(passage_ids, scores), start=1)
        ]
    return {"query_id": str(question["query_id"]), "split": str(question["split"]),
            "latency_ms": latency_ms, "hits": hits}
```

`run_search` 中:从 `index_report_data` 取 `retrieval_unit`;document 模式下不读/不校验 `chunks.jsonl`,`chunk_to_doc=None`:

```python
    index_report_data = validate_index(index, questions, index_report)
    retrieval_unit = index_report_data.get("retrieval_unit", "chunk")
    chunk_to_doc = None
    if retrieval_unit == "chunk":
        chunks_rows = _read_jsonl(chunks)
        chunk_to_doc = {str(row["chunk_id"]): str(row["doc_id"]) for row in chunks_rows}
    rows = [search_one(retriever, row, chunk_to_doc, top_k) for row in questions_rows]
```

同时放宽 `validate_index` 的 `text_mode` 门:document 模式报告仍为 `abstract_only`,门不变。

- [ ] **Step 5: 运行确认通过**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_graph_document.py tests/test_graph.py -v`
Expected: 全部通过

- [ ] **Step 6: Commit**

```bash
git add src/medical_graphrag/retrieval/graph.py src/medical_graphrag/retrieval/search_graph.py tests/test_graph_document.py
git commit -m "feat: 检索契约按 retrieval_unit 分支(document 直出 doc_id)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 8: 评测 document unit(不 collapse)

**Files:**
- Modify: `MedicalGraphRAG/src/medical_graphrag/evaluation/graph.py`
- Test: `MedicalGraphRAG/tests/test_graph_document.py`(追加)

- [ ] **Step 1: 追加失败测试(doc 级评测不接受 chunk_id)**

```python
from medical_graphrag.evaluation.graph import evaluate_graph_run


def test_evaluate_graph_run_document_no_collapse(tmp_path: Path) -> None:
    dataset = _write_pubmedqa_dataset(tmp_path)
    index_dir = tmp_path / "graph"
    config = GraphBuildConfig(
        retrieval_unit="document", passage_edge_mode="similarity",
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        ner_model="en_ner_bc5cdr_md", similarity_k=5, similarity_min_cosine=0.0,
    )
    report = build_graph_index(dataset, index_dir, build_config=config)
    rankings = tmp_path / "rankings.jsonl"
    rankings.write_text(
        '{"query_id":"q1","split":"dev","latency_ms":1.0,"hits":[{"doc_id":"PMID:1","rank":1,"score":0.9}]}\n',
        encoding="utf-8")
    run_context = {
        "index": report,
        "search": {
            "graph_sha256": report["graph_sha256"],
            "graph_build_report_sha256": report["graph_sha256"],
            "dataset_manifest_sha256": report["dataset_manifest_sha256"],
            "dataset_artifact_hashes": report["dataset_artifact_hashes"],
            "rankings_sha256": _sha(rankings),
            "questions_sha256": report["dataset_artifact_hashes"]["questions.jsonl"],
            "requested_top_k": 10,
            "query_count": 1, "min_hits": 1, "max_hits": 1, "short_ranking_count": 0,
            "hit_count_histogram": {"1": 1},
            "config": report["config"],
        },
        "index_report_sha256": _sha((tmp_path / "graph" / "graph_build.json")),
        "dataset_manifest_sha256": report["dataset_manifest_sha256"],
    }
    exp_dir = tmp_path / "exp"
    metrics = evaluate_graph_run(dataset, rankings, exp_dir, run_context=run_context)
    assert metrics["dev"]["recall@1"] == 1.0  # doc_id 直接命中 qrels
```

- [ ] **Step 2: 运行确认失败**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_graph_document.py::test_evaluate_graph_run_document_no_collapse -v`
Expected: FAIL("KeyError: 'chunk_id'")(因 collapse 读取 chunk_id)

- [ ] **Step 3: evaluate_graph_run 按 retrieval_unit 分支**

在 `evaluate_graph_run` 中,从 `run_context["index"]` 取 `retrieval_unit`,分支 collapse:

```python
    retrieval_unit = str(index_report.get("retrieval_unit", "chunk"))
    collapsed: dict[str, list[str]] = {}
    detailed: dict[str, list[dict[str, object]]] = {}
    for row in raw_rows:
        query_id = str(row["query_id"])
        if row["split"] != questions[query_id]["split"]:
            raise ValueError(f"split mismatch for {query_id}")
        if retrieval_unit == "document":
            ranking = [
                {"doc_id": str(hit["doc_id"]), "score": float(hit["score"]),
                 "rank": int(hit["rank"])}
                for hit in row["hits"]
            ]
        else:
            ranking = collapse_chunk_hits(row["hits"], metadata, min_unique_docs=0)
        collapsed[query_id] = [str(item["doc_id"]) for item in ranking]
        detailed[query_id] = [
            {**item, "title": documents[str(item["doc_id"])]["title"]}
            for item in ranking
        ]
```

`run_manifest` 追加:

```python
    run_manifest = {
        **run_context,
        "retrieval_unit": retrieval_unit,
        "aggregation": "none" if retrieval_unit == "document" else "max_chunk_score",
        ...
    }
```

- [ ] **Step 4: 运行确认通过**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_graph_document.py tests/test_graph.py -v`
Expected: 全部通过

- [ ] **Step 5: Commit**

```bash
git add src/medical_graphrag/evaluation/graph.py tests/test_graph_document.py
git commit -m "feat: 评测 document unit 直用 doc_id,不 collapse

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 9: document 级公平基线(BM25/Dense/Hybrid)

**Files:**
- Modify: `MedicalGraphRAG/src/medical_graphrag/retrieval/bm25.py`(export_document_collection)
- Modify: `MedicalGraphRAG/src/medical_graphrag/retrieval/dense.py`(build_dense_document_index)
- Modify: `MedicalGraphRAG/src/medical_graphrag/retrieval/search_bm25.py` / `search_dense.py`(document 搜索或复用)
- Test: `MedicalGraphRAG/tests/test_graph_document.py`(追加基线 smoke)

- [ ] **Step 1: 实现 export_document_collection(bm25.py)**

```python
def export_document_collection(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Lucene collection over documents.jsonl (document retrieval unit)."""
    manifest = validate_frozen_dataset(dataset_dir)
    documents = _read_jsonl(dataset_dir / "documents.jsonl")
    collection = [{"id": str(row["doc_id"]), "contents": str(row["content"])}
                  for row in documents]
    collection_path = output_dir / "collection/documents.jsonl"
    write_jsonl(collection_path, collection)
    report = {
        "retrieval_unit": "document",
        "source_artifact": "documents.jsonl",
        "source_artifact_sha256": manifest["artifact_hashes"]["documents.jsonl"],
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "document_count": len(collection),
        "collection_sha256": sha256_file(collection_path),
    }
    write_json(output_dir / "export_report.json", report)
    return report
```

- [ ] **Step 2: 实现 build_dense_document_index(dense.py)**

```python
def build_dense_document_index(
    dataset_dir: Path,
    output_dir: Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    overlap_tokens: int = 32,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Dense index over full documents (document retrieval unit).

    Uses embed_documents_full so the vectors are identical to those used for
    similarity kNN edges and graph passage prior (spec §5.1).
    """
    import faiss
    import numpy as np

    manifest = validate_frozen_dataset(dataset_dir)
    documents = _read_jsonl(dataset_dir / "documents.jsonl")
    texts = [str(row["content"]) for row in documents]
    doc_ids = [str(row["doc_id"]) for row in documents]
    embedder = _load_embedder(model_name)
    embeddings, window_counts, truncated = embed_documents_full(
        texts, embedder, overlap_tokens=overlap_tokens, batch_size=batch_size)
    if truncated != 0:
        raise ValueError("document embedding truncated tokens != 0")
    dim = int(embeddings.shape[1])
    embeddings_path = output_dir / "document_embeddings.npy"
    np.save(embeddings_path, embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    index_path = output_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    metadata_path = output_dir / "document_metadata.jsonl"
    write_jsonl(metadata_path,
                [{"doc_id": d} for d in doc_ids])
    report = {
        "retrieval_unit": "document",
        "source_artifact": "documents.jsonl",
        "source_artifact_sha256": manifest["artifact_hashes"]["documents.jsonl"],
        "embedding_model": model_name,
        "dim": dim,
        "index_type": INDEX_TYPE,
        "embeddings_sha256": sha256_file(embeddings_path),
        "index_sha256": sha256_file(index_path),
        "metadata_sha256": sha256_file(metadata_path),
        "window_coverage": {
            "window_count_min": min(window_counts),
            "window_count_mean": sum(window_counts) / max(len(window_counts), 1),
            "window_count_p95": _window_p95(sorted(window_counts)),
            "truncated_token_count": int(truncated),
        },
    }
    write_json(output_dir / "index_build.json", report)
    return report
```

> 在 `dense.py` 增加 `_window_p95(sorted_counts)`(同 `graph._percentile` 实现)。

- [ ] **Step 3: 实现 document 搜索(与 search_dense.py 同构)**

新增 `retrieval/search_document.py`:

```python
"""Document-level Dense/BM25 search over the same retrieval passages.

Adapts the chunk-level search contract to the document unit: hits carry
``doc_id`` directly (no chunk→doc collapse). Query vector = full question
embedding; doc vectors = embed_documents_full output (spec §5.1).
"""
import json
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file
from medical_graphrag.retrieval.dense import DEFAULT_EMBEDDING_MODEL


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_hit_counts(rows: list[dict[str, object]], *, requested_top_k: int) -> dict[str, object]:
    counts = [len(row["hits"]) for row in rows]
    histogram = Counter(counts)
    return {
        "requested_top_k": requested_top_k,
        "min_hits": min(counts),
        "max_hits": max(counts),
        "short_ranking_count": sum(count < requested_top_k for count in counts),
        "hit_count_histogram": {str(c): f for c, f in sorted(histogram.items())},
    }


def _validate_dense_index(index, embeddings, metadata, questions, report_path) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("retrieval_unit") != "document":
        raise ValueError("index must be a document retrieval unit")
    for path, field in ((index, "index_sha256"), (embeddings, "embeddings_sha256"),
                        (metadata, "metadata_sha256")):
        if sha256_file(path) != report.get(field):
            raise ValueError(f"{field} SHA-256 mismatch")
    if sha256_file(questions) != report.get("dataset_artifact_hashes", {}).get("questions.jsonl"):
        raise ValueError("questions SHA-256 mismatch")
    return report


def run_dense_document_search(
    *,
    index: Path,
    embeddings: Path,
    metadata: Path,
    index_report: Path,
    questions: Path,
    output: Path,
    report: Path,
    top_k: int = 100,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, object]:
    import faiss
    import numpy as np

    from sentence_transformers import SentenceTransformer

    index_report_data = _validate_dense_index(index, embeddings, metadata, questions, index_report)
    if embedding_model != index_report_data["embedding_model"]:
        raise ValueError("requested embedding model does not match index report")
    faiss_index = faiss.read_index(str(index))
    embedder = SentenceTransformer(embedding_model)
    questions_rows = _read_jsonl(questions)
    metadata_rows = _read_jsonl(metadata)
    doc_ids = [str(row["doc_id"]) for row in metadata_rows]
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("duplicate doc_id in metadata")
    if index_report_data["document_count"] != len(doc_ids):
        raise ValueError("metadata count does not match index report")

    rows = []
    for question in questions_rows:
        started = time.perf_counter()
        vector = np.asarray(
            embedder.encode(str(question["question"]), normalize_embeddings=True,
                            show_progress_bar=False), dtype="float32",
        ).reshape(1, -1)
        scores, indices = faiss_index.search(vector, top_k)
        latency_ms = round((time.perf_counter() - started) * 1000, 6)
        hits = []
        for rank in range(top_k):
            doc_index = int(indices[0, rank])
            if doc_index < 0:
                break
            hits.append({"doc_id": doc_ids[doc_index], "rank": rank + 1,
                         "score": float(scores[0, rank])})
        rows.append({"query_id": str(question["query_id"]), "split": str(question["split"]),
                     "latency_ms": latency_ms, "hits": hits})
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate query_id in search output")
    hit_summary = summarize_hit_counts(rows, requested_top_k=top_k)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                              for r in rows), encoding="utf-8")
    rankings_sha256 = sha256_file(output)
    report.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "command": ["search_document.run_dense_document_search", str(questions), str(output)],
        "query_count": len(rows), **hit_summary,
        "retrieval_unit": "document",
        "embedding_model": index_report_data["embedding_model"],
        "dim": index_report_data["dim"],
        "index_type": index_report_data["index_type"],
        "embeddings_sha256": index_report_data["embeddings_sha256"],
        "index_sha256": index_report_data["index_sha256"],
        "index_report_sha256": sha256_file(index_report),
        "dataset_manifest_sha256": index_report_data["dataset_manifest_sha256"],
        "questions_sha256": sha256_file(questions),
        "rankings_sha256": rankings_sha256,
    }
    report.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")
    return report_data
```

> BM25-document 同理:对 `export_document_collection` 产出的 Lucene collection 跑 `LuceneSearcher`(Pyserini,与 `search_bm25.py` 同构),docid 即 `doc_id`,产出同构 hits。Hybrid-document 直接复用现有 `evaluation/hybrid.py` 的 RRF(它已按 doc_id 聚合),输入两个 document 级 raw_rankings。

- [ ] **Step 4: 追加基线 smoke 测试**

在 `test_graph_document.py`:

```python
def test_dense_document_index_builds(tmp_path: Path) -> None:
    dataset = _write_pubmedqa_dataset(tmp_path)
    out = tmp_path / "dense_doc"
    from medical_graphrag.retrieval.dense import build_dense_document_index
    report = build_dense_document_index(dataset, out)
    assert report["retrieval_unit"] == "document"
    assert report["dim"] == 768
    assert report["source_artifact"] == "documents.jsonl"
```

- [ ] **Step 5: 运行确认通过**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_graph_document.py -v`
Expected: 通过

- [ ] **Step 6: Commit**

```bash
git add src/medical_graphrag/retrieval/bm25.py src/medical_graphrag/retrieval/dense.py src/medical_graphrag/retrieval/search_document.py tests/test_graph_document.py
git commit -m "feat: document 级 BM25/Dense/Hybrid 公平基线

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 10: run_pipeline + CLI 装配

**Files:**
- Modify: `MedicalGraphRAG/src/medical_graphrag/run_pipeline.py`
- Modify: `MedicalGraphRAG/src/medical_graphrag/cli.py`

- [ ] **Step 1: 在 run_pipeline.py 增加 document 级 runner**

```python
def run_graph_document(
    dataset: str,
    *,
    git_commit: str,
    docker_image: str = DOCKER_IMAGE_DEFAULT,
    root: Path = ROOT,
    profile: str = "similarity",
    top_k: int = 100,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Build document-level graph (EP or similarity), search, evaluate."""
    dataset_dir = _dataset_dir(dataset, root)
    assert profile in {"ep", "similarity"}
    name = f"graph_document_{profile}_v1"
    idx_dir = root / "indexes" / dataset / name
    out_dir = root / "outputs" / dataset / name
    exp_dir = root / "experiments" / dataset / name
    build_config = GraphBuildConfig(
        retrieval_unit="document",
        passage_edge_mode="similarity" if profile == "similarity" else "none",
        embedding_model=GRAPH_DEFAULT_EMBEDDING_MODEL,
        ner_model=DEFAULT_NER_MODEL,
    )
    build_graph_index(dataset_dir, idx_dir, build_config=build_config,
                      batch_size=batch_size)
    run_graph_search(
        index=idx_dir,
        index_report=idx_dir / "graph_build.json",
        questions=dataset_dir / "questions.jsonl",
        chunks=dataset_dir / "chunks.jsonl",
        output=out_dir / "raw_rankings.jsonl",
        report=out_dir / "search_run.json",
        top_k=top_k,
        ner_model=DEFAULT_NER_MODEL,
        embedding_model=GRAPH_DEFAULT_EMBEDDING_MODEL,
    )
    context = _build_context(
        git_commit=git_commit, docker_image=docker_image,
        command=["cli", "run", "graph-document", "--dataset", dataset,
                 "--profile", profile],
        extra={
            "index": json.loads((idx_dir / "graph_build.json").read_text(encoding="utf-8")),
            "search": json.loads((out_dir / "search_run.json").read_text(encoding="utf-8")),
            "index_report_sha256": sha256_file(idx_dir / "graph_build.json"),
            "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        },
    )
    return evaluate_graph_run(dataset_dir, out_dir / "raw_rankings.jsonl",
                              exp_dir, run_context=context)
```

新增 `run_dense_document` / `run_bm25_document` / `run_hybrid_document`(对称实现,分别调用 Task 9 的构建与搜索函数),并注册:

```python
RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "bm25": run_bm25,
    "dense": run_dense,
    "graph": run_graph,
    "hybrid": run_hybrid,
    "reranker": run_reranker,
    "bm25-document": run_bm25_document,
    "dense-document": run_dense_document,
    "graph-document": run_graph_document,
    "hybrid-document": run_hybrid_document,
}
```

- [ ] **Step 2: cli.py 增加 profile 参数**

```python
    run.add_argument("--profile", choices=["ep", "similarity"], default="similarity",
                     help="graph-document 的边策略(ep=仅实体边, similarity=相似度软边)")
```

`_run_command` 中对 `graph-document` 传 `profile=args.profile`。

- [ ] **Step 3: 运行全部测试确认无回归**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/ -q`
Expected: 全绿

- [ ] **Step 4: Commit**

```bash
git add src/medical_graphrag/run_pipeline.py src/medical_graphrag/cli.py
git commit -m "feat: run_pipeline/cli 装配 document 级检索器与 profile

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Task 11: Phase 1 在 pubmedqa_hard_v1 实跑 + 审阅包

**Files:** 无(运行产物)

- [ ] **Step 1: 实跑 document 级五路**

```bash
cd MedicalGraphRAG
GIT=$(git rev-parse HEAD)
for ret in bm25-document dense-document graph-document hybrid-document; do
  /opt/venv/bin/python -m medical_graphrag.cli run "$ret" --dataset pubmedqa_hard_v1 \
    --git-commit "$GIT" --docker-image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
done
# graph-document 两个 profile
/opt/venv/bin/python -m medical_graphrag.cli run graph-document --dataset pubmedqa_hard_v1 \
  --profile ep --git-commit "$GIT" --docker-image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
```

- [ ] **Step 2: 汇总指标表(五路 per-dataset dev/test)**

```bash
for d in experiments/pubmedqa_hard_v1/bm25_document_v1 \
         experiments/pubmedqa_hard_v1/dense_document_v1 \
         experiments/pubmedqa_hard_v1/graph_document_ep_v1 \
         experiments/pubmedqa_hard_v1/graph_document_similarity_v1 \
         experiments/pubmedqa_hard_v1/hybrid_document_v1; do
  echo "== $d =="; cat "$d/metrics.json"
done
```

- [ ] **Step 3: 检查 similarity 图诊断(边数/isolated/度分布/权分布)**

```bash
cat indexes/pubmedqa_hard_v1/graph_document_similarity_v1/graph_build.json | \
  /opt/venv/bin/python -c "import json,sys; r=json.load(sys.stdin); \
  print('edges_by_type:', r['edge_count_by_type']); \
  print('weight_stats:', r['edge_weight_stats_by_type']); \
  print('diag:', r['passage_passage_diagnostics']); \
  print('window:', r['window_coverage'])"
```

> 若 `similarity` 边为 0 或 isolated 率过高,记录到审阅包"已知限制",不回调参数(规格 §7.1)。

- [ ] **Step 3b: 记录工程层指标(规格 §11)**

```bash
# 索引构建墙钟时间(用 time 包裹 build 命令重新计时一次即可)
time /opt/venv/bin/python -m medical_graphrag.cli run graph-document \
  --dataset pubmedqa_hard_v1 --profile similarity \
  --git-commit "$(git rev-parse HEAD)" --docker-image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
# 索引空间占用
du -sh indexes/pubmedqa_hard_v1/graph_document_similarity_v1
du -sh indexes/pubmedqa_hard_v1/graph_document_ep_v1
# CPU RAM / GPU VRAM(容器内 nvidia-smi;RAM 用 /usr/bin/time -v)
```

将墙钟时间、`du -sh`、RAM/VRAM 峰值写入审阅包"工程指标"小节。

- [ ] **Step 4: 提取 3 成功 + 3 退化案例**

```bash
cat experiments/pubmedqa_hard_v1/graph_document_similarity_v1/cases.json
```

- [ ] **Step 5: 产出审阅包(规格 §15 全项)**

```bash
git diff --stat
git log --oneline -15
cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/ -q
```

将以下内容写入 `docs/superpowers/plans/2026-08-08-per-dataset-edge-policy.md` 末尾的"Phase 1 结果附录":
1. 五路 dev/test 指标表;
2. similarity 图边数 / isolated rate / degree min-mean-P50-P95-P99-max / weight 分布;
3. 3 成功 + 3 退化案例(query_id + gold_rank);
4. 已知限制(如 isolated 率高、相似度边稀疏、chunk-level 与 document-level 不可直接对比)。

- [ ] **Step 6: Commit 结果附录**

```bash
git add docs/superpowers/plans/2026-08-08-per-dataset-edge-policy.md
git commit -m "docs: Phase 1 pubmedqa document 级结果附录

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 2: SciFact/NFCorpus 复用验证

### Task 12: 固定配置复用验证

**Files:** 无(运行验证)

- [ ] **Step 1: 在 scifact_v1 / nfcorpus_v1 上重跑 Task 11 的 5 条命令,不改任何算法参数**
- [ ] **Step 2: 确认两个数据集独立建库、独立评测,目录互不覆盖**
- [ ] **Step 3: 确认 NFCorpus 多相关 qrels 保留全部 relevant(不退化 first-gold)**
- [ ] **Step 4: 确认 test split 未用于调参;若指标异常,记录到"已知限制"而非回调 k/threshold**
- [ ] **Step 5: Commit(仅文档)**

```bash
git add docs/superpowers/plans/2026-08-08-per-dataset-edge-policy.md
git commit -m "docs: Phase 2 scifact/nfcorpus 固定配置复用验证结果

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 3: 长文本 Adjacent 结构验证

### Task 13: MedRAG adapter + adjacent 索引

**Files:**
- Create: `MedicalGraphRAG/src/medical_graphrag/data/medrag_adapter.py`
- Test: `MedicalGraphRAG/tests/test_medrag_adapter.py`

- [ ] **Step 1: 写失败测试(adapter 规则)**

```python
import pytest

from medical_graphrag.data.medrag_adapter import adapt_medrag_chunks


def test_textbook_adapter_groups_by_file_stem():
    rows = [
        {"id": "Anatomy_Gray_0", "title": "t", "content": "a"},
        {"id": "Anatomy_Gray_1", "title": "t", "content": "b"},
        {"id": "Anatomy_Gray_2", "title": "t", "content": "c"},
    ]
    passages = adapt_medrag_chunks(rows, doc_id="Anatomy_Gray")
    assert [p.order for p in passages] == [0, 1, 2]
    assert all(p.doc_id == "Anatomy_Gray" for p in passages)


def test_statpearls_adapter_never_mixes_articles():
    rows = [
        {"id": "article-1_0", "title": "t", "content": "a"},
        {"id": "article-2_0", "title": "t", "content": "b"},
    ]
    passages = adapt_medrag_chunks(rows, doc_id="article-1")
    assert len(passages) == 1
    assert passages[0].passage_id == "article-1_0"


def test_adapter_rejects_unparseable_order():
    with pytest.raises(ValueError, match="order"):
        adapt_medrag_chunks([{"id": "Anatomy_Gray_x", "title": "t", "content": "c"}],
                            doc_id="Anatomy_Gray")


def test_adapter_rejects_duplicate_order():
    with pytest.raises(ValueError, match="duplicate"):
        adapt_medrag_chunks(
            [{"id": "Anatomy_Gray_1", "title": "t", "content": "a"},
             {"id": "Anatomy_Gray_1", "title": "t", "content": "b"}],
            doc_id="Anatomy_Gray")
```

- [ ] **Step 2: 运行确认失败**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_medrag_adapter.py -v`
Expected: FAIL("ModuleNotFoundError: medrag_adapter")

- [ ] **Step 3: 实现 adapter**

```python
"""MedRAG corpus → RetrievalPassage adapter.

Rules (spec §5.2): file stem is the document ID (textbook stem = doc, StatPearls
article stem = doc); the trailing integer in the original ``id`` is the chunk
order. Prefix mismatch / unparseable order / duplicate order → fail, never skip.
"""
import json
from pathlib import Path

from medical_graphrag.data.retrieval_passages import RetrievalPassage


def adapt_medrag_chunks(
    rows: list[dict[str, object]],
    *,
    doc_id: str,
) -> list[RetrievalPassage]:
    prefix = doc_id + "_"
    passages: list[RetrievalPassage] = []
    seen_orders: set[int] = set()
    for row in rows:
        raw_id = str(row["id"])
        if not raw_id.startswith(prefix):
            raise ValueError(f"id {raw_id!r} does not match doc_id prefix {prefix!r}")
        suffix = raw_id[len(prefix):]
        if not suffix.isdigit():
            raise ValueError(f"id {raw_id!r} has unparseable order {suffix!r}")
        order = int(suffix)
        if order in seen_orders:
            raise ValueError(f"doc {doc_id} has duplicate order {order}")
        seen_orders.add(order)
        passages.append(RetrievalPassage(
            passage_id=raw_id,
            doc_id=doc_id,
            order=order,
            title=str(row.get("title", "")),
            content=str(row.get("content", "")),
            source=str(row.get("source", "")),
        ))
    return passages


def adapt_medrag_file(path: Path) -> list[RetrievalPassage]:
    doc_id = path.stem
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return adapt_medrag_chunks(rows, doc_id=doc_id)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd MedicalGraphRAG && /opt/venv/bin/python -m pytest tests/test_medrag_adapter.py -v`
Expected: 4 passed

- [ ] **Step 5: 小 fixture 验证 adjacent 建图 + PPR 可搜索**

```bash
# 用 2 个 textbook 文件的迷你子集跑 graph build(adjacent 模式)
/opt/venv/bin/python -m medical_graphrag.cli run graph-adjacent --dataset textbooks_fixture \
  --git-commit "$(git rev-parse HEAD)" --docker-image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
```

> `graph-adjacent` runner:以 MedRAG 目录为输入,经 `adapt_medrag_file` 生成 chunk passages,`GraphBuildConfig(retrieval_unit="chunk", passage_edge_mode="adjacent")` 建图。无 qrels 时不报 Recall/MRR/nDCG(规格 §11)。

- [ ] **Step 6: Commit**

```bash
git add src/medical_graphrag/data/medrag_adapter.py tests/test_medrag_adapter.py
git commit -m "feat: MedRAG 长文本 adapter + adjacent 结构验证

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 备注

- Phase 1 是门禁:未通过代码/实验审阅不得进入 Phase 2(规格 §14)。
- Reranker-document 为可选(规格 §10),只在已有 reranker 流程自然支持时补跑,不阻塞首轮五路闭环。
- 所有"提升 X%"结论只能发生在同一 retrieval unit / 同一 manifest / 同一 query split 之间(规格 §10)。
- 历史 chunk-level 实验(目录 `graph_abstract_only` 等)保持原样。
- 本计划审阅方为外部 LLM(规格 §15),提交时需附 `git diff --stat`、测试摘要、真实指标与 3+3 案例,不得只报"pytest passed"。
