# BM25 Hard Retrieval Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结的 `pubmedqa_hard_v1` 上，用 WSL Docker 内的 Pyserini/Lucene 产出可复现的 BM25 文档级硬指标、延迟和案例分析。

**Architecture:** Windows 项目代码负责验证数据、导出 Pyserini collection、chunk→document 聚合与统一评测；Docker 脚本只负责 Lucene 建索引和 Top-100 chunk 查询。大体积索引与逐题原始排名保持 ignored，小型 `metrics.json`、`run_manifest.json`、`cases.json` 进入 Git。

**Tech Stack:** Python 3.11、pytest、Pyserini 0.22.1、Lucene/Java、Docker `llm-pytorch`、JSONL/TSV。

---

## 文件结构

- Create: `MedicalGraphRAG/src/medical_graphrag/retrieval/__init__.py` — 检索包入口。
- Create: `MedicalGraphRAG/src/medical_graphrag/retrieval/bm25.py` — 冻结输入校验、Pyserini collection 导出、chunk→doc 聚合。
- Create: `MedicalGraphRAG/src/medical_graphrag/evaluation/bm25.py` — 读取原始排名、按 split 评测、延迟统计、案例与运行记录。
- Create: `MedicalGraphRAG/scripts/build_pyserini_index.py` — 容器内构建索引并记录耗时、大小和版本。
- Create: `MedicalGraphRAG/scripts/search_pyserini_bm25.py` — 容器内检索 Top-100 chunk 并记录逐题延迟。
- Modify: `MedicalGraphRAG/src/medical_graphrag/cli.py` — 增加 `export-pyserini`、`evaluate-bm25`。
- Modify: `MedicalGraphRAG/README.md` — 增加准确的 Windows/Docker 运行手册。
- Modify: `MedicalGraphRAG/.gitignore` — 明确忽略 indexes/outputs，允许 experiments JSON。
- Modify: `MedicalGraphRAG/tests/test_git_tracking.py` — 验证代码和紧凑实验记录不会被忽略。
- Create: `MedicalGraphRAG/tests/test_bm25_export.py` — 输入校验与导出测试。
- Create: `MedicalGraphRAG/tests/test_bm25_collapse.py` — 聚合与稳定排序测试。
- Create: `MedicalGraphRAG/tests/test_bm25_evaluation.py` — split、指标、延迟、案例与 manifest 测试。
- Create: `MedicalGraphRAG/tests/test_pyserini_scripts.py` — 索引/检索脚本纯 Python 边界测试。
- Modify: `MedicalGraphRAG/tests/test_cli.py` — 新子命令冒烟。
- Create after real run: `MedicalGraphRAG/experiments/pubmedqa_hard_v1/bm25_abstract_only/{metrics.json,run_manifest.json,cases.json}`。
- Modify after real run: `STUDY_PROGRESS.md` — 记录真实命令、指标、版本与下一步。

### Task 1: 冻结输入校验与 Pyserini collection 导出

**Files:**
- Create: `MedicalGraphRAG/src/medical_graphrag/retrieval/__init__.py`
- Create: `MedicalGraphRAG/src/medical_graphrag/retrieval/bm25.py`
- Create: `MedicalGraphRAG/tests/test_bm25_export.py`

- [ ] **Step 1: 写输入哈希、字段泄漏与 ID 对齐的失败测试**

```python
import hashlib
import json
from pathlib import Path

import pytest

from medical_graphrag.retrieval.bm25 import export_pyserini_collection


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset(root: Path) -> None:
    root.mkdir()
    questions = root / "questions.jsonl"
    documents = root / "documents.jsonl"
    chunks = root / "chunks.jsonl"
    qrels = root / "qrels.tsv"
    questions.write_text(
        '{"query_id":"q1","question":"alpha?","answer":"yes","long_answer":"x","split":"dev"}\n',
        encoding="utf-8",
    )
    documents.write_text(
        '{"doc_id":"d1","title":"LEAK TITLE","content":"alpha body","source":"pubmedqa","year":null}\n',
        encoding="utf-8",
    )
    chunks.write_text(
        '{"chunk_id":"c1","doc_id":"d1","order":0,"title":"LEAK TITLE","content":"alpha body","source":"pubmedqa"}\n',
        encoding="utf-8",
    )
    qrels.write_text("query_id\tdoc_id\trelevance\nq1\td1\t1\n", encoding="utf-8")
    artifacts = {path.name: _sha(path) for path in (questions, documents, chunks, qrels)}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "counts": {"questions": 1, "documents": 1, "chunks": 1, "qrels": 1},
                "artifact_hashes": artifacts,
            }
        ),
        encoding="utf-8",
    )


def test_export_uses_content_only_and_keeps_metadata_separate(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    _write_dataset(dataset)
    summary = export_pyserini_collection(dataset, output)
    collection = json.loads((output / "collection/chunks.jsonl").read_text(encoding="utf-8"))
    metadata = json.loads((output / "chunk_metadata.jsonl").read_text(encoding="utf-8"))
    assert collection == {"contents": "alpha body", "id": "c1"}
    assert "LEAK TITLE" not in collection["contents"]
    assert metadata["chunk_id"] == collection["id"]
    assert metadata["doc_id"] == "d1"
    assert summary["chunk_count"] == 1


def test_export_rejects_tampered_artifact(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    with (dataset / "chunks.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch.*chunks.jsonl"):
        export_pyserini_collection(dataset, tmp_path / "output")


def test_export_rejects_duplicate_chunk_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    row = (dataset / "chunks.jsonl").read_text(encoding="utf-8")
    (dataset / "chunks.jsonl").write_text(row + row, encoding="utf-8")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    manifest["counts"]["chunks"] = 2
    manifest["artifact_hashes"]["chunks.jsonl"] = _sha(dataset / "chunks.jsonl")
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate chunk_id: c1"):
        export_pyserini_collection(dataset, tmp_path / "output")
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `python -m pytest tests/test_bm25_export.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'medical_graphrag.retrieval'`。

- [ ] **Step 3: 实现冻结输入验证和原子导出**

`retrieval/__init__.py`：

```python
"""Retrieval backends and shared ranking utilities."""
```

`retrieval/bm25.py` 核心接口：

```python
import json
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file, write_jsonl


ARTIFACT_NAMES = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def validate_frozen_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ARTIFACT_NAMES:
        actual = sha256_file(dataset_dir / name)
        expected = manifest["artifact_hashes"][name]
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
    return manifest


def export_pyserini_collection(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    manifest = validate_frozen_dataset(dataset_dir)
    chunks = _read_jsonl(dataset_dir / "chunks.jsonl")
    if len(chunks) != manifest["counts"]["chunks"]:
        raise ValueError("chunk count does not match manifest")
    seen: set[str] = set()
    collection = []
    metadata = []
    for row in chunks:
        chunk_id = str(row["chunk_id"])
        content = str(row["content"]).strip()
        doc_id = str(row["doc_id"]).strip()
        if chunk_id in seen:
            raise ValueError(f"duplicate chunk_id: {chunk_id}")
        if not chunk_id or not content or not doc_id:
            raise ValueError("chunk_id, doc_id, and content must not be empty")
        seen.add(chunk_id)
        collection.append({"id": chunk_id, "contents": content})
        metadata.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "order": int(row["order"]),
                "title": str(row["title"]),
                "source": str(row["source"]),
            }
        )
    write_jsonl(output_dir / "collection/chunks.jsonl", collection)
    write_jsonl(output_dir / "chunk_metadata.jsonl", metadata)
    return {
        "chunk_count": len(collection),
        "collection_sha256": sha256_file(output_dir / "collection/chunks.jsonl"),
        "metadata_sha256": sha256_file(output_dir / "chunk_metadata.jsonl"),
    }
```

- [ ] **Step 4: 运行导出测试并确认通过**

Run: `python -m pytest tests/test_bm25_export.py -v`

Expected: `3 passed`。

- [ ] **Step 5: 运行完整测试并提交**

Run: `python -m pytest -v`

Expected: 原有 25 项和新增 3 项全部通过。

```powershell
git add MedicalGraphRAG/src/medical_graphrag/retrieval MedicalGraphRAG/tests/test_bm25_export.py
git commit -m "feat: export frozen chunks for Pyserini"
```

### Task 2: Chunk 命中聚合为稳定文档排名

**Files:**
- Modify: `MedicalGraphRAG/src/medical_graphrag/retrieval/bm25.py`
- Create: `MedicalGraphRAG/tests/test_bm25_collapse.py`

- [ ] **Step 1: 写最大分数、稳定并列与唯一文档下限的失败测试**

```python
import pytest

from medical_graphrag.retrieval.bm25 import collapse_chunk_hits


def test_collapse_uses_max_score_and_stable_ties() -> None:
    metadata = {
        "c1": {"doc_id": "d1"},
        "c2": {"doc_id": "d1"},
        "c3": {"doc_id": "d2"},
        "c4": {"doc_id": "d0"},
    }
    hits = [
        {"chunk_id": "c1", "chunk_rank": 1, "score": 4.0},
        {"chunk_id": "c3", "chunk_rank": 2, "score": 5.0},
        {"chunk_id": "c4", "chunk_rank": 3, "score": 5.0},
        {"chunk_id": "c2", "chunk_rank": 4, "score": 6.0},
    ]
    ranking = collapse_chunk_hits(hits, metadata, min_unique_docs=2)
    assert [item["doc_id"] for item in ranking] == ["d1", "d2", "d0"]
    assert ranking[0]["score"] == 6.0
    assert ranking[0]["best_chunk_id"] == "c2"
    assert ranking[1]["best_chunk_rank"] == 2


def test_collapse_rejects_unknown_chunk() -> None:
    with pytest.raises(ValueError, match="unknown chunk_id: missing"):
        collapse_chunk_hits(
            [{"chunk_id": "missing", "chunk_rank": 1, "score": 1.0}],
            {},
            min_unique_docs=1,
        )


def test_collapse_rejects_short_document_ranking() -> None:
    with pytest.raises(ValueError, match="expected at least 2 unique documents, got 1"):
        collapse_chunk_hits(
            [{"chunk_id": "c1", "chunk_rank": 1, "score": 1.0}],
            {"c1": {"doc_id": "d1"}},
            min_unique_docs=2,
        )
```

- [ ] **Step 2: 运行测试并确认函数尚不存在**

Run: `python -m pytest tests/test_bm25_collapse.py -v`

Expected: FAIL with `ImportError: cannot import name 'collapse_chunk_hits'`。

- [ ] **Step 3: 实现最大分数聚合和确定性排序**

在 `retrieval/bm25.py` 中增加：

```python
from collections.abc import Mapping, Sequence


def collapse_chunk_hits(
    hits: Sequence[Mapping[str, object]],
    metadata: Mapping[str, Mapping[str, object]],
    *,
    min_unique_docs: int = 10,
) -> list[dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for hit in hits:
        chunk_id = str(hit["chunk_id"])
        if chunk_id not in metadata:
            raise ValueError(f"unknown chunk_id: {chunk_id}")
        doc_id = str(metadata[chunk_id]["doc_id"])
        candidate = {
            "doc_id": doc_id,
            "score": float(hit["score"]),
            "best_chunk_id": chunk_id,
            "best_chunk_rank": int(hit["chunk_rank"]),
        }
        current = best.get(doc_id)
        if current is None or (
            -float(candidate["score"]), int(candidate["best_chunk_rank"]), chunk_id
        ) < (
            -float(current["score"]),
            int(current["best_chunk_rank"]),
            str(current["best_chunk_id"]),
        ):
            best[doc_id] = candidate
    ranking = sorted(
        best.values(),
        key=lambda item: (-float(item["score"]), int(item["best_chunk_rank"]), str(item["doc_id"])),
    )
    if len(ranking) < min_unique_docs:
        raise ValueError(
            f"expected at least {min_unique_docs} unique documents, got {len(ranking)}"
        )
    return ranking
```

- [ ] **Step 4: 运行聚合测试与完整测试**

Run: `python -m pytest tests/test_bm25_collapse.py -v`

Expected: `3 passed`。

Run: `python -m pytest -v`

Expected: 全部通过。

- [ ] **Step 5: 提交聚合逻辑**

```powershell
git add MedicalGraphRAG/src/medical_graphrag/retrieval/bm25.py MedicalGraphRAG/tests/test_bm25_collapse.py
git commit -m "feat: collapse BM25 chunks into document rankings"
```

### Task 3: 容器内可审计的索引与检索脚本

**Files:**
- Create: `MedicalGraphRAG/scripts/build_pyserini_index.py`
- Create: `MedicalGraphRAG/scripts/search_pyserini_bm25.py`
- Create: `MedicalGraphRAG/tests/test_pyserini_scripts.py`

- [ ] **Step 1: 写命令构造和查询输出的失败测试**

测试通过 `importlib.util` 从文件加载脚本，避免 Windows 测试环境安装 Pyserini：

```python
import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_index_command_is_fixed() -> None:
    module = _load("build_pyserini_index")
    command = module.build_command(Path("collection"), Path("index"), threads=8)
    assert command[-12:] == [
        "--collection", "JsonCollection", "--input", "collection", "--index", "index",
        "--generator", "DefaultLuceneDocumentGenerator", "--threads", "8", "--stemmer", "porter",
    ]


def test_search_one_query_records_rank_score_and_doc_id() -> None:
    module = _load("search_pyserini_bm25")

    class Hit:
        def __init__(self, docid: str, score: float) -> None:
            self.docid = docid
            self.score = score

    class Searcher:
        def search(self, query: str, k: int):
            assert query == "alpha?"
            assert k == 1
            return [Hit("c1", 2.5)]

    result = module.search_one(
        Searcher(),
        {"query_id": "q1", "question": "alpha?", "split": "dev"},
        {"c1": "d1"},
        top_k=1,
        clock=iter([1.0, 1.012]).__next__,
    )
    assert result["query_id"] == "q1"
    assert result["split"] == "dev"
    assert result["latency_ms"] == 12.0
    assert result["hits"] == [
        {"chunk_id": "c1", "doc_id": "d1", "chunk_rank": 1, "score": 2.5}
    ]
```

- [ ] **Step 2: 运行测试并确认脚本不存在**

Run: `python -m pytest tests/test_pyserini_scripts.py -v`

Expected: FAIL with `FileNotFoundError`。

- [ ] **Step 3: 实现索引包装脚本**

`scripts/build_pyserini_index.py` 提供固定命令，并在成功后写 `index_build.json`：

```python
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def build_command(collection: Path, index: Path, threads: int) -> list[str]:
    return [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection", "--input", str(collection),
        "--index", str(index), "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads), "--stemmer", "porter",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    subprocess.run(build_command(args.collection, args.index, args.threads), check=True)
    elapsed = time.perf_counter() - started
    size = sum(path.stat().st_size for path in args.index.rglob("*") if path.is_file())
    import pyserini
    java_version = subprocess.run(
        ["java", "-version"], text=True, capture_output=True, check=True
    ).stderr.splitlines()[0]
    report = {
        "elapsed_seconds": elapsed,
        "index_bytes": size,
        "pyserini_version": pyserini.__version__,
        "python_version": platform.python_version(),
        "java_version": java_version,
        "threads": args.threads,
        "stemmer": "porter",
        "command": build_command(args.collection, args.index, args.threads),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 实现 Top-100 查询脚本**

`scripts/search_pyserini_bm25.py` 延迟导入 Pyserini，显式设置参数：

```python
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def search_one(searcher, question, chunk_to_doc, top_k, clock: Callable[[], float] = time.perf_counter):
    started = clock()
    raw_hits = searcher.search(str(question["question"]), k=top_k)
    latency_ms = round((clock() - started) * 1000, 6)
    hits = []
    for rank, hit in enumerate(raw_hits, start=1):
        if hit.docid not in chunk_to_doc:
            raise ValueError(f"unknown chunk_id from Pyserini: {hit.docid}")
        hits.append(
            {
                "chunk_id": hit.docid,
                "doc_id": chunk_to_doc[hit.docid],
                "chunk_rank": rank,
                "score": float(hit.score),
            }
        )
    return {
        "query_id": str(question["query_id"]),
        "split": str(question["split"]),
        "latency_ms": latency_ms,
        "hits": hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    args = parser.parse_args()
    import pyserini
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(args.index)
    searcher.set_bm25(k1=args.k1, b=args.b)
    questions = _read_jsonl(args.questions)
    metadata = _read_jsonl(args.metadata)
    chunk_to_doc = {str(row["chunk_id"]): str(row["doc_id"]) for row in metadata}
    rows = [search_one(searcher, row, chunk_to_doc, args.top_k) for row in questions]
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate query_id in search output")
    if any(len(row["hits"]) != args.top_k for row in rows):
        raise ValueError(f"every query must return exactly {args.top_k} hits")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "command": sys.argv,
                "query_count": len(rows),
                "top_k": args.top_k,
                "k1": args.k1,
                "b": args.b,
                "pyserini_version": pyserini.__version__,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 运行脚本测试和完整测试**

Run: `python -m pytest tests/test_pyserini_scripts.py -v`

Expected: `2 passed`。

Run: `python -m pytest -v`

Expected: 全部通过。

- [ ] **Step 6: 提交容器脚本**

```powershell
git add MedicalGraphRAG/scripts MedicalGraphRAG/tests/test_pyserini_scripts.py
git commit -m "feat: add auditable Pyserini BM25 runners"
```

### Task 4: Split-aware 评测、延迟、案例和 CLI

**Files:**
- Create: `MedicalGraphRAG/src/medical_graphrag/evaluation/bm25.py`
- Modify: `MedicalGraphRAG/src/medical_graphrag/cli.py`
- Create: `MedicalGraphRAG/tests/test_bm25_evaluation.py`
- Modify: `MedicalGraphRAG/tests/test_cli.py`

- [ ] **Step 1: 写 dev/test 隔离与输出契约的失败测试**

```python
import json
from pathlib import Path

from medical_graphrag.evaluation.bm25 import evaluate_bm25_run


def test_evaluation_reports_splits_separately(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "questions.jsonl").write_text(
        '\n'.join([
            '{"query_id":"q1","question":"a","answer":"yes","long_answer":"x","split":"dev"}',
            '{"query_id":"q2","question":"b","answer":"no","long_answer":"y","split":"test"}',
        ]) + '\n', encoding="utf-8"
    )
    (dataset / "documents.jsonl").write_text(
        '\n'.join([
            '{"doc_id":"d1","title":"one","content":"a","source":"pubmedqa","year":null}',
            '{"doc_id":"d2","title":"two","content":"b","source":"pubmedqa","year":null}',
            '{"doc_id":"x","title":"other","content":"z","source":"medrag_pubmed","year":null}',
        ]) + '\n', encoding="utf-8"
    )
    (dataset / "chunks.jsonl").write_text(
        '\n'.join([
            '{"chunk_id":"c1","doc_id":"d1","order":0,"title":"one","content":"gold alpha text","source":"pubmedqa"}',
            '{"chunk_id":"c2","doc_id":"d2","order":0,"title":"two","content":"gold beta text","source":"pubmedqa"}',
            '{"chunk_id":"cx","doc_id":"x","order":0,"title":"other","content":"other text","source":"medrag_pubmed"}',
        ]) + '\n', encoding="utf-8"
    )
    (dataset / "qrels.tsv").write_text(
        "query_id\tdoc_id\trelevance\nq1\td1\t1\nq2\td2\t1\n", encoding="utf-8"
    )
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        '\n'.join([
            '{"chunk_id":"c1","doc_id":"d1","order":0,"title":"one","source":"pubmedqa"}',
            '{"chunk_id":"c2","doc_id":"d2","order":0,"title":"two","source":"pubmedqa"}',
            '{"chunk_id":"cx","doc_id":"x","order":0,"title":"other","source":"medrag_pubmed"}',
        ]) + '\n', encoding="utf-8"
    )
    rankings = tmp_path / "rankings.jsonl"
    rankings.write_text(
        '\n'.join([
            '{"query_id":"q1","split":"dev","latency_ms":10,"hits":[{"chunk_id":"c1","doc_id":"d1","chunk_rank":1,"score":3},{"chunk_id":"cx","doc_id":"x","chunk_rank":2,"score":2}]}',
            '{"query_id":"q2","split":"test","latency_ms":20,"hits":[{"chunk_id":"cx","doc_id":"x","chunk_rank":1,"score":3},{"chunk_id":"c2","doc_id":"d2","chunk_rank":2,"score":2}]}',
        ]) + '\n', encoding="utf-8"
    )
    output = tmp_path / "experiment"
    result = evaluate_bm25_run(
        dataset, metadata, rankings, output,
        min_unique_docs=2,
        run_context={"git_commit": "abc", "index": {"elapsed_seconds": 1.0}},
    )
    assert set(result) == {"dev", "test"}
    assert result["dev"]["sample_count"] == 1
    assert result["dev"]["recall@1"] == 1.0
    assert result["test"]["recall@1"] == 0.0
    assert result["test"]["recall@5"] == 1.0
    assert result["test"]["latency_ms"]["p50"] == 20.0
    assert json.loads((output / "metrics.json").read_text(encoding="utf-8")) == result
    assert (output / "run_manifest.json").exists()
    assert (output / "cases.json").exists()
```

同时修改 `test_cli.py`：

```python
def test_cli_lists_pipeline_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "medical_graphrag.cli", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    for command in ("fetch-pubmedqa", "audit", "build", "export-pyserini", "evaluate-bm25"):
        assert command in result.stdout
```

- [ ] **Step 2: 运行测试并确认评测模块和 CLI 子命令尚不存在**

Run: `python -m pytest tests/test_bm25_evaluation.py tests/test_cli.py -v`

Expected: evaluation import FAIL；修复 import 前不得开始实现指标文件。

- [ ] **Step 3: 实现 split-aware evaluator**

`evaluation/bm25.py` 使用以下公开接口，内部复用 `evaluate_rankings` 与 `collapse_chunk_hits`：

```python
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file, write_json
from medical_graphrag.evaluation.retrieval import evaluate_rankings
from medical_graphrag.retrieval.bm25 import collapse_chunk_hits


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _qrels(path: Path) -> dict[str, str]:
    rows = path.read_text(encoding="utf-8").splitlines()[1:]
    result = {}
    for row in rows:
        query_id, doc_id, relevance = row.split("\t")
        if relevance != "1" or query_id in result:
            raise ValueError("qrels must contain one relevance=1 row per query")
        result[query_id] = doc_id
    return result


def evaluate_bm25_run(
    dataset_dir: Path,
    metadata_path: Path,
    rankings_path: Path,
    output_dir: Path,
    *,
    min_unique_docs: int = 10,
    run_context: dict[str, Any],
) -> dict[str, Any]:
    questions = {str(row["query_id"]): row for row in _jsonl(dataset_dir / "questions.jsonl")}
    documents = {str(row["doc_id"]): row for row in _jsonl(dataset_dir / "documents.jsonl")}
    chunks = _jsonl(dataset_dir / "chunks.jsonl")
    metadata_rows = _jsonl(metadata_path)
    metadata = {str(row["chunk_id"]): row for row in metadata_rows}
    raw_rows = _jsonl(rankings_path)
    if len(raw_rows) != len(questions) or {str(row["query_id"]) for row in raw_rows} != set(questions):
        raise ValueError("ranking query set does not match questions")
    qrels = _qrels(dataset_dir / "qrels.tsv")
    if set(qrels) != set(questions) or any(doc_id not in documents for doc_id in qrels.values()):
        raise ValueError("qrels do not resolve one existing document for every question")
    collapsed = {}
    latencies = {}
    detailed = {}
    for row in raw_rows:
        query_id = str(row["query_id"])
        if row["split"] != questions[query_id]["split"]:
            raise ValueError(f"split mismatch for {query_id}")
        latency = float(row["latency_ms"])
        if not math.isfinite(latency) or latency < 0:
            raise ValueError(f"invalid latency for {query_id}")
        ranking = collapse_chunk_hits(row["hits"], metadata, min_unique_docs=min_unique_docs)
        collapsed[query_id] = [str(item["doc_id"]) for item in ranking]
        detailed[query_id] = [
            {**item, "title": documents[str(item["doc_id"])]["title"]} for item in ranking
        ]
        latencies[query_id] = latency
    metrics = {}
    for split in ("dev", "test"):
        ids = [query_id for query_id, row in questions.items() if row["split"] == split]
        split_qrels = {query_id: qrels[query_id] for query_id in ids}
        split_rankings = {query_id: collapsed[query_id] for query_id in ids}
        values = [latencies[query_id] for query_id in ids]
        split_metrics = evaluate_rankings(split_qrels, split_rankings, ks=(1, 5, 10))
        metrics[split] = {
            "sample_count": len(ids),
            **split_metrics,
            "latency_ms": {
                "mean": statistics.fmean(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            },
        }
    cases = {}
    test_ids = [query_id for query_id, row in questions.items() if row["split"] == "test"]
    gold_ranks = {
        query_id: (
            collapsed[query_id].index(qrels[query_id]) + 1
            if qrels[query_id] in collapsed[query_id] else None
        )
        for query_id in test_ids
    }
    success_all = [query_id for query_id in test_ids if gold_ranks[query_id] is not None and gold_ranks[query_id] <= 10]
    failure_all = [query_id for query_id in test_ids if gold_ranks[query_id] is None or gold_ranks[query_id] > 10]
    success = success_all[:5]
    failure = failure_all[:5]
    for label, ids in (("success", success), ("failure", failure)):
        cases[label] = [
            {
                "query_id": query_id,
                "question": questions[query_id]["question"],
                "gold_doc_id": qrels[query_id],
                "gold_title": documents[qrels[query_id]]["title"],
                "gold_rank": gold_ranks[query_id],
                "gold_chunk_excerpt": next(
                    str(chunk["content"])[:500]
                    for chunk in chunks if chunk["doc_id"] == qrels[query_id]
                ),
                "top_documents": detailed[query_id][:10],
            }
            for query_id in ids
        ]
    cases["summary"] = {
        "success_at_10_available": len(success_all),
        "failure_at_10_available": len(failure_all),
        "saved_per_group_max": 5,
    }
    run_manifest = {
        **run_context,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rankings_sha256": sha256_file(rankings_path),
        "question_count": len(questions),
        "document_count": len(documents),
        "chunk_metadata_count": len(metadata),
        "split_counts": {name: value["sample_count"] for name, value in metrics.items()},
        "chunk_top_k": 100,
        "aggregation": "max_chunk_score",
        "bm25": {"k1": 0.9, "b": 0.4},
        "text_mode": "abstract_only",
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(output_dir / "cases.json", cases)
    return metrics
```

- [ ] **Step 4: 增加 CLI 子命令和参数转发**

在 `cli.py` 的 parser 中增加：

```python
export = subparsers.add_parser("export-pyserini")
export.add_argument("--dataset-dir", type=Path, required=True)
export.add_argument("--output-dir", type=Path, required=True)

evaluate = subparsers.add_parser("evaluate-bm25")
evaluate.add_argument("--dataset-dir", type=Path, required=True)
evaluate.add_argument("--metadata", type=Path, required=True)
evaluate.add_argument("--rankings", type=Path, required=True)
evaluate.add_argument("--index-report", type=Path, required=True)
evaluate.add_argument("--search-report", type=Path, required=True)
evaluate.add_argument("--output-dir", type=Path, required=True)
evaluate.add_argument("--git-commit", required=True)
evaluate.add_argument("--docker-image", required=True)
```

在 `main()` 中增加：

```python
if args.command == "export-pyserini":
    export_pyserini_collection(args.dataset_dir, args.output_dir)
    return 0
if args.command == "evaluate-bm25":
    index_report = json.loads(args.index_report.read_text(encoding="utf-8"))
    search_report = json.loads(args.search_report.read_text(encoding="utf-8"))
    dataset_manifest = validate_frozen_dataset(args.dataset_dir)
    evaluate_bm25_run(
        args.dataset_dir,
        args.metadata,
        args.rankings,
        args.output_dir,
        run_context={
            "git_commit": args.git_commit,
            "host_platform": platform.platform(),
            "host_python_version": platform.python_version(),
            "docker_image": args.docker_image,
            "evaluation_command": sys.argv,
            "index": index_report,
            "search": search_report,
            "dataset_manifest_sha256": sha256_file(args.dataset_dir / "manifest.json"),
            "dataset_artifact_hashes": dataset_manifest["artifact_hashes"],
        },
    )
    return 0
```

并在文件顶部显式导入 `json`、`platform`、`sys`、`sha256_file`、`validate_frozen_dataset`、`export_pyserini_collection`、`evaluate_bm25_run`。

- [ ] **Step 5: 运行聚焦测试与完整测试**

Run: `python -m pytest tests/test_bm25_evaluation.py tests/test_cli.py -v`

Expected: 新评测测试和 CLI 测试全部通过。

Run: `python -m pytest -v`

Expected: 全部通过。

- [ ] **Step 6: 提交评测与 CLI**

```powershell
git add MedicalGraphRAG/src/medical_graphrag/evaluation/bm25.py MedicalGraphRAG/src/medical_graphrag/cli.py MedicalGraphRAG/tests/test_bm25_evaluation.py MedicalGraphRAG/tests/test_cli.py
git commit -m "feat: evaluate split-aware BM25 retrieval runs"
```

### Task 5: 运行手册、忽略规则与可追踪产物

**Files:**
- Modify: `MedicalGraphRAG/.gitignore`
- Modify: `MedicalGraphRAG/tests/test_git_tracking.py`
- Modify: `MedicalGraphRAG/README.md`

- [ ] **Step 1: 写 Git 追踪失败测试**

在 `test_git_tracking.py` 增加路径断言：

```python
def _is_ignored(path: str) -> bool:
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path],
        cwd=repository_root,
        check=False,
    )
    return result.returncode == 0


def test_bm25_experiment_json_is_trackable() -> None:
    assert not _is_ignored(
        "MedicalGraphRAG/experiments/pubmedqa_hard_v1/bm25_abstract_only/metrics.json"
    )


def test_bm25_large_outputs_stay_ignored() -> None:
    assert _is_ignored("MedicalGraphRAG/indexes/pubmedqa_hard_v1/bm25_abstract_only/segments_1")
    assert _is_ignored(
        "MedicalGraphRAG/outputs/pubmedqa_hard_v1/bm25_abstract_only/raw_rankings.jsonl"
    )
```

- [ ] **Step 2: 运行追踪测试并确认实验 JSON 当前策略**

Run: `python -m pytest tests/test_git_tracking.py -v`

Expected: 若 experiment JSON 被意外忽略则 FAIL；indexes/outputs 必须保持 ignored。

- [ ] **Step 3: 明确项目局部忽略规则**

在 `MedicalGraphRAG/.gitignore` 增加：

```gitignore
indexes/
outputs/
!experiments/
!experiments/**/*.json
```

若根 `.gitignore` 的父目录规则阻止否定规则生效，则只增加必要的父目录例外；用 `git check-ignore -v` 验证，不放开任意 `.jsonl` 原始排名。

- [ ] **Step 4: 在 README 写入唯一正式运行命令**

加入下列命令，Docker 内项目路径与已确认挂载保持一致：

```powershell
python -m medical_graphrag.cli export-pyserini `
  --dataset-dir data/processed/pubmedqa_hard_v1 `
  --output-dir outputs/pubmedqa_hard_v1/bm25_abstract_only

docker exec llm-pytorch python "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/scripts/build_pyserini_index.py" `
  --collection "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/outputs/pubmedqa_hard_v1/bm25_abstract_only/collection" `
  --index "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/indexes/pubmedqa_hard_v1/bm25_abstract_only" `
  --report "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/outputs/pubmedqa_hard_v1/bm25_abstract_only/index_build.json" `
  --threads 8

docker exec llm-pytorch python "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/scripts/search_pyserini_bm25.py" `
  --index "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/indexes/pubmedqa_hard_v1/bm25_abstract_only" `
  --questions "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/data/processed/pubmedqa_hard_v1/questions.jsonl" `
  --metadata "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/outputs/pubmedqa_hard_v1/bm25_abstract_only/chunk_metadata.jsonl" `
  --output "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/outputs/pubmedqa_hard_v1/bm25_abstract_only/raw_rankings.jsonl" `
  --report "/workspace/code_list/some tricks/LLMLeanring/MedicalGraphRAG/outputs/pubmedqa_hard_v1/bm25_abstract_only/search_run.json" `
  --top-k 100 --k1 0.9 --b 0.4

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

README 同时声明：dev 用于调试，test 是主报告；不可混合 1,000 题；没有 GPU 显存指标，因为 BM25/Lucene 运行在 CPU。

- [ ] **Step 5: 验证 README 命令入口和 Git 规则**

Run: `python -m medical_graphrag.cli export-pyserini --help`

Expected: exit 0，显示 `--dataset-dir` 和 `--output-dir`。

Run: `python -m medical_graphrag.cli evaluate-bm25 --help`

Expected: exit 0，显示八个必需参数。

Run: `python -m pytest tests/test_git_tracking.py tests/test_cli.py -v`

Expected: 全部通过。

- [ ] **Step 6: 提交文档和追踪规则**

```powershell
git add MedicalGraphRAG/.gitignore MedicalGraphRAG/tests/test_git_tracking.py MedicalGraphRAG/README.md
git commit -m "docs: add reproducible BM25 runbook"
```

### Task 6: 全量真实运行、独立验证与学习进度更新

**Files:**
- Create: `MedicalGraphRAG/experiments/pubmedqa_hard_v1/bm25_abstract_only/metrics.json`
- Create: `MedicalGraphRAG/experiments/pubmedqa_hard_v1/bm25_abstract_only/run_manifest.json`
- Create: `MedicalGraphRAG/experiments/pubmedqa_hard_v1/bm25_abstract_only/cases.json`
- Modify: `STUDY_PROGRESS.md`

- [ ] **Step 1: 在运行前执行完整测试并记录基线 commit**

Run: `python -m pytest -v`

Expected: 所有测试通过，0 failed。

Run: `git status --short`

Expected: 空输出。

Run: `git rev-parse HEAD`

Expected: 输出当前实现 commit，并在最终 `run_manifest.json` 中完全一致。

- [ ] **Step 2: 导出 7,562 个 abstract-only chunk**

Run: Task 5 README 中的 `export-pyserini` 命令。

Expected:

- `collection/chunks.jsonl` 和 `chunk_metadata.jsonl` 均为 7,562 行；
- collection 中没有 title/source/answer 字段；
- 导出摘要中的输入哈希与冻结 manifest 一致。

- [ ] **Step 3: 在 `llm-pytorch` 中构建 Lucene 索引**

Run: Task 5 README 中的 `build_pyserini_index.py` 命令。

Expected:

- exit 0；
- 索引目录非空；
- `index_build.json` 中 `elapsed_seconds > 0`、`index_bytes > 0`、`pyserini_version == "0.22.1"`。

- [ ] **Step 4: 在完整 1,000 条 query 上检索 Top-100 chunk**

Run: Task 5 README 中的 `search_pyserini_bm25.py` 命令。

Expected:

- `raw_rankings.jsonl` 恰有 1,000 行和 1,000 个唯一 query ID；
- 每行恰有 100 个 chunk hit；
- `k1=0.9`、`b=0.4`，索引文本仅为 abstract content；
- 每题延迟有限且非负。

- [ ] **Step 5: 生成 dev/test 分离的真实指标与案例**

Run: Task 5 README 中的 `evaluate-bm25` 命令。

Expected:

- `metrics.json` 只有 `dev`、`test` 两个一级指标组；
- 两个 split 的 `sample_count` 都是 500；
- 每组包含 Recall@1/5/10、MRR@10、nDCG@10、mean/P50/P95 latency；
- 所有指标来自脚本输出，不手工填写；
- `cases.json` 保存 test 成功与失败案例。

- [ ] **Step 6: 独立复算关键结构与指标**

使用一条只读验证命令重新读取 qrels、raw rankings 和 metrics：

```powershell
python -c "import json; from pathlib import Path; p=Path('outputs/pubmedqa_hard_v1/bm25_abstract_only/raw_rankings.jsonl'); rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]; m=json.loads(Path('experiments/pubmedqa_hard_v1/bm25_abstract_only/metrics.json').read_text(encoding='utf-8')); assert len(rows)==1000; assert len({r['query_id'] for r in rows})==1000; assert all(len(r['hits'])==100 for r in rows); assert m['dev']['sample_count']==m['test']['sample_count']==500; print({'queries':len(rows),'dev_recall@10':m['dev']['recall@10'],'test_recall@10':m['test']['recall@10']})"
```

Expected: assertions 全部通过并打印真实 dev/test Recall@10。

- [ ] **Step 7: 人工阅读案例并更新学习进度**

在 `STUDY_PROGRESS.md` 记录：

- 当前阶段仍为阶段 1 MedRAG 基础检索；
- 完整导出、索引、检索和评测命令；
- dev/test 的真实 Recall@1/5/10、MRR@10、nDCG@10、mean/P50/P95 latency；
- 索引时间、索引空间、Pyserini/Java/Docker 版本、Git commit；
- 至少一个 BM25 成功原因和一个失败原因；
- 显存峰值记为“不适用：CPU Lucene BM25”，不能填写推测数字；
- 下一步是解释 BM25 词项匹配后再开始 Dense baseline。

- [ ] **Step 8: 最终验证并提交真实实验记录**

Run: `python -m pytest -v`

Expected: 全部通过，0 failed。

Run: `git diff --check`

Expected: 空输出。

Run: `git status --short`

Expected: 只出现三个 experiment JSON 与 `STUDY_PROGRESS.md`。

```powershell
git add MedicalGraphRAG/experiments/pubmedqa_hard_v1/bm25_abstract_only/metrics.json MedicalGraphRAG/experiments/pubmedqa_hard_v1/bm25_abstract_only/run_manifest.json MedicalGraphRAG/experiments/pubmedqa_hard_v1/bm25_abstract_only/cases.json STUDY_PROGRESS.md
git commit -m "exp: record PubMedQA BM25 hard baseline"
```

完成后重新运行 `git status --short`，必须为空；最终汇报引用实际 commit 和实际脚本输出，不根据预期值描述性能。
