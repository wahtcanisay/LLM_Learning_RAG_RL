# MedicalGraphRAG Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the standalone `MedicalGraphRAG/` project and a deterministic PubMedQA hard-retrieval data pipeline that produces validated questions, documents, chunks, qrels, audit, manifest, and retrieval metrics.

**Architecture:** Keep `MedRAG/` and `LinearRAG/` read-only. Implement a small Python package under `MedicalGraphRAG/src/medical_graphrag` with separate modules for schemas, PubMedQA ingestion, MedRAG distractor sampling, chunking, benchmark assembly, validation, and metrics. Expose the pipeline through `python -m medical_graphrag.cli`, with all behavior driven by one versioned JSON config.

**Tech Stack:** Python 3.12, standard library, `transformers` tokenizer, `pytest`, JSONL/TSV artifacts, SHA-256 manifests.

---

## File map

Create the following project without running `git init` inside it:

```text
MedicalGraphRAG/
├── .gitignore                         # Ignore raw and large generated data
├── README.md                          # Project scope and current runnable commands
├── pyproject.toml                     # Package metadata and test configuration
├── configs/
│   └── pubmedqa_hard_v1.json          # All fixed dataset parameters
├── data/
│   ├── raw/README.md                  # Expected source files and provenance
│   └── processed/README.md            # Generated-artifact policy
├── src/medical_graphrag/
│   ├── __init__.py                    # Package version
│   ├── cli.py                         # fetch, audit, and build commands
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schemas.py                 # Validated immutable records
│   │   ├── io.py                      # Atomic JSON/JSONL/TSV and hashing helpers
│   │   ├── pubmedqa.py                # Official PQA-L loading and split assignment
│   │   ├── medrag_pubmed.py           # Deterministic shard/reservoir distractor sampling
│   │   ├── chunking.py                # Document-safe token chunking
│   │   └── benchmark.py               # Audit and full artifact assembly
│   └── evaluation/
│       ├── __init__.py
│       └── retrieval.py               # Recall, MRR, and binary nDCG metrics
└── tests/
    ├── fixtures/
    │   ├── ori_pqal_small.json
    │   ├── test_ground_truth_small.json
    │   └── medrag_pubmed/
    │       ├── pubmed23n0001.jsonl
    │       └── pubmed23n0002.jsonl
    ├── test_schemas.py
    ├── test_pubmedqa.py
    ├── test_medrag_pubmed.py
    ├── test_chunking.py
    ├── test_benchmark.py
    └── test_retrieval_metrics.py
```

## Task 1: Scaffold the standalone project

**Files:**
- Create: `MedicalGraphRAG/pyproject.toml`
- Create: `MedicalGraphRAG/.gitignore`
- Create: `MedicalGraphRAG/README.md`
- Create: `MedicalGraphRAG/src/medical_graphrag/__init__.py`
- Create: `MedicalGraphRAG/src/medical_graphrag/data/__init__.py`
- Create: `MedicalGraphRAG/src/medical_graphrag/evaluation/__init__.py`
- Create: `MedicalGraphRAG/tests/test_package.py`

- [ ] **Step 1: Write the failing package import test**

```python
# MedicalGraphRAG/tests/test_package.py
from medical_graphrag import __version__


def test_package_version() -> None:
    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Add packaging configuration**

```toml
# MedicalGraphRAG/pyproject.toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "medical-graphrag"
version = "0.1.0"
description = "Reproducible medical retrieval and GraphRAG experiments"
requires-python = ">=3.10"
dependencies = [
  "transformers>=4.44,<5",
]

[project.optional-dependencies]
dev = ["pytest>=8,<9"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 3: Add package files**

```python
# MedicalGraphRAG/src/medical_graphrag/__init__.py
__version__ = "0.1.0"
```

```python
# MedicalGraphRAG/src/medical_graphrag/data/__init__.py
"""Dataset construction utilities."""
```

```python
# MedicalGraphRAG/src/medical_graphrag/evaluation/__init__.py
"""Deterministic evaluation utilities."""
```

- [ ] **Step 4: Add data exclusions**

```gitignore
# MedicalGraphRAG/.gitignore
.pytest_cache/
__pycache__/
*.py[cod]
*.egg-info/

data/raw/*
!data/raw/README.md

data/processed/*
!data/processed/README.md
!data/processed/pubmedqa_hard_v1/
data/processed/pubmedqa_hard_v1/*
!data/processed/pubmedqa_hard_v1/README.md
!data/processed/pubmedqa_hard_v1/audit_20.json
!data/processed/pubmedqa_hard_v1/manifest.json
```

- [ ] **Step 5: Add the initial README**

```markdown
# MedicalGraphRAG

Standalone implementation for reproducible medical retrieval experiments.

The sibling `MedRAG/` and `LinearRAG/` directories are read-only references and
data sources. This project owns all new cleaning, retrieval, evaluation, tests,
and experiment records.

The first milestone is `pubmedqa_hard_v1`: 1,000 PubMedQA gold documents plus
4,000 deterministic MedRAG PubMed distractors with document-level qrels.
```

- [ ] **Step 6: Run the test**

Run from `MedicalGraphRAG/`:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/test_package.py -v
```

Expected: `1 passed`.

- [ ] **Step 7: Commit the scaffold**

```powershell
git add MedicalGraphRAG/.gitignore MedicalGraphRAG/README.md MedicalGraphRAG/pyproject.toml MedicalGraphRAG/src MedicalGraphRAG/tests/test_package.py
git commit -m "chore: scaffold MedicalGraphRAG project"
```

## Task 2: Define strict records and atomic artifact I/O

**Files:**
- Create: `MedicalGraphRAG/src/medical_graphrag/data/schemas.py`
- Create: `MedicalGraphRAG/src/medical_graphrag/data/io.py`
- Create: `MedicalGraphRAG/tests/test_schemas.py`

- [ ] **Step 1: Write failing schema tests**

```python
# MedicalGraphRAG/tests/test_schemas.py
import pytest

from medical_graphrag.data.schemas import Document, PubMedQARecord, Qrel


def test_pubmedqa_record_rejects_invalid_label() -> None:
    with pytest.raises(ValueError, match="answer"):
        PubMedQARecord(
            pmid="1",
            question="Question?",
            contexts=("Evidence",),
            answer="unknown",
            long_answer="Conclusion",
            year="2020",
            split="dev",
        )


def test_document_requires_content() -> None:
    with pytest.raises(ValueError, match="content"):
        Document("PMID:1", "Title", "", "pubmedqa", "2020")


def test_qrel_requires_binary_relevance() -> None:
    with pytest.raises(ValueError, match="relevance"):
        Qrel("1", "PMID:1", 2)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_schemas.py -v
```

Expected: collection fails because `medical_graphrag.data.schemas` does not exist.

- [ ] **Step 3: Implement immutable validated records**

```python
# MedicalGraphRAG/src/medical_graphrag/data/schemas.py
from dataclasses import asdict, dataclass
from typing import Any


VALID_ANSWERS = frozenset({"yes", "no", "maybe"})
VALID_SPLITS = frozenset({"dev", "test"})


@dataclass(frozen=True)
class PubMedQARecord:
    pmid: str
    question: str
    contexts: tuple[str, ...]
    answer: str
    long_answer: str
    year: str | None
    split: str

    def __post_init__(self) -> None:
        if not self.pmid.strip():
            raise ValueError("pmid must not be empty")
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if not self.contexts or any(not value.strip() for value in self.contexts):
            raise ValueError("contexts must contain non-empty strings")
        if self.answer not in VALID_ANSWERS:
            raise ValueError(f"answer must be one of {sorted(VALID_ANSWERS)}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")


@dataclass(frozen=True)
class Question:
    query_id: str
    question: str
    answer: str
    long_answer: str
    split: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    content: str
    source: str
    year: str | None

    def __post_init__(self) -> None:
        if not self.doc_id.strip():
            raise ValueError("doc_id must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.content.strip():
            raise ValueError("content must not be empty")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    order: int
    title: str
    content: str
    source: str


@dataclass(frozen=True)
class Qrel:
    query_id: str
    doc_id: str
    relevance: int = 1

    def __post_init__(self) -> None:
        if self.relevance != 1:
            raise ValueError("relevance must equal 1 in pubmedqa_hard_v1")


def record_dict(record: object) -> dict[str, Any]:
    return asdict(record)
```

- [ ] **Step 4: Implement atomic I/O and hashing**

```python
# MedicalGraphRAG/src/medical_graphrag/data/io.py
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _atomic_text(path, text)


def write_qrels(path: Path, rows: Iterable[tuple[str, str, int]]) -> None:
    lines = ["query_id\tdoc_id\trelevance\n"]
    lines.extend(f"{query_id}\t{doc_id}\t{relevance}\n" for query_id, doc_id, relevance in rows)
    _atomic_text(path, "".join(lines))
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_schemas.py -v
```

Expected: `3 passed`.

- [ ] **Step 6: Commit schemas and I/O**

```powershell
git add MedicalGraphRAG/src/medical_graphrag/data/schemas.py MedicalGraphRAG/src/medical_graphrag/data/io.py MedicalGraphRAG/tests/test_schemas.py
git commit -m "feat: add benchmark data contracts"
```

## Task 3: Load PubMedQA and assign the official split

**Files:**
- Create: `MedicalGraphRAG/src/medical_graphrag/data/pubmedqa.py`
- Create: `MedicalGraphRAG/tests/fixtures/ori_pqal_small.json`
- Create: `MedicalGraphRAG/tests/fixtures/test_ground_truth_small.json`
- Create: `MedicalGraphRAG/tests/test_pubmedqa.py`

- [ ] **Step 1: Add minimal fixtures**

Create `MedicalGraphRAG/tests/fixtures/ori_pqal_small.json` with:

```json
{
  "100": {
    "QUESTION": "Does treatment help?",
    "CONTEXTS": ["Background.", "Results show benefit."],
    "YEAR": "2020",
    "final_decision": "yes",
    "LONG_ANSWER": "Treatment helped."
  },
  "200": {
    "QUESTION": "Is exposure safe?",
    "CONTEXTS": ["Observed adverse events."],
    "YEAR": "2021",
    "final_decision": "no",
    "LONG_ANSWER": "Exposure was not safe."
  }
}
```

Create `MedicalGraphRAG/tests/fixtures/test_ground_truth_small.json` with:

```json
{
  "200": "no"
}
```

- [ ] **Step 2: Write failing loader tests**

```python
# MedicalGraphRAG/tests/test_pubmedqa.py
from pathlib import Path

from medical_graphrag.data.pubmedqa import load_pubmedqa


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_pubmedqa_assigns_official_test_split() -> None:
    records = load_pubmedqa(
        FIXTURES / "ori_pqal_small.json",
        FIXTURES / "test_ground_truth_small.json",
    )
    by_id = {record.pmid: record for record in records}
    assert by_id["100"].split == "dev"
    assert by_id["200"].split == "test"
    assert by_id["200"].answer == "no"
    assert by_id["100"].contexts == ("Background.", "Results show benefit.")
```

- [ ] **Step 3: Run the test to verify failure**

Run:

```powershell
python -m pytest tests/test_pubmedqa.py -v
```

Expected: import fails because `pubmedqa.py` does not exist.

- [ ] **Step 4: Implement the loader**

```python
# MedicalGraphRAG/src/medical_graphrag/data/pubmedqa.py
import json
from pathlib import Path

from medical_graphrag.data.schemas import PubMedQARecord


def load_pubmedqa(pqal_path: Path, ground_truth_path: Path) -> list[PubMedQARecord]:
    pqal = json.loads(pqal_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    missing = sorted(set(ground_truth) - set(pqal), key=int)
    if missing:
        raise ValueError(f"official test PMIDs missing from PQA-L: {missing[:5]}")

    records: list[PubMedQARecord] = []
    for pmid in sorted(pqal, key=int):
        raw = pqal[pmid]
        answer = ground_truth.get(pmid, raw.get("final_decision"))
        if pmid in ground_truth and answer != raw.get("final_decision"):
            raise ValueError(f"ground-truth label mismatch for PMID {pmid}")
        records.append(
            PubMedQARecord(
                pmid=pmid,
                question=str(raw.get("QUESTION", "")).strip(),
                contexts=tuple(str(value).strip() for value in raw.get("CONTEXTS", [])),
                answer=str(answer).strip().lower(),
                long_answer=str(raw.get("LONG_ANSWER", "")).strip(),
                year=str(raw["YEAR"]).strip() if raw.get("YEAR") is not None else None,
                split="test" if pmid in ground_truth else "dev",
            )
        )
    return records
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_pubmedqa.py -v
```

Expected: `1 passed`.

- [ ] **Step 6: Commit the loader**

```powershell
git add MedicalGraphRAG/src/medical_graphrag/data/pubmedqa.py MedicalGraphRAG/tests/fixtures/ori_pqal_small.json MedicalGraphRAG/tests/fixtures/test_ground_truth_small.json MedicalGraphRAG/tests/test_pubmedqa.py
git commit -m "feat: load official PubMedQA splits"
```

## Task 4: Sample deterministic MedRAG distractors

**Files:**
- Create: `MedicalGraphRAG/src/medical_graphrag/data/medrag_pubmed.py`
- Create: `MedicalGraphRAG/tests/fixtures/medrag_pubmed/pubmed23n0001.jsonl`
- Create: `MedicalGraphRAG/tests/fixtures/medrag_pubmed/pubmed23n0002.jsonl`
- Create: `MedicalGraphRAG/tests/test_medrag_pubmed.py`

- [ ] **Step 1: Add two three-row JSONL fixtures**

```jsonl
{"id":"s1_0","title":"Alpha","content":"Alpha abstract.","contents":"Alpha. Alpha abstract."}
{"id":"s1_1","title":"Beta","content":"Beta abstract.","contents":"Beta. Beta abstract."}
{"id":"s1_2","title":"Gamma","content":"Gamma abstract.","contents":"Gamma. Gamma abstract."}
```

```jsonl
{"id":"s2_0","title":"Delta","content":"Delta abstract.","contents":"Delta. Delta abstract."}
{"id":"s2_1","title":"Epsilon","content":"Epsilon abstract.","contents":"Epsilon. Epsilon abstract."}
{"id":"s2_2","title":"Zeta","content":"Zeta abstract.","contents":"Zeta. Zeta abstract."}
```

- [ ] **Step 2: Write failing deterministic-sampling tests**

```python
# MedicalGraphRAG/tests/test_medrag_pubmed.py
from pathlib import Path

from medical_graphrag.data.medrag_pubmed import sample_distractors


SHARDS = Path(__file__).parent / "fixtures" / "medrag_pubmed"


def test_sample_distractors_is_repeatable() -> None:
    first, first_shards = sample_distractors(
        SHARDS, seed=7, shard_count=2, per_shard=3, target_count=4,
        excluded_titles=set(), excluded_content_hashes=set(),
    )
    second, second_shards = sample_distractors(
        SHARDS, seed=7, shard_count=2, per_shard=3, target_count=4,
        excluded_titles=set(), excluded_content_hashes=set(),
    )
    assert [item.doc_id for item in first] == [item.doc_id for item in second]
    assert first_shards == second_shards
    assert len(first) == 4
    assert all(item.source == "medrag_pubmed" for item in first)
```

- [ ] **Step 3: Run the test to verify failure**

Run:

```powershell
python -m pytest tests/test_medrag_pubmed.py -v
```

Expected: import fails because `medrag_pubmed.py` does not exist.

- [ ] **Step 4: Implement stable normalization, reservoir sampling, and selection**

```python
# MedicalGraphRAG/src/medical_graphrag/data/medrag_pubmed.py
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from medical_graphrag.data.schemas import Document


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def content_hash(title: str, content: str) -> str:
    value = normalize_text(title) + "\n" + normalize_text(content)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _shard_rng(seed: int, name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def reservoir_sample(path: Path, count: int, seed: int) -> list[dict[str, object]]:
    rng = _shard_rng(seed, path.name)
    reservoir: list[dict[str, object]] = []
    seen = 0
    for line in path.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        item = json.loads(line)
        seen += 1
        if len(reservoir) < count:
            reservoir.append(item)
        else:
            index = rng.randrange(seen)
            if index < count:
                reservoir[index] = item
    return reservoir


def sample_distractors(
    shard_dir: Path,
    *,
    seed: int,
    shard_count: int,
    per_shard: int,
    target_count: int,
    excluded_titles: set[str],
    excluded_content_hashes: set[str],
) -> tuple[list[Document], list[str]]:
    shards = sorted(path for path in shard_dir.glob("*.jsonl") if path.stat().st_size > 0)
    if not shards:
        raise ValueError(f"no non-empty JSONL shards found in {shard_dir}")
    order = shards.copy()
    random.Random(seed).shuffle(order)
    selected_names: list[str] = []
    candidates: dict[str, Document] = {}

    for path in order:
        selected_names.append(path.name)
        for raw in reservoir_sample(path, per_shard, seed):
            local_id = str(raw.get("id", "")).strip()
            title = str(raw.get("title", "")).strip()
            content = str(raw.get("content", "")).strip()
            if not local_id or not title or not content:
                continue
            digest = content_hash(title, content)
            if normalize_text(title) in excluded_titles or digest in excluded_content_hashes:
                continue
            candidates.setdefault(
                digest,
                Document(f"MEDRAG:{local_id}", title, content, "medrag_pubmed", None),
            )
        minimum_shards_used = min(shard_count, len(shards))
        if len(selected_names) >= minimum_shards_used and len(candidates) >= target_count:
            break

    ranked = sorted(
        candidates.values(),
        key=lambda item: hashlib.sha256(f"{seed}:{item.doc_id}".encode("utf-8")).hexdigest(),
    )
    if len(ranked) < target_count:
        raise ValueError(f"only {len(ranked)} unique distractors available; need {target_count}")
    return ranked[:target_count], selected_names
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_medrag_pubmed.py -v
```

Expected: `1 passed`.

- [ ] **Step 6: Commit the sampler**

```powershell
git add MedicalGraphRAG/src/medical_graphrag/data/medrag_pubmed.py MedicalGraphRAG/tests/fixtures/medrag_pubmed MedicalGraphRAG/tests/test_medrag_pubmed.py
git commit -m "feat: sample deterministic PubMed distractors"
```

## Task 5: Chunk without crossing document boundaries

**Files:**
- Create: `MedicalGraphRAG/src/medical_graphrag/data/chunking.py`
- Create: `MedicalGraphRAG/tests/test_chunking.py`

- [ ] **Step 1: Write failing chunk-boundary tests**

```python
# MedicalGraphRAG/tests/test_chunking.py
from medical_graphrag.data.chunking import chunk_sections


class CharacterCodec:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(value) for value in ids)


def test_chunk_sections_never_mix_sections_or_documents() -> None:
    chunks = chunk_sections(
        doc_id="PMID:1",
        title="Title",
        sections=("abcdefgh", "XYZ"),
        source="pubmedqa",
        tokenizer=CharacterCodec(),
        max_tokens=5,
        overlap=2,
    )
    assert [chunk.chunk_id for chunk in chunks] == ["PMID:1#0", "PMID:1#1", "PMID:1#2"]
    assert [chunk.content for chunk in chunks] == ["abcde", "defgh", "XYZ"]
    assert all(chunk.doc_id == "PMID:1" for chunk in chunks)
```

- [ ] **Step 2: Run the test to verify failure**

Run:

```powershell
python -m pytest tests/test_chunking.py -v
```

Expected: import fails because `chunking.py` does not exist.

- [ ] **Step 3: Implement token-safe section chunking**

```python
# MedicalGraphRAG/src/medical_graphrag/data/chunking.py
from typing import Protocol, Sequence

from medical_graphrag.data.schemas import Chunk


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> Sequence[int]: ...
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str: ...


def chunk_sections(
    *,
    doc_id: str,
    title: str,
    sections: tuple[str, ...],
    source: str,
    tokenizer: Tokenizer,
    max_tokens: int,
    overlap: int,
) -> list[Chunk]:
    if max_tokens <= 0 or overlap < 0 or overlap >= max_tokens:
        raise ValueError("require max_tokens > overlap >= 0")
    chunks: list[Chunk] = []
    order = 0
    stride = max_tokens - overlap
    for section in sections:
        token_ids = list(tokenizer.encode(section, add_special_tokens=False))
        if not token_ids:
            continue
        start = 0
        while start < len(token_ids):
            window = token_ids[start : start + max_tokens]
            content = tokenizer.decode(window, skip_special_tokens=True).strip()
            if content:
                chunks.append(Chunk(f"{doc_id}#{order}", doc_id, order, title, content, source))
                order += 1
            if start + max_tokens >= len(token_ids):
                break
            start += stride
    if not chunks:
        raise ValueError(f"document {doc_id} produced no chunks")
    return chunks
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m pytest tests/test_chunking.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit chunking**

```powershell
git add MedicalGraphRAG/src/medical_graphrag/data/chunking.py MedicalGraphRAG/tests/test_chunking.py
git commit -m "feat: add document-safe chunking"
```

## Task 6: Assemble, audit, validate, and manifest the benchmark

**Files:**
- Create: `MedicalGraphRAG/configs/pubmedqa_hard_v1.json`
- Create: `MedicalGraphRAG/src/medical_graphrag/data/benchmark.py`
- Create: `MedicalGraphRAG/tests/test_benchmark.py`

- [ ] **Step 1: Add the fixed configuration**

```json
{
  "version": "pubmedqa_hard_v1",
  "seed": 20260803,
  "gold_document_count": 1000,
  "dev_query_count": 500,
  "test_query_count": 500,
  "distractor_count": 4000,
  "initial_shard_count": 40,
  "candidates_per_shard": 120,
  "tokenizer": "sentence-transformers/all-mpnet-base-v2",
  "max_tokens": 512,
  "overlap": 64,
  "audit_query_count": 20,
  "retrieval_text_mode": "abstract_only",
  "comparison_text_mode": "title_abstract"
}
```

- [ ] **Step 2: Write failing assembly tests**

```python
# MedicalGraphRAG/tests/test_benchmark.py
from pathlib import Path

from medical_graphrag.data.benchmark import assemble_records, validate_records
from medical_graphrag.data.medrag_pubmed import sample_distractors
from medical_graphrag.data.pubmedqa import load_pubmedqa


FIXTURES = Path(__file__).parent / "fixtures"


def test_fixture_benchmark_has_one_qrel_per_question() -> None:
    pubmedqa = load_pubmedqa(
        FIXTURES / "ori_pqal_small.json",
        FIXTURES / "test_ground_truth_small.json",
    )
    gold_titles = {record.question.lower() for record in pubmedqa}
    distractors, _ = sample_distractors(
        FIXTURES / "medrag_pubmed",
        seed=7,
        shard_count=2,
        per_shard=3,
        target_count=2,
        excluded_titles=gold_titles,
        excluded_content_hashes=set(),
    )
    questions, documents, qrels = assemble_records(pubmedqa, distractors)
    validate_records(questions, documents, qrels, gold_count=2, distractor_count=2)
    assert len(questions) == 2
    assert len(documents) == 4
    assert len(qrels) == 2
    assert {qrel.query_id for qrel in qrels} == {question.query_id for question in questions}
```

- [ ] **Step 3: Run the test to verify failure**

Run:

```powershell
python -m pytest tests/test_benchmark.py -v
```

Expected: import fails because `benchmark.py` does not exist.

- [ ] **Step 4: Implement record assembly and invariant validation**

```python
# MedicalGraphRAG/src/medical_graphrag/data/benchmark.py
from collections import Counter
from pathlib import Path

from medical_graphrag.data.io import sha256_file
from medical_graphrag.data.schemas import Document, PubMedQARecord, Qrel, Question


def assemble_records(
    pubmedqa: list[PubMedQARecord],
    distractors: list[Document],
) -> tuple[list[Question], list[Document], list[Qrel]]:
    questions = [
        Question(record.pmid, record.question, record.answer, record.long_answer, record.split)
        for record in pubmedqa
    ]
    gold_documents = [
        Document(
            doc_id=f"PMID:{record.pmid}",
            title=record.question,
            content="\n\n".join(record.contexts),
            source="pubmedqa",
            year=record.year,
        )
        for record in pubmedqa
    ]
    qrels = [Qrel(record.pmid, f"PMID:{record.pmid}", 1) for record in pubmedqa]
    return questions, gold_documents + distractors, qrels


def validate_records(
    questions: list[Question],
    documents: list[Document],
    qrels: list[Qrel],
    *,
    gold_count: int,
    distractor_count: int,
) -> None:
    query_ids = [item.query_id for item in questions]
    doc_ids = [item.doc_id for item in documents]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("duplicate query_id")
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("duplicate doc_id")
    if len(documents) != gold_count + distractor_count:
        raise ValueError("unexpected document count")
    qrel_counts = Counter(item.query_id for item in qrels)
    if set(qrel_counts) != set(query_ids) or any(value != 1 for value in qrel_counts.values()):
        raise ValueError("each query must have exactly one qrel")
    if any(item.doc_id not in set(doc_ids) for item in qrels):
        raise ValueError("qrel references missing document")


def artifact_hashes(output_dir: Path) -> dict[str, str]:
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    return {name: sha256_file(output_dir / name) for name in names}
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_benchmark.py -v
```

Expected: `1 passed`.

- [ ] **Step 6: Commit assembly logic**

```powershell
git add MedicalGraphRAG/configs/pubmedqa_hard_v1.json MedicalGraphRAG/src/medical_graphrag/data/benchmark.py MedicalGraphRAG/tests/test_benchmark.py
git commit -m "feat: assemble validated PubMedQA benchmark"
```

## Task 7: Add retrieval hard metrics

**Files:**
- Create: `MedicalGraphRAG/src/medical_graphrag/evaluation/retrieval.py`
- Create: `MedicalGraphRAG/tests/test_retrieval_metrics.py`

- [ ] **Step 1: Write exact metric tests**

```python
# MedicalGraphRAG/tests/test_retrieval_metrics.py
from medical_graphrag.evaluation.retrieval import evaluate_rankings


def test_single_gold_metrics_are_exact() -> None:
    qrels = {"q1": "d1", "q2": "d2"}
    rankings = {
        "q1": ["d1", "x", "y"],
        "q2": ["x", "d2", "y"],
    }
    result = evaluate_rankings(qrels, rankings, ks=(1, 2, 10))
    assert result["recall@1"] == 0.5
    assert result["recall@2"] == 1.0
    assert result["recall@10"] == 1.0
    assert result["mrr@10"] == 0.75
    assert round(result["ndcg@10"], 6) == round((1.0 + 1.0 / 1.584962500721156) / 2, 6)
```

- [ ] **Step 2: Run the test to verify failure**

Run:

```powershell
python -m pytest tests/test_retrieval_metrics.py -v
```

Expected: import fails because `retrieval.py` does not exist.

- [ ] **Step 3: Implement Recall, MRR, and binary nDCG**

```python
# MedicalGraphRAG/src/medical_graphrag/evaluation/retrieval.py
import math
from collections.abc import Mapping, Sequence


def evaluate_rankings(
    qrels: Mapping[str, str],
    rankings: Mapping[str, Sequence[str]],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    if not qrels:
        raise ValueError("qrels must not be empty")
    totals = {f"recall@{k}": 0.0 for k in ks}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for query_id, gold_doc_id in qrels.items():
        ranking = list(dict.fromkeys(rankings.get(query_id, ())))
        rank = ranking.index(gold_doc_id) + 1 if gold_doc_id in ranking else None
        for k in ks:
            totals[f"recall@{k}"] += float(rank is not None and rank <= k)
        reciprocal_ranks.append(1.0 / rank if rank is not None and rank <= 10 else 0.0)
        ndcgs.append(1.0 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0)
    count = len(qrels)
    result = {name: value / count for name, value in totals.items()}
    result["mrr@10"] = sum(reciprocal_ranks) / count
    result["ndcg@10"] = sum(ndcgs) / count
    return result
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m pytest tests/test_retrieval_metrics.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit metrics**

```powershell
git add MedicalGraphRAG/src/medical_graphrag/evaluation/retrieval.py MedicalGraphRAG/tests/test_retrieval_metrics.py
git commit -m "feat: add retrieval hard metrics"
```

## Task 8: Expose fetch, audit, and build commands

**Files:**
- Create: `MedicalGraphRAG/src/medical_graphrag/cli.py`
- Modify: `MedicalGraphRAG/src/medical_graphrag/data/benchmark.py`
- Modify: `MedicalGraphRAG/README.md`
- Create: `MedicalGraphRAG/data/raw/README.md`
- Create: `MedicalGraphRAG/data/processed/README.md`
- Create: `MedicalGraphRAG/tests/test_cli.py`

- [ ] **Step 1: Write a failing CLI help test**

```python
# MedicalGraphRAG/tests/test_cli.py
import subprocess
import sys


def test_cli_lists_pipeline_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "medical_graphrag.cli", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "fetch-pubmedqa" in result.stdout
    assert "audit" in result.stdout
    assert "build" in result.stdout
```

- [ ] **Step 2: Run the test to verify failure**

Run:

```powershell
python -m pytest tests/test_cli.py -v
```

Expected: command fails because `cli.py` does not exist.

- [ ] **Step 3: Implement the command parser and official downloads**

```python
# MedicalGraphRAG/src/medical_graphrag/cli.py
import argparse
import urllib.request
from pathlib import Path

from medical_graphrag.data.benchmark import audit_benchmark, build_benchmark


PQA_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
GROUND_TRUTH_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"


def fetch_pubmedqa(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, url in (("ori_pqal.json", PQA_URL), ("test_ground_truth.json", GROUND_TRUTH_URL)):
        destination = output_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medical-graphrag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch-pubmedqa")
    fetch.add_argument("--output-dir", type=Path, default=Path("data/raw/pubmedqa"))

    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--pubmedqa-dir", type=Path, required=True)
    audit.add_argument("--medrag-pubmed-dir", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--pubmedqa-dir", type=Path, required=True)
    build.add_argument("--medrag-pubmed-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "fetch-pubmedqa":
        fetch_pubmedqa(args.output_dir)
        return 0
    if args.command == "audit":
        audit_benchmark(args.config, args.pubmedqa_dir, args.medrag_pubmed_dir, args.output_dir)
        return 0
    if args.command == "build":
        build_benchmark(args.config, args.pubmedqa_dir, args.medrag_pubmed_dir, args.output_dir)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add tested `audit_benchmark()` and `build_benchmark()` orchestration**

First extend `tests/test_benchmark.py` with an orchestration test that uses the fixture files and `CharacterCodec` from the chunking test:

```python
import json

from medical_graphrag.data.benchmark import audit_benchmark


class CharacterCodec:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(value) for value in ids)


def test_fixture_audit_writes_passing_report(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "gold_document_count": 2,
                "dev_query_count": 1,
                "test_query_count": 1,
                "distractor_count": 2,
                "initial_shard_count": 2,
                "candidates_per_shard": 3,
                "tokenizer": "fixture-tokenizer",
                "max_tokens": 5,
                "overlap": 2,
                "audit_query_count": 1,
                "retrieval_text_mode": "abstract_only",
                "comparison_text_mode": "title_abstract"
            }
        ),
        encoding="utf-8",
    )
    report = audit_benchmark(
        config_path,
        FIXTURES,
        FIXTURES / "medrag_pubmed",
        tmp_path / "output",
        tokenizer=CharacterCodec(),
    )
    assert report["passed"] is True
    assert len(report["items"]) == 1
    assert (tmp_path / "output" / "audit_20.json").exists()
```

Run `python -m pytest tests/test_benchmark.py -v`; expected failure is that `audit_benchmark` is absent.

Then extend `benchmark.py`. Add these imports:

```python
import json
from typing import Any

from medical_graphrag.data.chunking import Tokenizer, chunk_sections
from medical_graphrag.data.io import write_json, write_jsonl, write_qrels
from medical_graphrag.data.medrag_pubmed import content_hash, normalize_text, sample_distractors
from medical_graphrag.data.pubmedqa import load_pubmedqa
from medical_graphrag.data.schemas import Chunk, record_dict
```

Append the following orchestration code:

```python
def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "seed", "gold_document_count", "dev_query_count", "test_query_count",
        "distractor_count", "initial_shard_count", "candidates_per_shard",
        "tokenizer", "max_tokens", "overlap", "audit_query_count",
        "retrieval_text_mode", "comparison_text_mode",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    if config["retrieval_text_mode"] != "abstract_only":
        raise ValueError("primary retrieval_text_mode must be abstract_only")
    if config["comparison_text_mode"] != "title_abstract":
        raise ValueError("comparison_text_mode must be title_abstract")
    return config


def _load_sources(
    config: dict[str, Any],
    pubmedqa_dir: Path,
    medrag_dir: Path,
) -> tuple[list[PubMedQARecord], list[Document], list[str]]:
    pubmedqa = load_pubmedqa(
        pubmedqa_dir / "ori_pqal.json",
        pubmedqa_dir / "test_ground_truth.json",
    )
    split_counts = Counter(record.split for record in pubmedqa)
    expected_splits = {
        "dev": int(config["dev_query_count"]),
        "test": int(config["test_query_count"]),
    }
    if len(pubmedqa) != int(config["gold_document_count"]) or split_counts != expected_splits:
        raise ValueError(
            f"unexpected PubMedQA counts: total={len(pubmedqa)}, splits={dict(split_counts)}"
        )
    excluded_titles = {normalize_text(record.question) for record in pubmedqa}
    excluded_hashes = {
        content_hash(record.question, "\n\n".join(record.contexts)) for record in pubmedqa
    }
    distractors, shard_names = sample_distractors(
        medrag_dir,
        seed=int(config["seed"]),
        shard_count=int(config["initial_shard_count"]),
        per_shard=int(config["candidates_per_shard"]),
        target_count=int(config["distractor_count"]),
        excluded_titles=excluded_titles,
        excluded_content_hashes=excluded_hashes,
    )
    return pubmedqa, distractors, shard_names


def _build_chunks(
    pubmedqa: list[PubMedQARecord],
    documents: list[Document],
    tokenizer: Tokenizer,
    config: dict[str, Any],
) -> list[Chunk]:
    gold_sections = {f"PMID:{record.pmid}": record.contexts for record in pubmedqa}
    chunks: list[Chunk] = []
    for document in documents:
        sections = gold_sections.get(document.doc_id, (document.content,))
        chunks.extend(
            chunk_sections(
                doc_id=document.doc_id,
                title=document.title,
                sections=sections,
                source=document.source,
                tokenizer=tokenizer,
                max_tokens=int(config["max_tokens"]),
                overlap=int(config["overlap"]),
            )
        )
    return chunks


def _resolve_tokenizer(config: dict[str, Any], tokenizer: Tokenizer | None) -> Tokenizer:
    if tokenizer is not None:
        return tokenizer
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(config["tokenizer"]))


def audit_benchmark(
    config_path: Path,
    pubmedqa_dir: Path,
    medrag_dir: Path,
    output_dir: Path,
    *,
    tokenizer: Tokenizer | None = None,
) -> dict[str, object]:
    config = _load_config(config_path)
    pubmedqa, distractors, shard_names = _load_sources(config, pubmedqa_dir, medrag_dir)
    count = int(config["audit_query_count"])
    audit_records = [record for record in pubmedqa if record.split == "test"][:count]
    if len(audit_records) != count:
        raise ValueError(f"only {len(audit_records)} test records available for {count}-item audit")

    questions, gold_documents, qrels = assemble_records(audit_records, [])
    validate_records(questions, gold_documents, qrels, gold_count=count, distractor_count=0)
    chunks = _build_chunks(
        audit_records,
        gold_documents,
        _resolve_tokenizer(config, tokenizer),
        config,
    )
    gold_by_id = {document.doc_id: document for document in gold_documents}
    chunk_ids = {document.doc_id: [] for document in gold_documents}
    for chunk in chunks:
        chunk_ids[chunk.doc_id].append(chunk.chunk_id)

    distractor_hashes = {content_hash(item.title, item.content) for item in distractors}
    items: list[dict[str, object]] = []
    for record in audit_records:
        gold_doc_id = f"PMID:{record.pmid}"
        duplicate = content_hash(record.question, "\n\n".join(record.contexts)) in distractor_hashes
        qrel_count = sum(item.query_id == record.pmid for item in qrels)
        item_passed = qrel_count == 1 and bool(chunk_ids[gold_doc_id]) and not duplicate
        items.append(
            {
                "query_id": record.pmid,
                "gold_doc_id": gold_doc_id,
                "split": record.split,
                "answer": record.answer,
                "context_count": len(record.contexts),
                "context_nonempty": all(bool(value.strip()) for value in record.contexts),
                "qrel_count": qrel_count,
                "chunk_ids": chunk_ids[gold_doc_id],
                "title_leak_risk": normalize_text(record.question)
                == normalize_text(gold_by_id[gold_doc_id].title),
                "duplicate_with_distractor": duplicate,
                "passed": item_passed,
            }
        )
    report: dict[str, object] = {
        "passed": len(items) == count and all(bool(item["passed"]) for item in items),
        "audit_query_count": count,
        "selected_shards": shard_names,
        "items": items,
    }
    write_json(output_dir / "audit_20.json", report)
    if not report["passed"]:
        raise ValueError("20-question audit failed; inspect audit_20.json")
    return report


def build_benchmark(
    config_path: Path,
    pubmedqa_dir: Path,
    medrag_dir: Path,
    output_dir: Path,
    *,
    tokenizer: Tokenizer | None = None,
) -> dict[str, object]:
    audit_path = output_dir / "audit_20.json"
    if not audit_path.exists():
        raise ValueError("run the audit command before build")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise ValueError("existing audit did not pass")

    config = _load_config(config_path)
    pubmedqa, distractors, shard_names = _load_sources(config, pubmedqa_dir, medrag_dir)
    questions, documents, qrels = assemble_records(pubmedqa, distractors)
    validate_records(
        questions,
        documents,
        qrels,
        gold_count=int(config["gold_document_count"]),
        distractor_count=int(config["distractor_count"]),
    )
    chunks = _build_chunks(pubmedqa, documents, _resolve_tokenizer(config, tokenizer), config)
    write_jsonl(output_dir / "questions.jsonl", (record_dict(item) for item in questions))
    write_jsonl(output_dir / "documents.jsonl", (record_dict(item) for item in documents))
    write_jsonl(output_dir / "chunks.jsonl", (record_dict(item) for item in chunks))
    write_qrels(
        output_dir / "qrels.tsv",
        ((item.query_id, item.doc_id, item.relevance) for item in qrels),
    )
    manifest: dict[str, object] = {
        "dataset": "pubmedqa_hard_v1",
        "config": config,
        "source_hashes": {
            "ori_pqal.json": sha256_file(pubmedqa_dir / "ori_pqal.json"),
            "test_ground_truth.json": sha256_file(pubmedqa_dir / "test_ground_truth.json"),
        },
        "selected_shards": shard_names,
        "counts": {
            "questions": len(questions),
            "documents": len(documents),
            "chunks": len(chunks),
            "qrels": len(qrels),
        },
        "artifact_hashes": artifact_hashes(output_dir),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
```

Run `python -m pytest tests/test_benchmark.py -v`; expected: both assembly and orchestration tests pass.

- [ ] **Step 5: Run CLI and complete-suite tests**

Run:

```powershell
python -m pytest -v
python -m medical_graphrag.cli --help
```

Expected: all tests pass; help lists `fetch-pubmedqa`, `audit`, and `build`.

- [ ] **Step 6: Document source and generated-data policies**

`data/raw/README.md` must list the two official PubMedQA URLs and state that raw files are ignored. `data/processed/README.md` must state that large generated files are ignored and reconstructed from config plus manifest. Expand the project README with exact fetch, audit, build, and test commands.

- [ ] **Step 7: Commit the runnable pipeline**

```powershell
git add MedicalGraphRAG/src/medical_graphrag/cli.py MedicalGraphRAG/src/medical_graphrag/data/benchmark.py MedicalGraphRAG/tests/test_cli.py MedicalGraphRAG/README.md MedicalGraphRAG/data/raw/README.md MedicalGraphRAG/data/processed/README.md
git commit -m "feat: expose benchmark data pipeline"
```

## Task 9: Run the real 20-question gate

**Files:**
- Generate: `MedicalGraphRAG/data/raw/pubmedqa/ori_pqal.json` (ignored)
- Generate: `MedicalGraphRAG/data/raw/pubmedqa/test_ground_truth.json` (ignored)
- Generate: `MedicalGraphRAG/data/processed/pubmedqa_hard_v1/audit_20.json`
- Modify: `MedicalGraphRAG/README.md`

- [ ] **Step 1: Fetch official PubMedQA files**

Run from `MedicalGraphRAG/`:

```powershell
python -m medical_graphrag.cli fetch-pubmedqa --output-dir data/raw/pubmedqa
```

Expected: two JSON files exist and parse successfully.

- [ ] **Step 2: Run the real audit against local MedRAG PubMed**

Run:

```powershell
python -m medical_graphrag.cli audit `
  --config configs/pubmedqa_hard_v1.json `
  --pubmedqa-dir data/raw/pubmedqa `
  --medrag-pubmed-dir ../MedRAG/corpus/pubmed/chunk `
  --output-dir data/processed/pubmedqa_hard_v1
```

Expected: `audit_20.json` contains `"passed": true` and exactly 20 items. If it fails, stop and record the actual data defect; do not run `build`.

- [ ] **Step 3: Verify audit invariants independently**

Run:

```powershell
python -c "import json,pathlib; p=pathlib.Path('data/processed/pubmedqa_hard_v1/audit_20.json'); d=json.loads(p.read_text(encoding='utf-8')); assert d['passed'] is True; assert len(d['items']) == 20; assert all(x['query_id'] == x['gold_doc_id'].removeprefix('PMID:') for x in d['items']); print('AUDIT_20_OK')"
```

Expected: `AUDIT_20_OK`.

- [ ] **Step 4: Record the exact audit command in README**

Add the command, source hashes, elapsed time, selected shard count, and audit result. Do not record Recall@k because no retriever has run yet.

- [ ] **Step 5: Commit only the small audit evidence and documentation**

```powershell
git add MedicalGraphRAG/data/processed/pubmedqa_hard_v1/audit_20.json MedicalGraphRAG/README.md
git commit -m "data: verify PubMedQA twenty-question audit"
```

## Final verification

- [ ] Run from `MedicalGraphRAG/`:

```powershell
python -m pytest -v
python -m medical_graphrag.cli --help
```

Expected: all tests pass and all three pipeline commands are listed.

- [ ] Run from the repository root:

```powershell
git status --short
git log --oneline -10
```

Expected: no generated large data is staged; the pre-existing user change to `STUDY_PROGRESS.md` remains untouched.

- [ ] Confirm the milestone boundary:

This plan ends after the real 20-question audit. It does not claim the full 5,000-document dataset is built and does not report retrieval results. The next plan begins with the full deterministic build, then BM25 and retrieval-metric execution.
