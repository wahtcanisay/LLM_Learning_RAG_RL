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
