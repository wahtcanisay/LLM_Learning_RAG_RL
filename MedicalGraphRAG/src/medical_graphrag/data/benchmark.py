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
    document_ids = set(doc_ids)
    if any(item.doc_id not in document_ids for item in qrels):
        raise ValueError("qrel references missing document")


def artifact_hashes(output_dir: Path) -> dict[str, str]:
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    return {name: sha256_file(output_dir / name) for name in names}
