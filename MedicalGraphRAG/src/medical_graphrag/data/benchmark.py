from collections import Counter
import json
from pathlib import Path
from typing import Any

from medical_graphrag.data.chunking import Tokenizer, chunk_sections
from medical_graphrag.data.io import sha256_file, write_json, write_jsonl, write_qrels
from medical_graphrag.data.medrag_pubmed import content_hash, normalize_text, sample_distractors
from medical_graphrag.data.pubmedqa import load_pubmedqa
from medical_graphrag.data.schemas import (
    Chunk,
    Document,
    PubMedQARecord,
    Qrel,
    Question,
    record_dict,
)


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
    if len(questions) != gold_count:
        raise ValueError("unexpected question count")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("duplicate query_id")
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("duplicate doc_id")
    if len(documents) != gold_count + distractor_count:
        raise ValueError("unexpected document count")
    source_counts = Counter(item.source for item in documents)
    expected_sources = Counter({"pubmedqa": gold_count, "medrag_pubmed": distractor_count})
    if source_counts != expected_sources:
        raise ValueError(f"unexpected document source counts: {dict(source_counts)}")
    qrel_counts = Counter(item.query_id for item in qrels)
    if set(qrel_counts) != set(query_ids) or any(value != 1 for value in qrel_counts.values()):
        raise ValueError("each query must have exactly one qrel")
    document_ids = set(doc_ids)
    if any(item.doc_id not in document_ids for item in qrels):
        raise ValueError("qrel references missing document")


def artifact_hashes(output_dir: Path) -> dict[str, str]:
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    return {name: sha256_file(output_dir / name) for name in names}


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "seed",
        "gold_document_count",
        "dev_query_count",
        "test_query_count",
        "distractor_count",
        "initial_shard_count",
        "candidates_per_shard",
        "tokenizer",
        "max_tokens",
        "overlap",
        "audit_query_count",
        "retrieval_text_mode",
        "comparison_text_mode",
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
    expected_splits = Counter(
        {
            "dev": int(config["dev_query_count"]),
            "test": int(config["test_query_count"]),
        }
    )
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

    return AutoTokenizer.from_pretrained(str(config["tokenizer"]), local_files_only=True)


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
    gold_chunks: dict[str, list[dict[str, object]]] = {
        document.doc_id: [] for document in gold_documents
    }
    for chunk in chunks:
        chunk_ids[chunk.doc_id].append(chunk.chunk_id)
        gold_chunks[chunk.doc_id].append(
            {"chunk_id": chunk.chunk_id, "order": chunk.order, "content": chunk.content}
        )

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
                "question": record.question,
                "answer": record.answer,
                "long_answer": record.long_answer,
                "context_count": len(record.contexts),
                "contexts": list(record.contexts),
                "context_nonempty": all(bool(value.strip()) for value in record.contexts),
                "qrel_count": qrel_count,
                "chunk_ids": chunk_ids[gold_doc_id],
                "gold_chunks": gold_chunks[gold_doc_id],
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
