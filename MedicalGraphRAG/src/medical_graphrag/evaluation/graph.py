"""LinearGraphRetriever evaluation.

Chunk unit: graph search emits chunk-level rankings exactly like BM25/Dense, so
the evaluation reuses the same folding (max chunk score per document), the same
``evaluate_rankings`` metrics and the same split discipline.

Document unit: graph search already emits document-level hits
(``{doc_id, rank, score}``), so evaluation skips chunk→doc collapse (P0-5).

``write_graph_pair_cases`` produces the Graph-EP vs Graph-Sim paired comparison
required by the review (P1-5).
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medical_graphrag.data.io import sha256_file, write_json, write_jsonl
from medical_graphrag.evaluation.retrieval import (
    evaluate_rankings,
    first_gold_rank,
    read_jsonl,
    read_qrels,
    validate_hit_rows,
)
from medical_graphrag.retrieval.bm25 import collapse_chunk_hits, validate_frozen_dataset


def _validate_run_context(
    run_context: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    rankings_path: Path,
    dataset_manifest_sha256: str,
    dataset_artifact_hashes: dict[str, str],
    *,
    retrieval_unit: str,
) -> dict[str, Any]:
    try:
        index_report = run_context["index"]
        search_report = run_context["search"]
    except KeyError as error:
        raise ValueError(f"missing run report: {error.args[0]}") from error
    if search_report.get("graph_sha256") != index_report.get("graph_sha256"):
        raise ValueError("search report does not match index report")
    # P0-6: graph_build_report_sha256 must be the graph build report FILE hash,
    # matching the supplied index-report file hash — never graph_sha256.
    if search_report.get("graph_build_report_sha256") != run_context.get(
        "index_report_sha256"
    ):
        raise ValueError("search report does not match the supplied index report file")
    for report in (index_report, search_report):
        if report.get("dataset_manifest_sha256") != dataset_manifest_sha256:
            raise ValueError("run report does not match frozen dataset manifest")
    if index_report.get("dataset_artifact_hashes") != dataset_artifact_hashes:
        raise ValueError("index report does not match frozen dataset artifacts")
    if search_report.get("rankings_sha256") != sha256_file(rankings_path):
        raise ValueError("search report rankings SHA-256 mismatch")
    if search_report.get("questions_sha256") != dataset_artifact_hashes.get(
        "questions.jsonl"
    ):
        raise ValueError("search report questions SHA-256 mismatch")
    if str(search_report.get("retrieval_unit", "chunk")) != retrieval_unit:
        raise ValueError("search report retrieval unit does not match expected unit")
    requested_top_k = int(search_report["requested_top_k"])
    if requested_top_k <= 0:
        raise ValueError("requested_top_k must be positive")
    counts = [len(row["hits"]) for row in raw_rows]
    histogram = Counter(counts)
    actual_summary = {
        "query_count": len(raw_rows),
        "requested_top_k": requested_top_k,
        "min_hits": min(counts),
        "max_hits": max(counts),
        "short_ranking_count": sum(count < requested_top_k for count in counts),
        "hit_count_histogram": {
            str(count): frequency for count, frequency in sorted(histogram.items())
        },
    }
    reported_summary = {key: search_report.get(key) for key in actual_summary}
    if actual_summary != reported_summary:
        raise ValueError("search report hit summary mismatch")
    validate_hit_rows(raw_rows, retrieval_unit=retrieval_unit)
    return search_report


def evaluate_graph_run(
    dataset_dir: Path,
    rankings_path: Path,
    output_dir: Path,
    *,
    min_unique_docs: int = 0,
    run_context: dict[str, Any],
) -> dict[str, Any]:
    dataset_manifest = validate_frozen_dataset(dataset_dir)
    dataset_manifest_sha256 = sha256_file(dataset_dir / "manifest.json")
    questions = {
        str(row["query_id"]): row for row in read_jsonl(dataset_dir / "questions.jsonl")
    }
    documents = {
        str(row["doc_id"]): row for row in read_jsonl(dataset_dir / "documents.jsonl")
    }
    chunks = read_jsonl(dataset_dir / "chunks.jsonl")
    metadata = {
        str(row["chunk_id"]): {
            "chunk_id": str(row["chunk_id"]),
            "doc_id": str(row["doc_id"]),
            "order": int(row["order"]),
            "title": str(row["title"]),
            "source": str(row["source"]),
        }
        for row in chunks
    }
    raw_rows = read_jsonl(rankings_path)
    if len(raw_rows) != len(questions) or {
        str(row["query_id"]) for row in raw_rows
    } != set(questions):
        raise ValueError("ranking query set does not match questions")
    retrieval_unit = str(run_context["index"].get("retrieval_unit", "chunk"))
    search_report = _validate_run_context(
        run_context,
        raw_rows,
        rankings_path,
        dataset_manifest_sha256,
        dataset_manifest["artifact_hashes"],
        retrieval_unit=retrieval_unit,
    )

    qrels = read_qrels(dataset_dir / "qrels.tsv")
    if set(qrels) != set(questions) or any(
        doc_id not in documents for doc_ids in qrels.values() for doc_id in doc_ids
    ):
        raise ValueError("qrels do not resolve an existing document for every question")

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
            ranking = collapse_chunk_hits(
                row["hits"], metadata, min_unique_docs=min_unique_docs
            )
        collapsed[query_id] = [str(item["doc_id"]) for item in ranking]
        detailed[query_id] = [
            {**item, "title": documents[str(item["doc_id"])]["title"]}
            for item in ranking
        ]

    metrics: dict[str, Any] = {}
    for split in ("dev", "test"):
        ids = [
            query_id for query_id, row in questions.items() if row["split"] == split
        ]
        if not ids:
            continue
        split_qrels = {query_id: qrels[query_id] for query_id in ids}
        split_rankings = {query_id: collapsed[query_id] for query_id in ids}
        metrics[split] = {
            "sample_count": len(ids),
            **evaluate_rankings(split_qrels, split_rankings, ks=(1, 5, 10)),
        }

    test_ids = [
        query_id for query_id, row in questions.items() if row["split"] == "test"
    ]
    gold_ranks = {
        query_id: first_gold_rank(collapsed[query_id], qrels[query_id])
        for query_id in test_ids
    }
    success_all = [
        query_id
        for query_id in test_ids
        if gold_ranks[query_id] is not None and gold_ranks[query_id] <= 10
    ]
    failure_all = [
        query_id
        for query_id in test_ids
        if gold_ranks[query_id] is None or gold_ranks[query_id] > 10
    ]
    cases: dict[str, Any] = {}
    for label, ids in (("success", success_all[:5]), ("failure", failure_all[:5])):
        cases[label] = [
            {
                "query_id": query_id,
                "question": questions[query_id]["question"],
                "gold_doc_id": qrels[query_id][0],
                "gold_title": documents[qrels[query_id][0]]["title"],
                "gold_rank": gold_ranks[query_id],
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
        "split_counts": {
            name: value["sample_count"] for name, value in metrics.items()
        },
        "top_k": int(search_report["requested_top_k"]),
        "retrieval_unit": retrieval_unit,
        "aggregation": "none" if retrieval_unit == "document" else "max_chunk_score",
        "graph": {
            "ner_model": search_report["ner_model"],
            "embedding_model": search_report["embedding_model"],
            "config": search_report["config"],
            "passage_edge_mode": run_context["index"].get("passage_edge_mode", "none"),
            "adjacent_passage_edges": run_context["index"].get("adjacent_passage_edges", False),
        },
        "text_mode": "abstract_only",
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(output_dir / "cases.json", cases)
    return metrics


def write_graph_pair_cases(
    dataset_dir: Path,
    ep_rankings: Path,
    sim_rankings: Path,
    output: Path,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Per-question Graph-EP vs Graph-Sim comparison (P1-5).

    Both rankings must be document-unit raw rankings over the same questions.
    For every test question, records query_id, gold doc ids, both gold ranks,
    rank delta, improvement/degradation/no-change classification, the Sim docs
    that entered the top-k relative to EP (new similarity neighbours), and a
    short text summary. Writes a JSONL; returns summary counts.
    """
    documents = {
        str(row["doc_id"]): row for row in read_jsonl(dataset_dir / "documents.jsonl")
    }
    questions = {
        str(row["query_id"]): row for row in read_jsonl(dataset_dir / "questions.jsonl")
    }
    qrels = read_qrels(dataset_dir / "qrels.tsv")

    def _doc_ranking(rows_path: Path) -> dict[str, list[str]]:
        rows = read_jsonl(rows_path)
        return {
            str(row["query_id"]): [str(hit["doc_id"]) for hit in row["hits"][:top_k]]
            for row in rows
        }

    ep = _doc_ranking(ep_rankings)
    sim = _doc_ranking(sim_rankings)
    if set(ep) != set(sim):
        raise ValueError("EP and Sim rankings must cover the same query set")
    missing = set(questions) - set(ep)
    if missing:
        raise ValueError(f"rankings missing questions: {sorted(missing)}")

    rows_out: list[dict[str, Any]] = []
    counts = {"improvement": 0, "degradation": 0, "no_change": 0}
    for query_id in sorted(ep):
        row = questions[query_id]
        if row["split"] != "test":
            continue
        gold = qrels[query_id]
        ep_rank = first_gold_rank(ep[query_id], gold)
        sim_rank = first_gold_rank(sim[query_id], gold)
        if sim_rank is not None and (ep_rank is None or sim_rank < ep_rank):
            label = "improvement"
        elif ep_rank is not None and (sim_rank is None or sim_rank > ep_rank):
            label = "degradation"
        else:
            label = "no_change"
        counts[label] += 1
        ep_set = set(ep[query_id])
        sim_set = set(sim[query_id])
        new_neighbours = [
            {
                "doc_id": doc_id,
                "title": documents[doc_id]["title"],
                "sim_rank": rank,
            }
            for rank, doc_id in enumerate(sim[query_id], start=1)
            if doc_id not in ep_set
        ]
        rows_out.append({
            "query_id": query_id,
            "question": row["question"],
            "gold_doc_ids": gold,
            "gold_rank_ep": ep_rank,
            "gold_rank_sim": sim_rank,
            "rank_delta": (sim_rank - ep_rank) if (ep_rank is not None and sim_rank is not None) else None,
            "label": label,
            "new_similarity_neighbours": new_neighbours[:10],
            "top_docs_ep": ep[query_id][:10],
            "top_docs_sim": sim[query_id][:10],
            "reason": _pair_reason(ep[query_id], sim[query_id], gold),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows_out)
    summary = {
        "total_test_questions": len(rows_out),
        "improvement": counts["improvement"],
        "degradation": counts["degradation"],
        "no_change": counts["no_change"],
        "output_sha256": sha256_file(output),
    }
    return summary


def _pair_reason(ep_rank: list[str], sim_rank: list[str], gold: list[str]) -> str:
    ep_pos = {doc: rank for rank, doc in enumerate(ep_rank, start=1)}
    sim_pos = {doc: rank for rank, doc in enumerate(sim_rank, start=1)}
    gold = set(gold)
    moved_up = [doc for doc in sim_pos if doc not in gold and sim_pos[doc] < ep_pos.get(doc, 10**6)]
    moved_down = [doc for doc in ep_pos if doc not in gold and ep_pos[doc] < sim_pos.get(doc, 10**6)]
    reason = []
    if moved_up:
        reason.append(f"similarity raised {len(moved_up)} non-gold docs into top-k: {moved_up[:5]}")
    if moved_down:
        reason.append(f"similarity pushed out {len(moved_down)} EP top-k docs: {moved_down[:5]}")
    if not reason:
        reason.append("no top-k membership change outside the gold docs")
    return "; ".join(reason)
