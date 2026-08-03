from pathlib import Path

import pytest

from medical_graphrag.data.benchmark import assemble_records, validate_records
from medical_graphrag.data.medrag_pubmed import normalize_text, sample_distractors
from medical_graphrag.data.pubmedqa import load_pubmedqa
from medical_graphrag.data.schemas import Qrel


FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_records():
    pubmedqa = load_pubmedqa(
        FIXTURES / "ori_pqal_small.json",
        FIXTURES / "test_ground_truth_small.json",
    )
    gold_titles = {normalize_text(record.question) for record in pubmedqa}
    distractors, _ = sample_distractors(
        FIXTURES / "medrag_pubmed",
        seed=7,
        shard_count=2,
        per_shard=3,
        target_count=2,
        excluded_titles=gold_titles,
        excluded_content_hashes=set(),
    )
    return pubmedqa, distractors


def test_fixture_benchmark_has_one_qrel_per_question() -> None:
    pubmedqa, distractors = _fixture_records()
    questions, documents, qrels = assemble_records(pubmedqa, distractors)
    validate_records(questions, documents, qrels, gold_count=2, distractor_count=2)
    assert len(questions) == 2
    assert len(documents) == 4
    assert len(qrels) == 2
    assert {qrel.query_id for qrel in qrels} == {question.query_id for question in questions}


def test_validate_records_rejects_duplicate_qrel() -> None:
    pubmedqa, distractors = _fixture_records()
    questions, documents, qrels = assemble_records(pubmedqa, distractors)
    with pytest.raises(ValueError, match="exactly one qrel"):
        validate_records(
            questions,
            documents,
            qrels + [Qrel(qrels[0].query_id, qrels[0].doc_id)],
            gold_count=2,
            distractor_count=2,
        )
