import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from medical_graphrag.data.benchmark import (
    _resolve_tokenizer,
    assemble_records,
    audit_benchmark,
    build_benchmark,
    validate_records,
)
from medical_graphrag.data.medrag_pubmed import normalize_text, sample_distractors
from medical_graphrag.data.pubmedqa import load_pubmedqa
from medical_graphrag.data.schemas import Qrel


FIXTURES = Path(__file__).parent / "fixtures"


class CharacterCodec:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(value) for value in ids)


def _fixture_config(tmp_path: Path) -> Path:
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
                "comparison_text_mode": "title_abstract",
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _fixture_pubmedqa(tmp_path: Path) -> Path:
    pubmedqa_dir = tmp_path / "pubmedqa"
    pubmedqa_dir.mkdir()
    shutil.copyfile(FIXTURES / "ori_pqal_small.json", pubmedqa_dir / "ori_pqal.json")
    shutil.copyfile(
        FIXTURES / "test_ground_truth_small.json",
        pubmedqa_dir / "test_ground_truth.json",
    )
    return pubmedqa_dir


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


def test_fixture_audit_and_build_write_valid_artifacts(tmp_path: Path) -> None:
    config_path = _fixture_config(tmp_path)
    pubmedqa_dir = _fixture_pubmedqa(tmp_path)
    output_dir = tmp_path / "output"
    tokenizer = CharacterCodec()

    report = audit_benchmark(
        config_path,
        pubmedqa_dir,
        FIXTURES / "medrag_pubmed",
        output_dir,
        tokenizer=tokenizer,
    )
    manifest = build_benchmark(
        config_path,
        pubmedqa_dir,
        FIXTURES / "medrag_pubmed",
        output_dir,
        tokenizer=tokenizer,
    )

    assert report["passed"] is True
    assert len(report["items"]) == 1
    assert manifest["counts"]["questions"] == 2
    assert manifest["counts"]["documents"] == 4
    assert manifest["counts"]["qrels"] == 2
    assert set(manifest["artifact_hashes"]) == {
        "questions.jsonl",
        "documents.jsonl",
        "chunks.jsonl",
        "qrels.tsv",
    }
    assert (output_dir / "audit_20.json").exists()
    assert (output_dir / "manifest.json").exists()


def test_resolve_tokenizer_never_checks_network(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **kwargs: object) -> object:
            calls.append((name, kwargs))
            return object()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))
    _resolve_tokenizer({"tokenizer": "fixture-tokenizer"}, None)
    assert calls == [("fixture-tokenizer", {"local_files_only": True})]
