import json
from pathlib import Path

import pytest

from medical_graphrag.data.pubmedqa import load_pubmedqa


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_pubmedqa_assigns_official_test_split() -> None:
    records = load_pubmedqa(
        FIXTURES / "ori_pqal_small.json",
        FIXTURES / "test_ground_truth_small.json",
    )
    assert [record.pmid for record in records] == ["100", "200"]
    assert [record.split for record in records] == ["dev", "test"]
    assert records[0].contexts == ("Background.", "Results show benefit.")
    assert records[1].answer == "no"


def test_load_pubmedqa_rejects_ground_truth_mismatch(tmp_path: Path) -> None:
    ground_truth = tmp_path / "ground_truth.json"
    ground_truth.write_text(json.dumps({"200": "yes"}), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        load_pubmedqa(FIXTURES / "ori_pqal_small.json", ground_truth)
