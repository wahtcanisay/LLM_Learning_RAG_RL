import json
import sys
from pathlib import Path

import pytest
import yaml
from transformers import AutoTokenizer


MEDSEARCH_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MEDSEARCH_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from MedSearch.sft.train import build_dataloader, load_config


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct",
        local_files_only=True,
    )
    tokenizer.padding_side = "right"
    return tokenizer


def test_load_config_reads_the_minimal_yaml_schema(tmp_path):
    expected = {
        "stage": "stage_a",
        "data_path": "data.jsonl",
        "model_path": "model",
        "max_length": 1024,
        "batch_size": 2,
        "shuffle": False,
        "num_workers": 0,
        "use_qlora": True,
        "resume_from_checkpoint": None,
    }
    config_path = tmp_path / "stage_a.yaml"
    config_path.write_text(yaml.safe_dump(expected), encoding="utf-8")

    assert load_config(config_path) == expected


@pytest.mark.parametrize("stage", ["stage_a", "stage_b"])
def test_build_dataloader_connects_config_to_the_existing_data_pipeline(
    tmp_path, tokenizer, stage
):
    if stage == "stage_a":
        samples = [
            {
                "messages": [
                    {"role": "user", "content": "short question"},
                    {"role": "assistant", "content": "short answer"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "a somewhat longer question"},
                    {"role": "assistant", "content": "a somewhat longer answer"},
                ]
            },
        ]
        data_path = tmp_path / "samples.jsonl"
        data_path.write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples),
            encoding="utf-8",
        )
        max_length = 1024
    else:
        samples = [
            {
                "question": "short question?",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "trajectory": [
                    {"type": "answer", "thinking": "reason", "answer": "A"}
                ],
            },
            {
                "question": "a somewhat longer question?",
                "options": {"A": "aa", "B": "bb", "C": "cc", "D": "dd"},
                "trajectory": [
                    {
                        "type": "answer",
                        "thinking": "a somewhat longer reason",
                        "answer": "B",
                    }
                ],
            },
        ]
        data_path = tmp_path / "samples.json"
        data_path.write_text(json.dumps(samples), encoding="utf-8")
        max_length = 4096

    config = {
        "stage": stage,
        "data_path": str(data_path),
        "model_path": "unused-in-this-test",
        "max_length": max_length,
        "batch_size": 2,
        "shuffle": False,
        "num_workers": 0,
    }

    dataloader = build_dataloader(config, tokenizer, project_root=PROJECT_ROOT)
    batch = next(iter(dataloader))
    assert len(dataloader.dataset) == 2
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] <= max_length


@pytest.mark.parametrize("stage", ["stage_a", "stage_b"])
def test_checked_in_configs_share_runtime_control_schema(stage):
    config = load_config(MEDSEARCH_ROOT / "configs" / f"{stage}.yaml")
    assert isinstance(config["use_qlora"], bool)
    checkpoint = config["resume_from_checkpoint"]
    assert checkpoint is None or isinstance(checkpoint, str)
