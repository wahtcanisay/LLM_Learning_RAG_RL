"""教师子进程的文件往返与错误传播；使用本地小模型，不下载权重。"""
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM


def process_scorer():
    name = "MedSearch.distillation.teacher_process"
    assert importlib.util.find_spec(name) is not None, "Teacher subprocess entry is missing"
    return importlib.import_module(name).score_in_subprocess


def failure_rows():
    return [
        {"original_index": 2, "group_id": "g", "trajectory_id": "trace-2",
         "prompt_ids": [10, 11], "response_ids": [20, 21, 40, 30],
         "response_mask": [1, 1, 0, 1], "references": ["must not reach teacher"]},
        {"original_index": 0, "group_id": "g", "trajectory_id": "trace-0",
         "prompt_ids": [12], "response_ids": [22, 31], "response_mask": [1, 1]},
    ]


def test_empty_failure_batch_never_starts_process(tmp_path):
    score = process_scorer()
    with patch("subprocess.run", side_effect=AssertionError("Teacher must be skipped")):
        assert score([], "not-a-checkpoint", tmp_path / "unused") == []
    assert not (tmp_path / "unused").exists()


def test_cpu_process_roundtrip_matches_existing_scorer(tmp_path):
    score = process_scorer()
    torch.manual_seed(42)
    model_dir = tmp_path / "teacher"
    Qwen2ForCausalLM(Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
        bos_token_id=1, eos_token_id=2,
    )).save_pretrained(model_dir)
    failures = failure_rows()
    result = score(failures, model_dir, tmp_path / "jobs", device="cpu")

    from MedSearch.distillation.teacher import load_teacher
    from MedSearch.rl.distillation_hook import score_failed_batch
    expected = score_failed_batch(load_teacher(str(model_dir), "cpu"), failures)
    for actual, reference in zip(result, expected, strict=True):
        assert actual["original_index"] == reference["original_index"]
        assert actual["trajectory_id"] == reference["trajectory_id"]
        for field in ("teacher_logprobs", "opd_mask"):
            torch.testing.assert_close(actual[field], reference[field])
            assert actual[field].device.type == "cpu" and actual[field].dtype == torch.float32
            assert not actual[field].requires_grad
    job = next((tmp_path / "jobs").iterdir())
    requests = [json.loads(line) for line in (job / "request.jsonl").read_text().splitlines()]
    assert all(set(row) == {"original_index", "group_id", "trajectory_id", "prompt_ids", "response_ids", "response_mask"}
               for row in requests)
    report = json.loads((job / "result.json").read_text())
    assert report["rows"] == 2 and report["scored_tokens"] == 5
    assert report["frozen"] and report["eval"] and report["dtype"] == "torch.bfloat16"
    assert not Path(f"/proc/{report['pid']}").exists()


def test_process_failure_raises_and_does_not_publish_scores(tmp_path):
    score = process_scorer()
    with pytest.raises(subprocess.CalledProcessError):
        score(failure_rows(), tmp_path / "missing-teacher", tmp_path / "failed-job", device="cpu")
    job = next((tmp_path / "failed-job").iterdir())
    assert (job / "request.jsonl").is_file() and (job / "worker.log").is_file()
    assert not (job / "result.pt").exists()
