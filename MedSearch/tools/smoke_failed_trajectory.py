"""验收真实 7B 的分段教师评分与原轨迹回填。"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from MedSearch.distillation.teacher import load_teacher
from MedSearch.rl.distillation_hook import score_failed_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("output_json")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prompt = encode("<|im_start|>user\nWhat supports aspirin after myocardial infarction?"
                    "<|im_end|>\n<|im_start|>assistant\n")
    first = encode("<search>aspirin secondary prevention myocardial infarction</search>")
    observation = encode("<information>Smoke fixture: aspirin is used in secondary prevention.</information>")
    last = encode("<answer>Aspirin supports secondary prevention.</answer>")
    response = first + observation + last
    mask = [1] * len(first) + [0] * len(observation) + [1] * len(last)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    teacher = load_teacher(args.checkpoint_dir, "cuda")
    result = score_failed_trajectory(teacher, prompt, response, mask, True)
    torch.cuda.synchronize()
    scores = result["teacher_logprobs"]
    torch.testing.assert_close(result["opd_mask"], torch.tensor(mask, dtype=torch.float32))
    assert scores.shape == (len(response),) and torch.isfinite(scores).all()
    assert torch.count_nonzero(scores[len(first):len(first) + len(observation)]) == 0
    for value in result.values():
        assert value.device.type == "cpu" and value.dtype == torch.float32
        assert not value.requires_grad
    success = score_failed_trajectory(None, prompt, response, mask, False)
    assert all(torch.count_nonzero(value) == 0 for value in success.values())
    report = {
        "checkpoint": args.checkpoint_dir,
        "dtype": str(next(teacher.parameters()).dtype),
        "response_tokens": len(response),
        "scored_tokens": sum(mask),
        "observation_tokens": len(observation),
        "score_min": scores[result["opd_mask"].bool()].min().item(),
        "score_max": scores[result["opd_mask"].bool()].max().item(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "status": "passed",
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
