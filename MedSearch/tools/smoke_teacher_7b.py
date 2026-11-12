"""对固定 Search-R1 教师做一次真实 BF16 加载与 sampled-token 评分。"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from MedSearch.distillation.teacher import load_teacher, score_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("output_json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 7B teacher smoke")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, local_files_only=True)
    prompt_ids = tokenizer.encode(
        "<|im_start|>user\nWhat evidence supports aspirin after myocardial infarction?"
        "<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    action_ids = tokenizer.encode(
        "<search>aspirin secondary prevention myocardial infarction</search>",
        add_special_tokens=False,
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    teacher = load_teacher(args.checkpoint_dir, device="cuda")
    prefix = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    target = torch.tensor([action_ids], dtype=torch.long, device="cuda")
    scores = score_tokens(teacher, prefix, target, mode="sampled")
    torch.cuda.synchronize()
    if scores.shape != target.shape or not torch.isfinite(scores).all():
        raise AssertionError("teacher scores are missing or non-finite")

    report = {
        "model_type": teacher.config.model_type,
        "config_vocab_size": teacher.config.vocab_size,
        "dtype": str(next(teacher.parameters()).dtype),
        "prompt_tokens": len(prompt_ids),
        "action_tokens": len(action_ids),
        "score_shape": list(scores.shape),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
