"""短时实测 3B actor → CPU → 7B 教师 → 3B actor 的单卡资源周期。"""

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from MedSearch.distillation.teacher import load_teacher, score_tokens
from MedSearch.rl.distillation_hook import prepare_teacher_phase, restore_student_phase


def memory() -> dict:
    free, total = torch.cuda.mem_get_info()
    with open("/proc/meminfo", encoding="ascii") as handle:
        available_kib = next(
            int(line.split()[1]) for line in handle if line.startswith("MemAvailable:")
        )
    return {
        "cuda_used_gib": round((total - free) / 2**30, 3),
        "host_available_gib": round(available_kib / 2**20, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("student_dir")
    parser.add_argument("teacher_dir")
    parser.add_argument("output_json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tokenizer = AutoTokenizer.from_pretrained(args.student_dir, local_files_only=True)
    prefix_ids = tokenizer.encode(
        "<|im_start|>user\nWhat evidence supports aspirin after myocardial infarction?"
        "<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    target_ids = tokenizer.encode(
        "<search>aspirin secondary prevention myocardial infarction</search>",
        add_special_tokens=False,
    )
    report = {"prompt_tokens": len(prefix_ids), "action_tokens": len(target_ids), "phases": {}}
    torch.cuda.reset_peak_memory_stats()

    def record(name, started):
        torch.cuda.synchronize()
        report["phases"][name] = {
            "seconds": round(time.perf_counter() - started, 2),
            "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            **memory(),
        }
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    student = AutoModelForCausalLM.from_pretrained(
        args.student_dir, dtype=torch.bfloat16, local_files_only=True
    ).eval().requires_grad_(False).to("cuda")
    prefix = torch.tensor([prefix_ids], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        first_token = int(student(input_ids=prefix).logits[0, -1].argmax())
    record("student_first_forward", started)

    # FSDP param_offload=true 在 eval/train context 退出时自动执行这次搬移。
    started = time.perf_counter()
    student.to("cpu")
    del prefix
    gc.collect()
    torch.cuda.empty_cache()
    if next(student.parameters()).device.type != "cpu":
        raise AssertionError("student parameters are still on GPU")
    calls = []

    def worker_to(device, *, model, optimizer, grad):
        calls.append({"device": device, "model": model, "optimizer": optimizer, "grad": grad})
        if model:
            student.to("cuda" if device == "device" else "cpu")

    runtime = SimpleNamespace(
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                actor=SimpleNamespace(fsdp_config=SimpleNamespace(param_offload=True))
            )
        ),
        actor_rollout_wg=SimpleNamespace(to=worker_to),
    )
    prepare_teacher_phase(runtime)
    record("student_offload_and_prepare", started)

    started = time.perf_counter()
    teacher = load_teacher(args.teacher_dir, "cuda")
    prefix = torch.tensor([prefix_ids], dtype=torch.long, device="cuda")
    target = torch.tensor([target_ids], dtype=torch.long, device="cuda")
    scores = score_tokens(teacher, prefix, target, mode="sampled")
    if scores.shape != target.shape or not bool(torch.isfinite(scores).all()):
        raise AssertionError("teacher score is missing or non-finite")
    report["teacher_score_shape"] = list(scores.shape)
    record("teacher_load_and_score", started)

    started = time.perf_counter()
    del teacher, scores, prefix, target
    gc.collect()
    torch.cuda.empty_cache()
    restore_student_phase(runtime)
    record("teacher_exit_and_restore_hook", started)

    started = time.perf_counter()
    # FSDP eval_mode 会在下一次 actor 前向前自动恢复参数。
    student.to("cuda")
    prefix = torch.tensor([prefix_ids], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        second_token = int(student(input_ids=prefix).logits[0, -1].argmax())
    record("student_restored_forward", started)
    if first_token != second_token:
        raise AssertionError("student action token changed across resource switch")
    if [call["device"] for call in calls] != ["cpu", "device"]:
        raise AssertionError("phase hook call order changed")
    report["first_action_token_id"] = first_token
    report["restored_action_token_id"] = second_token
    report["hook_calls"] = calls
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
