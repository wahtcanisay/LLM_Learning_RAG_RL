"""真实 7B 子进程与调度函数验收；使用合成 token，不训练或评测能力。"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from transfer_queue import KVBatchMeta
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from MedSearch.distillation.teacher_process import score_in_subprocess
from MedSearch.rl.distillation_hook import run_failure_teacher_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prompt = encode("<|im_start|>user\nEngineering smoke: find supporting evidence.<|im_end|>\n<|im_start|>assistant\n")
    search = encode("<search>engineering smoke evidence</search>")
    observation = encode("<information>Synthetic observation for an interface check.</information>")
    answer = encode("<answer>Engineering fixture only.</answer>")
    response = search + observation + answer
    mask = [1] * len(search) + [0] * len(observation) + [1] * len(answer)

    def repeated(text, size):
        tokens = encode(text)
        return (tokens * (size // len(tokens) + 1))[:size]

    # 一条分段动作、一条当前最大单次输入/生成形状；后者仅测长度和显存。
    examples = [(prompt, response, mask, 0),
                (repeated("Engineering context. ", 4096), repeated("Engineering action. ", 500), [1] * 500, 0),
                (prompt, answer, [1] * len(answer), 1), ([10], [0], [0], None)]
    rows = [{"uid": "group-0" if i < 3 else "padding", "prompts": torch.tensor(p),
             "responses": torch.tensor(r), "response_mask": torch.tensor(m),
             "extra_fields": {"reward_extra_info": {"acc": acc}, "references": ["must not reach teacher"]}
             if acc is not None else {}} for i, (p, r, m, acc) in enumerate(examples)]
    data = list_of_dict_to_tensordict(rows)
    meta = KVBatchMeta(partition_id="teacher-process-smoke", keys=[f"trace-{i}" for i in range(4)],
                       tags=[{}, {}, {}, {"is_padding": True}])
    events = []
    # 这里只记录资源调度调用；没有启动真实学生/Ray/FSDP/optimizer。
    runtime = SimpleNamespace(
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(fsdp_config=SimpleNamespace(param_offload=True)))),
        actor_rollout_wg=SimpleNamespace(to=lambda device, **kwargs: events.append(device)),
    )
    free_before, total = torch.cuda.mem_get_info()

    def score_fn(failures):
        events.append("score")
        return score_in_subprocess(failures, args.teacher, output / "jobs")

    result = run_failure_teacher_phase(runtime, meta, data, score_fn)
    free_after, _ = torch.cuda.mem_get_info()
    assert events == ["cpu", "score", "device"] and result is data
    for i, row in enumerate(rows):
        expected_mask = row["response_mask"].float() if i < 2 else torch.zeros_like(row["response_mask"], dtype=torch.float32)
        torch.testing.assert_close(data[i]["opd_mask"], expected_mask)
        scores = data[i]["teacher_logprobs"]
        assert scores.shape == expected_mask.shape and torch.isfinite(scores).all()
        assert scores.dtype == torch.float32 and scores.device.type == "cpu" and not scores.requires_grad
        assert torch.count_nonzero(scores[expected_mask == 0]) == 0
    job = next((output / "jobs").iterdir())
    worker = json.loads((job / "result.json").read_text())
    assert worker["rows"] == 2 and worker["frozen"] and worker["eval"]
    assert worker["dtype"] == "torch.bfloat16" and not Path(f"/proc/{worker['pid']}").exists()
    assert torch.cuda.memory_allocated() == 0  # 父进程没有加载教师。
    report = {
        "status": "passed", "teacher": str(Path(args.teacher).resolve()), "worker": worker,
        "scope": "real 7B child and real TensorDict; resource calls recorded only; no student/Ray/FSDP/optimizer",
        "synthetic_fixture_not_on_policy_rollout": True, "events": events,
        "worker_exit_code": 0, "worker_pid_gone": True,
        "input_shapes": [{"prefix": len(p), "response": len(r), "model_tokens": sum(m)} for p, r, m, _ in examples],
        "gpu_used_before_gib": round((total - free_before) / 2**30, 3),
        "gpu_used_after_gib": round((total - free_after) / 2**30, 3),
        "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            "MedSearch/distillation/teacher_process.py", "MedSearch/distillation/teacher.py",
            "MedSearch/rl/distillation_hook.py", "MedSearch/tools/smoke_teacher_process.py")},
        "request_sha256": hashlib.sha256((job / "request.jsonl").read_bytes()).hexdigest(),
        "result_sha256": hashlib.sha256((job / "result.pt").read_bytes()).hexdigest(),
    }
    (output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
