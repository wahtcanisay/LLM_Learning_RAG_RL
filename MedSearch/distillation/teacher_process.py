"""短生命周期教师评分进程；通过本批任务文件与 CPU 张量传递结果。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch


def score_in_subprocess(failure_batch, teacher_path, output_dir, *, device="cuda") -> list[dict]:
    """保存失败 token，等待教师进程退出，再返回可供回填的 CPU 分数。"""
    if not failure_batch:
        return []
    # 只向教师传身份与实际轨迹 token，不传 references/gold 等批次字段。
    fields = ("original_index", "group_id", "trajectory_id", "prompt_ids", "response_ids", "response_mask")
    requests = [{key: row[key] for key in fields} for row in failure_batch]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix="teacher-", dir=output_dir))
    source, destination = job / "request.jsonl", job / "result.pt"
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests), encoding="utf-8")
    with (job / "worker.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, "-m", "MedSearch.distillation.teacher_process",
             "--input", str(source), "--output", str(destination),
             "--teacher", str(Path(teacher_path).resolve()), "--device", device],
            cwd=Path(__file__).resolve().parents[2], check=True,
            stdout=log, stderr=subprocess.STDOUT,
        )
    # 仅加载本进程刚生成的可信文件；非零退出不会走到这里。
    results = torch.load(destination, map_location="cpu", weights_only=True)
    expected_ids = [(row["original_index"], row["trajectory_id"]) for row in requests]
    if [(row["original_index"], row["trajectory_id"]) for row in results] != expected_ids:
        raise ValueError(f"Teacher results do not cover the requested trajectories: {job}")
    for request, result in zip(requests, results):
        mask = torch.tensor(request["response_mask"], dtype=torch.float32)
        for field in ("teacher_logprobs", "opd_mask"):
            value = result[field]
            if (value.shape != mask.shape or value.device.type != "cpu" or value.dtype != torch.float32
                    or value.requires_grad or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid teacher {field}: {job}")
        if not torch.equal(result["opd_mask"], mask) or torch.count_nonzero(result["teacher_logprobs"][mask == 0]):
            raise ValueError(f"Teacher scores must cover student actions only: {job}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    # 子进程内导入与加载；父进程不持有教师模型。
    from MedSearch.distillation.teacher import load_teacher
    from MedSearch.rl.distillation_hook import score_failed_batch

    requests = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    started = time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    teacher = load_teacher(args.teacher, args.device) if requests else None
    results = score_failed_batch(teacher, requests)
    destination = Path(args.output)
    temporary = destination.with_suffix(".partial.pt")
    torch.save(results, temporary)
    temporary.replace(destination)
    report = {
        "pid": os.getpid(), "rows": len(results),
        "scored_tokens": sum(int(row["opd_mask"].sum()) for row in results),
        "dtype": str(next(teacher.parameters()).dtype) if teacher is not None else None,
        "frozen": teacher is None or not any(p.requires_grad for p in teacher.parameters()),
        "eval": teacher is None or not teacher.training,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)
        if args.device.startswith("cuda") else None,
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
