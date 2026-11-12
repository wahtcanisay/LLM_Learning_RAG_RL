"""一次加载教师，将 SFT 动作位置的 Top-32 分布写入 Parquet。"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from MedSearch.distillation.teacher import load_teacher, score_tokens
from MedSearch.rl.distillation_hook import build_teacher_score_jobs


SCHEMA = pa.schema([
    ("trajectory_id", pa.string()), ("question_id", pa.string()),
    ("prompt_ids", pa.list_(pa.int32())), ("response_ids", pa.list_(pa.int32())),
    ("response_mask", pa.list_(pa.int8())),
    ("target_positions", pa.list_(pa.int32())),
    ("teacher_topk_ids", pa.list_(pa.list_(pa.int32()))),
    ("teacher_topk_logprobs", pa.list_(pa.list_(pa.float32()))),
])


def score_record(model, record):
    """返回训练 token 与有效目标位置的缓存；不复制 gold/参考答案字段。"""
    prompt, response, mask = (record[k] for k in ("prompt_ids", "response_ids", "response_mask"))
    if not prompt or len(response) != len(mask) or any(m not in (0, 1) for m in mask):
        raise ValueError("Invalid prompt/response/mask in cache input")
    device = next(model.parameters()).device
    positions, ids, logprobs = [], [], []
    for job in build_teacher_score_jobs(prompt, response, mask):
        result = score_tokens(model,
                              torch.tensor([job["prefix_ids"]], dtype=torch.long, device=device),
                              torch.tensor([job["target_ids"]], dtype=torch.long, device=device),
                              mode="topk", k=32)
        top_ids = result["topk_ids"].squeeze(0).detach().cpu()
        top_probs = result["topk_logprobs"].squeeze(0).detach().cpu().float()
        if top_ids.shape != (len(job["target_ids"]), 32) or top_probs.shape != top_ids.shape:
            raise ValueError("Top-32 output shape does not match target tokens")
        if not torch.isfinite(top_probs).all() or not ((top_ids >= 0) & (top_ids < 151665)).all():
            raise ValueError("Non-finite scores or unmapped token IDs")
        start = len(prompt) + job["response_start"]
        positions.extend(range(start, start + len(job["target_ids"])))
        ids.extend(top_ids.tolist())
        logprobs.extend(top_probs.tolist())
    if not positions:
        raise ValueError("SFT trajectory has no model-generated target tokens")
    return {"trajectory_id": str(record.get("trajectory_id", record.get("question_id", ""))),
            "question_id": str(record.get("question_id", "")),
            "prompt_ids": prompt, "response_ids": response, "response_mask": mask,
            "target_positions": positions, "teacher_topk_ids": ids,
            "teacher_topk_logprobs": logprobs}


def validate_cache(path):
    """流式重读，核对目标位置、Top-32 长度及 Parquet float32 落盘类型。"""
    source = pq.ParquetFile(path)
    if source.schema_arrow != SCHEMA:
        raise ValueError("Unexpected Parquet cache schema")
    rows = tokens = 0
    for batch in source.iter_batches(batch_size=16):
        for row in batch.to_pylist():
            positions = [len(row["prompt_ids"]) + i for i, m in enumerate(row["response_mask"]) if m == 1]
            if row["target_positions"] != positions:
                raise ValueError("Cached target positions do not match action mask")
            if len(row["teacher_topk_ids"]) != len(positions) or len(row["teacher_topk_logprobs"]) != len(positions):
                raise ValueError("Cached scores do not match target positions")
            ids = torch.tensor(row["teacher_topk_ids"])
            probs = torch.tensor(row["teacher_topk_logprobs"])
            if ids.shape != (len(positions), 32) or probs.shape != ids.shape:
                raise ValueError("Invalid cached Top-32 shape")
            if not torch.isfinite(probs).all() or not ((ids >= 0) & (ids < 151665)).all():
                raise ValueError("Invalid cached scores or token IDs")
            rows += 1
            tokens += len(positions)
    return {"rows": rows, "scored_tokens": tokens}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    output = Path(args.output)
    manifest_path = output.with_suffix(".manifest.json")
    temporary = output.with_suffix(".partial.parquet")
    if any(p.exists() for p in (output, manifest_path, temporary)):
        raise FileExistsError("Use a new output path; existing cache files are not overwritten")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = load_teacher(args.teacher, "cuda")  # 整次任务只加载一次。
    count = 0
    with pq.ParquetWriter(temporary, SCHEMA, compression="zstd") as writer:
        with Path(args.input).open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = score_record(model, json.loads(line))
                writer.write_table(pa.Table.from_pylist([row], schema=SCHEMA))
                count += 1
                if args.limit is not None and count >= args.limit:
                    break
    torch.cuda.synchronize()
    report = validate_cache(temporary)
    if count == 0 or report["rows"] != count:
        raise ValueError("Cache row count mismatch or empty input")
    report.update({"status": "complete", "scope": "subset" if args.limit else "full",
                   "input": args.input, "input_sha256": file_hash(args.input),
                   "teacher": args.teacher, "teacher_revision": args.teacher_revision,
                   "teacher_file_hashes": {name: file_hash(Path(args.teacher) / name)
                                            for name in ("config.json", "tokenizer.json", "tokenizer_config.json")},
                   "k": 32, "temperature": 1, "mapped_vocab_size": 151665,
                   "position_convention": "input_ids index; logits at position-1 predicts target",
                   "output": str(output), "output_sha256": file_hash(temporary),
                   "elapsed_seconds": round(time.perf_counter() - started, 2),
                   "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)})
    temporary.replace(output)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
