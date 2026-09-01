"""Evaluate frozen Stage-A dev data and export samples for human review."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
import subprocess

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GROUPS = ("single", "multi", "reasoning", "general")
STRICT_THINK_ANSWER = re.compile(
    r"\s*<think>(?P<think>.+?)</think>\s*"
    r"<answer>(?P<answer>.+?)</answer>\s*",
    re.DOTALL,
)


def stratified_sample(samples: list[dict], per_group: int, seed: int) -> list[dict]:
    """Select the same number of frozen examples from every represented group."""
    if per_group <= 0:
        raise ValueError("per_group must be positive")

    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["mixture_group"]].append(sample)

    unknown = sorted(set(grouped) - set(GROUPS))
    if unknown:
        raise ValueError(f"unknown Stage-A mixture groups: {unknown}")

    rng = random.Random(seed)
    selected = []
    for group in GROUPS:
        candidates = grouped.get(group, [])
        if not candidates:
            continue
        if len(candidates) < per_group:
            raise ValueError(
                f"mixture group {group!r} has {len(candidates)} samples, "
                f"fewer than per_group={per_group}"
            )
        selected.extend(rng.sample(candidates, per_group))
    return selected


def split_prompt_and_reference(sample: dict) -> tuple[list[dict], str]:
    """Use all turns before the final assistant message as the generation prompt."""
    messages = sample.get("messages")
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Stage-A evaluation sample must end with an assistant message")
    return messages[:-1], messages[-1]["content"]


def analyze_tags(text: str) -> dict:
    """Describe exact `<think>` and `<answer>` structure without judging content."""
    think_open = text.count("<think>")
    think_close = text.count("</think>")
    answer_open = text.count("<answer>")
    answer_close = text.count("</answer>")
    has_think = bool(think_open or think_close)

    complete_think = False
    if think_open == think_close == 1:
        start = text.index("<think>") + len("<think>")
        end = text.index("</think>")
        complete_think = start < end and bool(text[start:end].strip())

    strict = STRICT_THINK_ANSWER.fullmatch(text)
    strict_think_answer = bool(
        strict
        and strict.group("think").strip()
        and strict.group("answer").strip()
    )
    return {
        "think_open_count": think_open,
        "think_close_count": think_close,
        "answer_open_count": answer_open,
        "answer_close_count": answer_close,
        "has_think": has_think,
        "complete_think": complete_think,
        "malformed_think": has_think and not complete_think,
        "strict_think_answer": strict_think_answer,
    }


def aggregate_nll(rows: list[tuple[str, float, int]]) -> dict:
    """Aggregate summed NLL by supervised assistant-token count."""
    totals = defaultdict(lambda: {"summed_nll": 0.0, "supervised_tokens": 0, "examples": 0})
    for group, summed_nll, supervised_tokens in rows:
        if supervised_tokens <= 0:
            raise ValueError("each loss row must contain supervised assistant tokens")
        for key in ("overall", group):
            totals[key]["summed_nll"] += float(summed_nll)
            totals[key]["supervised_tokens"] += int(supervised_tokens)
            totals[key]["examples"] += 1

    result = {}
    ordered_keys = ["overall", *(group for group in GROUPS if group in totals)]
    for key in ordered_keys:
        total = totals[key]
        nll = total["summed_nll"] / total["supervised_tokens"]
        result[key] = {
            "examples": total["examples"],
            "supervised_tokens": total["supervised_tokens"],
            "nll": nll,
            "perplexity": math.exp(nll) if nll < 709 else math.inf,
        }
    return result


def summarize_tag_records(records: list[dict]) -> dict:
    """Report tag rates and repeated-generation stability overall and by group."""
    def summarize(subset: list[dict]) -> dict:
        count = len(subset)
        if not count:
            return {}

        prompts = defaultdict(list)
        for record in subset:
            prompts[record["sample_id"]].append(record)

        def expected(record: dict) -> bool:
            tags = record["tag_analysis"]
            if record["mixture_group"] == "reasoning":
                return tags["strict_think_answer"]
            return not tags["has_think"]

        consistent = 0
        stable_expected = 0
        for prompt_records in prompts.values():
            signatures = {
                (
                    record["tag_analysis"]["has_think"],
                    record["tag_analysis"]["complete_think"],
                    record["tag_analysis"]["strict_think_answer"],
                )
                for record in prompt_records
            }
            consistent += len(signatures) == 1
            stable_expected += all(expected(record) for record in prompt_records)

        reasoning = [record for record in subset if record["mixture_group"] == "reasoning"]
        normal = [record for record in subset if record["mixture_group"] != "reasoning"]
        return {
            "generations": count,
            "prompts": len(prompts),
            "has_think_rate": sum(record["tag_analysis"]["has_think"] for record in subset) / count,
            "complete_think_rate": sum(record["tag_analysis"]["complete_think"] for record in subset) / count,
            "malformed_think_rate": sum(record["tag_analysis"]["malformed_think"] for record in subset) / count,
            "strict_think_answer_rate": sum(
                record["tag_analysis"]["strict_think_answer"] for record in subset
            ) / count,
            "max_token_truncation_rate": sum(
                record.get("hit_max_new_tokens", False) for record in subset
            ) / count,
            "missing_expected_think_answer_rate": (
                sum(not record["tag_analysis"]["strict_think_answer"] for record in reasoning)
                / len(reasoning)
                if reasoning
                else 0.0
            ),
            "unexpected_think_rate": (
                sum(record["tag_analysis"]["has_think"] for record in normal) / len(normal)
                if normal
                else 0.0
            ),
            "tag_signature_consistency_rate": consistent / len(prompts),
            "stable_expected_behavior_rate": stable_expected / len(prompts),
        }

    result = {"overall": summarize(records)}
    for group in GROUPS:
        subset = [record for record in records if record["mixture_group"] == group]
        if subset:
            result[group] = summarize(subset)
    return result


def write_review_markdown(path: str | Path, records: list[dict], metadata: dict) -> None:
    """Write generated answers with an intentionally blank human-review rubric."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage-A 医疗对话人工抽样审阅",
        "",
        "```json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
        "```",
    ]
    for index, record in enumerate(records, 1):
        lines.extend(
            [
                "",
                f"## {index}. {record['sample_id']} / {record['mixture_group']} / repeat {record['repeat_index']}",
                "",
                f"来源：`{record['source']}`",
                "",
                "### 对话上下文",
                "",
                "```json",
                json.dumps(record["prompt_messages"], ensure_ascii=False, indent=2),
                "```",
                "",
                "### 参考回答",
                "",
                record["reference"],
                "",
                "### 模型回答",
                "",
                record["generated"],
                "",
                "### 自动标签检查",
                "",
                "```json",
                json.dumps(record["tag_analysis"], ensure_ascii=False, indent=2),
                "```",
                "",
                "### 人工评分（留空待审）",
                "",
                "- 医学正确性（0/1/2）：",
                "- 相关性与完整性（0/1/2）：",
                "- 幻觉（是/否）：",
                "- 安全风险（是/否）：",
                "- 标签行为（通过/失败）：",
                "- 备注：",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_generation_settings(repeats: int, temperature: float, top_p: float) -> None:
    """Reject settings that cannot measure stochastic prompt-level stability."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if repeats > 1 and temperature == 0:
        raise ValueError("repeats > 1 requires temperature > 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")


def calculate_loss_row(model, encoded: dict, group: str) -> tuple[str, float, int]:
    """Return group, summed NLL and supervised-token count for one example."""
    device = next(model.parameters()).device
    batch = {
        key: torch.tensor([value], dtype=torch.long, device=device)
        for key, value in encoded.items()
    }
    supervised_tokens = int((batch["labels"] != -100).sum().item())
    if supervised_tokens == 0:
        raise ValueError("evaluation example has no supervised assistant tokens")
    with torch.inference_mode():
        loss = model(**batch, use_cache=False).loss.float()
    if not torch.isfinite(loss):
        raise ValueError("evaluation loss is not finite")
    return group, float(loss.item()) * supervised_tokens, supervised_tokens


def main(argv: list[str] | None = None) -> None:
    """Evaluate one local Base or Stage-A adapter on the frozen Stage-A dev set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args(argv)

    validate_generation_settings(args.repeats, args.temperature, args.top_p)
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    from peft import PeftModel
    from transformers import AutoTokenizer, set_seed

    from .data import load_samples
    from .preprocess import encode_stage_a
    from .train import build_quantization_config, load_base_model, load_config

    config_path = Path(args.config)
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    if config["stage"] != "stage_a":
        raise ValueError("Stage-A evaluator requires a stage_a config")
    samples = load_samples(data_path)
    selected = stratified_sample(samples, args.per_group, args.seed)

    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "right"

    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtypes[config["dtype"]]
    quantization = build_quantization_config(dtype) if config["use_qlora"] else None
    model = load_base_model(config, quantization_config=quantization, dtype=dtype)
    if quantization is None and torch.cuda.is_available():
        model = model.to("cuda")

    resolved_adapter = None
    if args.adapter_path:
        resolved_adapter = Path(args.adapter_path)
        if not resolved_adapter.is_absolute():
            resolved_adapter = PROJECT_ROOT / resolved_adapter
        if not resolved_adapter.exists():
            raise FileNotFoundError(f"adapter path does not exist: {resolved_adapter}")
        model = PeftModel.from_pretrained(model, str(resolved_adapter), is_trainable=False)
    model.eval()
    set_seed(args.seed)

    loss_rows = []
    for index, sample in enumerate(samples, 1):
        encoded = encode_stage_a(sample, tokenizer, config["max_length"])
        loss_rows.append(calculate_loss_row(model, encoded, sample["mixture_group"]))
        if index % 25 == 0 or index == len(samples):
            print(f"loss {index}/{len(samples)}", flush=True)

    records = []
    device = next(model.parameters()).device
    for sample_index, sample in enumerate(selected, 1):
        prompt_messages, reference = split_prompt_and_reference(sample)
        rendered = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)
        for repeat_index in range(args.repeats):
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.temperature > 0:
                generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)
            with torch.inference_mode():
                output = model.generate(**inputs, **generation_kwargs)
            generated_ids = output[0, inputs["input_ids"].shape[1] :]
            ended_with_eos = bool(
                generated_ids.numel()
                and generated_ids[-1].item() == tokenizer.eos_token_id
            )
            generated = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
            records.append(
                {
                    "variant": args.variant,
                    "sample_id": sample["id"],
                    "mixture_group": sample["mixture_group"],
                    "source": sample["source"],
                    "repeat_index": repeat_index,
                    "prompt_messages": prompt_messages,
                    "reference": reference,
                    "generated": generated,
                    "generated_tokens": int(generated_ids.numel()),
                    "ended_with_eos": ended_with_eos,
                    "hit_max_new_tokens": (
                        generated_ids.numel() >= args.max_new_tokens and not ended_with_eos
                    ),
                    "tag_analysis": analyze_tags(generated),
                }
            )
        print(f"generate {sample_index}/{len(selected)}", flush=True)

    metadata = {
        "variant": args.variant,
        "config": str(config_path),
        "data": str(data_path),
        "model": str(model_path),
        "adapter": str(resolved_adapter) if resolved_adapter else None,
        "dev_examples": len(samples),
        "sampled_prompts": len(selected),
        "per_group": args.per_group,
        "repeats": args.repeats,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "quantized_4bit": config["use_qlora"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
    }
    metrics = {
        "metadata": metadata,
        "dev_loss": aggregate_nll(loss_rows),
        "tag_behavior": summarize_tag_records(records),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "generations.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    write_review_markdown(output_dir / "review.md", records, metadata)


if __name__ == "__main__":
    main()
