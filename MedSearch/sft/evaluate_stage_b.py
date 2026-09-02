"""Check only the first Stage-B action on frozen protocol-dev prompts.

This is an SFT gate: it does not retrieve documents, inject information, or
judge query relevance.  It answers only whether a checkpoint emits one legal,
non-empty ``<search>`` action after the shared Stage-B prompt.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess

import torch

from .preprocess import format_stage_b_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_first_action(text: str) -> dict:
    """Validate the single action generated in the first Stage-B turn."""
    matches = list(ACTION_RE.finditer(text))
    first = matches[0] if matches else None
    action = first.group(1) if first else None
    content = first.group(2).strip() if first else ""
    thinks = list(THINK_RE.finditer(text))
    valid_think = bool(
        first
        and len(thinks) == 1
        and thinks[0].start() < first.start()
        and thinks[0].group(1).strip()
    )
    open_tags = text.count("<search>") + text.count("<answer>")
    close_tags = text.count("</search>") + text.count("</answer>")
    valid_action = bool(
        first
        and content
        and valid_think
        and len(matches) == 1
        and open_tags == close_tags == 1
    )
    return {
        "action": action,
        "content": content,
        "valid_think": valid_think,
        "valid_action": valid_action,
        "malformed": not valid_action,
    }


def summarize_first_actions(actions: list[dict]) -> dict:
    """Aggregate one first action per prompt."""
    if not actions:
        raise ValueError("at least one action record is required")
    count = len(actions)
    return {
        "examples": count,
        "valid_think_rate": sum(item["valid_think"] for item in actions) / count,
        "valid_search_rate": sum(
            item["valid_action"] and item["action"] == "search" for item in actions
        )
        / count,
        "nonempty_query_rate": sum(
            item["valid_action"] and item["action"] == "search" and bool(item["content"])
            for item in actions
        )
        / count,
        "direct_answer_rate": sum(
            item["valid_action"] and item["action"] == "answer" for item in actions
        )
        / count,
        "malformed_action_rate": sum(item["malformed"] for item in actions) / count,
    }


def main(argv: list[str] | None = None) -> None:
    """Generate once for every protocol-dev prompt and export first-turn rates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("invalid generation settings")

    from peft import PeftModel
    from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, set_seed

    from .train import build_quantization_config, load_base_model, load_config

    class StopOnAction(StoppingCriteria):
        def __init__(self, tokenizer):
            self.sequences = [
                tokenizer.encode(tag, add_special_tokens=False)
                for tag in ("</search>", " </search>", "</answer>", " </answer>")
            ]

        def __call__(self, input_ids, scores, **_kwargs):
            return any(
                input_ids.shape[1] >= len(sequence)
                and torch.equal(
                    input_ids[0, -len(sequence) :],
                    torch.tensor(sequence, device=input_ids.device),
                )
                for sequence in self.sequences
            )

    def resolve(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    config_path = resolve(args.config)
    data_path = resolve(args.data)
    adapter_path = resolve(args.adapter_path) if args.adapter_path else None
    output_dir = resolve(args.output_dir)
    if adapter_path is not None and not adapter_path.exists():
        raise FileNotFoundError(f"adapter path does not exist: {adapter_path}")
    samples = json.loads(data_path.read_text(encoding="utf-8"))
    config = load_config(config_path)
    model_path = resolve(config["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[config["dtype"]]
    quantization = build_quantization_config(dtype) if config["use_qlora"] else None
    model = load_base_model(config, quantization_config=quantization, dtype=dtype)
    if quantization is None and torch.cuda.is_available():
        model = model.to("cuda")
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    set_seed(args.seed)
    device = next(model.parameters()).device
    stopper = StoppingCriteriaList([StopOnAction(tokenizer)])

    records = []
    for index, sample in enumerate(samples, 1):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": format_stage_b_prompt(sample)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        generation = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "stopping_criteria": stopper,
        }
        if args.temperature > 0:
            generation.update(temperature=args.temperature, top_p=args.top_p)
        with torch.inference_mode():
            output = model.generate(**inputs, **generation)
        generated_ids = output[0, inputs["input_ids"].shape[1] :]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        records.append(
            {
                "sample_id": sample["id"],
                "difficulty": sample["difficulty"],
                "generated": generated,
                "generated_tokens": int(generated_ids.numel()),
                **parse_first_action(generated),
            }
        )
        print(f"generate {index}/{len(samples)}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": str(config_path),
        "data": str(data_path),
        "adapter": str(adapter_path) if adapter_path else None,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {"metadata": metadata, "first_turn": summarize_first_actions(records)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "generations.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
