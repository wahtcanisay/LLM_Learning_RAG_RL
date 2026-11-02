"""Accept real SFT checkpoint save/reload/resume and one action inference."""
import argparse
import gc
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, set_seed

from MedSearch.rl.protocol import parse_action
from MedSearch.sft.data import SFTDataset, build_data_collator, load_samples
from MedSearch.sft.train import attach_lora, build_trainer, load_base_model

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "MedSearch/outputs/distillation/sft_checkpoint_smoke_20261013_final"


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['top32', 'ce'], default='top32')
    args = parser.parse_args()
    if args.method == 'ce':
        OUT = ROOT / 'MedSearch/outputs/distillation/sft_ce_checkpoint_smoke_20261013'
    if list(OUT.glob("checkpoint-*")):
        raise FileExistsError(f"Do not overwrite an existing acceptance run: {OUT}")
    set_seed(20261013)
    started = time.monotonic()
    cache = ROOT / ("MedSearch/data/distillation/top32_sft1000_20261011/train.parquet" if args.method == 'top32'
                    else 'MedSearch/data/distillation/ce_sft1000_20261013/train.parquet')
    rows = load_samples(cache)
    longest = max(rows, key=lambda r: len(r["prompt_ids"]) + len(r["response_ids"]))
    dev_path = ROOT / "MedSearch/data/distillation/pools_v2_20260926_devclean/dev.parquet"
    dev = pq.read_table(dev_path)
    config = {
        "model_path": "models/Qwen2.5-3B-Instruct", "dtype": "bfloat16", "adapter_path": None,
        "batch_size": 1, "num_workers": 0, "lora_rank": 8, "lora_alpha": 16,
        "lora_dropout": 0., "lora_target_modules": ["q_proj", "v_proj"],
        "training": {
            "output_dir": str(OUT), "seed": 20261013, "max_steps": 1,
            "learning_rate": 1e-4, "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True, "save_strategy": "steps", "save_steps": 1,
            "save_only_model": False, "report_to": "none", "logging_steps": 1,
            "lr_scheduler_type": "constant", "disable_tqdm": True,
        },
    }
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["model_path"], local_files_only=True)
    tokenizer.padding_side = "right"
    dataset = SFTDataset([longest, longest], tokenizer, "search_distill", 4096)
    loader = DataLoader(dataset, batch_size=1, collate_fn=build_data_collator(tokenizer))
    if args.method == 'ce':
        assert 'teacher_topk_ids' not in loader.collate_fn([dataset[0]])
    model = attach_lora(load_base_model(config, dtype=torch.bfloat16), config)
    trainer = build_trainer(config, model, tokenizer, loader)
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    checkpoint = OUT / "checkpoint-1"
    required = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt",
                "trainer_state.json", "rng_state.pth"]
    assert all((checkpoint / name).is_file() for name in required)

    def last_logits(current_model):
        current_model.eval()
        batch = trainer._prepare_inputs(loader.collate_fn([dataset[0]]))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return current_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, -1].float().cpu()

    # A normal initial question prompt; no teacher history, future targets or gold.
    inference_row = rows[0]
    prefix = inference_row["prompt_ids"]

    class EndAction(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            new_ids = input_ids[0, len(prefix):]
            text = tokenizer.decode(new_ids[-16:], skip_special_tokens=False)
            done = "</search>" in text or "</answer>" in text
            return torch.full((input_ids.shape[0],), done, device=input_ids.device, dtype=torch.bool)

    def action(current_model):
        current_model.eval()
        ids = torch.tensor([prefix], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = current_model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=500, do_sample=False, use_cache=True,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([EndAction()]))
        tokens = output[0, len(prefix):].cpu().tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        return tokens, text, parse_action(text)

    reference = last_logits(model)
    adapter_reference = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    saved = load_file(str(checkpoint / "adapter_model.safetensors"))
    print(json.dumps({"saved_adapter_max_difference": max((saved[k].float() - v.float()).abs().max().item() for k, v in adapter_reference.items()),
                      "adapter_dtype": str(next(iter(adapter_reference.values())).dtype)}), flush=True)
    before_ids, before_text, before_action = action(model)
    print(json.dumps({"checkpoint": str(checkpoint), "longest_length": len(dataset[0]["input_ids"]),
                      "action_before_reload": before_action.kind if before_action else None,
                      "action_text": before_text}), flush=True)
    base = model.unload()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    reloaded_state = get_peft_model_state_dict(reloaded)
    print(json.dumps({"reloaded_adapter_max_difference": max((reloaded_state[k].detach().cpu().float() - v.float()).abs().max().item() for k, v in adapter_reference.items()),
                      "reloaded_adapter_dtype": str(next(iter(reloaded_state.values())).dtype)}), flush=True)
    actual = last_logits(reloaded)
    max_difference = (actual - reference).abs().max().item()
    assert torch.equal(actual, reference), max_difference
    after_ids, after_text, after_action = action(reloaded)
    assert before_ids == after_ids
    assert after_action is not None, after_text

    config["adapter_path"] = str(checkpoint)
    config["training"]["max_steps"] = 2
    resumed = build_trainer(config, reloaded, tokenizer, loader)
    before_resume = {name: p.detach().cpu().clone() for name, p in reloaded.named_parameters() if p.requires_grad}
    resumed_result = resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed_result.global_step == 2
    delta_sq = sum((p.detach().cpu().float() - before_resume[name].float()).square().sum().item()
                   for name, p in reloaded.named_parameters() if name in before_resume)
    assert delta_sq > 0
    report = {
        "status": "passed", "config": config,
        "trajectory_id": longest["trajectory_id"], "longest_length": len(dataset[0]["input_ids"]),
        "saved_checkpoint_files": sorted(p.name for p in checkpoint.iterdir()),
        "reload_logits_max_abs_difference": max_difference, "action_tokens_identical": True,
        "action_inference_context": "initial question prompt; no teacher history, future target or gold",
        "inference_question_id": inference_row["question_id"],
        "action_text": after_text, "action_kind": after_action.kind, "action_content": after_action.content,
        "action_quality_verified": False, "before_resume_step": result.global_step,
        "after_resume_step": resumed_result.global_step, "resume_adapter_delta_sq": delta_sq,
        "native_resume_loads": ["adapter", "optimizer", "scheduler", "rng", "trainer_state"],
        "dev_path": str(dev_path), "dev_rows": dev.num_rows, "dev_columns": dev.column_names,
        "dev_has_teacher_cache": "teacher_topk_ids" in dev.column_names,
        "elapsed_seconds": time.monotonic() - started,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
