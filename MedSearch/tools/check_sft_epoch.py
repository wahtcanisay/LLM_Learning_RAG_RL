"""Verify the saved cold-start adapter and one initial-prompt action."""
import argparse
import json
from pathlib import Path

import torch
import wandb
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from MedSearch.rl.protocol import parse_action
from MedSearch.sft.data import load_samples
from MedSearch.sft.train import PROJECT_ROOT, load_base_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['top32', 'ce'], default='top32')
    args = parser.parse_args()
    output = PROJECT_ROOT / f"MedSearch/outputs/distillation/sft_{args.method}_epoch1_20261013"
    report = json.loads((output / "run_report.json").read_text())
    assert report["status"] == "completed"
    adapter_path = output / "adapter"
    adapter = load_file(str(adapter_path / "adapter_model.safetensors"))
    checkpoint = load_file(str(output / "checkpoint-125/adapter_model.safetensors"))
    assert adapter.keys() == checkpoint.keys()
    assert all(torch.equal(adapter[key], checkpoint[key]) for key in adapter)
    lora_b_norm = sum(value.float().square().sum().item() for key, value in adapter.items() if "lora_B" in key) ** .5
    assert lora_b_norm > 0  # 新建 LoRA 的 B 为零；保存的 B 已实际更新。
    config = report["config"]
    tokenizer = AutoTokenizer.from_pretrained(PROJECT_ROOT / config["model_path"], local_files_only=True)
    model = PeftModel.from_pretrained(load_base_model(config, dtype=torch.bfloat16), adapter_path).to("cuda").eval()
    row = load_samples(PROJECT_ROOT / config["data_path"])[0]
    prefix = row["prompt_ids"]

    class EndAction(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            text = tokenizer.decode(input_ids[0, len(prefix):][-16:], skip_special_tokens=False)
            return torch.full((input_ids.shape[0],), "</search>" in text or "</answer>" in text,
                              device=input_ids.device, dtype=torch.bool)

    ids = torch.tensor([prefix], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=500,
                                   do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id, stopping_criteria=StoppingCriteriaList([EndAction()]))
    text = tokenizer.decode(generated[0, len(prefix):], skip_special_tokens=True)
    action = parse_action(text)
    assert action is not None, text
    verification = {"checkpoint_adapter_identical": True, "lora_b_norm": lora_b_norm,
                    "question_id": row["question_id"], "inference_context": "training question initial prompt, no teacher history or gold",
                    "action_text": text, "action_kind": action.kind if action else None,
                    "action_valid": action is not None, "qa_accuracy_evaluated": False}
    # 只读核对线上 run，确认指标已到服务器。
    run = wandb.Api(timeout=30).run(report["wandb_url"].replace("https://wandb.ai/", "").replace("/runs/", "/"))
    verification["wandb_server_state"] = run.state
    verification["wandb_server_summary"] = {key: run.summary.get(key) for key in ["global_step", "train_loss", "checkpoint_saved"]}
    history = run.history(keys=["train/global_step", "train/ce_loss", "train/top32_loss", "train/loss"], pandas=False)
    verification["wandb_server_loss_rows"] = len(history)
    assert run.summary.get("global_step") == 125
    assert history
    (output / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
