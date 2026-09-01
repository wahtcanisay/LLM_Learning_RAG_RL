"""Teacher-only observations around the unchanged student training entry.

Runs one real GPU update; intentionally NOT a pytest test.
Only --save-roundtrip opts into saving and reloading the student's adapter.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import peft
import torch
import transformers
import yaml
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from MedSearch.sft import train


def parameter_digest(parameter):
    """Read all parameter bytes, copying at most 16M BF16 bytes per chunk."""
    digest = hashlib.sha256()
    for chunk in parameter.detach().reshape(-1).split(8 * 1024 * 1024):
        raw = chunk.cpu().contiguous().view(torch.uint8).numpy()
        digest.update(raw)
    return digest.hexdigest()


def probe_token_ids(tokenizer, messages):
    """Transformers 5 returns a BatchEncoding by default; request raw IDs."""
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False,
    )


def probe_last_logits(model, inputs):
    """Use identical BF16 autocast whether or not Accelerate wrapped forward."""
    model.eval()
    with torch.no_grad(), torch.autocast(inputs["input_ids"].device.type, dtype=torch.bfloat16):
        return model(**inputs, use_cache=False).logits[:, -1, :].float().cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-roundtrip", action="store_true")
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This teacher probe requires exactly one visible CUDA GPU")
    original_run = train.run_training
    original_save = train.save_adapter
    reference = {}

    def observe_run(trainer, *, resume_from_checkpoint=None):
        assert resume_from_checkpoint is None
        model = trainer.model
        train.describe_model(model)
        if trainer.args.max_steps != 1:
            raise ValueError("Probe only allows one optimizer update")
        assert bool(getattr(model, "is_loaded_in_4bit", False)) is args.qlora
        quantization = None
        bnb_bias_ids = set()
        quantized_layers = []
        if args.qlora:
            import bitsandbytes as bnb
            quantization = model.config.quantization_config.to_dict()
            quantized_layers = [layer for layer in model.modules() if isinstance(layer, bnb.nn.Linear4bit)]
            assert len(quantized_layers) == 252
            assert all(isinstance(layer.weight, bnb.nn.Params4bit) and layer.weight.dtype == torch.uint8
                       and layer.weight.quant_state.quant_type == "nf4" and layer.weight.quant_state.nested
                       for layer in quantized_layers)
            bnb_bias_ids = {id(layer.bias) for layer in quantized_layers if layer.bias is not None}
        execution = {
            "qlora": args.qlora, "quantization_config": quantization,
            "argv": argv, "linear4bit_layers": len(quantized_layers),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (Path(trainer.args.output_dir) / "teacher_execution.json").write_text(
            json.dumps(execution, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("PROBE: snapshot frozen-base hashes and LoRA weights", flush=True)
        frozen_before = {
            name: parameter_digest(p) for name, p in model.named_parameters() if not p.requires_grad
        }
        bias_before = {
            name: parameter_digest(p.float()) for name, p in model.named_parameters() if id(p) in bnb_bias_ids
        }
        if args.save_roundtrip:
            reference["frozen_hashes"] = frozen_before
        lora_before = {
            name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad
        }
        lengths = [len(trainer.train_dataset[i]["input_ids"]) for i in range(len(trainer.train_dataset))]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        metrics = original_run(trainer)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        gpu_peak = {
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        changed_lora = []
        for name, p in model.named_parameters():
            if name in lora_before:
                value = p.detach().cpu()
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite LoRA parameter: {name}")
                if not torch.equal(lora_before[name], value):
                    changed_lora.append(name)
        print("PROBE: verify all frozen-base parameter hashes", flush=True)
        changed_base = []
        bias_dtype_casts = []
        for name, p in model.named_parameters():
            if name not in frozen_before:
                continue
            assert not p.requires_grad and p.grad is None
            if parameter_digest(p) == frozen_before[name]:
                continue
            # bnb Linear4bit.forward converts frozen bias storage from the
            # preparation's FP32 to BF16. Only allow exactly preserved values.
            if (args.qlora and id(p) in bnb_bias_ids and p.dtype == torch.bfloat16
                    and parameter_digest(p.float()) == bias_before[name]):
                bias_dtype_casts.append(name)
            else:
                changed_base.append(name)
        if args.qlora:
            assert all(isinstance(layer.weight, bnb.nn.Params4bit) and layer.weight.dtype == torch.uint8
                       and layer.weight.quant_state.quant_type == "nf4" and layer.weight.quant_state.nested
                       for layer in quantized_layers)
        train.describe_model(model)
        gradients = [entry["grad_norm"] for entry in trainer.state.log_history if "grad_norm" in entry]
        report = {
            **execution,
            "global_step": trainer.state.global_step,
            "train_loss": metrics["train_loss"],
            "grad_norms": gradients,
            "sample_token_lengths": lengths,
            "frozen_tensors_checked": len(frozen_before),
            "changed_base_parameters": changed_base,
            "bnb_bias_dtype_casts_with_exact_values": bias_dtype_casts,
            "lora_tensors_checked": len(lora_before),
            "changed_lora_parameters": changed_lora,
            "run_training_wall_seconds": elapsed,
            "torch_cuda_peak": gpu_peak,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        if args.qlora:
            report["versions"]["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
            report["versions"]["accelerate"] = importlib.metadata.version("accelerate")
        report["passed"] = (
            trainer.state.global_step == metrics["global_step"] == 1
            and math.isfinite(metrics["train_loss"])
            and bool(gradients) and all(math.isfinite(g) for g in gradients)
            and any(g > 0 for g in gradients)
            and bool(changed_lora) and not changed_base
        )
        (Path(trainer.args.output_dir) / "parameter_verification.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        if not report["passed"]:
            raise AssertionError("Real 3B update checks failed; inspect parameter_verification.json")
        print(f"PROBE: passed; changed LoRA tensors={len(changed_lora)}, changed base tensors=0", flush=True)
        return metrics

    def observe_save(model, tokenizer, output_dir):
        """Observe native student saving without retaining the training model."""
        print("PROBE: capture fixed-input logits and save adapter", flush=True)
        messages = [{"role": "user", "content": "What does a thermometer measure?"}]
        ids = probe_token_ids(tokenizer, messages)
        inputs = {
            "input_ids": torch.tensor([ids], device=model.device),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long, device=model.device),
        }
        logits = probe_last_logits(model, inputs)
        if not torch.isfinite(logits).all():
            raise AssertionError("Nonfinite reference logits")
        state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in peft.get_peft_model_state_dict(model, save_embedding_layers=False).items()
        }
        assert original_save(model, tokenizer, output_dir) is None
        saved = load_file(Path(output_dir) / "adapter_model.safetensors")
        assert saved.keys() == state.keys()
        assert all("lora_" in key for key in saved)
        assert all(torch.equal(saved[key], tensor) for key, tensor in state.items())
        reference.update(
            adapter_dir=Path(output_dir), messages=messages,
            inputs={key: value.cpu() for key, value in inputs.items()}, logits=logits,
            state=state, vocab=tokenizer.get_vocab(), special_tokens=tokenizer.special_tokens_map,
            chat_template=tokenizer.chat_template, padding_side=tokenizer.padding_side,
            attention_backend=model.config._attn_implementation,
        )
        save_file(
            {"logits": logits, **reference["inputs"]},
            Path(output_dir).parent / "roundtrip_reference.safetensors",
        )

    train.run_training = observe_run
    if args.save_roundtrip:
        train.save_adapter = observe_save
    try:
        config = train.load_config(PROJECT_ROOT / "MedSearch/configs/stage_a.yaml")
        sample_count = config["batch_size"] * config["training"]["gradient_accumulation_steps"]
        source_path = Path(config["data_path"])
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        samples = train.load_samples(source_path)
        if len(samples) < sample_count:
            raise ValueError(f"smoke needs {sample_count} distinct samples")
        selected = random.Random(config["training"]["seed"]).sample(samples, sample_count)
        output_root = PROJECT_ROOT / "MedSearch/outputs/smoke"
        output_root.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="stage_a-", dir=output_root))
        smoke_data = output_dir / "samples.jsonl"
        smoke_data.write_text(
            "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in selected),
            encoding="utf-8",
        )
        config["data_path"] = str(smoke_data)
        config["use_qlora"] = args.qlora
        config["resume_from_checkpoint"] = None
        config["training"].update(
            output_dir=str(output_dir),
            max_steps=1,
            logging_steps=1,
            logging_nan_inf_filter=False,
            skip_memory_metrics=False,
            save_strategy="no",
        )
        smoke_config = output_dir / "smoke_config.yaml"
        smoke_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        manifest = {
            "config": config,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
            ).strip(),
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    Path(train.__file__),
                    Path(train.__file__).with_name("data.py"),
                    Path(train.__file__).with_name("preprocess.py"),
                ]
            },
            "data_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "smoke_data_sha256": hashlib.sha256(smoke_data.read_bytes()).hexdigest(),
            "sample_ids": [sample["id"] for sample in selected],
        }
        (output_dir / "run_config.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        argv = ["--config", str(smoke_config)]
        train.main(argv)
    finally:
        train.run_training = original_run
        train.save_adapter = original_save

    if args.save_roundtrip:
        # main's Trainer/model are out of scope: release them before loading again.
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("PROBE: reload local base and saved adapter after releasing Trainer", flush=True)
        adapter_dir = reference["adapter_dir"]
        config = json.loads((adapter_dir.parent / "run_config.json").read_text())["config"]
        config["adapter_path"] = str(adapter_dir)
        tokenizer = transformers.AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
        assert tokenizer.get_vocab() == reference["vocab"]
        assert tokenizer.special_tokens_map == reference["special_tokens"]
        assert tokenizer.chat_template == reference["chat_template"]
        assert tokenizer.padding_side == reference["padding_side"]
        ids = probe_token_ids(tokenizer, reference["messages"])
        assert ids == reference["inputs"]["input_ids"][0].tolist()
        if args.qlora:
            base = train.load_base_model(
                config,
                quantization_config=train.build_quantization_config(torch.bfloat16),
                dtype=torch.bfloat16,
            )
            base = train.prepare_qlora_model(
                base,
                use_gradient_checkpointing=config["training"]["gradient_checkpointing"],
            )
            model = train.attach_lora(base, config)
        else:
            model = train.attach_lora(
                train.load_base_model(config, dtype=torch.bfloat16), config
            ).to("cuda")
        train.describe_model(model)
        assert model.config._attn_implementation == reference["attention_backend"]
        state = peft.get_peft_model_state_dict(model, save_embedding_layers=False)
        assert state.keys() == reference["state"].keys()
        assert all(torch.equal(value.cpu(), reference["state"][key]) for key, value in state.items())
        frozen = {name: parameter_digest(p) for name, p in model.named_parameters() if not p.requires_grad}
        assert frozen == reference["frozen_hashes"]
        logits = probe_last_logits(model, {k: v.to("cuda") for k, v in reference["inputs"].items()})
        save_file({"logits": logits}, adapter_dir.parent / "roundtrip_reloaded_logits.safetensors")
        torch.testing.assert_close(logits, reference["logits"], atol=1e-5, rtol=1e-5)
        files = {p.name: p.stat().st_size for p in adapter_dir.iterdir() if p.is_file()}
        assert "config.json" not in files
        assert not list(adapter_dir.glob("model*.safetensors"))
        assert not list(adapter_dir.glob("*.bin"))
        report = {
            "passed": True,
            "mode": "QLoRA" if args.qlora else "BF16 LoRA",
            "adapter_dir": str(adapter_dir), "files_bytes": files,
            "adapter_tensors_checked": len(state), "frozen_base_tensors_checked": len(frozen),
            "tokenizer_equal": True, "trainable_reload_checked": True,
            "probe_messages": reference["messages"], "input_ids": ids,
            "compared_logits": "last input position, full vocabulary",
            "logits_max_abs_diff": (logits - reference["logits"]).abs().max().item(),
            "logits_exact_equal": torch.equal(logits, reference["logits"]),
            "atol": 1e-5, "rtol": 1e-5,
            "attention_backend": reference["attention_backend"],
            "inference_precision": "BF16 autocast for both pre-save and reloaded model",
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (adapter_dir.parent / "adapter_roundtrip_verification.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"PROBE: adapter roundtrip passed; logits max diff={report['logits_max_abs_diff']}", flush=True)


if __name__ == "__main__":
    main()
