#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "--inside" ]]; then
    exec docker exec -w '/workspace/code_list/some tricks/LLMLeanring' \
        -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -e TOKENIZERS_PARALLELISM=false \
        llm-pytorch bash MedSearch/docs/tests/check_qlora_loading.sh --inside "${1:-load}"
fi
mode="${2:-load}"
case "$mode" in
    load) prefix=qlora-load ;;
    --backward) prefix=qlora-backward ;;
    *) printf 'Unknown check mode: %s\n' "$mode" >&2; exit 2 ;;
esac
mkdir -p MedSearch/docs/run_logs
log_dir=$(mktemp -d "MedSearch/docs/run_logs/$prefix-XXXXXXXX")
printf 'QLoRA loading evidence: %s\n' "$log_dir"
python -u - "$mode" <<'PY' 2>&1 | tee "$log_dir/check.log"
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import bitsandbytes as bnb
import torch
from transformers import AutoTokenizer, set_seed
from MedSearch.sft import train

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
config = train.load_config("MedSearch/configs/stage_a.yaml")
set_seed(config["training"]["seed"])
quantization = train.build_quantization_config(torch.bfloat16)
assert quantization is not None and quantization.load_in_4bit
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
started = time.perf_counter()
model = train.load_base_model(
    config,
    quantization_config=quantization,
    dtype=torch.bfloat16,
)
torch.cuda.synchronize()
load_seconds = time.perf_counter() - started
assert model.is_loaded_in_4bit
layers = {name: layer for name, layer in model.named_modules() if isinstance(layer, bnb.nn.Linear4bit)}
assert len(layers) == model.config.num_hidden_layers * len(config["lora_target_modules"])
assert all(isinstance(layer.weight, bnb.nn.Params4bit) for layer in layers.values())
assert all(layer.weight.dtype == torch.uint8 for layer in layers.values())
assert all(layer.weight.quant_state.quant_type == "nf4" for layer in layers.values())
assert all(layer.weight.quant_state.nested for layer in layers.values())
assert all(p.device == torch.device("cuda:0") for p in model.parameters())
assert not any("lora_" in name for name, _ in model.named_parameters())
tokenizer = AutoTokenizer.from_pretrained(train.PROJECT_ROOT / config["model_path"], local_files_only=True)
details = {}
scope = "Qwen 3B 4-bit base loading and one forward; no LoRA/backward/training"
if sys.argv[1] == "--backward":
    def tensor_hash(tensor):
        return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    packed_before = {name: tensor_hash(layer.weight) for name, layer in layers.items()}
    states_before = {name: layer.weight.quant_state for name, layer in layers.items()}
    base = model
    model = train.prepare_qlora_model(model, use_gradient_checkpointing=True)
    assert model is base and model.is_gradient_checkpointing
    assert all(not p.requires_grad for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters() if not isinstance(p, bnb.nn.Params4bit))
    assert all(tensor_hash(layer.weight) == packed_before[name] for name, layer in layers.items())
    assert all(layer.weight.quant_state is states_before[name] for name, layer in layers.items())
    frozen = list(model.named_parameters())
    frozen_hashes = {name: tensor_hash(p) for name, p in frozen}
    # bitsandbytes Linear4bit.forward may cast its frozen bias to compute_dtype.
    bnb_bias_ids = {id(layer.bias) for layer in layers.values() if layer.bias is not None}
    bias_float_hashes = {name: tensor_hash(p.float()) for name, p in frozen if id(p) in bnb_bias_ids}
    print("Preparation passed; attaching existing LoRA interface", flush=True)
    model = train.attach_lora(model, config)
    description = train.describe_model(model)
    lora = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert len(lora) == 2 * len(layers)
    assert all("lora_" in name for name in lora)
    lora_before = {name: tensor_hash(p) for name, p in lora.items()}
    tokenizer.padding_side = "right"
    # One actual Stage-A sample, existing encoder/mask/collator, no data rewrite.
    probe_config = {**config, "batch_size": 1, "shuffle": False, "num_workers": 0}
    loader = train.build_dataloader(probe_config, tokenizer)
    inputs = {name: value.to("cuda:0") for name, value in next(iter(loader)).items()}
    supervised = int((inputs["labels"][:, 1:] != -100).sum())
    assert supervised > 0
    model.train()
    model.zero_grad(set_to_none=True)
    started_backward = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**inputs, use_cache=False)
    assert torch.isfinite(output.loss) and output.loss.requires_grad
    output.loss.backward()
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - started_backward
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in lora.values())
    nonzero = [name for name, p in lora.items() if torch.count_nonzero(p.grad).item() > 0]
    assert nonzero, "No LoRA gradient signal"
    # Initially B=0: A gradients may legitimately be zero on the first backward.
    assert all(not p.requires_grad and p.grad is None for _, p in frozen)
    changed = [(name, p) for name, p in frozen if tensor_hash(p) != frozen_hashes[name]]
    print(json.dumps({"base_byte_changes": [{"name": name, "dtype": str(p.dtype),
        "is_bnb_bias": id(p) in bnb_bias_ids} for name, p in changed]}), flush=True)
    # Accept ONLY the documented FP32 -> BF16 bias storage conversion, with
    # exact original FP32 numeric values restored; every other byte must match.
    assert all(id(p) in bnb_bias_ids and p.dtype == torch.bfloat16
               and tensor_hash(p.float()) == bias_float_hashes[name] for name, p in changed)
    assert all(tensor_hash(p) == lora_before[name] for name, p in lora.items())
    assert all(isinstance(layer.weight, bnb.nn.Params4bit) and layer.weight.dtype == torch.uint8
               and layer.weight.quant_state is states_before[name] for name, layer in layers.items())
    logits = output.logits
    scope = "Qwen 3B NF4 preparation, new LoRA and one Stage-A backward; no optimizer step/save/Trainer"
    details = {
        "gradient_checkpointing": model.is_gradient_checkpointing, "use_reentrant": False,
        "lora_config": {key: value for key, value in config.items() if key.startswith("lora_")},
        "parameter_counts": description, "lora_tensors": len(lora),
        "finite_lora_gradient_tensors": len(lora), "nonzero_lora_gradient_tensors": len(nonzero),
        "nonzero_lora_a": sum("lora_A" in name for name in nonzero),
        "nonzero_lora_b": sum("lora_B" in name for name in nonzero),
        "frozen_base_tensors": len(frozen), "base_grads_absent": True,
        "base_value_preserved": True, "base_byte_hash_changes": len(changed),
        "bnb_bias_dtype_casts": len(changed), "other_base_byte_hashes_unchanged": True,
        "adapter_hashes_unchanged": True,
        "packed_weights_preserved_by_preparation": True, "quantization_state_identity_preserved": True,
        "loss": output.loss.item(), "forward_backward_seconds": backward_seconds,
        "batch_shape": list(inputs["input_ids"].shape), "max_length": config["max_length"],
        "supervised_shifted_tokens": supervised, "sample_index": 0,
        "sample_id": loader.dataset.samples[0].get("id"), "data_path": config["data_path"],
        "data_sha256": hashlib.sha256((train.PROJECT_ROOT / config["data_path"]).read_bytes()).hexdigest(),
        "optimizer_steps": 0,
    }
else:
    messages = [{"role": "user", "content": "What does a thermometer measure?"}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to("cuda:0")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**inputs, use_cache=False).logits
torch.cuda.synchronize()
assert logits.shape[:2] == inputs["input_ids"].shape
assert logits.shape[-1] == model.config.vocab_size
assert torch.isfinite(logits).all()
source = Path(train.__file__)
print(json.dumps({
    "passed": True, "scope": scope, **details,
    "model_path": config["model_path"], "dtype": config["dtype"],
    "seed": config["training"]["seed"], "quantization": quantization.to_dict(),
    "is_loaded_in_4bit": model.is_loaded_in_4bit, "linear4bit_layers": len(layers),
    "weight_storage_dtype": "torch.uint8", "all_nf4_double_quant": True,
    "all_parameters_on_cuda0": True,
    "actual_parameter_devices": sorted({str(p.device) for p in model.parameters()}),
    "logits_shape": list(logits.shape), "finite_logits": True,
    "load_seconds": load_seconds, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    "versions": {name: importlib.metadata.version(name) for name in
                 ["torch", "transformers", "bitsandbytes", "accelerate", "peft"]},
    "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "train_py_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "data_py_sha256": hashlib.sha256(Path("MedSearch/sft/data.py").read_bytes()).hexdigest(),
    "preprocess_py_sha256": hashlib.sha256(Path("MedSearch/sft/preprocess.py").read_bytes()).hexdigest(),
    "probe_sha256": hashlib.sha256(Path("MedSearch/docs/tests/check_qlora_loading.sh").read_bytes()).hexdigest(),
}, ensure_ascii=False, indent=2))
PY
