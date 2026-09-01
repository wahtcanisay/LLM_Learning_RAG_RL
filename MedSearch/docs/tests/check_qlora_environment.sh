#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "--inside" ]]; then
    exec docker exec -w '/workspace/code_list/some tricks/LLMLeanring' \
        -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        llm-pytorch bash MedSearch/docs/tests/check_qlora_environment.sh --inside
fi
mkdir -p MedSearch/docs/run_logs
log_dir=$(mktemp -d MedSearch/docs/run_logs/qlora-env-XXXXXXXX)
printf 'QLoRA environment evidence: %s\n' "$log_dir"
python -u - <<'PY' 2>&1 | tee "$log_dir/check.log"
import json
import importlib.metadata
import torch
import bitsandbytes as bnb

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.is_bf16_supported()
torch.manual_seed(20260831)
torch.cuda.reset_peak_memory_stats()
layer = bnb.nn.Linear4bit(
    128, 64, bias=False, compute_dtype=torch.bfloat16,
    compress_statistics=True, quant_type="nf4",
).to("cuda")
assert isinstance(layer.weight, bnb.nn.Params4bit)
assert layer.weight.dtype == torch.uint8
assert layer.weight.quant_state is not None
assert layer.weight.quant_state.nested
with torch.no_grad():
    output = layer(torch.randn(2, 128, device="cuda", dtype=torch.bfloat16))
torch.cuda.synchronize()
assert output.shape == (2, 64) and torch.isfinite(output).all()
print(json.dumps({
    "passed": True, "scope": "tiny NF4 double-quant CUDA forward only; no Qwen loading or training",
    "versions": {name: importlib.metadata.version(name) for name in
                 ["bitsandbytes", "torch", "transformers", "accelerate", "peft", "numpy", "packaging"]},
    "torch_cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
    "compute_capability": torch.cuda.get_device_capability(0),
    "bitsandbytes_path": bnb.__file__, "weight_dtype": str(layer.weight.dtype),
    "double_quant": layer.weight.quant_state.nested, "output_dtype": str(output.dtype),
    "output_shape": list(output.shape), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
}, indent=2))
PY
