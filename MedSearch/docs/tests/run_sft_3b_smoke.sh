#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--inside" ]]; then
    exec docker exec -w '/workspace/code_list/some tricks/LLMLeanring' \
        -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -e TOKENIZERS_PARALLELISM=false \
        llm-pytorch bash MedSearch/docs/tests/run_sft_3b_smoke.sh --inside "$@"
fi
shift

mkdir -p MedSearch/docs/run_logs
log_dir=$(mktemp -d MedSearch/docs/run_logs/3b-smoke-XXXXXXXX)
printf 'Teacher run logs: %s\n' "$log_dir"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
    --format=csv -lms 200 -f "$log_dir/gpu_memory.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT
python -u MedSearch/docs/tests/run_sft_3b_smoke.py "$@" 2>&1 | tee "$log_dir/run.log"
