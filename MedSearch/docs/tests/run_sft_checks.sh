#!/usr/bin/env bash
set -euo pipefail

# Teacher-only runner: existing container, local assets, CPU checks, no downloads.
if [ "$#" -eq 0 ]; then
    set -- MedSearch/docs/tests
fi
docker exec -w '/workspace/code_list/some tricks/LLMLeanring' \
    -e CUDA_VISIBLE_DEVICES= -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e TOKENIZERS_PARALLELISM=false \
    llm-pytorch python -m pytest -q -p no:cacheprovider "$@"
