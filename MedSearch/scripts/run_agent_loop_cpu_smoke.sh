#!/usr/bin/env bash
set -euo pipefail

project_dir="/workspace/code_list/some tricks/LLMLeanring"
export PYTHONPATH="$project_dir/MedSearch/.deps/verl:$project_dir"
cd "$project_dir"
python -m pytest MedSearch/tests/test_agent_loop_smoke.py -q
