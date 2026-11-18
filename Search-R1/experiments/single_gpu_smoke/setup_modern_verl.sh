#!/usr/bin/env bash
set -euo pipefail

VERL_COMMIT='e78efedbe56591cd53ac107150abb7565bb9b8b1'
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
VERL_DIR="${PROJECT_ROOT}/.deps/verl"
VERL_SHM_PATCHER="${PROJECT_ROOT}/modern_verl/patch_verl_for_wsl.py"

if [[ ! -d "${VERL_DIR}/.git" ]]; then
    git clone --filter=blob:none https://github.com/verl-project/verl.git "${VERL_DIR}"
fi

git -c safe.directory="${VERL_DIR}" -C "${VERL_DIR}" fetch --depth 1 origin "${VERL_COMMIT}"
git -c safe.directory="${VERL_DIR}" -C "${VERL_DIR}" checkout --detach "${VERL_COMMIT}"

# WSL2/Docker can advertise CUDA IPC yet reject the handle in vLLM's spawned
# worker. Keep upstream's default on native Linux and add an opt-in SHM fallback.
python "${VERL_SHM_PATCHER}" "${VERL_DIR}"

# Keep the already verified RTX 5090 PyTorch/vLLM stack; only switch the veRL
# editable source and install the small Search-R1 adapter package.
python -m pip install --break-system-packages --no-deps --no-build-isolation -e "${VERL_DIR}"
python -m pip install --break-system-packages --no-deps --no-build-isolation -e "${PROJECT_ROOT}/modern_verl"
python -m pip install --break-system-packages -r "${PROJECT_ROOT}/modern_verl/requirements-runtime.txt"

python - <<'PY'
import importlib.metadata
import verl

print("verl_source", verl.__file__)
print("verl_version", importlib.metadata.version("verl"))
print("adapter_version", importlib.metadata.version("search-r1-modern-verl"))
PY
