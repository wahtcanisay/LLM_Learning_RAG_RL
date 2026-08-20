"""Enable veRL's existing SHM weight transport behind an opt-in env var."""

from pathlib import Path
import sys


platform_file = Path(sys.argv[1]) / "verl" / "plugin" / "platform" / "platform_cuda.py"
source = platform_file.read_text(encoding="utf-8")
old = """    def is_ipc_supported(self) -> bool:
        return True
"""
new = """    def is_ipc_supported(self) -> bool:
        # CUDA IPC handles are not portable across every WSL2/Docker process
        # boundary. veRL already has a shared-memory fallback; expose it without
        # changing the default behavior on native Linux.
        return os.getenv("VERL_FORCE_SHM_WEIGHT_TRANSFER", "0") != "1"
"""

if new in source:
    print("veRL shared-memory override already installed")
elif source.count(old) == 1:
    platform_file.write_text(source.replace(old, new), encoding="utf-8")
    print("installed veRL shared-memory override")
else:
    raise SystemExit(f"unexpected upstream veRL CUDA platform source: {platform_file}")
