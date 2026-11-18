"""Historical cleanup procedure; the 2026-10-13 run is already complete."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path.cwd().resolve()
PROJECT = (ROOT / "MedSearch").resolve()
ARCHIVE = PROJECT / "docs/archive/retired_scripts_20261013.zip"
MANIFEST = PROJECT / "docs/audits/cleanup_20261013.json"


def checked(path, parent):
    resolved = path.resolve(strict=True)
    assert resolved.is_relative_to(parent.resolve()) and resolved != parent.resolve(), resolved
    assert path.is_file() and not path.is_symlink(), path
    return path


def record(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


scripts = sorted(p for p in (PROJECT / "docs/tests").iterdir() if p.is_file() and p.suffix in (".py", ".sh"))
scripts += sorted((PROJECT / "scripts").glob("run_medsearch_dapo*.sh"))
scripts += [PROJECT / "run_medsearch_dapo_v9_verifier.sh"]
scripts += [PROJECT / "tools/smoke_sft_top32.py",
            ROOT / "tmp/run_sft_top32_smoke.py", ROOT / "tmp/inspect_sft_action_context.py",
            ROOT / "tmp/inventory_medsearch_cleanup.py"]
scripts = [checked(p, ROOT) for p in scripts]
assert len(scripts) == len(set(scripts))
script_records = [record(p) for p in scripts]

obsolete_checkpoint_roots = [PROJECT / "outputs/distillation" / name for name in (
    "sft_checkpoint_smoke_20261013", "sft_checkpoint_smoke_20261013_probe2",
    "sft_checkpoint_smoke_20261013_verified")]
checkpoints = []
for directory in obsolete_checkpoint_roots:
    assert directory.resolve().is_relative_to((PROJECT / "outputs/distillation").resolve())
    assert not directory.is_symlink()
    checkpoints += [checked(p, directory) for p in directory.rglob("*") if p.is_file()]
checkpoint_records = [record(p) for p in checkpoints]

assert not ARCHIVE.exists() and not MANIFEST.exists(), "Never overwrite a cleanup record"
ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in scripts:
        archive.write(path, path.relative_to(ROOT).as_posix())
    # Keep configuration evidence for failed runs, without disposable weight/state binaries.
    for path in checkpoints:
        if path.suffix in (".json", ".md", ".jinja"):
            archive.write(path, path.relative_to(ROOT).as_posix())
with zipfile.ZipFile(ARCHIVE) as archive:
    assert archive.testzip() is None
    for item in script_records:
        assert hashlib.sha256(archive.read(item["path"])).hexdigest() == item["sha256"]

manifest = {
    "status": "planned", "date": "2026-10-13", "scripts_archive": ARCHIVE.relative_to(ROOT).as_posix(),
    "reason": "retired Stage A/B suite, superseded DAPO launchers, completed no-save smoke and temporary probes",
    "removed_scripts": script_records, "removed_failed_checkpoint_files": checkpoint_records,
    "preserved_verified_checkpoint": "MedSearch/outputs/distillation/sft_checkpoint_smoke_20261013_final",
    "preserved_current_tests": "MedSearch/tests", "dev_evaluation": "deferred by user",
}
MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
for path in scripts + checkpoints:
    expected = next(item for item in script_records + checkpoint_records if item["path"] == path.relative_to(ROOT).as_posix())
    assert record(path) == expected, f"File changed during audit: {path}"
    checked(path, ROOT).unlink()
for directory in obsolete_checkpoint_roots:
    for child in sorted((p for p in directory.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        assert child.resolve().is_relative_to(directory.resolve()) and not child.is_symlink()
        child.rmdir()
    directory.rmdir()
assert all(not (ROOT / item["path"]).exists() for item in script_records + checkpoint_records)
assert (PROJECT / "outputs/distillation/sft_checkpoint_smoke_20261013_final/checkpoint-1/adapter_model.safetensors").is_file()
manifest["status"] = "completed"
manifest["released_bytes"] = sum(item["bytes"] for item in script_records + checkpoint_records) - ARCHIVE.stat().st_size
MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"scripts_removed": len(scripts), "failed_checkpoint_files_removed": len(checkpoints),
                  "released_mib": round(manifest["released_bytes"] / 1024**2, 2), "archive": str(ARCHIVE),
                  "manifest": str(MANIFEST)}, ensure_ascii=False))
