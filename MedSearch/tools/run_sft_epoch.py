"""Run the frozen cold-start epoch and record online W&B metrics."""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import wandb

from MedSearch.sft.train import PROJECT_ROOT, load_config, main as train_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['top32', 'ce'], default='top32')
    args = parser.parse_args()
    config_path = PROJECT_ROOT / f"MedSearch/configs/sft_{args.method}_epoch1.yaml"
    config = load_config(config_path)
    output = PROJECT_ROOT / config["training"]["output_dir"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    cache = PROJECT_ROOT / config["data_path"]
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    expected_digest = ("83dd481cbb2f248b407490367d787a42f0dbde021b70ddacd647c717ed45c4e5" if args.method == 'top32'
                       else json.loads((cache.parent / 'manifest.json').read_text())['ce_sha256'])
    assert digest == expected_digest
    assert pq.read_metadata(cache).num_rows == 1000
    assert config["eval_data_path"] is None
    gate_name = 'sft_checkpoint_smoke_20261013_final' if args.method == 'top32' else 'sft_ce_checkpoint_smoke_20261013'
    gate = PROJECT_ROOT / f"MedSearch/outputs/distillation/{gate_name}/report.json"
    assert json.loads(gate.read_text())["status"] == "passed"
    metadata = {
        "config": config, "samples": 1000, "cache_sha256": digest,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "source_sha256": {
            str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (PROJECT_ROOT / "MedSearch/sft").glob("*.py")
        },
        "teacher_revision": "74d46a43224daa612b5f27174e1c58b35587ee92",
        "loss_formula": "CE + 0.1 * Top32" if args.method == 'top32' else 'CE',
        "teacher_k": 32 if args.method == 'top32' else None, "temperature": 1,
        "dev_evaluated": False, "checkpoint_gate": str(gate.relative_to(PROJECT_ROOT)),
    }
    run = wandb.init(project="medsearch_r1", name=config["training"]["run_name"], config=metadata, mode="online")
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.yaml")
    metadata.update(wandb_url=run.url, wandb_run_id=run.id, status="running")
    report_path = output / "run_report.json"
    report_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        train_main(["--config", str(config_path)])
        metrics = json.loads((output / "train_metrics.json").read_text())
        assert metrics["global_step"] == 125
        checkpoint = output / "checkpoint-125"
        required = ["adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"]
        assert all((checkpoint / name).is_file() for name in required)
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        assert state["global_step"] == 125 and state["epoch"] == 1.0
        adapter = output / "adapter/adapter_model.safetensors"
        assert adapter.is_file()
        metadata.update(status="completed", metrics=metrics, log_history=state["log_history"],
                        elapsed_seconds=time.monotonic() - started,
                        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                        cuda_peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                        adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest())
        run.summary.update({**metrics, "cuda_peak_allocated_gib": metadata["cuda_peak_allocated_gib"],
                            "elapsed_seconds": metadata["elapsed_seconds"], "checkpoint_saved": True})
        report_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": metadata["status"], "metrics": metrics, "wandb_url": run.url,
                          "cuda_peak_allocated_gib": metadata["cuda_peak_allocated_gib"]}), flush=True)
    except BaseException as error:
        metadata.update(status="failed", error_type=type(error).__name__, elapsed_seconds=time.monotonic()-started)
        report_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        run.finish(exit_code=1)
        raise
    run.finish()


if __name__ == "__main__":
    main()
