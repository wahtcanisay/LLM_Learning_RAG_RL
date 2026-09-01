"""Finalize the frozen Stage-A files without rebuilding retired cleaning stages."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BAD_GENERAL_IDS = {
    "fd7ef9554a46e565b2b1880e143ea1d4d30078c99948d484cb43622eb56bfd4b",
    "79c037161f347b782bc241857bb07d1818f3a79992b4d8cfe88a3d1998ef54b2",
}
REPLACEMENT_IDS = (
    "19db9c590e7c7dff434e95ae312ac919aef007e899bc6e2b474da53ca94a0342",
    "d13e455d15d612c386889cf164acfeea16938981232d7134e067d9b51a743d0e",
)


def _has_replacement_character(row):
    return any("\ufffd" in message.get("content", "") for message in row.get("messages", []))


def finalize_rows(rows, bad_ids, replacements):
    """Add stable reasoning IDs and replace explicitly rejected general rows."""
    output = []
    removed = 0
    for source in rows:
        row = dict(source)
        if row.get("id") in bad_ids:
            removed += 1
            continue
        if not row.get("id") and row.get("mixture_group") == "reasoning":
            row["id"] = f"medical_o1_{row['provenance']['index']}"
        output.append(row)

    existing = {row.get("id") for row in output}
    additions = [dict(row) for row in replacements if row.get("id") not in existing]
    if any(_has_replacement_character(row) for row in additions):
        raise ValueError("replacement contains U+FFFD")
    if removed != len(additions):
        raise ValueError(f"removed {removed} rows but received {len(additions)} new replacements")
    output.extend(additions)
    if len({row.get("id") for row in output}) != len(output):
        raise ValueError("final IDs are missing or duplicated")
    return output


def _read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _load_replacements(raw_dir, model_path):
    import pyarrow.dataset as ds
    from transformers import AutoTokenizer

    dataset = ds.dataset(str(raw_dir), format="parquet")
    raw_rows = dataset.to_table(filter=ds.field("prompt_id").isin(REPLACEMENT_IDS)).to_pylist()
    by_id = {row["prompt_id"]: row for row in raw_rows}
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    replacements = []
    for row_id in REPLACEMENT_IDS:
        row = by_id[row_id]
        tokens = len(tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False))
        if tokens > 1024:
            raise ValueError(f"replacement {row_id} has {tokens} tokens")
        replacements.append({
            "id": row_id,
            "messages": row["messages"],
            "source": "ultrachat_200k",
            "sample_type": "general_dialogue",
            "mixture_group": "general",
            "provenance": {
                "source": "HuggingFaceH4/ultrachat_200k",
                "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
                "split": "train_sft",
                "prompt_id": row_id,
                "replacement_for_corrupt_source": True,
            },
        })
    return replacements


def update_manifest(manifest):
    output = dict(manifest)
    output["finalization"] = {
        "reasoning_id_format": "medical_o1_{provenance.index}",
        "corrupt_general_replacements": len(BAD_GENERAL_IDS),
        "removed_ids": sorted(BAD_GENERAL_IDS),
        "replacement_ids": list(REPLACEMENT_IDS),
    }
    return output


def main(root=PROJECT_ROOT):
    data = root / "data" / "processed" / "sft_stage_a"
    train_path, dev_path = data / "stage_a_train.jsonl", data / "stage_a_dev.jsonl"
    train, dev = _read_jsonl(train_path), _read_jsonl(dev_path)
    replacements = _load_replacements(
        root / "data" / "raw" / "general" / "ultrachat_200k",
        root.parent / "models" / "Qwen2.5-3B-Instruct",
    )
    train = finalize_rows(train, BAD_GENERAL_IDS, replacements)
    dev = finalize_rows(dev, BAD_GENERAL_IDS, [])
    _write_jsonl(train_path, train)
    _write_jsonl(dev_path, dev)
    manifest_path = data / "stage_a_manifest.json"
    manifest = update_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"train": len(train), "dev": len(dev), "bad_ids": sorted(BAD_GENERAL_IDS), "replacement_ids": list(REPLACEMENT_IDS)}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
