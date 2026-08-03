import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from medical_graphrag.data.schemas import Document


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def content_hash(title: str, content: str) -> str:
    value = normalize_text(title) + "\n" + normalize_text(content)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _shard_rng(seed: int, name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def reservoir_sample(path: Path, count: int, seed: int) -> list[dict[str, object]]:
    rng = _shard_rng(seed, path.name)
    reservoir: list[dict[str, object]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            seen += 1
            if len(reservoir) < count:
                reservoir.append(item)
            else:
                index = rng.randrange(seen)
                if index < count:
                    reservoir[index] = item
    return reservoir


def sample_distractors(
    shard_dir: Path,
    *,
    seed: int,
    shard_count: int,
    per_shard: int,
    target_count: int,
    excluded_titles: set[str],
    excluded_content_hashes: set[str],
) -> tuple[list[Document], list[str]]:
    shards = sorted(path for path in shard_dir.glob("*.jsonl") if path.stat().st_size > 0)
    if not shards:
        raise ValueError(f"no non-empty JSONL shards found in {shard_dir}")
    order = shards.copy()
    random.Random(seed).shuffle(order)
    selected_names: list[str] = []
    candidates: dict[str, Document] = {}

    for path in order:
        selected_names.append(path.name)
        for raw in reservoir_sample(path, per_shard, seed):
            local_id = str(raw.get("id", "")).strip()
            title = str(raw.get("title", "")).strip()
            content = str(raw.get("content", "")).strip()
            if not local_id or not title or not content:
                continue
            digest = content_hash(title, content)
            if normalize_text(title) in excluded_titles or digest in excluded_content_hashes:
                continue
            candidates.setdefault(
                digest,
                Document(f"MEDRAG:{local_id}", title, content, "medrag_pubmed", None),
            )
        minimum_shards_used = min(shard_count, len(shards))
        if len(selected_names) >= minimum_shards_used and len(candidates) >= target_count:
            break

    ranked = sorted(
        candidates.values(),
        key=lambda item: hashlib.sha256(f"{seed}:{item.doc_id}".encode("utf-8")).hexdigest(),
    )
    if len(ranked) < target_count:
        raise ValueError(f"only {len(ranked)} unique distractors available; need {target_count}")
    return ranked[:target_count], selected_names
