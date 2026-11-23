import json
from pathlib import Path

manifest = Path("MedSearch/data/medrag_textbooks_full.manifest")
output = Path("MedSearch/data/retrieval/medrag_textbooks.jsonl")
output.parent.mkdir(parents=True, exist_ok=True)
count = 0
with output.open("w", encoding="utf-8") as dst:
    for raw_path in manifest.read_text(encoding="utf-8").splitlines():
        raw_path = raw_path.strip()
        if not raw_path or raw_path.startswith("#"):
            continue
        with Path(raw_path).open(encoding="utf-8") as src:
            for line in src:
                item = json.loads(line)
                dst.write(json.dumps({
                    "id": str(count),
                    "title": item["title"],
                    "content": item["content"],
                    "contents": item.get("contents") or f"{item['title']}. {item['content']}",
                }, ensure_ascii=False) + "\n")
                count += 1
print(f"wrote={count} output={output} bytes={output.stat().st_size}")
