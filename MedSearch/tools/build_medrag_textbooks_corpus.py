"""Merge the selected MedRAG textbook chunks into Search-R1 JSONL format."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("MedSearch/data/medrag_textbooks_subset.manifest"))
    parser.add_argument("--output", type=Path, default=Path("MedSearch/data/retrieval/medrag_textbooks.jsonl"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as dst:
        for raw_path in args.manifest.read_text(encoding="utf-8").splitlines():
            raw_path = raw_path.strip()
            if not raw_path or raw_path.startswith("#"):
                continue
            source = Path(raw_path).stem
            with Path(raw_path).open(encoding="utf-8") as src:
                for line in src:
                    item = json.loads(line)
                    contents = item.get("contents") or f"{item['title']}. {item['content']}"
                    dst.write(json.dumps({
                        "id": f"{source}:{item['id']}",
                        "title": item["title"],
                        "content": item["content"],
                        "contents": contents,
                    }, ensure_ascii=False) + "\n")
                    count += 1
    print(f"wrote={count} output={args.output} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
