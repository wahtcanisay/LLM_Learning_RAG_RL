"""Stream local Textbooks and StatPearls into one numeric-id BM25 collection."""

import argparse
import json
from pathlib import Path


def build_combined_corpus(textbooks: Path, statpearls: Path, output: Path) -> int:
    if output.exists():
        raise FileExistsError(output)
    sources = [textbooks, *sorted(statpearls.glob("*.jsonl"))]
    if len(sources) == 1:
        raise ValueError(f"no StatPearls chunks in {statpearls}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    count = 0
    with partial.open("x", encoding="utf-8") as destination:
        for source in sources:
            with source.open(encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    destination.write(json.dumps({
                        "id": str(count),
                        "title": item["title"],
                        "content": item["content"],
                        "contents": item.get("contents") or f"{item['title']}. {item['content']}",
                    }, ensure_ascii=False) + "\n")
                    count += 1
    partial.replace(output)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--textbooks", type=Path, required=True)
    parser.add_argument("--statpearls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = build_combined_corpus(args.textbooks, args.statpearls, args.output)
    print(f"wrote={count} bytes={args.output.stat().st_size} output={args.output}")
