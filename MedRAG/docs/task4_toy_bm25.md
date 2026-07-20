# Task 4：最小 BM25/Lucene 闭环

在 MedRAG 容器中执行，不要使用完整医学语料：

```bash
cd "/workspace/code_list/some tricks/LLMLeanring/MedRAG"
rm -rf /tmp/medrag_task4_index
mkdir -p /tmp/medrag_task4_index

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input docs/task4_toy_collection \
  --index /tmp/medrag_task4_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1
```

查询并核对 `docid → source/index → JSONL`：

```bash
python - <<'PY'
import json
from pathlib import Path
from pyserini.search.lucene import LuceneSearcher

collection = Path("docs/task4_toy_collection")
searcher = LuceneSearcher("/tmp/medrag_task4_index")
hits = searcher.search("facial nerve", k=3)

for rank, hit in enumerate(hits, start=1):
    parts = hit.docid.split("_")
    source = "_".join(parts[:-1])
    index = int(parts[-1])
    jsonl_path = collection / f"{source}.jsonl"
    item = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[index])
    print({"rank": rank, "docid": hit.docid, "score": hit.score,
           "source": source, "index": index, "title": item["title"]})
PY
```

需要保留的输出：

- 建库命令的成功日志；
- 每个 hit 的 `docid` 和 `score`；
- 通过 `source/index` 找回的 `title`；
- 第一名是否为 `toy_0` / `Neurology`。

这次只验证 BM25 的 Lucene 后端，不建立 Dense/FAISS 索引。
