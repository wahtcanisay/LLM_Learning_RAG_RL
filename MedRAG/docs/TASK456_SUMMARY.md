# MedRAG Task 4～6 实验与学习总结

日期：2026-07-20

这三个任务使用同一批 3 条 toy JSONL 文档，依次验证：

```text
BM25/Lucene 建库与查询
→ Dense Embedding/FAISS 建库与查询
→ BM25 + Dense 的 RRF 融合
```

本次实验的目标是理解数据流和索引映射，不是生成正式 MedRAG 基线指标。

## 一、输入数据：toy JSONL

文件：`docs/task4_toy_collection/toy.jsonl`

```json
{"id":"toy_0","title":"Neurology","content":"The facial nerve controls muscles of facial expression.","contents":"Neurology. The facial nerve controls muscles of facial expression."}
{"id":"toy_1","title":"Cardiology","content":"Atrial fibrillation is an irregular heart rhythm.","contents":"Cardiology. Atrial fibrillation is an irregular heart rhythm."}
{"id":"toy_2","title":"Infectious disease","content":"Antibiotics are used to treat bacterial infections.","contents":"Infectious disease. Antibiotics are used to treat bacterial infections."}
```

四个字段的作用：

- `id`：chunk 的唯一标识；
- `title`：文档标题；
- `content`：正文；
- `contents`：标题和正文的拼接版本，作为 BM25 或 Dense 的主要输入文本。

这个文件是原始数据集合，不是索引。索引建立后仍然需要保留它，因为检索结果通常只提供位置、ID 和分数，最终文本还要从 JSONL 找回。

## 二、Task 4：BM25/Lucene

### 2.1 建库命令

在容器中执行：

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

参数含义：

| 参数 | 含义 |
|---|---|
| `--collection JsonCollection` | 输入目录中的文件是 JSON/JSONL 集合 |
| `--input` | 读取 toy JSONL 的目录 |
| `--index` | Lucene 索引输出目录 |
| `--generator DefaultLuceneDocumentGenerator` | 将 JSONL 文档转换成 Lucene 文档 |
| `--threads 1` | 使用一个索引线程，便于观察 toy 实验 |

### 2.2 建库输出

重要日志：

```text
DocumentCollection path: docs/task4_toy_collection
CollectionClass: JsonCollection
Generator: DefaultLuceneDocumentGenerator
Threads: 1
Language: en
Stemmer: porter
Keep stopwords? false
Index path: /tmp/medrag_task4_index
1 file found
3 docs added
Indexing Complete! 3 documents indexed
indexed:                3
unindexable:            0
empty:                  0
skipped:                0
errors:                 0
```

解释：

- `Language: en`：使用英文分析器；
- `Stemmer: porter`：使用 Porter 词干化；
- `Keep stopwords? false`：停用词不保留；
- `1 file found`：输入目录中发现一个 JSONL 文件；
- `3 docs added`：三行 JSONL 被视为三个 Lucene 文档；
- `unindexable/empty/skipped/errors` 全为 0：没有文档被拒绝、跳过或处理失败。

日志中的：

```text
Store document "contents" field? false
```

表示 `contents` 被用于建立倒排索引，但完整文本没有作为 Lucene stored field 保存。因此原始 JSONL 不能删除，后面仍要通过 `source/index` 找回文本。

### 2.3 BM25 查询命令

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
    item = json.loads(
        (collection / f"{source}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[index]
    )
    print({
        "rank": rank,
        "docid": hit.docid,
        "score": hit.score,
        "source": source,
        "index": index,
        "title": item["title"],
    })
PY
```

输出：

```python
{
    'rank': 1,
    'docid': 'toy_0',
    'score': 1.1835999488830566,
    'source': 'toy',
    'index': 0,
    'title': 'Neurology'
}
```

映射过程：

```text
Lucene docid toy_0
→ source = toy
→ index = 0
→ toy.jsonl 第 0 行
→ Neurology chunk
```

这里 BM25 只返回 `toy_0`，因为另外两条文档没有 `facial` 或 `nerve` 这些匹配词。`k=3` 表示最多返回三条，不保证一定返回三条。

### 2.4 BM25 warning

出现过：

```text
WARNING: You are running with Java 20 or later...
SyntaxWarning: invalid escape sequence '\\s'
```

它们分别是旧版 Lucene 的性能提示和 Pyserini 旧 Python 代码的语法提示，不影响本次索引和查询结果。

## 三、Task 5：Dense Embedding + FAISS

### 3.1 执行命令

```bash
python docs/task5_toy_dense.py
```

脚本位置：`docs/task5_toy_dense.py`

使用模型：

```text
sentence-transformers/all-MiniLM-L6-v2
```

这只是小型模型，用于验证 Dense/FAISS 数据流，不是 MedRAG 正式的 MedCPT 基线。

### 3.2 脚本做的事情

```text
读取 toy.jsonl
→ 取出 contents
→ SentenceTransformer 编码
→ 文档向量矩阵
→ 行归一化
→ FAISS IndexFlatIP.add()
→ 写入 embeddings.npy
→ 写入 metadatas.jsonl
→ 写入 faiss.index
→ 用同一模型编码 query
→ FAISS.search()
→ indices 映射回 metadata 和 JSONL
```

输出文件位于：

```text
/tmp/medrag_task5_dense/embeddings.npy
/tmp/medrag_task5_dense/metadatas.jsonl
/tmp/medrag_task5_dense/faiss.index
```

`metadatas.jsonl` 的核心映射是：

```json
{"index":0,"source":"toy"}
{"index":1,"source":"toy"}
{"index":2,"source":"toy"}
```

它的行顺序必须和 `index.add()` 添加向量的顺序完全一致。

### 3.3 Dense 输出

```python
{'rank': 1, 'faiss_position': 0, 'score': 0.7347357273101807, 'source': 'toy', 'index': 0, 'id': 'toy_0', 'title': 'Neurology'}
{'rank': 2, 'faiss_position': 2, 'score': 0.11410441994667053, 'source': 'toy', 'index': 2, 'id': 'toy_2', 'title': 'Infectious disease'}
{'rank': 3, 'faiss_position': 1, 'score': 0.09615442156791687, 'source': 'toy', 'index': 1, 'id': 'toy_1', 'title': 'Cardiology'}
```

`faiss_position` 是 FAISS 内部向量位置，不是最终文本。完整映射是：

```text
faiss_position = 2
→ metadatas.jsonl 第 2 行
→ source = toy, index = 2
→ toy.jsonl 第 2 行
→ toy_2 / Infectious disease
```

因为文档向量和 query 向量都做了归一化，`IndexFlatIP` 的内积可以近似看成 cosine similarity。分数越大越相似。

### 3.4 Dense warning

模型第一次运行会下载权重和 tokenizer。出现过的 `cached_download`、`clean_up_tokenization_spaces` warning 都是依赖未来版本提示，不是编码或 FAISS 失败。

## 四、Task 6：RRF 融合

### 4.1 执行命令

```bash
python docs/task6_toy_rrf.py
```

这个脚本复用 Task 4 和 Task 5 的索引，并直接调用：

```python
RetrievalSystem.merge()
```

它没有实例化真实 `Retriever`，避免再次触发语料下载或建库。

### 4.2 两路原始结果

```text
BM25_RESULTS [('toy_0', 1.1835999488830566)]
DENSE_RESULTS [('toy_0', 0.7347357273101807), ('toy_2', 0.11410441994667053), ('toy_1', 0.09615442156791687)]
```

注意：BM25 的 `1.1836` 和 Dense 的 `0.7347` 不可以直接相加，因为它们来自不同的分数空间。

### 4.3 RRF 输出

```python
{'rank': 1, 'id': 'toy_0', 'rrf_score': 0.019801980198019802, 'title': 'Neurology'}
{'rank': 2, 'id': 'toy_2', 'rrf_score': 0.00980392156862745, 'title': 'Infectious disease'}
{'rank': 3, 'id': 'toy_1', 'rrf_score': 0.009708737864077669, 'title': 'Cardiology'}
```

代码使用：

```text
rrf_k = 100
RRF contribution = 1 / (rrf_k + rank + 1)
```

代码中的 `rank` 从 0 开始，因此：

```text
toy_0：BM25 第 1 名 + Dense 第 1 名
      = 1/101 + 1/101
      = 0.019801980198019802

toy_2：只出现在 Dense 第 2 名
      = 1/102
      = 0.00980392156862745

toy_1：只出现在 Dense 第 3 名
      = 1/103
      = 0.009708737864077669
```

Task 6 验证了：

```text
各检索器分别排序
→ 按文档 ID 去重
→ 同一文档累加不同排名贡献
→ 按 RRF 分数排序
→ 截取最终 Top-k
```

### 4.4 Task 6 的边界

本次 RRF 中的 Dense 结果来自 MiniLM toy 向量，不是 MedCPT；因此结果用于验证 `merge()` 的数据结构和排名融合逻辑，不能称为 MedRAG 正式实验。

## 五、Task 4～6 的统一理解

```text
同一批 JSONL chunk
        │
        ├── BM25：词项 → Lucene 倒排索引 → docid/score
        │
        └── Dense：Embedding → FAISS 向量索引 → indices/score
                         │
                         └── metadata → source/index → JSONL chunk

BM25 结果 + Dense 结果
        ↓
RetrievalSystem.merge()
        ↓
RRF 排名融合
        ↓
最终候选 chunk
```

三者的职责：

- Task 4：验证稀疏词项检索和 Lucene docid 映射；
- Task 5：验证向量编码、FAISS 位置和 metadata 映射；
- Task 6：验证不同检索器结果不能直接合并原始分数，而要通过 RRF 按排名融合。

## 六、当前还没有完成的内容

- 没有在完整 Textbooks/PubMed/Wikipedia 语料上建立正式 BM25 索引；
- 没有使用 MedCPT 建立正式 Dense 基线；
- 没有运行真实医学 QA 评测；
- 没有统计 Recall@k、MRR、QA Accuracy、延迟和显存峰值；
- 没有调用 LLM 生成答案；
- 没有加入 Reranker。

下一步可以在已经下载的真实语料上选择一个小规模子集，先建立正式 BM25，再建立 MedCPT Dense 索引，最后再进入检索结果到 Prompt 的生成链路。
