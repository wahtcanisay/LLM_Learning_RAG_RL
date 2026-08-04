# MedicalGraphRAG —— 医学问答检索系统研究项目

> 一句话：**教电脑"怎么在医学资料里找答案"**，并且一步步用真实实验验证"哪种找法最靠谱"。

用生活化的话说：想象一个医学图书管理员，你问它"阿司匹林能缓解疼痛吗？"，它要先在 5000 本医学书里找到相关段落，再把段落交给一个"阅读助手"生成答案。我们这个项目研究的，就是**管理员该用什么策略找书**——是查关键词？理解意思？还是顺着概念网络跳？——并且用数据说话，哪种策略在什么场景下更好。

---

## 1. 我们最终要造什么

最终目标是一个 **MedSearch-R1** 医学搜索智能体：模型先形成初步医学判断，不确定就**主动去检索证据**，证据够了再提交带引用的答案。这就像"一个会自己查资料再答题的医生实习生"。

为了造它，项目分 5 个阶段，每个阶段打好一层地基：

```
阶段 1：MedRAG   —— 医学检索基础（语料 + 关键词/向量检索）      ✅ 完成
阶段 2：LinearRAG —— 图结构检索（顺着概念网络找）              ✅ 完成
阶段 3：MedicalGPT —— 让模型懂医学（领域微调 SFT）             🔜 进行中
阶段 4：Search-R1 —— 让模型学会"该搜就搜、够了就停"（强化学习）
阶段 5：MedSearch-R1 —— 全部组合成最终智能体
```

---

## 2. 零基础必懂概念

### 2.1 什么是 RAG（检索增强生成）？

**RAG = 先检索资料，再让大模型根据资料作答。**

比喻：闭卷考试 vs 开卷考试。普通大模型是"闭卷"——凭记忆答，容易忘、容易编。RAG 是"开卷"——先翻到相关章节（检索），再看着章节答题（生成）。所以 RAG 的核心第一步是**检索**：怎么在资料库里找到对的那几篇。

### 2.2 我们研究的 4 种检索器（"找书策略"）

| 检索器 | 比喻 | 原理 | 强项 |
|---|---|---|---|
| **BM25** | 图书馆按关键词查索引 | 数你问题里的词在书里出现几次、有多罕见 | 术语明确的问题（"阿司匹林"） |
| **Dense** | 按"意思"找人推荐 | 把文字变成向量（一串数字），找向量最接近的 | 换个说法也能匹配（"解热镇痛药"） |
| **Hybrid** | 两个管理员的结果合并 | 把 BM25 和 Dense 的排名用 RRF 融合 | 两路都强时互补 |
| **Graph** | 顺着"概念网络"跳 | 建一张"实体↔文章"图，从问题提到的实体沿边传播分数 | 多跳、跨实体问题（理论上） |

- **BM25**（关键词）：就像搜索引擎，只看词面匹配，快但死板。
- **Dense**（向量/语义）：把每段文字压缩成一串数字（向量），找"意思最像"的。能理解同义词，但更慢。
- **Hybrid**（RRF 融合）：把上面两个的排名列表合并——每个文档按"在两个列表里各排第几"打分相加。不比较原始分数（两路分数不可比），只比排名。
- **Graph**（图检索）：先做命名实体识别（找出文章里的疾病、药物），建一张"实体—文章"图，问题里的实体沿着图的边扩散分数，最后给文章排名。

### 2.3 怎么读指标（怎么判断"找得准不准"）

我们有 1000 道题，每道题知道**哪篇文章是正确答案**（gold）。检索器给每道题排出一个文章列表，我们看：

| 指标 | 生活化解释 | 数值含义（例子） |
|---|---|---|
| **Recall@k** | 正确答案里，有多少比例出现在前 k 名 | 0.9 意思：90% 的正确答案在前 k 名里 |
| **MRR@10** | 第一个正确答案平均排第几 | 0.5 ≈ 平均第 2 名就找到第一个对的 |
| **nDCG@10** | 相关的东西是不是都排前面（越相关越靠前越好） | 0.3 已算不错，完美是 1.0 |

注意：这些是**检索**指标（找到的文章对不对），不是**问答**指标（答得对不对）——检索命中 ≠ 答案正确，这是两件事。

### 2.4 我们在哪两个"考场"上考

**考场 A：pubmedqa_hard_v1（自建，单答案）**
- 1000 道医学选择题，5000 篇医学摘要（论文），7562 个片段；
- 每题恰好 1 篇正确答案；
- 难点：干扰项也是医学摘要，很接近。

**考场 B：nfcorpus_v1（BEIR 公开基准，多答案）**
- 323 道医学检索题，3633 篇医学文章；
- 每题平均 **38 篇**相关文章（多答案！）；
- 这是国际通用的检索基准（BEIR），有官方标注。

---

## 3. 实验结果（诚实版）

### 考场 A：pubmedqa_hard_v1（500 题测试集）

| 方法 | Recall@1 | Recall@10 | MRR@10 | nDCG@10 | 延迟 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.926 | 0.984 | 0.9458 | 0.9551 | 1.4ms |
| **Dense** | **0.966** | **0.994** | **0.9778** | **0.9819** | 16ms |
| Hybrid | 0.960 | 0.992 | 0.9738 | 0.9785 | 离线 |
| Graph | 0.800 | 0.958 | 0.8567 | 0.8812 | 169ms |

### 考场 B：nfcorpus_v1（323 题测试集）

| 方法 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.150 | 0.516 | 0.313 |
| Dense | 0.159 | 0.502 | 0.325 |
| **Hybrid** | **0.172** | **0.552** | **0.354** |
| Graph | 0.157 | 0.483 | 0.314 |

---

## 4. 关键结论（为什么这些结果有意思）

**① 没有"万能"检索器——看数据说话。**
- 考场 A（单答案、语义为主）：**Dense 赢**，因为题目和摘要意思接近，语义匹配碾压关键词。
- 考场 B（多答案、均衡）：**Hybrid 赢**，因为 BM25 和 Dense 势均力敌、信号互补，融合才有效。

**② 基准选择会改变"融合有没有用"的结论。** 同一个 Hybrid，在 A 上不如 Dense，在 B 上胜过两路单路。这说明**实验结果不能脱离数据解读**——这也是项目最想训练的研究素养。

**③ 图检索（Graph）在两个考场都没赢。** 我们忠实实现了 LinearRAG 的图算法（医学实体识别 + 图 + PPR），但它在这两个数据集上都不如传统检索。可能的原因：这两个基准的问题不是真正的"多跳推理"题，图的优势没发挥出来。这是个**诚实的负结果**——实现正确（有独立复算和代码审查背书），但价值未在现有数据上体现。

**④ 所有数字都经过审计。** 每个指标都能独立复算（误差 < 1e-15）、每步都有哈希绑定（防篡改）、有代码审查。**不编造结果**是项目的铁律。

---

## 5. 代码导航（每个文件夹干嘛）

```
MedicalGraphRAG/          ← 主项目：所有清洗/检索/评测/实验都在这里
  src/medical_graphrag/
    data/                 ← 数据加载、切块、冻结数据构建
    retrieval/            ← 4 个检索器 + 图构建
      bm25.py             ← BM25（Lucene）
      dense.py            ← Dense（FAISS）
      hybrid.py           ← RRF 融合
      graph.py            ← 图检索（移植 LinearRAG）
    evaluation/           ← 指标计算 + 各检索器评测
  scripts/                ← 容器内跑的构建/检索脚本
  experiments/            ← 实验结果（metrics.json 等）
  data/processed/         ← 冻结数据集（可重建，git 忽略）
  indexes/ outputs/       ← 索引和中间产物（git 忽略）

docs/superpowers/specs/   ← 每个阶段的设计文档（从设计看实现）
STUDY_PROGRESS.md         ← 学习进度（当前状态/下一步）
HANDOFF_*.md              ← 每个阶段的交接文档（给下一位 Agent）
LinearRAG/ MedRAG/        ← 只读参考实现与数据源
MedicalGPT/               ← 另一条并行线（GPT 在做的 SFT 预习）
```

---

## 6. 怎么亲手跑一遍（简版）

环境：WSL2 Docker 容器 `llm-pytorch`，Python 环境在 `/opt/venv`。

```bash
# 1. 建冻结数据（二选一，数据已建好，通常不用重跑）
#    python scripts/build_nfcorpus.py --raw-dir data/raw/nfcorpus --output-dir data/processed/nfcorpus_v1

# 2. 跑一个检索器（以 Dense 为例，用容器 venv）
cd MedicalGraphRAG
/opt/venv/bin/python scripts/build_faiss_dense_index.py \
  --dataset-dir data/processed/nfcorpus_v1 --output-dir outputs/nfcorpus_v1/dense_abstract_only
/opt/venv/bin/python scripts/search_faiss_dense.py \
  --index .../index.faiss --questions .../questions.jsonl --metadata .../chunk_metadata.jsonl \
  --output .../raw_rankings.jsonl --report .../search_run.json --index-report .../index_build.json \
  --top-k 100 --embedding-model models/all-mpnet-base-v2

# 3. 评测
/opt/venv/bin/python -m medical_graphrag.cli evaluate-dense \
  --dataset-dir data/processed/nfcorpus_v1 --metadata ... --rankings ... \
  --index-report ... --search-report ... --output-dir experiments/nfcorpus_v1/dense_abstract_only \
  --git-commit $(git rev-parse HEAD) --docker-image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
```

> 完整命令看各阶段设计文档（`docs/superpowers/specs/`）和交接文档（`HANDOFF_*.md`）。

---

## 7. 当前状态 & 下一步

**已完成**：检索层全部落地——4 个检索器 × 2 个基准，全部真实指标 + 审计。

**进行中**：阶段 3 MedicalGPT SFT（让模型懂医学，另一条并行线已在预习）。

**后续**：Reranker（Qwen3-Reranker）、阶段 4 Search-R1 强化学习、阶段 5 MedSearch-R1 组合。

**铁律**：只改一个变量、指标必须真实脚本输出、封闭基准不外推、不跳步。

---

*如果你想深入某一块（比如"图检索到底怎么建图"、"RRF 为什么用排名不用分数"、"Recall@k 和 nDCG 差在哪"），随时问我，我按初学者能懂的方式讲。*
