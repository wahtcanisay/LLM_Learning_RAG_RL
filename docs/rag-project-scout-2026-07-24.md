# 2026-07-24 RAG 项目选型调研

## 目标

寻找同时满足以下条件的 RAG 项目：

- 2025-2026 年仍活跃，思想不局限于 BM25、Dense 和普通 RRF；
- 能展示完整或关键的现代 RAG 流程；
- 核心算法代码可见，尽量少依赖前后端、微服务和多层框架；
- RTX 5090 32GB 能完成最小复现；
- 能形成有真实基线、指标和失败分析的求职项目。

本报告中的 star、提交时间和代码规模是 2026-07-24 的快照。时间和显存是工程估算，不是已经运行得到的实验结果。

## Search Queries

- GitHub：`RAG created:>=2025-01-01 stars:>100`
- GitHub：`GraphRAG retrieval augmented generation 2025`
- GitHub：`vectorless RAG reasoning retrieval`
- GitHub：`reranker guided graph search RAG`
- GitHub：`agentic RAG small language model`
- ICLR 2026：[Information Retrieval / RAG 分类页](https://papernotes.org/ICLR2026/information_retrieval/)
- 对候选进一步核验：官方 README、核心源码目录、OpenReview、许可证、提交记录和最小运行入口

## GitHub Popular Scan

本轮补充使用 GitHub 官方公开 REST API，按以下条件检索并去重：

- `topic:rag created:>=2025-01-01 archived:false`
- `topic:retrieval-augmented-generation created:>=2025-01-01 archived:false`
- `rag in:name created:>=2025-01-01 archived:false`
- `graphrag created:>=2025-01-01 archived:false`
- `rag created:>=2025-01-01 pushed:>=2026-04-01 archived:false`

`topic:rag` 的高 star 结果混入大量 Agent 平台、教程、PDF 解析器和只把 RAG 当附属功能的项目，因此 star 只用于发现候选，不能代替代码适配性判断。

| Repo | Stars | 创建/最近推送 | 源码树快照 | 结论 |
|---|---:|---|---:|---|
| [Graphify](https://github.com/Graphify-Labs/graphify) | 95,030 | 2026-04-03 / 2026-07-22 | 284 个代码文件，约 4.5 MB | 热度最高，但面向代码库 AST/知识图谱，不是医学 RAG；工程体量大 |
| [PageIndex](https://github.com/VectifyAI/PageIndex) | 34,350 | 2025-04-01 / 2026-07-24 | 15 个代码文件，约 258 KB | **热门侧最佳匹配**；语义树、无向量库、reasoning retrieval，核心较小 |
| [RAG-Anything](https://github.com/HKUDS/RAG-Anything) | 22,395 | 2025-06-06 / 2026-07-20 | 62 个代码文件，约 889 KB | 多模态流程完整，但建立在 LightRAG、MinerU 和多解析器之上 |
| [Memvid](https://github.com/memvid/memvid) | 16,038 | 2025-05-27 / 2026-07-14 | 158 个代码文件，约 2.3 MB | 新颖的单文件 Agent Memory/RAG 存储层，偏 Rust 检索基础设施 |
| [LEANN](https://github.com/StarTrail-org/LEANN) | 12,725 | 2025-06-09 / 2026-07-19 | 167 个代码文件，约 1.9 MB | MLSys 2026，低存储 ANN 很新，但 DiskANN/HNSW/C++ 子模块较重 |
| [Local Deep Research](https://github.com/LearningCircuit/local-deep-research) | 8,770 | 2025-02-09 / 2026-07-23 | 2,309 个代码文件，约 30.6 MB | Agentic research 产品完整，但远超快速源码学习范围 |
| [DeepSearcher](https://github.com/zilliztech/deep-searcher) | 8,013 | 2025-02-08 / 2025-11-19 | 145 个代码文件，约 677 KB | Deep Research + 私有数据检索，依赖 Milvus 等组件，近期活跃度较弱 |
| [PixelRAG](https://github.com/StarTrail-org/PixelRAG) | 7,109 | 2026-05-29 / 2026-07-16 | 174 个代码文件，约 5.5 MB | 极新；页面渲染为图像 tile，再用 Qwen3-VL-Embedding 建索引；最小 PDF 可运行，但代码和训练工程偏厚 |
| [VideoRAG](https://github.com/HKUDS/VideoRAG) | 3,206 | 2025-02-03 / 2026-03-18 | 93 个代码文件，约 664 KB | KDD 2026，视频 GraphRAG；媒体处理链路和数据成本高 |
| [ViDoRAG](https://github.com/Alibaba-NLP/ViDoRAG) | 669 | 2025-02-25 / 2026-01-11 | 18 个代码文件，约 100 KB | **小而新的多模态候选**；GMM 动态混合检索 + actor-critic 迭代 Agent，但依赖 ColQwen/LlamaIndex/VLM |
| [LinearRAG](https://github.com/DEEP-PolyU/LinearRAG) | 526 | 2025-10-27 / 2026-07-05 | 7 个代码文件，约 55 KB | 热度不高，但 ICLR 2026 学术性和源码透明度最佳 |

## Candidate Table

| Repo | Fit | Minimum Run | 可做的修改 | Evidence | Risk |
|---|---|---|---|---|---|
| [LinearRAG](https://github.com/DEEP-PolyU/LinearRAG) | **Strong fit，主项目首选** | 1k-5k chunks，100-300 个问题，先做 retrieval-only | 医学 NER、统一 `search()`、Dense/Hybrid/Graph 对比 | ICLR 2026；Tri-Graph、实体激活、语义桥接和两阶段检索均在少量源码中 | 主文件约 600 行，需要自己拆解和补测试 |
| [GraphRAG-Benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) | **Strong fit，但只做评测层** | 只选医学数据和两种方法 | 给 LinearRAG 增加统一评测、任务分型和成本统计 | ICLR 2026；含医学领域，研究“何时图检索有效” | 不是新 retriever；完整 benchmark 依赖多个独立框架 |
| [PageIndex](https://github.com/VectifyAI/PageIndex) | **Strong fit，工业侧备选** | 选择少量医学长文档，运行语义树索引和 tree search | 本地 Qwen、医学文档树、与 Dense/LinearRAG 比较 | 约 34k stars；2026-07-24 仍活跃；核心检索脚本较短 | 不是本次 ICLR 2026 学术候选；公开指标必须自行复现后才能写简历 |
| [ViDoRAG](https://github.com/Alibaba-NLP/ViDoRAG) | **Strong fit，多模态方向备选** | ExampleDataset、少量 PDF 页面，先做 retrieval-only | 医学图文页面、GMM 动态 top-k、文本/视觉混合检索 | EMNLP 2025；核心入口只有 ingestion、search、agent 和 eval 等少量文件 | ColQwen、LlamaIndex 和 VLM 增加环境成本；仓库无明确许可证 |
| [Reranker-Guided Search](https://github.com/xuhaike/Reranker-Guided-Search) | **Strong fit，适合作为检索增强模块** | 小数据、固定 reranker 调用预算 | 重写 200-300 行最小版，接现有 Dense index | ICLR 2026；让 reranker 指导近邻图搜索，而不是只重排固定 top-k | 主脚本超过 1000 行、star 少、无明确许可证；不适合单独包装成完整 RAG |
| [Expert Heads](https://github.com/Xuan-Van/ExpertHead) | **Strong fit，适合独立机制小项目** | 100-300 条 HotpotQA，短上下文 | 文档位置扰动、专家头稳定性和证据排序可视化 | ICLR 2026；直接使用 LLM attention head 的证据信号 | attention 为平方复杂度；不是完整检索系统，star 和维护活跃度低 |
| [BrowseNet](https://github.com/bisect-group/BrowseNet) | **Weak fit** | 预构建数据上的少量问题 | 医学实体图、查询 DAG、图遍历 | ICLR 2026；核心图遍历较直接 | 查询分解和回答依赖外部 API |
| [QAFD-RAG](https://github.com/Tarzanagh/QAFD-RAG) | **Weak fit** | 预构建 MuSiQue 图上的 retrieval-only | 作为 LinearRAG 的图检索对比 | ICLR 2026；代码和评测较完整 | OpenIE、双图和扩散组件较多，重建图依赖 API |
| [CF-RAG](https://github.com/CF-RAG/CF-RAG) | **Risky fit** | 运行发布的小规模反事实检索流程 | 增加反事实 query 和证据仲裁模块 | ICLR 2026；代码量不大，完整流程可见 | 低 star、少提交、无许可证，核心逻辑集中在大脚本 |
| [LDAR](https://github.com/ku-dmlab/LDAR) | **Risky fit，后续控制器候选** | 缩短上下文和数据，训练轻量策略 | 替代 R2RAG 的纯 prompt 路由器 | ICLR 2026；学习 distraction-aware adaptive retrieval | 长上下文训练耗时，继承工程较多，不满足快速上手 |
| [MiniRAG](https://github.com/HKUDS/MiniRAG) | **Weak fit** | Qwen 3B 和小型数据集 | 小模型图检索基线 | 约 2k stars；面向小语言模型的异构图 RAG | 基于 LightRAG，数据库、API 和服务适配较多，底层学习路径不够短 |
| [Youtu-GraphRAG](https://github.com/TencentCloudADP/Youtu-GraphRAG) | **Not recommended now** | 不建议当前复现 | 仅阅读 schema-guided graph/agent 思想 | ICLR 2026，约 1.2k stars | 前后端、Docker、Agent 和多存储组件过重 |
| [Q-RAG](https://github.com/griver/Q-RAG) | **Not recommended now** | 只考虑后续 RL 阶段 | value-based multistep retriever | ICLR 2026 Oral | RL 组件多、仓库仍在重构、官方 reader 规模偏大 |
| [FrugalRAG](https://github.com/microsoft/FrugalRAG) | **Not recommended now** | 留到 Search-R1 阶段 | 成本约束和多跳搜索策略 | ICLR 2026；SFT/GRPO 代码已公开 | vLLM、检索服务、SFT 和 GRPO 组合过重 |
| [LEANN](https://github.com/StarTrail-org/LEANN) | **Not recommended for current goal** | 不建议作为完整 RAG 学习入口 | 以后用于低存储 ANN/检索基础设施 | 约 12.7k stars，MLSys 2026 | Python/C++、ANN 子模块和集成代码很多，偏检索基础设施 |

## Top Recommendations

### 1. PageIndex：GitHub 热门侧第一名

PageIndex 最同时满足“新、热门、完整流程、核心代码较小”：

- 先把长文档组织为类似目录的语义树；
- 再让 LLM 在树上推理和选择页面；
- 不依赖 Vector DB，也不按固定 token 人工切块；
- 自托管仓库提供完整 agentic vectorless RAG 示例；
- 适合用医学指南、教材章节或长 PDF 做可解释检索。

风险是它不是本次 ICLR 2026 论文项目，而且 tree building 和 reasoning retrieval 都会消耗 LLM token。README 中的 FinanceBench 结果不能直接写入我们的简历。

### 2. LinearRAG + GraphRAG-Benchmark：学术侧第一名

这是当前最强组合，但两者职责不同：

- LinearRAG 提供新算法：关系无关 Tri-Graph、实体激活、语义桥接和段落聚合；
- GraphRAG-Benchmark 提供可信评测：区分事实检索、复杂推理、摘要等任务，并包含医学领域；
- MedRAG 提供现有医学 chunk、BM25/Dense/Hybrid 基线；
- R2RAG 只保留已经掌握的动态控制思想，不再阅读外围服务代码。

这个组合能形成“底层算法 + 医学迁移 + 基线对比 + 成本分析”，比单纯运行一个热门框架更适合算法实习简历。

### 3. ViDoRAG：多模态简洁备选

如果岗位明确偏多模态 RAG，ViDoRAG 比 RAG-Anything 和 PixelRAG 更适合源码学习：

- 核心代码约 18 个文件；
- 流程覆盖 PDF 页面化、视觉/文本 embedding、GMM 动态混合召回、actor-critic 迭代生成和评测；
- 数据集提供 reference page，可以真实评测页面召回；
- 5090 32GB 可先只运行 ColQwen/BGE retrieval-only。

它的不足是 GitHub 热度只有约 669 stars，环境依赖比纯文本 LinearRAG 多，而且仓库无明确许可证。若只看热度，可选择 PixelRAG；若看代码透明度，ViDoRAG 更优。

### 检索增强模块：Reranker-Guided Search

RGS 不适合单独作为“完整 RAG 项目”，但很适合作为 LinearRAG/MedRAG 中的第二个算法增强点：

- 固定相同 reranker 调用预算；
- 比较顺序重排与 reranker-guided graph search；
- 记录 Recall@10、nDCG@10、访问文档数和延迟；
- 最小版本可以自己重写，避免复制千行单体脚本。

如果更想展示 Transformer 内部机制而不是搜索算法，可把第三项换成 Expert Heads。

## Study Plan

### Corpus Fit

| 条件 | LinearRAG | PageIndex |
|---|---|---|
| 直接使用 MedRAG `id/title/content` chunks | **适合**：图节点和语义桥接以 chunk/段落为输入 | **不适合直接使用**：核心优势依赖长文档的页码、章节和层级 |
| 官方医疗数据 | **有**：`medical/chunks.json` 与 `medical/questions.json`，并给出 biomedical SciSpacy 配置 | **没有专门医疗 benchmark**：README 只说明适用于 medical literature |
| 形成检索指标 | **快**：问题、答案和 evidence 已配套 | **需要额外构造**：必须准备带 gold page/section 的医学长文档问题 |
| 索引成本 | 关系无关构图，不需要 LLM 做关系抽取 | tree building 和 query-time reasoning 都需要 LLM |
| HR/面试辨识度 | ICLR 2026、GraphRAG、医学迁移和对比实验 | 34k+ stars、vectorless RAG、语义树和可解释页面检索 |

因此，“快速跑一个演示”可以选 PageIndex；“快速做出可信的算法简历项目”应先选 LinearRAG。

当前不立即运行新仓库。下一次只做一个任务：

1. 阅读 LinearRAG README 和 `src` 下 6 个核心 Python 文件，画出“实体抽取、Tri-Graph 构建、查询实体激活、两阶段检索、返回段落”的调用链。

完成标准：

- 能指出三层节点和边分别来自什么数据；
- 能解释为什么不需要 LLM 做关系抽取；
- 能定位 query-time retrieval 的入口；
- 能说明官方 `medical/chunks.json` 与 MedRAG JSONL 之间需要怎样适配；
- 不运行生成模型，不下载大数据，不声称已有指标。

## Resume-Safe Claims After Completion

以下内容只有完成对应代码、实验和日志后才能写入简历：

- 在统一医学 RAG 框架中实现 BM25、Dense、Hybrid 与 LinearGraphRetriever 接口；
- 将 LinearRAG 的关系无关 Tri-Graph 迁移到医学 chunk，并比较普通 NER 与医学 NER；
- 在固定语料、问题集和 top-k 下报告 Recall@k、MRR、QA 准确率、延迟、索引时间和索引空间；
- 在相同 reranker 调用预算下比较顺序重排和 Reranker-Guided Search；
- 对简单题、多跳题和跨实体题分别分析收益与失败原因。

当前只能安全表述为“完成候选调研和源码可行性评估”，不能表述为“复现成功”或“性能提升”。

## Interview Grilling Questions

1. LinearRAG 的 Tri-Graph 与传统知识图谱有什么区别？
2. 不抽取关系为什么还能支持多跳检索？
3. 实体激活、语义桥接和段落聚合分别解决什么问题？
4. 为什么 GraphRAG 在简单事实题上可能不如 Dense Retrieval？
5. 如何证明效果来自图结构，而不是更强的 embedding 或更大的 top-k？
6. PageIndex 的语义树检索与向量检索在索引成本、召回方式和失败模式上有什么差异？
7. Reranker-Guided Search 为什么可能优于“先 Dense top-k，再统一 rerank”？
8. 固定 reranker 预算时，应该记录哪些质量和成本指标？
9. 检索 Recall 提升但 QA 准确率下降，可能由哪些环节造成？
10. 医学 NER 错误如何传播到图构建和最终召回？

## Decision

**最终建议：先做 LinearRAG 官方 medical 数据的最小复现，再迁移 MedRAG 的 Textbooks/StatPearls 小子集；PageIndex 作为第二个医学长文档热门项目，不与第一轮实验同时推进。**

RAG-Anything、PixelRAG、LEANN、Memvid 和 Deep Research 项目虽然热门，但不符合当前“快速阅读底层实现”的约束。ViDoRAG 只在明确投递多模态 RAG 岗位时替代 PageIndex。R2RAG 的外围工程代码停止学习，只保留已经理解的动态路由、查询改写、证据充分性和停止控制思想。
