# 维护记录

- 2026-07-30 及之前：由 GPT 维护本学习进度。
- 2026-07-31：由 DeepSeek 接续维护，并完成图节点/边阅读记录与路线调整。
- 2026-08-01 起：GPT 根据 `STUDY_PROGRESS_deepseek.md` 同步最新进度并继续维护；根目录 `STUDY_PROGRESS.md` 恢复为主进度文件。

# 当前阶段

- 阶段 1：MedRAG 基础代码、toy BM25/Dense/RRF 和四套语料核验已完成；当前回到阶段 1，使用 `MedicalGraphRAG/pubmedqa_hard_v1` 补齐正式 BM25 检索硬指标。
- 阶段 2：LinearRAG 默认主干代码第一轮阅读已完成；图检索端到端实验与医学迁移尚未开始。
- 阶段 3：MedicalGPT LoRA/QLoRA SFT。
- 阶段 4：Search-R1 搜索强化学习。
- 阶段 5：MedSearch-R1 医学证据搜索 Agent。
- R2RAG 已于 2026-07-31 从路线剔除，详见“路线变更记录”。

# 当前项目

**Medical GraphRAG + Agent**：MedRAG 提供医学语料与基础检索基线，LinearRAG 提供关系无关图检索，MedicalGPT/Search-R1 提供领域 SFT 与搜索 RL，MedSearch-R1 组合为医学搜索 Agent。

当前先完成并理解 `MedicalGraphRAG/pubmedqa_hard_v1` 的正式 BM25 基线；随后再继续 Dense baseline 和 LinearRAG 官方 medical 小数据验证。

## 当前执行焦点（2026-08-03）

新项目目录为 `MedicalGraphRAG/`，与 `MedRAG/`、`LinearRAG/` 平级。当前先补齐阶段 1 的可比较基线：使用 PubMedQA PQA-L 的 1,000 个 gold documents、4,000 个确定性 MedRAG PubMed distractors 和 document-level qrels，运行 BM25 并产出 Recall@1/5/10、MRR@10、nDCG@10。BM25 完成前，不启动 LinearRAG 医学迁移实验。

## 当前执行焦点（2026-08-04，阶段 3 临时预习）

- 因当前可用额度有限，学习者明确要求暂时开始 MedicalGPT；该变更记为阶段 3 代码阅读预习，不代表阶段 1 的 Hybrid 或阶段 2 医学迁移已经完成。
- 官方来源为 `https://github.com/shibing624/MedicalGPT`。Git 协议连续出现 TLS 中断和 443 超时，本地目前使用 GitHub 官方 `codeload` 的 `main` 分支源码快照；Git 历史和精确 commit 尚未补齐，后续网络恢复后必须校验。
- 本次只读并注释 `scripts/run_sft.sh → training/supervised_finetuning.py`，不下载模型或外部数据，不启动训练。
- MedicalGPT 的当前 GRPO 默认面向 GSM8K，奖励为答案正确性和 `<think>/<answer>` 格式；代码中没有 Search/Inspect 工具环境、检索状态或多轮搜索 rollout，因此不能把它等同于 Search-R1。

## LinearRAG 真实学习边界

- 已亲自阅读：`LinearRAG/readme.md`、`LinearRAG/run.py`、`src/config.py`、`src/embedding_store.py`、`src/ner.py`、`src/evaluate.py`、`src/utils.py` 的相关辅助函数，以及 `src/LinearRAG.py` 默认非向量化主干：初始化、离线 `index()`、正式图构建、在线 `retrieve()`、Seed Entity、实体传播、Passage 先验、PPR、Top-k 与 `qa()`。
- 尚未验证：官方 medical 数据上的端到端最小运行、真实耗时/显存/指标；向量化实体传播和属性关键词增强属于可选支线，暂不计入默认主干完成条件。
- 2026-07-31 正式图节点/边任务已完成（学习者决定不再复述）。记录：Q1/Q4 确认为笔误（Entity 误写 Sentence）；Q2 正确表述为“边权是归一化比例（0～1），不是原始次数”；Q3 术语为“正则表达式”。
- 助手添加注释、执行语法检查或检查调用链，只算材料准备，不算学习者完成。
- LinearRAG 官方源码版本：`bcc94e66c221f798801255efba09311d6fbcd8d6`。

## 已确认的路线与边界

- 顺序保持为：MedRAG → LinearRAG → MedicalGPT → Search-R1 → MedSearch-R1。
- 最终阶段使用 `MedSearch-R1`；设计文档位于 `docs/superpowers/specs/2026-07-18-medsearch-r1-design.md`。
- 不依赖 MIMIC-CDM，不模拟真实临床检查，不声称提供临床诊断。
- 暂缓：MedRAG 原生全量索引/生成/评测、PageIndex 候选项目。
- 不同时修改图构建、Embedding、重排和生成模型；每次只验证一个主要变量。

# 本周目标

本周先完成 `pubmedqa_hard_v1` 的 BM25 正式基线，再回到 LinearRAG 官方 medical 小数据端到端运行。当前已经能够区分：

1. 对象初始化与 Embedding 缓存；
2. NER 和离线建图；
3. 在线实体传播、Passage 打分和 PPR；
4. 生成指标与检索指标。

## LinearRAG 阅读顺序

1. 对象初始化与 Embedding 缓存

   `config.py → LinearRAG.__init__() → load_embedding_store() → embedding_store.py → compute_mdhash_id()`。
2. NER 数据契约

   `SpacyNER.__init__() → batch_ner() → extract_entities_sentences() → question_ner()`。
3. 离线索引总流程

   `index() → load_existing_data() → merge_ner_results() → save_ner_results() → extract_nodes_and_edges()`。
4. 正式图节点和边

   `add_entity_to_passage_edges() → add_adjacent_passage_edges() → augment_graph() → add_nodes() → add_edges()`。
5. 在线入口与 Seed Entity

   `retrieve() → get_seed_entities()`，第一遍只读默认非向量化分支。
6. 实体传播、Passage 先验和 PPR

   `calculate_entity_scores() → dense_passage_retrieval() → calculate_passage_scores() → run_ppr()`。
7. 生成与评测

   `qa() → evaluate.py → normalize_answer()`。

前一项未完成复述和检查题，不进入下一项。

# 今日唯一任务

2026-08-08：冻结“单数据集独立建库 + 摘要 document-level Similarity 软边 + 长文本同文档 Adjacent `1.0` 边”的设计，并形成交给 DS 的代码修改规格。旧的合并大库方案停止实施。

# 完成标准

- [x] 明确放弃合并大库、跨库图和 merged qrels；
- [x] 摘要型数据使用完整 `documents.jsonl`，并规定 token-window 全覆盖聚合，禁止静默截断；
- [x] Similarity 与 Adjacent 通过 graph profile 互斥，长文本 Adjacent 严格限同文档连续 order、权重 `1.0`；
- [x] 明确同 retrieval unit 的 BM25/Dense/Hybrid/Graph 公平基线与历史结果隔离；
- [x] 写出测试、审计、分阶段验收和 DS 回交材料；
- [x] 用户复核书面规格；
- [x] DS 提交 implementation plan `7b4c6cb`，Codex 完成审阅并判定存在阻断项，禁止按原计划直接实施；
- [ ] DS 按审阅交接提交 Phase 1 v2 计划并完成代码，之后由 Codex 审阅代码与真实日志。

# 已完成

## Per-Dataset Edge Policy 实施计划审阅（2026-08-08）

- 审阅 `docs/superpowers/plans/2026-08-08-per-dataset-edge-policy.md`（commit `7b4c6cb`），确认方向符合冻结设计，但存在测试必失败、ranking/report 契约不闭环、核心实现占位和三路 embedding 未真实共用等阻断问题。
- 审阅交接保存于 `docs/superpowers/reviews/2026-08-08-per-dataset-edge-policy-plan-review.md`；要求 DS 先修订仅覆盖 PubMedQA Phase 1 的 v2 计划，再修改代码。
- 本次未修改检索代码、未运行新实验、未产生新指标；下一门禁为 DS 回交代码 commit、pytest 日志、五路 runner 产物和 embedding artifact 哈希后，由 Codex 进行代码审阅。

## 单数据集边策略设计交接（2026-08-08）

- 审计并废弃 `docs/superpowers/specs/2026-08-07-merged-library-build-switch-design.md` 的合并大库方向，保留文件作为决策历史但禁止实施。
- 新规格位于 `docs/superpowers/specs/2026-08-08-per-dataset-document-edge-policy-design.md`：摘要型数据按完整 document 建 Similarity kNN 软边，textbooks/statpearls 按同文档连续 chunk 建 Adjacent `1.0` 边。
- 发现并封堵完整摘要 embedding 的隐性截断风险：规定 tokenizer window 全覆盖、mean-window-then-L2 聚合，Dense/Graph/Similarity 共用同一冻结 document embedding。
- 本次只修改设计与进度文档，未修改检索代码、未运行新实验、未产生任何新指标。

## MedicalGPT SFT 阅读材料准备（2026-08-04）

- 从官方 GitHub `main` 分支源码快照建立 `MedicalGPT/`；Git clone 因网络错误未补齐历史，不能记录未经验证的 commit。
- 为 `MedicalGPT/scripts/run_sft.sh` 加入参数分组、双卡示例和 QLoRA 开关说明。
- 为 `MedicalGPT/training/supervised_finetuning.py` 标注参数解析、模板与数据加载、loss mask、量化加载、LoRA 注入、collator、Trainer、恢复/保存和 perplexity 边界。
- 静态验证：`python -m py_compile MedicalGPT/training/supervised_finetuning.py` 返回 0；`C:\Program Files\Git\bin\bash.exe -n MedicalGPT/scripts/run_sft.sh` 返回 0。
- 与下载 ZIP 中的原始文件执行 `git diff --no-index`，确认两份源码只增加注释和解释性 docstring，没有训练逻辑变化。

## MedicalGPT SFT 调用链复述完成（2026-08-09）

- 完成 `supervised_finetuning.py` 调用链学习：参数 dataclass → tokenizer → loss mask → 量化/LoRA 注入 → Trainer，逐函数讲解 + 小问题自答（ModelArguments/DataArguments/ScriptArguments、preprocess_function、find_all_linear_names、SavePeftModelTrainer、print_trainable_parameters、check_and_optimize_memory）。
- 4 道概念题复述通过：① LoRA 冻结主干、`W=W0+BA` 的 A∈ℝ^{d_in×r}、B∈ℝ^{r×d_out}、r 决定可训练参数量；② QLoRA = 4bit 基模(NF4+DQ) + BF16 adapter，`load_in_4bit` 只是加载方式、`qlora=True` 才是完整方案；③ `train_on_inputs=False` 时用户问题保留在 input_ids 作条件、labels 标 -100 不算 loss（P(y|x) 读取 vs 学习）；④ 切模型改 `--model_name_or_path`，显存杠杆 = 量化 → batch → max_length → gradient checkpointing（非 lora_rank）。
- 追加题通过：MedicalGPT GRPO 是静态答案奖励、无 Search/Inspect 工具环境、无多轮 rollout；Search-R1 是 reason+search 交错、retrieved token mask、学会"何时搜/搜什么/何时停"——二者机制不同不能等同。
- 已补 4 处学习注释（commit `48f6dfb`）：可训练参数转 fp32、GC 与 use_cache 互斥、enable_input_require_grads、lm_head fp32 hook；py_compile 通过。

## MedicalGraphRAG（2026-08-03）

- 新建独立项目 `MedicalGraphRAG/`；实现 PubMedQA 加载、MedRAG PubMed 确定性干扰采样、文档边界安全切块、questions/documents/chunks/qrels 组装、原子 I/O、manifest、审计和检索硬指标函数。
- 测试：`cd MedicalGraphRAG && python -m pytest -q`，真实结果为 `25 passed`。
- 20 题人工审计门禁：20/20 通过，0 空 context、0 exact duplicate gold/distractor，涉及 40 个确定性 PubMed 分片和 75 个 gold chunks；证据位于 `MedicalGraphRAG/data/processed/pubmedqa_hard_v1/audit_20.json`。
- 完整数据构建命令：`python -m medical_graphrag.cli build --config configs/pubmedqa_hard_v1.json --pubmedqa-dir data/raw/pubmedqa --medrag-pubmed-dir ../MedRAG/corpus/pubmed/chunk --output-dir data/processed/pubmedqa_hard_v1`。
- 完整数据真实结果：1,000 questions、5,000 documents（1,000 gold + 4,000 distractors）、7,562 chunks、1,000 qrels；500 dev + 500 official test；构建耗时 13.18 秒。
- 数据配置：seed `20260803`，tokenizer `sentence-transformers/all-mpnet-base-v2`，max tokens `512`，overlap `64`，主检索文本 `abstract_only`。
- manifest：`MedicalGraphRAG/data/processed/pubmedqa_hard_v1/manifest.json`，SHA-256 为 `cf9b75917bb6c73ff5e5d1862293e31caf86ec5d93c05c24f40760c83b727baa`；四个数据产物哈希均已独立复核。
- BM25 环境：WSL2 `LLM-Ubuntu-22.04`，Docker 容器 `llm-pytorch`，Pyserini `0.22.1`；旧 toy Lucene index `/tmp/medrag_task4_index` 仍可读取。
- 正式 BM25 参数：`abstract_only`、Porter stemmer、移除 stopwords、`k1=0.9`、`b=0.4`、请求最多 100 chunks、按 `doc_id` 取最大 chunk score。
- 正式索引（审计重跑）：7,562 chunks，耗时 `2.580839597` 秒，空间 `1,254,677` bytes；容器 Python 3.12.3、Java 21.0.11、Pyserini 0.22.1。
- 正式检索：1,000/1,000 query 完成；995 题返回 100 hits，5 题为短排名，最少 3 hits；短排名分布已写入 `search_run.json`。
- dev（500 题）：Recall@1 `0.920`、Recall@5 `0.972`、Recall@10 `0.978`、MRR@10 `0.943250`、nDCG@10 `0.951905`；最终审计重跑 mean/P50/P95 延迟 `1.394/1.143/2.431 ms`。
- official test（500 题，主结果）：Recall@1 `0.926`、Recall@5 `0.974`、Recall@10 `0.984`、MRR@10 `0.945825`、nDCG@10 `0.955147`；最终审计重跑 mean/P50/P95 延迟 `1.359/1.179/2.247 ms`。
- 独立复算：直接从 raw hits、metadata 和 qrels 重新聚合，dev/test 五项检索指标与 `metrics.json` 在 `1e-12` 绝对误差内逐项一致。
- 显存峰值：不适用；本实验是 CPU Lucene BM25，没有调用 GPU。QA Accuracy：未评测，不能由检索指标推断。
- 结果位置：`MedicalGraphRAG/experiments/pubmedqa_hard_v1/bm25_abstract_only/`；raw rankings、collection、index 位于 ignored `outputs/`、`indexes/`。
- 最终审计重跑代码 commit：`49f655e`；数据 manifest SHA-256：`cf9b75917bb6c73ff5e5d1862293e31caf86ec5d93c05c24f40760c83b727baa`；questions SHA-256：`cb957619d30d8885e685e334652abbc6376263278c7de337c35cf3537ce56982`；collection SHA-256：`8651101da23e625c4324e6e0d97018039c2cefd97f539c74bfd69d7fb202360c`；index SHA-256：`24d98c4f6ce12c6aba2e8f7e7aa34c9b5594b92c0caca7a67c122b70f927a274`；raw rankings SHA-256：`3c2376b93f9c7982c28e2d706d942da0e3390c27e1fa9d9e092c09639aa28487`。
- 审计链：导出、建索引、搜索、评测逐级校验 frozen dataset、collection、metadata、index 与前序报告哈希；评测同时复算 query 数、命中数量分布、短排名数量，并从已验证搜索报告读取 Top-k、k1、b 和 text mode，不再硬编码实验标签。
- 2026-08-04：Dense 基线完成（`dense_abstract_only`）。embedding 用 `all-mpnet-base-v2`（= LinearRAG 同款；本地路径 `models/all-mpnet-base-v2`，438MB，经断点续传下载器 `scripts/download_embedding_model.py` 获取，HF 直连中途断线、hf-mirror 无 ETag 被拒的网络坑已记录）；FAISS IndexFlatIP、768 维、归一化、Top-100 候选深度与 BM25 对齐、按 doc_id 取 max 折叠。
- Dense test（500 题，主结果）：Recall@1 `0.966`、Recall@5 `0.992`、Recall@10 `0.994`、MRR@10 `0.977786`、nDCG@10 `0.981885`、mean 延迟 `16.080 ms`（按次运行波动，13~16ms 区间）。全面超过 BM25（Recall@1 +4.0、Recall@10 +1.0、MRR@10 +3.2、nDCG@10 +2.7）。
- Dense dev（500 题）：Recall@1 `0.966`、Recall@5 `0.994`、Recall@10 `0.994`、MRR@10 `0.978233`、nDCG@10 `0.982254`。
- Dense 案例：test 仅 3 题 gold 掉出 Top-10（BM25 8 题）；`11570976`（"Is it Crohn's disease?" rank 63）与 `18359123`（"Is it better to be big?" rank 57）与 BM25 是同一批短模糊 query 失败，可作为 Hybrid 是否缓解的观察点。
- 2026-08-04：Hybrid = RRF(BM25+Dense) 完成（`hybrid_rrf`）。复用两路已审计 raw rankings，按 doc_id 折叠后 RRF(k=60) 融合，不重跑任何检索器。
- Hybrid test（500 题）：Recall@1 `0.960`、Recall@5 `0.990`、Recall@10 `0.992`、MRR@10 `0.973833`、nDCG@10 `0.978454`；dev：Recall@1 `0.958`、Recall@10 `0.996`。
- **负结果（如实记录）**：Hybrid 全面低于 Dense 单路（Recall@1 0.960 vs 0.966、nDCG@10 0.978 vs 0.982）。已知失败查询 `11570976`（Dense 63→Hybrid 114）与 `18359123`（57→113）反而更糟。原因：此基准上 BM25 严格弱于 Dense（摘要偏语义匹配，词项是噪音），RRF 让 BM25 高位错误文档稀释 Dense 信号。设计假设"融合互补信号超过单路"在此基准被证伪，不做无依据调参追逐。
- 2026-08-04：LinearGraphRetriever 完成（`graph_abstract_only`）。医学 NER（BC5CDR）提取 8378 实体、19178 Entity-Passage 边、42342 句子桥；igraph（Entity+Passage 节点）；在线检索 = seed entities → Entity→Sentence→Entity BFS 传播 → passage 先验 → PPR（damping 0.85、passage_ratio 1.5、threshold 0.5、top_k_sentence 1、max_iterations 3）。相邻边 v1 默认关闭（pubmedqa 短摘要）。
- Graph test（500 题）：Recall@1 `0.800`、Recall@5 `0.942`、Recall@10 `0.958`、MRR@10 `0.861929`、nDCG@10 `0.885909`、mean 延迟 `168.690 ms`（PPR+NER 代价，比 Dense 慢 ~10×）；dev：Recall@1 `0.796`、Recall@10 `0.960`。
- **四路对比（test）**：Dense（R@1 0.966）> Hybrid（0.960）> BM25（0.926）> **Graph（0.800）**。图检索在 `pubmedqa_hard_v1` 垫底——短摘要 + 事实题不是图的主场，实体传播 + PPR 相对强 Dense 先验是噪音。印证需要方案 B（`linearrag_medical_v1` 图主场）。独立复算误差 ~1e-15。
- 当前完整测试：`72 passed`。

## MedRAG（2026-07-16 至 2026-07-23）

- 仓库来源：`https://github.com/gzxiong/MedRAG.git`；记录版本：`7599a72`。
- 已阅读：生成入口、模板、Retriever、BM25/Dense 分支、RRF 融合、教材与 PubMed 数据处理。
- 已解释：`id/title/content/contents` 契约、BM25 与 Dense 的区别、FAISS metadata 映射、RRF 去重与排名累加。
- 已完成 3 条 toy 文档上的 BM25、Dense 和 RRF 闭环；这些结果只验证代码与接口，不代表正式医学基线。
- 已核验 Textbooks、StatPearls、PubMed、Wikipedia 的 chunk；无未展开 LFS pointer。
- 已清理可恢复的 LFS 缓存和 StatPearls 原始中间文件，释放约 112.09 GiB。

### MedRAG 语料核验

| 语料 | JSONL 文件数 | 工作区大小 | 状态 |
|---|---:|---:|---|
| Textbooks | 18 | 201.76 MiB | 字段与 LFS 核验通过 |
| StatPearls | 9646 | 441.54 MiB | 字段核验通过 |
| PubMed | 1166 | 65.20 GiB | LFS 核验通过；官方含一个零字节分片 |
| Wikipedia | 646 | 42.54 GiB | 字段与 LFS 核验通过 |

## 路线变更记录

**2026-07-31：R2RAG 从学习路线剔除。**

- 官方仓库：`https://github.com/rmit-ir/NeurIPS-MMU-RAG`；源码保留在 `R2RAG/`（2026-07-22 至 2026-07-24 曾学习其核心思想，未运行 vLLM、外部搜索或正式实验）。
- 剔除判断：Router 与 `VanillaAgent` 的停止/改写决策主要依赖 prompt 输出和字符串匹配，无阈值、概率校准或可学习控制；后续 Search-R1/MedSearch-R1 将用 GRPO 学习搜索与停止策略，继续复现会重复且偏弱。
- 保留给 MedSearch-R1 的参考点：查询变体与融合召回、压缩轮次历史避免上下文膨胀、纯 prompt 停止控制的脆弱性。

## LinearRAG（2026-07-25 至今）

- 官方仓库：`https://github.com/DEEP-PolyU/LinearRAG`。
- 已下载源码并确认顶层调用链：`load_dataset → index → qa/retrieve → evaluate`。
- 已发现 `run.py` 硬编码 `CUDA_VISIBLE_DEVICES="4"`，正式运行前必须改为单卡可配置方式。
- 已完成 README、`run.py`、配置对象、初始化链和 `EmbeddingStore` 阅读。
- 2026-07-28：完成 10 道初始化与缓存检查题及补答；纠正了“Parquet 会保存映射字典”和“对象类型等于 Python 数据类型”的混淆。
- 2026-07-29：完成 Passage NER 与 Question NER 阅读。
- 2026-07-30：阅读 `index()` 的缓存复用、NER 合并、五个节点/映射返回值和文本→hash 映射主流程；缓存覆盖、set 顺序和无显式返回等边界题按学习者决定跳过，不计为已掌握。
- 2026-07-29：完成 `src/ner.py` 第一轮阅读，能够解释 `SpacyNER.__init__()`、`batch_ner()`、`extract_entities_sentences()` 和 `question_ner()` 的输入、输出、去重、大小写与离线/在线用途。
- 2026-07-29：确认 `doc.ents` 的元素是实体 `Span`，不是字典；Passage→Entity 和 Sentence→Entity 都按实体文本去重，同一实体重复出现只保留一个映射项，但 Entity–Passage 边权后续会重新统计原文出现次数。
- 2026-07-29：确认 Passage NER 保留原始大小写，Question NER 转为小写；问题无实体时 `question_ner()` 直接返回空集合 `set()`。
- 2026-07-29：确认 `max_workers` 当前没有通过 `n_process` 启用 spaCy 多进程，只参与 `batch_size` 计算；当 Passage 数少于 `max_workers` 时存在 `batch_size == 0` 的待验证风险。
- 2026-07-29：只读核验 MedRAG 的 Textbooks 和 StatPearls：两者均已切成带 `id/title/content/contents` 的 JSONL chunk；迁移 LinearRAG 需要转换为 `chunks.json`，并保留文档元数据以避免跨文档错误相邻边。
- 2026-07-31：正式图节点和两类边阅读完成并标记通过（学习者决定不再复述，与 2026-07-30 缓存边界题同例）。已确认 Entity–Passage 边是归一化比例，相邻 Passage 通过数字前缀和正则表达式建立固定权重边。
- 2026-08-01：完成 `retrieve() → get_seed_entities()` 默认非向量化分支阅读。能够解释有 Seed Entity 时进入图检索、无 Seed Entity 时回退 Dense Passage；确认 Seed 匹配位于 Question Entity–Corpus Entity 空间，Dense 回退位于 Question–Passage 空间；每个问题实体各执行一次 `argmax`，不同问题实体可以映射到同一个语料 Entity，当前实现没有最低相似度阈值。
- 2026-08-02：完成 `graph_search_with_seed_entities() → calculate_entity_scores()` 默认 BFS-style 分支阅读。能够解释逐层状态、Entity→Sentence→Entity 传播、Sentence Top-k 与全局去重、传播分数乘法、两次阈值剪枝和两个终止条件；确认同一 Entity 的图节点权重按路径累加，而 `new_entities` 只保留最后一次字典赋值。数值检查中明确：若两条路径产生的传播后分数为 0.6 和 0.7，则累计权重增加 1.3。
- 2026-08-03：完成 `calculate_passage_scores() → run_ppr() → retrieve() Top-k → qa() → Evaluator.evaluate()` 默认主干阅读。能够解释 Passage 先验由归一化 Dense 分数和带 tier 衰减的激活 Entity 奖励组成；确认 Node 权重负责问题相关的 PPR 重启位置，Edge 权重负责沿图传播的路线与比例，`damping` 平衡沿边传播和重启。PPR 对全部图节点打分后只保留 Passage，Top-k Passage 先进入生成 LLM 得到 `pred_answer`，随后 Evaluator 才使用 LLM Judge 和标准化包含指标比较 gold answer；二者均不是检索指标。

# 遇到的问题

- LinearRAG README 只说明数据下载位置，没有完整说明 `questions.json/chunks.json` 的代码契约；以后必须区分“README 明示”和“源码确认”。
- LinearRAG official medical 数据的上游医学基准尚未确认，不推测为 MedQA、PubMedQA 或其他数据集。
- 当前没有 gold passage 标注，不能用只有标准答案的数据声称得到 Recall@k、MRR 或 nDCG。
- R2RAG 已从主路线剔除，不再安排 `VanillaAgent` 的 `sid`、跨轮去重和引用映射验证。
- 根 `.gitignore` 的通用 `data/` 规则一度误伤 `MedicalGraphRAG/src/medical_graphrag/data/` 和 compact manifest；已用白名单与 5 项 Git 跟踪回归测试修复，大型 JSONL/TSV 仍保持忽略。
- tokenizer 对超过 512 tokens 的文本发出长度警告；当前切块器会在编码后按 512 tokens 滑窗，不把完整长序列送入模型。该警告不影响本次数据产物，但后续可单独抑制日志噪声。
- metadata 第 5,862 条 title 含 Unicode U+2029；`str.splitlines()` 会错误拆分合法 JSONL。搜索脚本与 evaluator 已改为文件对象逐行读取，并加入回归测试。
- `pyserini` 顶层模块没有 `__version__`；审计脚本已改用 `importlib.metadata.version("pyserini")`，索引报告成功记录 0.22.1。
- 5 个稀有词 query 的 Lucene 命中少于 100。原因是 BM25 不返回零词项匹配文档；当前保留短排名并记录完整分布，不做无依据补齐。
- 首次审计重跑时，容器直接执行脚本无法导入 `medical_graphrag`。原因是容器未安装当前 worktree 包；已让两个独立 Pyserini 脚本按 `__file__` 加入项目 `src/`，并以回归测试覆盖，随后真实索引与搜索成功。

# 下一步

**2026-08-05 进展：**

1. **nfcorpus 召回低是基准上限所致**（每题 ~38 相关，R@10 上限 0.263，我们达 57~65%）；BM25 nDCG 0.313 与公开 BEIR 一致。
2. **LinearRAG 内置 medical 定性检索通过**（多跳题两段证据都召回到 top-2）。
3. **Hybrid2 完成**：Qwen3-Reranker 三路重排（CrossEncoder 加载），nfcorpus 上 **nDCG 0.384 > Hybrid RRF 0.354 > Dense 0.325 > BM25 0.313**——cross-encoder 重排胜出，已审计合并。
4. **BEIR 多数据集接入（进行中）**：泛化 `build_beir.py`；构建 `scifact_v1`（300 题/5,183 文档/339 qrels）与 `trec_covid_v1`（50 题/171,332 文档/24,673 qrels）；**bioasq 因 HF 仓库 401（门禁）下载失败**，待找替代源。
5. **scifact 补齐五路**：Graph（R@10 0.682 / MRR 0.458 / nDCG 0.507）、**Hybrid2（R@10 0.895 / MRR 0.740 / nDCG 0.772）全面超过 Hybrid（0.838/0.669/0.705）**——cross-encoder 重排再次胜出。修复 `search_graph.py` search report 缺 `text_mode` 字段（d5e3007 引入的 rerank 校验必失败 bug），提交 `4442303`。
6. **trec_covid 放弃检索**：原始语料 24.6% 文档 `text` 为空（仅标题的会议摘要记录，非连续文本），3,135 条 qrels 指向空文本文档；title 回退无检索价值，决定不接入检索。
7. **HotpotQA 多跳基准接入（`hotpotqa_v1`）**：构建 `scripts/build_hotpotqa.py`，HF `hotpotqa/hotpot_qa` distractor validation——**7,405 问 / 66,581 文档 / 14,810 qrels（每问恰 2.0 gold）**。语料自洽（自带 8 干扰段/问），可靠 gold，Graph 可跑。NFKC+空白归一化解同标题段落 Unicode 变体。
8. **HotpotQA 图检索多跳验证（负结果）**：BM25（R@10 0.738 / MRR 0.798 / nDCG 0.663）、Dense（0.711/0.817/0.668）、Graph（0.680/0.784/0.638）、**Hybrid（0.800/0.852/0.731）——RRF 融合大幅胜出（R@10 +0.089 vs 单路最好 Dense）**。图检索第四个基准依然垫底；可能原因是 BC5CDR 医学 NER 在通用维基上实体稀疏（11,250 实体/66,581 段落）。
10. **HotpotQA Hybrid2 完成**：**R@10 0.898 / MRR 0.955 / nDCG 0.865，全面碾压 Hybrid（0.800/0.852/0.731）**——cross-encoder 重排在多跳场景价值最显著（R@10 +0.098、nDCG +0.134）。五路齐。
9. **LinearRAG medical 数据集调查**：`LinearRAG/dataset/medical/`（225 chunks + 2,062 问 4 种题型）evidence 为改写文本，全量匹配仅 0.35% 可对齐 chunk，无法建可靠 qrels；官方 `evaluate.py` 本就不使用 evidence。官方 HF `Zly0523/linear-rag` 的 hotpotqa 子集（1,000 问）与我们的 hotpotqa_v1 同源，无额外价值。→ medical 仅定性，hotpotqa_v1 为多跳验证主力。
10. **LinearRAG 论文检索效果调研**：论文检索指标只在 Medical (GraphRAG-Bench/NCCN) 上报告，是 **RAGAS 风格 Evidence Recall（LLM 判断，非标准 IR 指标）**，top-k=5、all-mpnet。复现需 LLM API（阶段 3 后）+ GraphRAG-Bench 官方数据。我们已有的 hotpotqa 标准 IR 指标是论文没有的视角（补充而非复现）。
11. **GraphRAG 对齐 LinearRAG 官方（已完成）**：发现移植时用错默认参数——官方 `damping=0.5 / passage_node_weight=0.05 / passage_ratio=2`（run.py），我们用 `0.85/0.5/1.5`；且官方 NER 排除 ORDINAL/CARDINAL。已全部对齐官方默认（`graph.py`/`build_graph_index.py`/测试更新）。重跑结果：pubmedqa R@10 0.982（原 0.958，+0.024）、scifact 0.705（原 0.682，+0.023）、hotpotqa 0.695（原 0.680，+0.015）、nfcorpus 0.158（原 0.157）。
12. **接口重构 + 统一 evaluate**（commit `5efb6e8`）：7 个 scripts（build_faiss/build_graph/build_pyserini/search_x3/rerank）抽取到 `src/medical_graphrag/retrieval/` 库函数，新增 `run_pipeline.py` 编排层，**`cli run <retriever> --dataset <name>` 成为唯一 evaluate 入口**；test_*_scripts.py 迁移到测 src 函数；79 tests 全绿。
13. **README 简历版总结 + 上传 GitHub**（commit `fe89a64`）：README 补充数据清洗小节 + 末尾简历版项目总结（三点：数据集清洗 / 检索 pipeline 三段 / 相对 baseline 提升）；根 README 已删，文档集中在 `MedicalGraphRAG/README.md`；**已推送 GitHub** `wahtcanisay/LLM_Learning_RAG_RL`（main 同步）。

**下一步（按序）：**
1. **Search-R1 模型选型（已调研）**：官方 Search-R1 用 Qwen2.5-7B 主（+26%）/3B 验证（+21%）；本地 RTX 5090 32GB + MedicalGPT 内置 GRPO（TRL，QLoRA 可配）。rollout≥4 下显存估算：3B≈11GB、4B≈13GB、7B≈20GB、8B≈22GB，均可行但 7B/8B 余量小且 rollout 慢（NF4 反量化 1.5-2×）。**建议：3B 起步跑通管线 → 稳定后上 4B/7B**（与设计文档 2026-07-18 一致）。
2. MIRAGE 端到端（已拉取仓库，等阶段 3 有 LLM 再跑）；
3. 阶段 3 MedicalGPT SFT（GPT 并行线，即将进入）。

**现有基准结果汇总**：`pubmedqa_hard_v1` Dense 最优；`nfcorpus_v1` Hybrid2 最优；`scifact_v1` Hybrid2 最优；`hotpotqa_v1` Hybrid2 最优。图检索在全部四个基准均未胜出。检索层 5 路（BM25/Dense/Hybrid/Graph/Hybrid2）完备，评测支持多相关 qrels。

# 待补知识

- spaCy/SciSpaCy 模型在通用实体与医学实体上的实际识别差异（待后续 toy 实验验证）；
- 官方 medical 数据的真实来源、字段契约和最小样本规模；
- 向量化传播与默认 BFS-style 分支的一致性（可选，端到端主干跑通后再验证）；
- Recall@k、MRR、nDCG 与 QA Accuracy 的评测边界。
- chunk-level ranking 到 document-level ranking 的聚合偏差，以及 Top-100 candidate depth 对 Recall@10 的影响。
- LinearRAG 相邻 Passage 边与 PubMed 短摘要不适配（2026-08-03 反思）：LinearRAG 用 `re.compile(r'^(\d+):')` 从 passage 文本提前缀建相邻边，而我们的 chunk `content` 无前缀、顺序在 `order` 字段，直接喂入建不出相邻边；照搬"全局排序连相邻"还会跨文档错边。迁移时应改为按 `(doc_id, order)` 只在文档内连接，并把相邻边作为单变量实验（预计对短摘要收益很小）；主信号是 Entity–Passage 边 + Entity→Sentence→Entity 传播 + PPR。

# 实验结果总表

以下均为 3 条 toy 文档上的接口验证，不是正式医学实验：

| 方法 | Query | Top-1 | Top-1 分数 | 验证内容 |
|---|---|---|---:|---|
| BM25/Lucene | `facial nerve` | `toy_0` | 1.1835999489 | 倒排索引、docid 与 chunk 映射 |
| Dense/FAISS | `facial nerve` | `toy_0` | 0.7347357273 | Embedding、IndexFlatIP 与 metadata |
| RRF | `facial nerve` | `toy_0` | 0.0198019802 | 两路排名按 `1/(k+rank)` 累加 |

正式 BM25 检索指标和延迟已实验；QA Accuracy 尚未实验，GPU 显存对 CPU Lucene BM25 不适用。

### MedicalGraphRAG 正式数据与 BM25 检索

| 数据版本 | Questions | Documents | Chunks | Qrels | 构建耗时 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| `pubmedqa_hard_v1` | 1,000 | 5,000 | 7,562 | 1,000 | 13.18 秒 | manifest 与哈希核验通过 |

| 方法 | Split | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean latency |
|---|---|---:|---:|---:|---:|---:|---:|
| BM25 abstract-only | dev (500) | 0.920 | 0.972 | 0.978 | 0.943250 | 0.951905 | 1.394 ms |
| BM25 abstract-only | official test (500) | 0.926 | 0.974 | 0.984 | 0.945825 | 0.955147 | 1.359 ms |
| Dense all-mpnet (IndexFlatIP) | dev (500) | 0.966 | 0.994 | 0.994 | 0.978233 | 0.982254 | 19.707 ms |
| Dense all-mpnet (IndexFlatIP) | official test (500) | 0.966 | 0.992 | 0.994 | 0.977786 | 0.981885 | 16.080 ms |
| Hybrid RRF (BM25+Dense, k=60) | dev (500) | 0.958 | 0.990 | 0.996 | 0.972905 | 0.978649 | 离线融合 n/a |
| Hybrid RRF (BM25+Dense, k=60) | official test (500) | 0.960 | 0.990 | 0.992 | 0.973833 | 0.978454 | 离线融合 n/a |
| Graph (BC5CDR Entity-Passage + PPR) | dev (500) | 0.768 | 0.956 | 0.976 | 0.850134 | 0.881531 | 168.0 ms |
| Graph (BC5CDR Entity-Passage + PPR) | official test (500) | 0.790 | 0.966 | 0.982 | 0.864191 | 0.893553 | 168.0 ms |

### NFCorpus（BEIR，多相关 qrels，test 323）

| 方法 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.150 | 0.516 | 0.313 |
| Dense all-mpnet (IndexFlatIP) | 0.159 | 0.502 | 0.325 |
| Hybrid RRF (BM25+Dense, k=60) | **0.172** | **0.552** | **0.354** |
| Graph (BC5CDR Entity-Passage + PPR) | 0.158 | 0.477 | 0.312 |

独立复算多相关指标误差 0/1e-16。

### SciFact（BEIR，科学声明核查，test 300）

| 方法 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.791 | 0.630 | 0.664 |
| Dense all-mpnet (IndexFlatIP) | 0.769 | 0.594 | 0.633 |
| Hybrid RRF (BM25+Dense, k=60) | 0.838 | 0.669 | 0.705 |
| Graph (BC5CDR Entity-Passage + PPR) | 0.705 | 0.456 | 0.509 |
| **Hybrid2 (Qwen3-Reranker)** | **0.895** | **0.740** | **0.772** |

（commit `4442303`；graph 检索在 scifact 上垫底——科学声明核查里词项/语义匹配强，实体传播信号弱。）

### HotpotQA（多跳硬 gold，test 7405）

| 方法 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.738 | 0.798 | 0.663 |
| Dense all-mpnet (IndexFlatIP) | 0.711 | 0.817 | 0.668 |
| Graph (BC5CDR Entity-Passage + PPR) | 0.695 | 0.797 | 0.650 |
| Hybrid RRF (BM25+Dense, k=60) | 0.800 | 0.852 | 0.731 |
| **Hybrid2 (Qwen3-Reranker)** | **0.898** | **0.955** | **0.865** |

（多跳 gold 每问恰 2.0 篇，R@10 显著低于单答案基准；RRF 融合胜出 = 词项+语义两路互补；**Hybrid2 重排大幅碾压（R@10 +0.098 vs Hybrid）——cross-encoder 在多跳场景价值最显著**。图检索依然垫底，BC5CDR 医学 NER 在通用维基上实体稀疏。）

该封闭基准只含 5,000 documents，且 PubMedQA 问题与 gold article 主题高度一致；不得把高 Recall 外推为全 PubMed 检索性能。主设置没有索引 title，但术语明确问题仍容易依靠 abstract 词项命中。

# 失败案例

| 日期 | 问题 | 根因 | 处理结果 |
|---|---|---|---|
| 2026-07-20 | Pyserini 导入时报 `Unable to find javac` | 环境只有 JRE，缺少 JDK 编译器 | 补齐 JDK 后 toy BM25 建库成功 |
| 2026-07-20 | `task6_toy_rrf.py` 找不到 `src` | 从 `docs/` 直接运行时项目根目录不在 `sys.path` | 根据 `__file__` 加入项目根目录后成功 |
| 2026-07-23 | Wikipedia Git LFS 多次出现 `EOF` | Hugging Face Batch API/并发压力 | 降低批量与并发后完成，`git lfs fsck` 通过 |
| 2026-08-03 | 根 `data/` 忽略规则隐藏源码包与 compact manifest | 通用 Git pattern 同时匹配源码目录和生成数据目录 | 添加 MedicalGraphRAG 精确白名单及回归测试，大型产物继续忽略 |
| 2026-08-03 | metadata JSONL 在第 5,862 条被解析为两段 | title 含 U+2029，`str.splitlines()` 将其误判为记录边界 | 改为文件对象逐行解析；Unicode 回归测试通过 |
| 2026-08-03 | 索引完成后审计脚本报 `pyserini.__version__` 不存在 | 错误假设顶层模块暴露版本属性 | 改用 distribution metadata；正式报告记录 Pyserini 0.22.1 |
| 2026-08-03 | 5/1,000 query 少于 100 Lucene hits | 稀有词 query 可匹配文档不足 100，BM25 不返回零匹配文档 | 保留真实短排名、记录分布、全部问题进入指标，不伪造补齐 |

## BM25 案例观察（official test）

- 成功：`7482275`（“Necrotizing fasciitis … hyperbaric oxygenation”）gold rank 1。`necrotizing fasciitis` 与 `hyperbaric oxygen` 是高区分度术语组合，gold abstract 的词项匹配明显强于普通干扰文档。
- 成功：`7547656`（epinephrine / uterine blood flow / pregnant ewes）gold rank 1。多个具体药物、生理指标和实验对象词项共同缩小候选范围。
- 失败：`11570976`（“Is it Crohn's disease?”）gold 未进 Top-100。query 过短，只提供疾病名；大量 Crohn's disease 文档共享该词，BM25 无法理解问题意图或语义证据差异。
- 失败：`18359123`（“Is it better to be big?”）gold 未进 Top-100。词项几乎都是通用词，缺少可区分的医学实体，BM25 排名被无关但词频匹配的文档占据。
- official test 共有 8 题 gold 未进 Top-10；已保存其中 5 题的 Top-10 文档、分数、gold rank/title/chunk excerpt。

已解决失败只保留根因和最终处理；后续新增失败必须附真实日志或可核验输出。
# 2026-08-08 代码审阅更新

- 已审阅 DS 的 per-dataset edge policy 实现，并在提交 `7377678` 修复影响实验正确性的核心问题。
- 旧 v1 完整摘要 embedding 存在 decode 后重新分词导致的窗口漂移：PubMedQA 6,037 个窗口中 205 个不一致；旧 v1 指标不得作为最终结论。
- 新实现直接编码冻结 token ID 窗口，Dense/Graph/Similarity 共用并严格校验同一 v2 embedding artifact。
- Similarity 边权从冻结向量重算；Adjacent 只连接同文档连续 order，权重固定 `1.0`，不跨 gap。
- Windows 针对性测试 `46 passed`；WSL Docker 为 `125 passed, 2 failed`，两项失败是 spaCy 3.8 与 BC5CDR 3.7 模型环境不兼容。
- v2 BM25-document 已完成：test Recall@10 `0.990`、MRR@10 `0.9619`、nDCG@10 `0.9689`；随后 embedding 阶段未使用 GPU且未生成产物，Dense/Hybrid/Graph 尚未完成。
- 详细审阅：`docs/superpowers/reviews/2026-08-08-per-dataset-edge-policy-code-review.md`。
