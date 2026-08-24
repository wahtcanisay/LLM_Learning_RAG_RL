# 当前阶段

**阶段 5：MedSearch-R1 医学证据搜索 Agent——SFT 数据设计。**

阶段 1、阶段 2 已完成并冻结，阶段 3 已完成最小 MedicalGPT SFT smoke，阶段 4 已完成
Search-R1 3B 单步 GRPO 保底架构。当前进入 MedSearch-R1，先确定 SFT 数据与隔离边界，
不立即编写完整训练工程。

```text
NQ / HotpotQA Parquet
  → Search-R1 多轮 SEARCH / ANSWER rollout
  → 原项目 HTTP 检索服务
  → 开放域 QA Exact Match reward
  → outcome-based GRPO
  → actor policy update
```

当前学习范围固定为 Search-R1 官方数据协议、Agent loop、检索接口、规则 reward、
GRPO advantage 和 actor update；暂不讨论任何领域迁移。

# 当前项目

**MedSearch-R1：基于领域微调与强化学习的医学证据搜索 Agent。**

当前设计入口：

- 项目书：`MedSearch/PROJECT_BOOK.md`
- SFT 训练管线参考：`MedicalGPT/`
- 后续多轮协议参考：`Search-R1/search_r1/llm_agent/generation.py`

# 本周目标

1. 冻结“对话型 Medical-SFT → Search100 + 对话 replay”两阶段 SFT 设计。
2. 审计候选单轮、多轮和 supervised rationale 数据的真实样本。
3. 建立 SFT、Search100、RL 与最终测试集的严格隔离表。

# 今日唯一任务

**2026-08-24：清理 MedSearch 的一次性数据准备产物，恢复可学习的最小工作区。**

当前只保留项目书、正式 Stage-A/B 数据和未来 RL 要复用的检索资产；一次性清洗、审计、
抽样、测试与说明材料统一移出活动目录。本任务不编写 SFT 训练代码。

# 完成标准

- `MedSearch` 根目录只保留一个项目说明文件；
- Stage-A 的 train/dev/manifest 和 Stage-B 的 Search100 均未受损；
- 一次性脚本、测试、样本与审计文件不再出现在活动目录；
- 活动文档不引用已归档的审计文件。

# 已完成

## 2026-08-24：MedSearch SFT 工作区瘦身

- 当前活动根目录只保留 `.gitignore`、`PROJECT_BOOK.md`、正式 `data/` 和一个可恢复的
  `archive/`；不再并列堆放审计说明、抽样目录和一次性脚本；
- Stage-A 活动数据只保留 `stage_a_train.jsonl`、`stage_a_dev.jsonl`、
  `stage_a_manifest.json`；Stage-B 活动数据只保留正式 Search100、医学检索语料与 BM25 索引；
- 清洗、复核、构造、验证脚本及其测试，一次性样本、审计 Markdown、候选判定与失败档案统一
  收进 `MedSearch/archive/data_preparation/`，不参与当前 SFT 编码与训练；
- `PROJECT_BOOK.md` 是当前唯一项目说明来源；旧审计文档不再被活动文档引用。

## 2026-08-23：Stage-B Search100 精品检索轨迹完成

- 从 MedQA/MedMCQA official train 构造 gold-blind 候选；query、检索和最终决策阶段均不读取 gold；
- 真实调用 Search-R1 BM25 医学服务，语料为 MedRAG Textbooks + StatPearls；检索 observation 和稳定
  `document_id` 全部保存在本地审计侧文件；
- 生成 190 条候选（含最初 10 条 pilot），由 `gpt-5.6-sol/xhigh` 按答案、证据、引用和题目质量终审；
- 独立二次复审替换 9 条医学答案正确但存在明显 OCR/拼写噪声的样本，最终 100/100 通过；
- 正式训练集精确保留 100 条：80 条一轮搜索、20 条两轮搜索；顶层字段仅为
  `id/question/options/trajectory`，trajectory 仅含 `search/information/answer`；
- 正式训练文件为 `MedSearch/data/stage_b/search100_training.json`；一次性审计与失败候选
  已在冻结结论和哈希后清理，不进入学生的 SFT 工作区；
- 正式文件 652,019 bytes，SHA-256
  `e9796395e17f6a9c29520cb48758ed9cdbb083c1d231f9562e4d2ef2b25725fb`；
- 清理前自动验证：100 个唯一 ID、字段无审计信息泄漏、answer 均属于 options、轨迹顺序合法；
  当时 53 项测试通过。一次性构建、审计与测试脚本随后按学习边界清理。

## 2026-08-23：Stage-A 医学三池全量清洗

- 数据清洗职责改由教师代理负责，学生从训练搭建开始学习；
- 77,846 条 raw 经确定性规则得到 drop 3,638、review 9,572、keep 候选 64,636；
- 三个 `gpt-5.6-sol` 对 review 做一次性二元终审：accept 7,500、reject 2,072，不留人工队列；
- 目标 Qwen2.5 tokenizer 长度闸门排除 2,494 条，最终 eligible pool 为 69,642 条；
- 三个低成本子智能体完成跨域盲审，修复了多轮 answer-confirmation 过度删除和单轮 false keep；
- 医学三池清洗结果作为下方 65/20/10/5 当前配方的输入；已剪除被替代的 70/20/10
  构建结果，避免学生误用旧 manifest；
- 当前有效统计、字段与冻结配方统一见 `MedSearch/PROJECT_BOOK.md` 和 Stage-A manifest。

## 2026-08-23：Stage-A 加入 5% 通用对话 replay

- 固定 UltraChat 200k 官方 revision `8049631c405ae6576f93f445c6b8166f76f5505a`，只下载并
  校验 `train_sft` 三个 Parquet 分片，共 207,865 条；行数、字节数和官方 SHA-256 一致；
- 确定性规则排除医学、代码、数学、工具动作、时效问法、模板拒答、截断和伪亲历，得到
  121,509 条候选；强模型分层抽检 240 条，accept 151、reject 89，并将 reject ID 固定排除；
- Qwen2.5 真实 chat template 长度闸门保留 59,133 条，排除 62,376 条超 1,024-token 样本；
- 固定 seed `20260823` 重建 65/20/10/5 Stage-A：single 10,663、multi 3,281、reasoning
  1,640、general 821；train 16,077、dev 328，user 重叠 0，强审 reject 命中 0；
- Qwen2.5 Instruct 已有通用指令能力，因此通用数据只作为 5% replay；0%/5%/10% 留作后续
  单变量消融，不把 5% 写成已验证最优值；
- 原 70/20/10 版本已被本配方替代，不再作为当前训练输入。

## 2026-08-23：全量下载验证与首轮风险抽检

- 五个入选文件已全量下载并完成 JSON 解析、记录数与 SHA-256 检查，共 77,846 条记录；
- 固定 seed `20260823` 生成单轮 90 条、多轮 40 条、medical-o1 40 条风险审计清单；
- 三个低成本子智能体只读初筛，共标记 `DROP 21`、`REVIEW 46`、
  `provisional KEEP 103`；该审计集刻意富集风险样本，比例不能外推全库；
- 多轮样本发现缺图和答案机械确认，MedQA-dialog 的 30%～40% 改为待验证上限；
- 当前有效的数据边界和最终选择统一见 `MedSearch/PROJECT_BOOK.md`。

## 2026-08-23：四类 SFT 候选数据各 10 条实物审计

- 固定 Meerkat revision `f0621a0b2eb9f6c48d6ec5299aecc03d6d350afc` 与 medical-o1
  revision `fc2c9e8a37b38f38da6d449564a8c350b244aef4`，通过镜像范围读取原仓库文件；
  没有下载全量数据。
- 四类候选各 10 条已完成抽查；冻结数据来源后，一次性样本和独立审计说明已清理。
- ChatDoctor-cleaned 抽样出现 C. difficile 被误判为 ulcerative colitis、持续胸痛被直接
  归为 gastritis 等安全问题，不能全量直接训练。
- MTS-dialog 的现有 target 是 clinical note generation，不是医生多轮回复；第一版排除。
- MedQA-dialog 是 5～17 messages 的 GPT-3.5 合成多轮考试对话，条件保留为小比例数据；
  缺图、答案确认模板和 MedQA 测试污染必须过滤。
- medical-o1 English 条件保留为少量 supervised rationale；需过滤缺图、异常题面，并
  避免把重复的 `Complex_CoT` 与 `Response` 无设计地串接。
- 当前尚未找到足够可靠的普通医学对话主数据源，因此不启动全量清洗或训练。

## 2026-08-23：Meerkat 第二批四个子集各 10 条审计

- 从同一固定 revision 取得 LiveQA、MedicationQA、MedInstruct-52K 和 MedBooks-18-CoT
  各 10 条；宿主机镜像连接超时后，使用现有 `llm-pytorch` 容器访问完全相同的固定 URL，
  文件写回同一工作区，没有改变数据版本。
- LiveQA 适合作为小型单轮质量锚点，但重复问题对应多个答案，需按规范化问题分组；
  MedicationQA 有过短和表格残片 target，只能过滤后保留。
- MedInstruct-52K 是多任务单轮 instruction，不是真实对话；可作为普通医学 assistant
  主体候选，但需过滤 `As an AI`、时效问题、模板答案和输入输出近复制。
- MedBooks-18-CoT 结构完整且论文有 CoT/textbook augmentation 消融支持，推荐作为第一版
  少量 rationale；它不承担普通对话，也不证明临床安全。
- 当前推荐配方为普通 instruction 75%～85%、多轮 5%～10%、rationale 10%～20%；
  该初步配方已被同日最终决策替换。

## 2026-08-23：阶段 A 数据配方冻结

- 第一版停用 ChatDoctor-cleaned、MTS-dialog 和 MedBooks-18-CoT；ChatDoctor 的医学标签
  与分诊噪声不依赖后续 RAG 修复。
- 普通单轮使用过滤后的 MedInstruct-52K 与去重后的 LiveQA/MedicationQA，目标占比
  45%～55%；多轮 MedQA-dialog 提高到 30%～40%。
- reasoning 只使用 medical-o1 English，目标占比 10%～15%。它的 `Complex_CoT` 作为
  SFT target 时仍属于 supervised rationale，不能写成无监督推理。
- medical-o1 第一版候选长度为 CoT 不超过 512 tokens、整条不超过 1024 tokens；使用
  Qwen2.5 tokenizer 统计后再冻结。超长样本整条过滤，不从中间截断推理。
- 总规模目标约 10K；清洗后不足时按真实数量训练，不用已过滤数据补齐。

## 2026-08-23：SFT 数据职责重新冻结

- 阶段 A 改为多数医学单轮/多轮对话加少量 supervised rationale，不再以
  MedQA/MedMCQA 选择题作为普通医学 SFT 主体。
- 阶段 B 固定为约 100 条真实检索支持的 Search100 加普通医学对话 replay；保留 replay
  是为了维持正常回答能力并降低协议过拟合与逢题必搜风险，但具体比例仍是实验超参数。
- 普通对话不强行添加 `<think>`；reasoning 子集只训练可见、可审计的医学推理说明，
  不把它描述为模型真实的内部思维。普通对话与 Agent 样本通过 system/task prompt 区分。
- MedQA/MedMCQA 官方 train 改作 Search100/RL 候选题池；冻结评测资产边界不变。
- 本次只更新项目书与数据审计，没有下载、转换、tokenization 或训练数据。

## 2026-08-21：MedSearch-R1 两阶段 SFT 与数据实物审计（阶段 A 配方已于 2026-08-23 替换）

- 当时先冻结两个 checkpoint 与 Search100 冷启动框架；其中“医学选择题 SFT + 医学 QA
  replay”的旧配方已被 2026-08-23 的“对话型 SFT + 普通医学对话 replay”替换。
- Search100 改为默认冷启动资产，不再以基础模型协议测试失败为前提；一轮/两轮/直答
  比例仍是待实验超参数。
- 本地 MIRAGE MedQA 文件真实为 1,273 条官方 test；MedMCQA 文件真实为 4,183 条
  official dev。二者均冻结为评测，不进入 SFT、Search100 或 RL train。
- MedMCQA dev 的 `cop` 为 1-4；`exp` 有 1,977 条为空，`topic_name` 有 3,760 条为空。
  因此解释和 topic 都不能设为必需字段，也不能为缺失解释伪造真值。
- 第一版统一字段、target 规则、loss mask 和隔离表已合并进
  `MedSearch/PROJECT_BOOK.md`。本次未下载、转换或训练数据，没有实验指标。

## 2026-08-21：MedQA/MedMCQA 官方 train 固定镜像下载与字段打印

- 固定 `openlifescienceai/medqa@153e61cdd129eb79d3c27f82cdf3bc5e018c11b0` 与
  `openlifescienceai/medmcqa@91c6572c454088bf71b679ad90aa8dffcd0d5868`；直连
  Hugging Face 超时后，通过镜像站下载相同仓库、revision 和路径。
- MedQA 文件 5,679,350 bytes、SHA-256
  `d7bac59f5cee804a006172d44f675559dc5160bdc84dbae05aeed9d5aa1b56f3`，PyArrow 读取
  10,178 行；字段为 `id/data/subject_name`，答案字段嵌套在 `data`。
- MedMCQA 文件 85,899,025 bytes、SHA-256
  `b119434ba551517a6ec0ba1f7e0b4c029165ed284a4704f262ce37c791c493c5`，PyArrow 读取
  182,822 行；Parquet 已把 `cop` 转成 0-3，因此映射是 `"ABCD"[cop]`。
- 两个数据集各 10 条完整字段已检查；一次性打印样本现已清理。MedQA 有 2 个
  精确重复题干且抽样存在缺失图像引用；MedMCQA 有 21,953 条空解释，抽样可见 OCR、
  页码和文本粘连噪声。
- `MedSearch/data/raw/` 已加入局部 `.gitignore`，原始外部数据不会被 `git add -A`
  提交。本次没有转换训练数据或启动训练，没有实验指标。

## 2026-08-20：最新版 veRL/FSDP2 的 3B GRPO 保底架构

- 固定官方 veRL commit `e78efedbe56591cd53ac107150abb7565bb9b8b1`，新增独立
  `Search-R1/modern_verl/` 适配包；Search-R1 XML action、HTTP retrieve、环境 token
  loss mask 和开放域 EM reward 均接入当前 `AgentLoopBase` / custom reward 接口。
- 单卡使用 Qwen2.5-3B-Instruct、LoRA rank `16`、FSDP2、异步 vLLM `0.27.1`、同题
  rollout `2`；没有下载或回退到 1.5B。
- WSL2/Docker 的 vLLM spawn worker 无法重建 CUDA IPC handle；保留 veRL 原有语义，
  通过可复现补丁启用其 CPU shared-memory weight transport。实际日志确认进入 fallback。
- 最终复跑退出码 `0`，完成 `training/global_step:1`：两条轨迹均为 4 turns，step
  `21.549 s`，生成 `13.102 s`，actor update `4.059 s`，权重同步 `3.271 s`。
- 外部 `nvidia-smi` 共采样 `5,420` 点，峰值 `23,117 MiB / 32,607 MiB`；actor 日志
  内部峰值 allocated `14.031 GiB`、reserved `14.789 GiB`。
- 最终 W&B run 为 `lx30qqpq`。两条 rollout 的 EM reward 都是 `0`，因此组内 advantage、
  policy loss、grad norm 都为 `0`；这证明完整训练路径可执行，不证明模型发生有效学习。
- 当前最新版 veRL 的 V1 trainer 还要求额外 TransferQueue 依赖；保底脚本明确使用
  `trainer.use_v1=False` 的稳定调度，但 actor/rollout/AgentLoop/reward 均采用当前接口。
- `data.dataloader_num_workers=0` 已消除 smoke 退出时 DataLoader worker 被杀的噪声；
  上游 W&B `Tracking.__del__` 仍可能在 Ray teardown 后打印 closed-transport traceback，
  发生于 step 和 final validation 之后且进程退出码为 `0`，记录为非训练路径清理告警。

## 2026-08-20：vLLM 单卡 hybrid-engine 权重交换修复

- 根因：旧 sharding manager 在 GPU 上物化 FSDP `state_dict` 后一次性唤醒 vLLM 的
  weights 与 KV cache，导致 Actor、交接张量、vLLM 权重在同一卡上重叠并 OOM。
- 修复后的时序：CPU state-dict → Actor 参数移出 GPU → vLLM 只唤醒 weights → 写入
  最新 Actor 权重 → 唤醒 KV → 生成 → vLLM level-2 sleep → Actor 参数恢复到 GPU。
- 真实 Ray/vLLM 运行连续完成两次交换和两次生成，退出码 `0`；不再出现
  `cumem_allocator.cpp:211`。
- `nvidia-smi` 每 `0.5s` 采样 `777` 点，峰值为 `32058 MiB`；原始记录位于
  `Search-R1/experiments/single_gpu_smoke/vllm_gpu_memory.csv`。
- 权重交换与已有兼容回归共 `10/10` 通过。该方案以 CPU 传输延迟换取单卡显存可行性，
  不声称具备多卡 NCCL 方案的吞吐。

## 2026-08-20：固定双跳样本无训练 Agent 闭环

- 固定子集：train 为 HotpotQA 原始行 `86094`，test 为 2WikiMultihopQA 原始行
  `41745`；随机种子 `20260819`。
- 使用本地 `Qwen2.5-3B-Instruct`、仓库原生 HF rollout、`max_turns=2`、BM25
  top-k `3` 完成无参数更新验证，进程退出码 `0`。
- 真实轨迹依次检索电影导演与 `Jeff Burr birthplace`，最终输出
  `<answer>Aurora, Ohio</answer>`，规则 EM reward 为 `1.0`。
- `ACTIVE_TRAJ_NUM: [1, 1, 1, 0]` 表明轨迹在最终回答后结束；完整证据保存在
  `Search-R1/experiments/single_gpu_smoke/inference.log`。
- 两次 observation 分别从 `471`、`514` tokens 截到 `256`；本题仍答对，但后续实验
  必须记录该截断，不能把一次成功外推为检索信息总是充分。
- 本次未启动外部显存采样器，因此 HF 运行的显存峰值记为“未知”，不以失败运行或
  纸面估算补填。

## 2026-08-20：官方 wiki-18 BM25 检索服务闸门

- 下载并校验官方 `PeterJinGo/wiki-18-bm25-index`：28 个索引文件，合计
  `2,297,611,339` bytes。
- 下载并校验官方 `PeterJinGo/wiki-18-corpus`：压缩文件 `5,123,307,260` bytes，
  SHA-256 `7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db`
  与 Hugging Face LFS oid 一致，`gzip -t` 通过。
- 发现上游文件虽命名为 `wiki-18.jsonl.gz`，实际是 tar.gz；正确提取出的
  `wiki_dump.jsonl` 为 `14,393,573,105` bytes、`21,015,324` 篇文档，字段为
  `{id, contents}`。
- 修复全局升级后的实际兼容问题：Pyserini 会连带导入 FAISS，旧 `faiss-cpu 1.8.0`
  与 NumPy `2.3.5` ABI 不兼容；已升级到 `faiss-cpu 1.15.0`。
- 作者原有 CPU BM25 server 已启动；真实 `POST /retrieve` 返回 HTTP 200、3 条带正文
  和分数的结果。请求与响应保存在 `Search-R1/experiments/single_gpu_smoke/`。
- 一次混用 Xet 分块缓存与普通 Range 续传产生了哈希不匹配文件，已隔离为
  `.corrupt-90124a`，未解压、未用于服务；待正确资产和服务验收后清理。

## 2026-08-20：vLLM 0.27.1 兼容迁移与训练资产下载

- 保持 Search-R1/veRL 上层同步调用方式不变，新增版本专用兼容层，将旧的
  `generate()`、`sync_model_weights()`、sleep/wake 和 cache 方法映射到 vLLM `0.27.1`。
- 该兼容层明确限定单卡 `TP=1`；由于 WSL2 不支持当前 vLLM V2 runner 所需的 CUDA
  UVA，并且跨进程同卡 CUDA IPC/NCCL 路径不可用，改用 vLLM 官方 public `LLM` 的
  单进程 V1 runner。未修改 Search-R1 的 trainer、rollout 或 GRPO 算法调用方式。
- 兼容层单元测试 `2/2` 通过，Python `compileall` 与 `git diff --check` 通过。
- RTX 5090 真实 smoke 通过并以退出码 `0` 输出 `SMOKE_OK`：初始生成是
  `Paris. The capital of Germany is Berlin`；将内存 actor 的末层 norm 权重置零并同步后，
  输出变为 `!!!!`，证明 actor 权重确实进入 vLLM，而非只验证导入。
- `RotaryEmbedding: Failed to load weights` 已定位为 vLLM 自建的
  `persistent=False` cos/sin 位置缓存不属于 HF `state_dict`；reload 代码保留原缓存，
  不是 actor 参数漏同步。
- Qwen2.5-3B-Instruct 两个 safetensors 分片已完整下载：`3,968,658,944` 与
  `2,203,268,048` bytes；模型位于工作区忽略目录 `models/Qwen2.5-3B-Instruct/`。
- 官方 `PeterJinGo/nq_hotpotqa_train` 数据已完整下载：train `169,615` 行、
  `355,663,891` bytes；test `51,713` 行、`70,370,337` bytes。文件位于
  `Search-R1/data/nq_hotpotqa_train/`，后续 smoke 会从全量文件抽取小子集。
- 没有执行仓库 `scripts/download.sh`：脚本硬编码作者目录、参数名与当前 Python CLI
  不一致，而且默认下载约 64.6 GB Dense Flat 索引；这与当前固定的 CPU BM25 路线不符。

## 2026-08-19：按用户决定直接升级 Docker 全局训练环境

- 容器 `llm-pytorch` 已直接升级为 PyTorch `2.13.0+cu130`、vLLM `0.27.1`、
  Transformers `5.15.0`、Ray `2.57.0`、W&B `0.28.2`、TensorDict `0.14.0`。
- NVIDIA driver `591.74` 声明支持 CUDA 13.1；真实 CUDA tensor 测试通过：RTX 5090、
  compute capability `(12, 0)`、`torch.ones(..., device='cuda') == 1.0`。
- Search-R1 已通过 editable/no-deps 安装；`verl.__file__` 指向工作区
  `/workspace/code_list/some tricks/LLMLeanring/Search-R1/verl/__init__.py`，确认使用本地源码。
- `RayPPOTrainer` 与 `RewardManager` 能导入；`ActorRolloutRefWorker` 在本地
  `verl/third_party/vllm/__init__.py` 主动拒绝 vLLM `0.27.1`：该兼容层只允许
  `0.3.1/0.4.2/0.5.4/0.6.3`。不得只删除版本检查后假定 API 兼容。
- vLLM `0.27.1` 要求 PyTorch `2.13.0`，但其元数据仍安装 TorchAudio `2.11.0`
  CUDA 12.8 build；因 Search-R1 是纯文本任务且二进制不兼容，已卸载 TorchAudio。
- 直接全局升级已如预期破坏旧项目依赖闭包：MedicalGraphRAG、sentence-transformers、
  LangChain 等要求旧 Transformers/NumPy/AnyIO/Packaging。当前阶段接受该代价，不回滚。
- 训练数据没有下载：`Search-R1/data` 中不存在 Parquet。wiki-18 只有约 2.08 GB 的
  `part_aa.incomplete`，而该分片完整大小约 42.9 GB，明显不是“只差一点”；不再续传
  64.6 GB Dense Flat 索引，训练 smoke 只准备少量 NQ 样本。

## 2026-08-19：Docker 与 Search-R1 运行环境审计

- 复用容器：`llm-pytorch`（容器 ID 前缀 `c8f4518d842b`），Linux/WSL2，项目在容器内
  映射为 `/workspace/code_list/some tricks/LLMLeanring/Search-R1`。
- GPU 透传正常：RTX 5090，32 GB；PyTorch `2.11.0+cu128`、CUDA 12.8、BF16 可用；
  根文件系统约 1007 GB，检查时约 912 GB 可用。
- 已有 `transformers 4.44.2`、`datasets 2.16.1`、`pyserini 0.22.1`、
  `faiss-cpu 1.8.0`；缺少 `vllm`、`ray`、`hydra-core`、`omegaconf`、`wandb`。
- 当前 Python 为 `/usr/bin/python` 3.12，并受 PEP 668 保护；直接全局 `pip install`
  被拒绝。后续必须在容器内创建隔离环境，不能使用 `--break-system-packages` 污染 MedRAG。
- 官方旧环境固定 Python 3.9、PyTorch 2.4/CUDA 12.1、vLLM 0.6.3；vLLM 0.6.3
  早于 RTX 5090/SM120 支持，不能在当前硬件上原样运行。不得直接执行旧 requirements。
- `vllm==0.9.0` 是首批可支持 RTX 5090 的版本，但依赖 dry-run 显示它会固定
  `torch==2.7.0`、`transformers>=4.51.1`、`ray>=2.43.0` 等，与容器当前全局环境及
  vendored veRL 的旧接口均不相同；只能放入隔离环境并先做兼容性闸门测试。
- Search-R1 官方的新版 veRL/SGLang 集成仍可作为参考，但当前 veRL 文档注明内置
  `SearchTool` 已移除；不能直接追随 latest，必须固定到仍包含该工具的官方快照。
- 官方 Dense Flat 索引为 64.6 GB，语料压缩包约 5.12 GB；当前 `data/wiki-18`
  只有约 2 GB 的未完成下载，没有可用索引或解压语料。官方 BM25 索引约 2.3 GB，
  更适合作为单卡训练时的独立 CPU 检索服务。
- `Qwen/Qwen2.5-3B-Instruct`、NQ/HotpotQA Parquet、W&B 登录信息均尚未就绪；
  当前没有产生训练 step、W&B run 或实验指标。

## 2026-08-19：Search-R1 源码阅读边界收束

- `fsdp_workers.py` 确认为 veRL 执行基础设施，不继续阅读 device mesh、FSDP、offload、
  vLLM 构建或 Ray RPC 细节。
- 当前掌握标准固定为：知道 actor+rollout worker 负责生成/更新，独立 reference 提供
  KL 对照，GRPO 不创建 critic；能够使用这些接口即可。
- Search-R1 特有的 data、Agent loop、reward、GRPO advantage、actor loss 与 `fit()`
  调度已经形成完整理解链，源码阅读阶段在此收束。
- 后续仅在真实运行出现明确错误时，沿堆栈按需查看 veRL 内部。

## 2026-08-19：路线剪枝——跳过 `_validate()` 深入阅读

- `_validate()` 使用独立 `val_files`；只有配置 `val_data_num` 时才从验证文件内部抽样，
  不是从当前训练 batch 临时切出测试样本。
- 它只生成验证轨迹、计算平均规则 reward 并写日志，不计算 advantage 或更新 actor；
  当前阶段不把它作为 Search-R1 核心机制继续阅读。
- 已保留一个运行阶段待核实问题：多轮路径创建 active 子批次时丢失 meta_info，静态代码
  显示 `do_sample/recompute_log_prob` 会回退到默认 `True`。
- 下一学习点直接进入 actor/rollout/reference worker 的角色与初始化边界。

## 2026-08-19：`RayPPOTrainer.fit()` 完整调度链阅读验收

- 已完整追踪一批题目从 `DataProto`、`n_agent` rollout、old/reference log-prob、
  rule reward、GRPO advantage、information loss mask 到 actor update 的执行顺序。
- 已能解释 `batch.union(final_gen_batch_output)` 是题目 metadata 与完整轨迹 tensor 的
  汇合点；删除后最先在 `_balance_batch()` 读取 `attention_mask` 时失败。
- 已能说明后续 `RewardManager` 缺少 prompts/responses/mask，reward 无法产生，继而
  advantage、ratio、loss 和反向更新均无法执行。
- 已进一步区分直接由 rollout/union 带回的字段与后续派生字段：完整轨迹、mask、
  `old_log_probs` 是直接输入；`ref_log_prob`、scores/rewards、advantages/returns 和
  `loss_mask` 是合并后逐阶段计算得到。
- 本次为源码阅读验收，没有运行模型或训练，没有新增实验指标。

## 2026-08-18：Search-R1 NQ 数据 → batch → uid 阅读验收

- 已读完 `nq_search.py`、`rl_dataset.py`、`ray_trainer.py`、`main_ppo.py`、`qa_em.py` 的指定代码段；已掌握的 reward/GRPO 重复段未重复阅读。
- 已能列出 Parquet 五个顶层字段，并解释 prompt 与 gold answer 隔离是字段协议与防止答案泄漏的共同要求。
- 已纠正 tensor shape：`__getitem__()` 返回单样本 `[max_prompt_length]`，`collate_fn()` 后为 `[batch_size, max_prompt_length]`，再按 `n_agent` 扩展 batch 维。
- 已理解 `n_agent` 在 rollout 前扩展轨迹，而 uid 不参与生成，只在后续 `compute_advantage()` 中提供分组标签。
- 已定位真正的分组代码：`compute_grpo_outcome_advantage()` 通过
  `id2score[index[i]].append(scores[i])` 按 uid 收集同题 rollout，并计算组内均值、标准差和 advantage。
- 已理解 `_read_files_and_tokenize()` 当前不 tokenize、`filter_prompts=True` 当前不生效，以及 `drop_last=True` 只丢弃不完整尾批次。
- 本次是源码复述验收，没有运行模型或训练，没有新增实验指标。

## 2026-08-16：GRPO advantage → actor policy loss 阅读验收

- 已能解释 `old_log_prob` 来自 rollout 行为策略，`log_prob` 来自更新中的当前 actor；`ratio` 是逐 token 的新旧策略概率比，不是整条 trajectory 的单一偏好分数。
- 已理解正/负 advantage 分别提高/降低已采样 token 的概率；PPO clip 裁剪 surrogate objective 的有利更新方向，不是把模型真实概率硬锁在区间内。
- 已理解 information 是外部检索 observation：参与 attention 以指导后续生成，但通过 `loss_mask` 排除在 policy loss 之外。
- 已区分 entropy bonus、reference-policy KL 与 `ppo_kl` 监控量：entropy 鼓励分布保持探索性；reference KL 限制当前 actor 偏离冻结的 SFT/reference 模型；`ppo_kl` 监控 current actor 与 rollout old actor 的变化。
- 已通过最小 entropy 判断：分布越均匀 entropy 越高，越集中 entropy 越低。该项是 veRL PPO actor 原有的可选正则，官方快照默认 `entropy_coeff=0.001`，不是本项目新增算法。
- 本次只完成代码阅读和概念验收，没有运行模型或训练，没有新增实验指标。

## 2026-08-14：Search-R1 reward → GRPO advantage 阅读验收

- 已能解释序列级 0/1 outcome reward 写在最后一个有效 response token，是因为结果只能在完整轨迹结束后判断，同时需适配 veRL 的 token-level reward 接口并避开 padding。
- 已区分 PPO/GAE 与 GRPO：GAE 使用 token value 从后向前传播 advantage；GRPO 不使用 value critic，而是对同题多条 rollout 的序列分数做组内标准化，再广播到有效 response token。
- 已正确判断全 1 或全 0 rollout group 的 advantage 为 0；进一步明确这表示该组没有策略梯度信号，是有效样本比例问题，不等于整个 GRPO 训练必然坍塌。
- 本次为源码复述验收，没有运行模型或训练，没有新增实验指标。

## 2026-08-13：Search-R1 reward 学习注释准备

- 已为 5 个 reward 主链函数补充输入类型、输出类型、Tensor shape、上游调用位置和下游数据流。
- 验证命令：

```cmd
python Search-R1\scripts\verify_source_notes.py
python -m compileall -q Search-R1\verl\trainer\main_ppo.py Search-R1\verl\utils\reward_score\qa_em.py Search-R1\verl\trainer\ppo\ray_trainer.py
git diff --check
```

- 验证结果：8 个 Python 文件和 1 个 Bash 文件保持 comment-only；3 个 reward + 3 个 action parser 样例通过；Python 编译通过。
- 边界：注释未改变执行逻辑；reward 代码复述已于 2026-08-14 验收，但没有运行训练或产生训练指标。

## 2026-08-13：Search-R1 多轮 rollout 主干阅读

- 已理解 `active_mask` 控制仅对未结束样本生成；search 注入 `<information>` 后继续，answer 标记结束，非法动作按当前 parser 路径处理。
- 已区分 `rollings`（下一轮输入）与 `original_right_side`（最终训练轨迹）。
- 已理解 `max_turns` 后仍有一次 `do_search=False` 的 final rollout，以及 `attention_mask` 与用于排除环境 information token 的 `info_mask`。
- 本次只做源码阅读和协议检查，未运行模型、检索服务或训练。

## 2026-08-12：Search-R1 主干源码学习注释

- 已注释数据处理、dataset、generation、tensor helper、retrieval server、reward、训练入口和 trainer 主链。
- 已建立 `Search-R1/docs/source_code_learning_zh.md` 和标准库验证脚本 `Search-R1/scripts/verify_source_notes.py`。
- 官方 8 GPU 配置不能直接用于单张 RTX 5090；单卡配置必须以后通过真实 smoke 确认。

## MedicalGPT SFT：已验证范围

- 已跑通 Qwen2.5-3B、BF16 LoRA 的 100 条 smoke：rank `8`、alpha `16`、batch `2`、gradient accumulation `4`、max length `1024`、1 epoch、12 steps、seed `20260809`。
- 真实结果：train loss `2.332629`、eval loss `2.008307`、perplexity `7.450692`、runtime `13.9074s`、整卡显存峰值 `17,113 MiB`。
- 结论边界：只证明训练链和 adapter 加载有效；3 条 validation 中 2 条输出变化，不代表医学能力提升。

## Medical GraphRAG / LinearRAG：冻结结论

- 已实现 BM25、Dense、RRF Hybrid、Graph、Qwen3-Reranker Hybrid2 的统一检索入口和四套真实基准评测。
- `MedicalGraphRAG/` 固定在 README 结果对应的 commit `fe89a64`；未验收扩展仅保留在 Git 历史。
- 图检索在四套基准均未胜出。该负结果保留，不再通过无依据调参追逐指标。

# 遇到的问题

| 问题 | 当前处理 |
|---|---|
| Search-R1 官方配置面向 8 GPU | 单卡 smoke 使用 3B、小 batch、2 条 rollout、短序列；参数只能通过真实运行确认。 |
| vendored Search-R1 兼容层只接受 vLLM ≤0.6.3 | 已新增 vLLM 0.27.1、单卡 TP=1 兼容层，并通过 CPU 交接与分阶段 wake 修复完整 Ray 路径的瞬时 OOM；真实两轮生成退出码为 `0`。 |
| 全局升级造成旧 MedRAG/GraphRAG 依赖冲突 | 用户明确选择不隔离；保留 `pip check` 结果，当前不回滚。 |
| 单卡 32GB 的 GRPO 资源边界未知 | 不做纸面外推；用真实日志记录 batch、rollout、长度、显存和耗时。 |
| 官方检索 observation 超过 `max_obs_length=256` | 当前按原逻辑截断；固定双跳样本仍得到 reward `1.0`，后续只把长度作为实验配置记录，不研究 retriever 内部。 |
| 多轮验证没有传递 `do_sample=False/recompute_log_prob=False` | 静态路径显示 active 子批次丢失 meta_info；当前先记录，不修改行为，运行阶段再实测。 |
| WSL2/Docker 跨 vLLM spawn 进程的 CUDA IPC 报 `invalid resource handle` | 为当前 veRL 增加环境变量控制的 shared-memory fallback 补丁；默认 native Linux 行为不变。 |
| smoke 的轨迹数小于默认 8 个 AgentLoop workers | 固定 `agent.num_workers=2`，与两条 rollout 等分。 |
| 两条 smoke trajectory 的 reward 都为 0 | advantage/loss/grad_norm 均为 0；保留为 GRPO 同组无差异信号的真实案例，不包装成训练提升。 |

# 下一步

- 当前唯一任务：学生阅读 `MedSearch/data/stage_b/search100_training.json` 中一条一轮轨迹和
  一条两轮轨迹，用自己的话解释四个顶层字段以及 `search/information/answer` 的数据流。
- 完成解释后，下一步才由学生编写极简渲染函数：把结构化 trajectory 转换成 Search-R1 标签文本，
  并验证 question/information 只参与 attention、search/answer 才计算 SFT CE loss。


# 待补知识

- MedQA-dialog 清洗后的实际保留量能否支持 30%～40% 多轮比例。
- medical-o1 在 512-token CoT 阈值下的保留率。
- Search100 一轮/两轮/直答比例和普通医学对话 replay 比例。
- Search100 是否需要过采样，才能在保留 replay 时学会稳定协议。

# 实验结果总表

| 数据集 / 实验 | 当前最优或关键结果 | 结论边界 |
|---|---|---|
| PubMedQA 封闭基准 | Dense：R@10 `0.994`、MRR `0.977786`、nDCG `0.981885` | 5,000 documents，不代表全量 PubMed。 |
| NFCorpus | Hybrid2 nDCG@10 `0.384` | 多相关 qrels；Graph 未胜出。 |
| SciFact | Hybrid2：R@10 `0.895`、MRR `0.740`、nDCG `0.772` | Graph 未胜出。 |
| HotpotQA | Hybrid2：R@10 `0.898`、MRR `0.955`、nDCG `0.865` | 每题 2 个 gold；Graph 未胜出。 |
| MedicalGPT 100 条 smoke | eval loss `2.008307`、PPL `7.450692` | 只验证训练管线。 |
| Search-R1 3B GRPO 单步 | step `21.549 s`；GPU peak `23,117 MiB`；W&B `lx30qqpq` | 两条 reward 均为 0，验证训练链路，不代表学习提升。 |

# 失败案例

| 场景 | 原因 | 保留结论 |
|---|---|---|
| PubMedQA 上 RRF Hybrid 低于 Dense | BM25 高位错误稀释 Dense 信号 | 融合不是默认更好。 |
| Graph 在四套基准均未胜出 | 实体传播信号与任务/语料不匹配 | 保留负结果，不包装为提升。 |
| TREC-COVID 未接入 | 24.6% 文档正文为空，部分 qrels 指向空文本 | 不用标题回退伪造正文检索。 |
| MedicalGPT 内部显存统计路径停滞 | Trainer 路径未形成完整产物 | 该次运行不计入；已用外部只读采样完成 smoke。 |
| Search-R1 单步两条 rollout 均答错 | 同题组 reward 全 0，GRPO advantage 与 gradient 为 0 | 保留为“同组无差异则无学习信号”的真实失败样本；扩大实验时监控有效组比例。 |

> 更早的逐日过程、已撤回方案和详细实验配置由 Git 历史、各子项目 README 与 `experiments/` 保留；本文件只维护当前有效状态。
