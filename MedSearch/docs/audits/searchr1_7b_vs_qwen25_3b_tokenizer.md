# Search-R1 7B / Qwen2.5 3B tokenizer 对照

日期：2026-09-26

## 对照对象

- 教师候选：[PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo](https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo)，revision `74d46a43224daa612b5f27174e1c58b35587ee92`。它列在 [Search-R1 官方 collection](https://huggingface.co/collections/PeterJinGo/search-r1) 中；蒸馏计划仍须最终固定教师 checkpoint。
- 学生：`Qwen/Qwen2.5-3B-Instruct`，本地 revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`。
- 只下载了教师 tokenizer 文件和小型 `config.json`，没有下载 7B 权重。教师文件保存在 `MedSearch/models/external_baselines/SearchR1-Qwen2.5-7B-Instruct-tokenizer/`。

## 结果

- 两侧实际 tokenizer 都有 151,665 个 token，完整 token→ID 映射完全相同：ID 错位 0、独有 token 0；ID 范围均为 0–151,664。
- BPE merges（忽略教师文件多出的版本头）、normalizer、pre-tokenizer、post-processor、special token 字符串和 chat template 一致。Search/answer 动作、医学术语/数值三个样例的 token IDs 完全一致。
- 模型输出层 `config.vocab_size` 不同：教师 152,064，学生 151,936。实际 tokenizer token IDs 都能被学生输出层表示；教师模型配置比学生多 128 行输出维度。相对真实 token ID 上界，学生/教师配置分别有 271/399 行额外输出行。这是输出维度 padding 差，不是新 tokenizer token 或 ID 位移。
- 两侧模型配置对 special ID 的命名不同：学生 `bos_token_id=151643`、未设 `pad_token_id`；教师未设 `bos_token_id`、`pad_token_id=151643`。Tokenizer 文件中两侧的 `<|endoftext|>`、`<|im_end|>` 和 chat template 一致；训练/打分 batch 应按真实长度构造 attention mask，不应仅通过 token ID 相等判断 padding。

## 蒸馏接线含义

按 token ID 做 A 档序列 CE 不需要扩充或重排学生 tokenizer。B 档 Top-k 和在线 OPD 需要共同 logits 支持：取教师 Top-k 前要保证目标 ID 落在学生输出维度内；建议对两侧未映射到真实 tokenizer token 的输出行做一致屏蔽后再归一化，并保留这项处理的配置与测试。尚未加载 7B 权重，因此还没有实测 padding 行是否会进入实际 Top-k；最终教师 checkpoint 若改变，需重跑本核对。

详细哈希及逐项结果见同目录 `searchr1_7b_vs_qwen25_3b_tokenizer.json`。
