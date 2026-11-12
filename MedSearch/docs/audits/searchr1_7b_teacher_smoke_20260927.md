# Search-R1 7B 教师下载与短评分验收

日期：2026-09-27

- 模型：`PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo`，revision `74d46a43224daa612b5f27174e1c58b35587ee92`。
- 项目目录：`MedSearch/models/external_baselines/SearchR1-Qwen2.5-7B-Instruct/`。共 16 个文件、7 个 safetensors 分片，文件总大小 30,478,412,494 字节。七个项目分片的 SHA256 均与该 revision 的 Hugging Face 下载元数据一致。

| 分片 | SHA256 |
|---|---|
| 00001 | `edd60e6e32accbfbeb58e119c0ca25a4de2e2b5e448f561d6940d70d1988bfb6` |
| 00002 | `a1938244e04fe85301398e1a4a97b57c401d26061d619eb3077f5b7878a13427` |
| 00003 | `9d75d4814a6764df95e1a8dd8068252df4679914137d0dd357c7879d42076755` |
| 00004 | `57a3c299358bea2686774ffeef8bfee1603536bda5cb6f88a1c599cb7f83b538` |
| 00005 | `0dc42bad4fffcd0f7e37f351fff852f9cf0aa887e88b073e03fd6280980c5f92` |
| 00006 | `c54df758d79acef06ee6d19b562b570d1658e24e54951378dc989f1f3ec8bb33` |
| 00007 | `2a07720200b7a96b1ed0cf8824472d62ddc079626f9683d8bc77bfd471a56b9a` |
- 环境：既有 WSL Docker 容器 `llm-pytorch`，单张 RTX 5090 32GB。模型从容器 `/tmp/medsearch_teacher_7b` 加载；项目目录副本经逐分片哈希核对。使用 `MedSearch/tools/smoke_teacher_7b.py` 调用当前 `load_teacher` 和 `score_tokens`，没有启动训练或付费 API。
- 真实 BF16 模型加载成功；20 个 prompt token + 16 个搜索动作 token 的 sampled 评分输出形状 `[1,16]`，分数均有限，范围 `[-19.6981, 0.0]`。该次进程 CUDA peak allocated 14.286 GiB，加载加评分 5.31 秒。原始结果见 `searchr1_7b_teacher_smoke_20260927.json`。
- 首次 smoke 失败于辅助脚本把 chat template 返回的字符串当作 token ID；改为显式 `tokenizer.encode(..., add_special_tokens=False)` 后重跑通过。学生核心 `load_teacher`、`score_tokens` 未因此修改。

此结果证明固定教师在当前环境能加载并给短动作评分。P2 要求的“学生 rollout → 判定 → 卸载/切换 → 教师评分 → 恢复学生前向”尚未实测；也未验收长上下文、长轨迹或持续训练显存。较长训练前仍需 checkpoint 保存、重载及一次动作推理验收。
