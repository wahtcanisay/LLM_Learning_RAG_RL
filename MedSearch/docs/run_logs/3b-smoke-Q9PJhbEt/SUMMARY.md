# 2026-08-31：首次保存重载探针失败（教师输入格式问题）

命令：`wsl.exe bash MedSearch/docs/tests/run_sft_3b_smoke.sh --save-roundtrip`，退出码 1。

- 训练部分通过：固定 8 条 Stage-A，BF16 LoRA，global_step=1，loss=1.173087239265442，
  Trainer runtime=6.5552 秒。252 个 LoRA 张量变化，434 个冻结基座张量哈希不变。
- 失败发生在教师 `observe_save` 构造固定输入时，尚未调用学生 `save_adapter`。
  错误为 `Could not infer dtype of tokenizers.Encoding`。
- 根因：Transformers 5.10.4 的 `apply_chat_template` 默认 `return_dict=True`；教师脚本
  错将返回结果当作整数 token ID 列表。不是学生保存接口或模型训练错误。
- 修正：教师探针显式设置 `return_dict=False`；补 CPU 测试验证返回整数列表、能构造二维
  tensor、与渲染文本再编码一致。完整回归 `32 passed in 10.48s`。
- 已完成训练的指标和配置保留于 `MedSearch/outputs/smoke/stage_a-8pqp9lor/`；
  adapter 子目录在失败前创建，但没有保存权重。不从 metrics 伪造 adapter。
- 原始日志 `run.log`、整卡显存记录 `gpu_memory.csv` 均保留；重新运行使用全新目录。

此次运行不是 QLoRA，没有修改学生主训练代码，不作为保存重载通过的证据。
