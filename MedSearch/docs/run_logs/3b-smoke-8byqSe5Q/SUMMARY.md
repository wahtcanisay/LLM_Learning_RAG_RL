# 2026-08-31：第二次保存重载探针失败（教师推理精度不一致）

命令：`wsl.exe bash MedSearch/docs/tests/run_sft_3b_smoke.sh --save-roundtrip`，退出码 1。

- 单步训练通过：global_step=1，loss=1.173087239265442，Trainer runtime=9.0682 秒；
  252 个 LoRA 张量变化，434 个冻结基座张量哈希不变。
- 学生保存函数实际成功。产物：`MedSearch/outputs/smoke/stage_a-2hm0nob1/adapter/`。
- 重载后的 504 个 LoRA 张量逐项相等、434 个基座哈希一致、tokenizer/模板检查通过。
- 失败于最后位置全词表 logits 对比：141,424/151,936 个元素超出 atol=rtol=1e-5，
  最大绝对误差 0.625。不能将该次记录判作输出一致性通过。
- 源码定位：Accelerate `prepare_model` 给训练模型 forward 保留了 BF16 autocast 包装；
  教师脚本重载后却使用普通前向。参数一致并不意味着两种计算精度的输出逐值相等。
- 修正仅在 docs 内教师探针：保存前后统一 BF16 autocast，不放宽阈值；新增 tiny CPU
  “带 AMP forward 包装/不带包装”一致性检查。完整 CPU 回归 `33 passed in 10.22s`。
- 下次运行将参考 logits 和重载 logits 也写入独立 safetensors 证据文件。

本目录保留原始日志、显存 CSV；adapter、配置和训练指标保留在对应 smoke 目录。
该运行是 BF16 LoRA，不是 QLoRA；教师未修改学生 main 或核心函数。
