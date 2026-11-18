# P2 单卡短生命周期实测（2026-09-27）

## 环境与输入

- 既有 WSL Docker `llm-pytorch`，RTX 5090 32 GB；未训练、未联网、未调用付费 API。
- 学生：本地 Qwen2.5-3B-Instruct BF16，容器副本 `/tmp/medsearch_student_3b`。两片 SHA256 与 Windows 原文件一致：`67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2`、`a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d`。
- 教师：固定 Search-R1 7B revision `74d46a43224daa612b5f27174e1c58b35587ee92`，此前七片哈希审计见 `searchr1_7b_teacher_smoke_20260927.md`。
- 辅助脚本：`tools/smoke_phase_switch.py`；调用学生已写的 `rl/distillation_hook.py` 和 `distillation/teacher.py`。原始数值见 `phase_switch_smoke_20260927.json`。

## 实测

输入为 20 个 prompt token 与 16 个 `<search>…</search>` 目标 token。直接 HF 3B 前向后，脚本模拟当前 veRL `param_offload=true` 的自动 CPU 卸载，调用 `prepare_teacher_phase`；加载 BF16 7B 并调用现有 `score_tokens`；教师释放后调用 `restore_student_phase`，再模拟 veRL 下一次 eval 前的自动恢复并完成 3B 前向。恢复前后学生下一 token ID 均为 `3862`，教师分数形状 `[1,16]` 且全部有限。

| 阶段 | 第二次运行耗时 | 阶段结束 GPU 已用 | PyTorch allocated 峰值 |
|---|---:|---:|---:|
| 3B 首次前向 | 2.35 s | 7.797 GiB | 5.892 GiB |
| 3B CPU 卸载与 prepare hook | 3.48 s | 1.780 GiB | 5.885 GiB |
| 7B 加载与评分 | 5.87 s | 16.157 GiB | 14.286 GiB |
| 教师释放与 restore hook | 0.30 s | 1.780 GiB | 14.258 GiB |
| 3B 恢复前向 | 1.19 s | 7.797 GiB | 5.892 GiB |

首次运行的 7B 加载与评分为 13.30 s，第二次为 5.87 s；同一短流程受文件缓存影响明显，不能用其中一个数值外推完整 RL 训练速度。第二次各阶段合计约 13.19 s，未包含 Ray、vLLM、检索、verifier、优化器与长序列开销。阶段结束 GPU 已用由 `torch.cuda.mem_get_info()` 取得；PyTorch 峰值仅覆盖本进程分配。

## 验收边界

该实测通过了单卡真实 3B/7B 权重的短前向、教师 sampled 评分与返回 3B 前向。hook 通过轻量 runtime 调用；学生参数搬移由脚本模拟 veRL 自动卸载，未启动真实 `RayPPOTrainer`、FSDP actor worker 或 vLLM rollout，也未运行 verifier 判定/失败批次或优化器恢复。因此工程书 P2 的完整“学生 rollout→判定→教师评分→恢复学生前向”仍待真实 veRL 小批验收，不能进入长训练。

## 复现命令（CMD）

容器已有上述本地模型副本及项目代码时：

```cmd
docker exec -w /tmp/medsearch_p1 llm-pytorch python -m MedSearch.tools.smoke_phase_switch /tmp/medsearch_student_3b /tmp/medsearch_teacher_7b /tmp/medsearch_p1/phase_switch_smoke.json
```
