# P6：V1 教师评分接线验收

日期：2026-11-09。基准 veRL commit：`23af6a7a2e8d6efeeb2adbe5d1689c7a24f503a3`。

## 本次改动

- `_compute_failure_log_prob` 按当前批次顺序读取轨迹，调用项目教师调度，向同一分区写回 `teacher_logprobs` 和 `opd_mask`。继续返回原 `KVBatchMeta`，保留 `extra_info`。
- `_step_once` 在 `_balance_batch` 后、`_compute_old_log_prob` 前调用桥接。`trainer.opd_coef` 默认为 0；正值时把 `trainer.teacher_path` 与运行目录下的 `teacher_scores/` 传给教师评分器。
- 仅导出 `trainer_base.py` 改动到 `patches/verl_v1_failure_teacher.patch`。`.deps/verl` 继续作为本地依赖。补丁去除了行尾多余空格，逻辑保持与验收版本一致。

在主仓库根目录，对固定 commit 的干净依赖应用补丁：

```bat
git -C MedSearch/.deps/verl apply --check ../../patches/verl_v1_failure_teacher.patch
git -C MedSearch/.deps/verl apply ../../patches/verl_v1_failure_teacher.patch
```

## 验收结果

WSL `llm-pytorch`，MedSearch 固定 veRL，隔离 CPU Ray 和真实 TransferQueue。执行实际 `_step_once` 和队列桥接，教师评分与训练阶段用模拟回调。

| 场景 | 结果 |
|---|---|
| 系数未配置或显式为 0 | 均跳过教师，原 GRPO 调用顺序保持不变 |
| 系数为正 | 重排后评分，旧策略概率计算前已能回读对应轨迹的教师字段；`temperature` 保留 |
| 教师评分报错 | 恢复调用执行，原异常向上传递，未进入旧概率计算或更新 |

四种情况通过，退出 0。原始验收脚本与日志保存在本地 `tmp/check_v1_teacher_step_20261109.py`、`tmp/v1_teacher_step_acceptance_20261109.log`。本次未加载真实 7B 或更新模型参数。下一项：在 actor 使用已有 `grpo_with_failure_opd_loss`，验证教师字段进入 microbatch，再做真实单步验收。
