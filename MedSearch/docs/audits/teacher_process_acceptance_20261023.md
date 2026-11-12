# 7B 短生命周期评分子进程验收

日期：2026-10-23。结论：**辅助子进程通过；V1 训练循环、真实学生卸载/恢复和 actor 合并 loss 尚未接线验收。**

## 本次实现

新增 `distillation/teacher_process.py`：

```python
score_in_subprocess(failure_batch, teacher_path, output_dir, *, device="cuda")
```

- 返回格式沿用 `score_failed_batch`，可直接作为现有 `run_failure_teacher_phase` 的 `score_fn`。
- 空失败列表直接返回，不启动进程。有失败时创建本批独立任务目录，保存 `request.jsonl`；只写原始索引、轨迹/组身份、prompt/response IDs和mask，不写gold/references。
- 通过当前Python解释器以模块方式启动子进程；一次加载冻结BF16教师，串行复用已有评分器。父进程没有教师模型。
- 全部评分完成后原子发布 `result.pt`，另存 `result.json` 与 `worker.log`。父进程等待退出0才读取可信CPU张量，并核对结果覆盖、身份、形状、dtype、有限值、动作mask和环境位置零分。
- 非零退出向上传递 `CalledProcessError`，不把错误伪装成空评分；现有调度函数的finally负责恢复学生。
- 每批文件用于证据和进程传递，不是按question_id跨策略复用的在线缓存。
- 用户核心 `distillation_hook.py`、`teacher.py` 与loss实现未被本次改写。

## 实测

当前既有WSL Docker `llm-pytorch`；本地目录 `models/external_baselines/SearchR1-Qwen2.5-7B-Instruct`，沿用此前固定教师目录。本次不下载模型、不调用付费API。

使用 `tools/smoke_teacher_process.py` 与真实veRL TensorDict：

| 项目 | 实际结果 |
|---|---|
| 失败输入 | 一条分段search/observation/answer；一条4096上下文＋500目标token形状检查 |
| 教师实际评分 | 2条、518个模型token；一次加载 |
| 正确项 / padding | OPD mask与分数均零，没有提交给教师 |
| 教师状态 | BF16、冻结、eval |
| 子进程退出 | 0；PID14211在返回后已不存在 |
| 分数落盘 | CPU float32，无梯度、有限值、与response对齐 |
| 调度顺序 | cpu → score → device |
| 教师allocated显存峰值 | 16.126 GiB |
| 评分前 / 退出后整卡占用 | 1.637 / 1.637 GiB |
| 子进程内加载＋评分时间 | 49.831秒；不是正式训练吞吐估算 |

合成token仅用于接口、长度和资源验收，不是当前学生on-policy rollout，不用于训练或能力评测。资源切换调用由轻量runtime记录，**没有实际加载学生、optimizer或启动Ray/FSDP/vLLM**；不能据此宣布完整P6通过。

机器报告、输入/结果及源码SHA256：`outputs/distillation/teacher_process_smoke_20261023/acceptance.json`。本批完整证据在该目录的 `jobs/teacher-t0acq4z7/`。

## 测试及命令

先写测试并实测入口缺失时3项失败；辅助入口实现后，本地小模型CPU真实子进程往返、空失败跳过及异常退出通过。最终联合原有调度/评分/回填测试：**28 passed，2条既有第三方tqdm弃用警告，退出0**。

项目根目录CMD：

```bat
C:\Users\dell\anaconda3\python.exe tmp\docker_medsearch.py --log tmp\teacher_process_regression_20261023.log python -m pytest MedSearch/tests/test_teacher_process.py MedSearch/tests/test_failure_teacher_phase.py MedSearch/tests/test_distillation_hook_jobs.py -q

C:\Users\dell\anaconda3\python.exe tmp\docker_medsearch.py --log tmp\teacher_process_7b_20261023.log python -m MedSearch.tools.smoke_teacher_process --teacher MedSearch/models/external_baselines/SearchR1-Qwen2.5-7B-Instruct --output-dir MedSearch/outputs/distillation/teacher_process_smoke_20261023
```

复跑GPU验收使用新输出目录，保留旧报告。red、CPU及最终测试日志分别为 `tmp/teacher_process_red_20261023.log`、`tmp/teacher_process_cpu_20261023.log`、`tmp/teacher_process_regression_20261023.log`。

## 下一项唯一核心编码任务

在 `MedSearch/.deps/verl/verl/trainer/ppo/v1/trainer_base.py`、`_compute_old_log_prob` 附近新增 `_compute_failure_teacher_scores(self, batch, score_fn)`：

1. 按 `batch.keys` 原顺序从TransferQueue读取 `uid/prompts/responses/response_mask/extra_fields`。
2. 调用现有 `run_failure_teacher_phase(self, batch, data, score_fn)`。
3. 只把返回data中的 `teacher_logprobs/opd_mask` 写回相同keys和partition；返回原batch，保留tags、extra_info与顺序。

本轮先完成该方法并单独验收读写；下一轮才在 `_balance_batch` 后、`_compute_old_log_prob` 前加入启用分支和实际callback配置。辅助 `score_fn` 可用 `functools.partial(score_in_subprocess, teacher_path=..., output_dir=...)` 绑定。真实Ray运行时的GPU可见性和学生资源状态仍须实测。
