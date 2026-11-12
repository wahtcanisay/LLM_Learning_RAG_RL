# P5：真实 GRPO 运行闭环与 checkpoint 验收

日期：2026-10-21。结论：**运行闭环、保存、原生重载和一次搜索→答案推理通过；本次没有非零参数更新，不代表训练效果提升。**

## 实际执行与产物

- WSL Docker `llm-pytorch`；真实 Ray / FSDP / vLLM，V1 同步训练器。
- 初始化：Qwen2.5-3B-Instruct + 定稿 CE+Top32 epoch8 adapter（checkpoint-1000）；LoRA r8、alpha16、q_proj/v_proj。
- 正式 RL correction04（164题）中一题，rollout n=2，seed42，单个 optimizer step，lr=1e-6；教师关闭，无 reference KL，不过滤全错组。
- prompt512、response4096、model4608；工具轮数2、单次动作500、observation500、BM25 top3；microbatch1，题目 mini-batch1。
- vLLM eager、max_num_seqs2、显式 KV cache512MiB。`save_freq=1`，没有沿用 V9 的禁用保存配置。
- 训练退出0；两条轨迹保存在 `outputs/distillation/grpo_p5_20261021/rollouts/1.jsonl`。
- checkpoint：`outputs/distillation/grpo_p5_20261021/checkpoints/global_step_1`。保存了模型、optimizer、RNG、scheduler、dataloader状态；模型文件约13.6GB，579个模型键、144个LoRA键，optimizer step值为1。
- 机器验收与文件SHA256：同目录 `P5_ACCEPTANCE.json`；较早的 `checkpoint_state_verification.json` 是重载前的阶段性记录，由最终验收报告覆盖其 pending 状态。

## 数值与边界

| 项目 | 实测结果 |
|---|---:|
| 训练轨迹 reward | -1 / -1 |
| GRPO advantage、PG loss、grad norm | 均为0 |
| 相对初始化的 LoRA 参数差平方和 | 0 |
| actor最大allocated显存（框架指标） | 23.5864 GB |
| actor最大reserved显存（框架指标） | 33.4492 GB |
| 整卡nvidia-smi观测峰值，包含其他进程 | 32088 MiB |
| 单步时间，不含初始化 | 179.3604秒 |
| 其中生成 / actor更新 / 保存 | 46.8787 / 7.0923 / 85.4262秒 |

两条轨迹奖励相同，组内相对优势为零；optimizer执行了step，但参数没有变化。这是纯GRPO全错组的预期结果，不是“已经学会”或模型损坏的证据。不能把本次单题结果当成整体训练准确率。

## 原生重载与推理

最终运行 `grpo_p5_reload_final_20261021` 退出0。日志确认恢复global_step1、model、optimizer、RNG、scheduler；原生resume流程读取dataloader状态。使用 `val_only=true`，没有新增训练步或checkpoint。

仅取旧dev一题作功能检查，保存 `outputs/distillation/grpo_p5_reload_final_20261021/validation/1.jsonl`：

- search：`muscles forming the anterior and posterior faucial folds of the tonsillar fossa`
- answer：`palatoglossus muscle and palatopharyngeus muscle`
- 本地verifier：score1、acc1、prob0.8824278712。不是完整dev评测或医学专家结论。
- 两次CPU暂存方式的原生重载产生相同输出文本。

## 为通过重载所作的最小修正

单卡NO_SHARD保存出的模型storage实际全部标记为`cuda:0`。原生直接`torch.load`重载报`CUDA driver error: device not ready`；同文件CPU加载成功。仅在veRL的 `utils/checkpoint/fsdp_checkpoint_manager.py` 模型加载行补 `map_location="cpu"`，避免直接在CUDA还原整份状态，再交由原生FSDP装载。

- veRL commit：`23af6a7a2e8d6efeeb2adbe5d1689c7a24f503a3`。
- 可移植补丁：`patches/verl_fsdp_checkpoint_cpu_load.patch`；反向check确认当前源码已应用。
- 没有修改GRPO、reward或actor loss算法。

## 命令（项目根目录CMD执行）

训练：

```bat
C:\Users\dell\anaconda3\python.exe tmp\docker_medsearch.py --log tmp\grpo_p5_20261021.log env MEDSEARCH_RUN_DIR=MedSearch/outputs/distillation/grpo_p5_20261021 bash MedSearch/scripts/run_medsearch_grpo_distill.sh
```

最终重载验证：

```bat
C:\Users\dell\anaconda3\python.exe tmp\docker_medsearch.py --log tmp\grpo_p5_reload_final_20261021.log env MEDSEARCH_RUN_DIR=MedSearch/outputs/distillation/grpo_p5_reload_final_20261021 bash tmp/p5_launcher_snapshot_20261021.sh trainer.resume_mode=resume_path trainer.resume_from_path=MedSearch/outputs/distillation/grpo_p5_20261021/checkpoints/global_step_1 trainer.val_before_train=true trainer.val_only=true data.val_max_samples=1 trainer.logger=[console]
```

首次训练实际launcher按当时manifest SHA256恢复并保存为 `outputs/distillation/grpo_p5_20261021/launcher_used.sh`；最终重载使用固定脚本副本。文件哈希与软件版本记录于机器报告，不能只凭当前工作区HEAD还原本次运行。

## 异常保留与下一步

- 首次训练driver退出0，但W&B offline退出清理阶段出现BrokenPipeError；checkpoint和step日志已落盘。没有声称日志完全无异常，没有在线上传或付费API调用。
- 初次重载失败日志 `tmp/grpo_p5_reload_20261021.log` 保留。
- CPU暂存首次重试已完成重载和推理，但助手在Bash运行中修改了一行注释，导致脚本后续读取偏移、wrapper退出127。日志 `tmp/grpo_p5_reload_retry01_20261021.log` 保留；使用固定副本重跑后退出0，最终依据为 `tmp/grpo_p5_reload_final_20261021.log`。
- **P6尚未完成**：已有失败提取、教师token评分、散射回填、OPD与合并loss函数，但没有在V1真实训练循环调用，也没有替换actor实际loss。7B是冻结教师，待训练的仍是3B。下一项核心工作是串行教师评分接线，随后验收全错组的非零OPD梯度及真实单卡更新；本次不直接开始较长OPD训练。
