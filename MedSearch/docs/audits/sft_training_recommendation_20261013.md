# 3B / 1000 条教师轨迹冷启动训练建议

日期：2026-10-13。状态：1000条、1 epoch、125步已完成。用户随后授权启用固定dev396题的1/3/5/8epoch对比；epoch1完整rollout与评分已保存，续训与后续评测运行中。最新方案见 `sft_epoch_comparison_20261013_plan.md`，状态见 `MedSearch/outputs/distillation/sft_epoch_comparison_20261013/pipeline_status.json`。下文暂停dev的条目保留为首轮历史决策，已由新指令更新。

## 当前输入与实现

- 原始学生：`models/Qwen2.5-3B-Instruct`；不从旧 RL / 医学 adapter 初始化。
- 固定缓存：`MedSearch/data/distillation/top32_sft1000_20261011/train.parquet`，SHA256 `83dd481cbb2f248b407490367d787a42f0dbde021b70ddacd647c717ed45c4e5`。冻结 Search-R1 7B revision `74d46a43224daa612b5f27174e1c58b35587ee92`；训练时无需教师常驻或重新 rollout。
- 当前 loss：CE + 0.1 * Top-32，T=1；提示和检索 observation 不作为目标，Top-32 不重新归一化，共同词表 151665 项。
- 只读统计命令：`C:\Users\dell\anaconda3\python.exe tmp\summarize_sft_training_inputs.py`。
- 1000 条、1000 个不同 question_id；136864 个有效目标 token；完整上下文共 1044758 tokens。长度 min=878，p50=1004，p90=1121，p99=1808，max=2071；359 条超过 1024，无超过 4096 的轨迹。
- response_mask 中环境片段计数：953 条含一个、47 条含两个；这是掩码片段统计，不代替工具日志的实际调用审计。这份数据偏向已有检索轨迹，不保证教会模型不检索。
- 已完成 8 条真实 3B LoRA 短验证。CE 1.0209→0.7310、Top-32 0.9075→0.6224，参数更新有效；这只是训练样本拟合证据，无 dev 泛化或保存恢复结论。

## 来源与适用范围

1. [DGPO ACL 2026 论文](https://aclanthology.org/2026.acl-long.1751.pdf)，§4.2、附录表8、B.1：冷启动 CE+forward KL；初始化配置5 epochs、batch64、lr1e-4、lambda1；外部检索 token 屏蔽。论文主要学生0.5–1B，不能将其配置称为我们3B LoRA的最优方案；我们保留项目已定0.1和Top-32近似。原论文正确轨迹筛选也不覆盖本项目冻结清洗规则。
2. [DGPO 官方代码](https://github.com/omron-sinicx/dgpo)：冷启动复用TAID仓库标准forward KL实现；没有要求重写一套训练框架。
3. [Hugging Face TRL PEFT指南](https://huggingface.co/docs/trl/en/peft_integration)：LoRA SFT示例lr2e-4、1 epoch、有效batch16；说明LoRA学习率可高于全参训练。这是起点示例，不是1000条搜索轨迹的调参结论。
4. [Hugging Face Smol课程SFT](https://huggingface.co/learn/smol-course/unit1/3)：lr1e-4为中等起点，2e-4更积极；建议监测validation loss及held-out输出，不能单凭训练loss选择epoch。
5. [PEFT LoRA文档](https://huggingface.co/docs/peft/en/developer_guides/lora)：q_proj/v_proj是常见默认覆盖；all-linear为更广覆盖选项。当前首轮沿用已验收的q/v配置；若数据拟合不足，后续只改变覆盖范围或rank中的一个变量。
6. [Search-R1官方仓库](https://github.com/PeterGriffinJin/Search-R1)：公开主线是搜索环境与RL训练，不能从其PPO配置直接推导本项目离线SFT的epoch和学习率。

## 建议第一轮参数

| 参数 | 建议 |
|---|---|
| 方法 | BF16 LoRA，继续现有Transformers DistillationTrainer |
| 数据 | 按用户本轮指示，1000条冷启动全部训练，沿用原固定dev，不新划100条验证 |
| 验证来源 | 原396题dev保留，当前不加载、不评测 |
| seed | 20261013 |
| 长度 | max_length=4096，当前1000条全部完整保留；不packing或重新分词 |
| LoRA | r8、alpha16、dropout0、q_proj/v_proj，保持真实smoke配置 |
| 学习率 | 1e-4，较smoke 2e-4保守；这是本项目建议 |
| batch | 每卡1、梯度累积8，有效batch8 |
| optimizer | AdamW、weight_decay0、max_grad_norm1 |
| schedule | cosine，warmup约5%；依实际总step取整数warmup_steps，避免版本接口差异 |
| 预算 | 第一轮建议1 epoch；记录训练结果，不在无dev结果时宣称最佳epoch，不默认连续5轮 |
| 保存 | epoch保存完整Trainer checkpoint，save_only_model=false；单独保存最终adapter |
| 验证 | 当前eval_strategy=no，不构造eval_dataloader；dev QA评测留到数据库匹配核查之后 |
| 日志 | 正式运行每5 optimizer steps；记录CE、Top-32、总loss与梯度、时间、显存 |

1000条/有效batch8为125个optimizer steps/epoch；这是步数计算，不是耗时预测。第一轮1 epoch约7个warmup steps。后续续训应明确新预算及scheduler：不能将已经衰减到零的1-epoch scheduler直接当成3-epoch训练配置。

用户决定后续验证用原来的dev，冷启动1000条不拆分；当前明确暂缓dev评测。原医学dev问题没有对应教师轨迹和缓存时，不能直接当同格式SFT验证轨迹。不把原dev/test搬进SFT训练，不额外请求教师生成dev轨迹。正式入口现已支持eval_data_path=None时不构造eval_dataloader。

## 验收与选择

- 真实checkpoint gate已通过，报告 `MedSearch/outputs/distillation/sft_checkpoint_smoke_20261013_final/report.json`：最长2071-token样本完成更新，adapter/optimizer/scheduler/RNG/trainer state均保存；统一BF16 autocast后重载logits最大差0，正常初始问题生成完整合法search动作，重载动作token一致；原生resume从step1推进至step2，adapter继续更新，CUDA peak allocated10.3618GiB。
- 此前辅助验收的第一次/第二次失败原因是前后autocast不一致；adapter文件和重载权重本身最大差0。统一精度后通过。动作辅助脚本先错误读取输入中的闭合标签导致提前停，又尝试在带强制最终回答提示的历史上下文推理产生缺少开标签的后缀；最终按正常初始问题入口，只检查新生成token，完整search动作验收通过。失败目录保留，未修改用户核心loss/Trainer。
- 长度上限4096不意味着每条都padding到4096；当前数据真实最大2071已覆盖，但不能将显存结果外推到未来真实4096-token轨迹。
- 当前仅监测训练CE、Top-32、总loss与参数更新，暂不计算dev指标。后续先核查教师/数据库的dev匹配情况，再做完整搜索QA评测；不因训练loss最低就认定医学答案最优。
- CE、Top-32与总loss分开记录。现有DistillationTrainer对已计算loss作detach并按日志窗口累计，复用Trainer.log上传分项，无额外模型前向、无改变loss数学或梯度；日志为窗口内microbatch平均。
- 当前训练数据也可能包含教师错误，SFT验证loss主要衡量模仿；医学QA指标单独诊断。最终冻结测试不能参与选checkpoint。
- 不保存短验证已经结束；正式训练必须显式启用checkpoint。不能继续使用save_strategy=no或V9 save_freq=-1。

## 当前下一任务

checkpoint保存/重载/动作推理预训练验收已完成。退役脚本清理记录见cleanup_20261013.json。正式配置为configs/sft_top32_epoch1.yaml，WSL启动命令为 `C:\Users\dell\anaconda3\python.exe tmp\run_sft_epoch.py`；1000条冷启动未重新划分，1 epoch训练已完成，完整checkpoint和adapter均保存。W&B run为 `https://wandb.ai/lixle-northwestern-polytechnic-university/medsearch_r1/runs/6ied183a`；后续按用户决定继续暂缓dev评测。
