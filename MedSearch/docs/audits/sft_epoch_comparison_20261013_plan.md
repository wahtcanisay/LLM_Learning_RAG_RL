# 1、3、5、8 epoch 的 dev 对比执行计划

用户于2026-10-13授权：先保存已有1epoch在dev上的rollout，再评分；随后训练到3/5/8epoch，比较并选下游RL初始化。

- 数据：冻结1000条Top-32缓存；固定原dev396题，不按命中或正确率筛题。重新核对原轨迹题干与dev标准化精确交集；历史近重复隔离记录保留，不将本次精确检查称为新的语义审计。
- 顺序：epoch1完整rollout→保存396条原始轨迹→本地评分→恢复checkpoint-125续训至累计epoch8→分别对checkpoint-375/625/1000完成相同rollout与评分。
- 调度：前1epoch使用已完成的单轮cosine；续训保留adapter/optimizer/RNG/Trainer状态，按1000总step的新cosine映射恢复非零学习率，显式重新计算step125的学习率。前125步不会因此重训。这是带一次调度调整的连续训练实验，不视作从头固定8epoch调度。
- 保存：后续每epoch完整checkpoint均保留，save_total_limit不限制；epoch3/5/8对应累计375/625/1000步。
- 搜索评测：原3B基模+对应LoRA adapter；vLLM原生LoRA。temperature0/top_p1/seed42/batch8，两轮搜索；总响应4096/模型4608，每动作500；BM25 Textbooks Top5、观察700token，现有8001服务。
- 轨迹：先保存prompt_ids/response_ids/action mask、环境片段、检索查询与文档hash、最终答案和终止原因；gold仅评分使用。生成阶段不调用verifier。
- 评分：复用medical-o1 verifier的原TEMPLATE、BF16、max_length1024、prob>0.5；生成完成后单独加载本地GPU verifier，释放搜索模型。各版本固定同一评分方式；与旧CPU BF16实现的数值差异记录在报告。
- 选择：主指标verifier判对率；同时比较词面得分、合法作答率、无效动作、截断、平均搜索次数与响应token。少数题净差需逐题回读，verifier误判不能当医学专家结论；近似持平优先较早epoch。保留各原始文件与W&B曲线。
- 验收：先8题LoRA搜索联调，完整396题回读及ID/哈希一致；续训前验证原生resume重设学习率后实际更新；最终候选重载动作推理。仅选RL初始化，不启动RL。
# 追加：纯 CE 同预算对照（用户授权，2026-10-13）

- 对照沿用原始 Qwen2.5-3B-Instruct、同1000条轨迹及顺序、prompt/response token和mask。只从缓存Parquet移除三列教师概率信息，保存逐值相同验收和hash，不重新生成数据。
- 复用现有 DistillationTrainer 的纯CE分支；LoRA、seed、batch/累积、学习率及长度完全相同。先1epoch/125步，再原生resume扩展cosine至累计1000步，保存3/5/8epoch，与既有Top32调度一致。
- 正式训练前先用最长真实轨迹完成纯CE一步保存、重载logits/动作一致及真实续训验收；第1轮后再次独立重载动作验收。
- 每组固定dev396题、相同解码/两轮BM25/预算，先完整保存rollout再用相同GPU verifier评分；不剔除无答案或格式失败。
- 单GPU串行执行，不调用付费API。CE独立产物根为 `MedSearch/outputs/distillation/sft_ce_epoch_comparison_20261013`，不覆盖Top32产物。最终汇总8个候选，不启动RL。
- 启动：`C:\Users\dell\anaconda3\python.exe tmp\docker_medsearch.py --log tmp\sft_ce_pipeline.log python -u -m MedSearch.tools.run_sft_comparison_pipeline --method ce`

