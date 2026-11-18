# MedAgent 初期工程汇报（2026-09-22）

## 1. 当前结论

MedAgent 已完成 DSH 候选执行底座的单角色 fake smoke、Python RoleRunner 边界、Planner 数据契约、有限 Search→Verify→Retry 闭环，以及公共审计 trace。当前可以在无付费 API 的条件下验证 Python 控制流和文件产物，但还不能宣称真实多角色 DSH、真实 3B 搜索、医学效果或完整三路由 workflow 已完成。

本地验证命令：

```cmd
cd /d "D:\code_list\some tricks\LLMLeanring\MedAgent"
python -m unittest discover -s tests -v
```

最新结果：`Ran 56 tests ... OK`。

## 2. 已经完成的工程工作

### DSH 与角色执行边界

- 固定 DSH `dsh-v0.1.5-rc.2` / commit `fb2c4b9e698e30edb738bca4cf0618587db7d203`。
- 使用源码 Python SDK 与同 commit Node runtime 跑通单 Planner 假 DeepSeek 流式链路。
- `RoleRunner` 在 fake backend 下实现角色 prompt、独立 session、完成状态、JSON 结果、usage、deadline 和 close 边界。
- 真实 Planner/Verifier 隔离、401/503、挂起后的线程/子进程退出仍未验证。

### Python 数据与控制流

- 完成 `Usage`、`RoleResult`、`Plan`、`SafetyResult` 及 `validate_plan()`。
- 完成 SearchTask/SearchResult/VerificationResult/SearchCycleResult。
- 完成最多两次搜索的 Search→Verify→Retry：支持 PASS、RETRY、ABSTAIN，并保证所有 fake 路径有终态。
- 该闭环尚未接本地 3B、BM25 或真实 Verifier。

### 公共审计 trace

- 完成顶层白名单 `redact()`。
- 完成不可覆盖的 run 目录和 `manifest.json`。
- 完成 `events.jsonl` 追加、run_id 检查和 sequence 去重。
- 完成 `answer.json`、`summary.json` 独占写入。
- 完成完成运行与中断运行的离线 `replay()`。
- replay 按当前决定只负责读取，不重复执行写入侧校验。

## 3. 主动延后或等待依赖的工作

| 项目 | 状态 | 原因或重新进入条件 |
|---|---|---|
| 预算账本 | 延后 | 当前无付费 API、无批量实验；付费调用、多层重试或批量评测前恢复 |
| 确定性安全 preflight | 延后 | 当前缺少审核后的医学规则集、否定/转述测试和风险评测 |
| 完整 RunState/迁移表 | 等待依赖 | DIRECT、EMERGENCY、Generator 尚未实现；现在编码只会形成空状态表 |
| 真实 DeepSeek API | 未运行 | 会产生费用；接入凭据前提醒用户并复核当日模型信息 |
| 真实 DSH 错误与清理 | 后续集成 | 需要真实 runtime 行为证据，fake 线程测试不能替代 |
| Generator/完整三路由 | 待实现 | 输出策略与相应服务尚未冻结 |
| V1/V2 实验 | 待实现 | P3 服务链和 P4 evaluation 完成后开始 |

## 4. 当前没有证明的事项

- 没有证明任何医学答案准确率、安全率或证据忠实度。
- 没有证明真实 DeepSeek Planner、Verifier、Generator 已能稳定协作。
- 没有证明本地 MedSearch-R1 checkpoint、服务协议或 BM25 端点可用。
- 没有产生真实 API 费用、延迟或 token 成本数据。
- 没有完成完整 Python `run()`、DIRECT/EMERGENCY 路线或 Generator。

## 5. 下一阶段

下一阶段按工程书进入 P3：先实现 `retrieval.py` 的只读 BM25 边界和 Evidence 规范化，再实现 `search_agent.py` 的本地 3B 动作循环，随后接正式 verification。等 DIRECT、EMERGENCY 和 Generator 具备实际调用路径后，再实现完整 workflow `run()` 和必要状态约束。

付费 API 不在下一小步中；需要真实 DeepSeek 调用时先提醒用户。
