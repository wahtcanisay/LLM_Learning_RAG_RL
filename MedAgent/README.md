# MedAgent

以DSH为候选角色执行底座的医学证据搜索Harness。Python负责医学状态机、角色输入与校验、预算、验证返工、证据和记忆治理；DSH只执行DeepSeek角色。3B SearchAgent保留独立标签协议，MedSearch资产只读复用。

## 阅读顺序

1. [PROJECT_BOOK.md](PROJECT_BOOK.md)：目标、业务主干、V1/V2边界。
2. [ENGINEERING_PLAN_V1_V2.md](ENGINEERING_PLAN_V1_V2.md)：完整目录、每文件函数、调用关系和P0～P6实施顺序。
3. [工程进度书](PROGRESS.md)：已经实际完成的步骤、证据、问题和下一任务。
4. [实验计划](refine-logs/EXPERIMENT_PLAN.md)与[执行状态](refine-logs/EXPERIMENT_TRACKER.md)。
5. [DSH边界证据](docs/decisions/dsh-boundary-validation.md)：真实版本、命令、通过和未通过项。

## 当前状态与唯一任务

2026-09-26：已有 RoleRunner fake 边界、Plan 契约、有限 Search→Verify→Retry、trace、Evidence 和 retrieval；具体 SearchAgent 基本 fake 循环通过 6 项测试。最近普通回归 65 项通过。真实 DSH 多角色隔离、错误清理、本地 3B/BM25 和完整三路由尚未验收，未调用付费 API。
当前 P3 的下一任务：核实真实单动作请求/响应与模板、stop、checkpoint；模型初始上下文只用 query，原题和反馈留在外层任务。Evidence 规范化、只读检索假响应及基本 SearchAgent 循环已验；真实 BM25 请求仍待服务可用时记录。精确可见片段与留档待真实接线时验收，具体边界见工程书第 4.4 节，不调用付费 API。

现有 `dsh_smoke/` 是验证资产，`medagent/` 是已开始实现的正式包。不要照旧src/medagent方案创建第二个包；按任务逐步建文件。UPSTREAM.lock固定源码SDK/runtime版本，不能假定该版本有可直接pip安装的发行包。

`.runtime/`、`runs/`、`memory_snapshots/`是生成物；`../MedSearch/`模型、训练、数据与索引不复制不修改。历史文档在docs/history及带时间实验快照中，仅供追溯。项目为研究用途，不提供临床诊疗建议。
