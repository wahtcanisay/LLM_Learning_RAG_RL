# MedAgent 工程与实验执行清单

更新：2026-09-16。以现有报告为证据，未在本轮重新执行。TODO不能解读为完成。

| ID | 任务 | 当前状态 | 证据/准入 |
|---|---|---|---|
| E0a | 单Planner假服务DSH smoke | 已有报告PASS | docs/decisions/dsh-boundary-validation.md；源码SDK+Node runtime |
| E0b / P0 | 正式RoleRunner最小实现与角色隔离 | TODO，学生当前唯一任务 | 双角色实际messages、session/system和结束事件；不能仅看返回JSON |
| E0c | 实际错误/暂态/挂起集成 | TODO | 请求次数与cleanup有界；不以线程假对象替代DSH实测 |
| E1 / P1-P2 | 契约、政策、预算、trace、假对象workflow | TODO | 三路由、硬校验、一次返工、弃答/失败 |
| E2 / P3 | 短真实API及3B/BM25服务接入 | NOT RUN | 费用提醒；版本和usage；保留原3B协议 |
| E3 / P4 | V1小样本与B0～B3评测 | TODO | 先工程smoke后固定数据；指标/成本/失败 |
| E4a / P5 | 数据用途与隔离清单 | TODO | 200 build、其余eval实际核验；旧dev不称未见test |
| E4b / P5 | 10题维护检查后至200题 | TODO | 每10题离线维护；上限100条、不凑数 |
| E4c / P6 | V2有/无记忆与可选LRU | TODO | 同题同预算，快照只读、成本分开 |
| E4d / P6 | MIRAGE优先外部验证 | TODO | 数据隔离与自定义协议；不把同域重建当迁移 |
