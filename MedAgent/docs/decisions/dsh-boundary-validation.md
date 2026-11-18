# DSH 边界验证

## 固定版本

- 仓库：`https://github.com/deepseek-ai/deepseek-harness.git`
- 标签：`dsh-v0.1.5-rc.2`
- commit：`fb2c4b9e698e30edb738bca4cf0618587db7d203`
- 许可证：MIT
- 本机 Windows Node：`v24.15.0`
- Python SDK：该标签源码中的 SDK；必须与同版本 runtime 一起构建或安装

官方 PyPI 项目当前只有 `0.0.0.dev0` 占位发布，GitHub Release 未附 wheel。因此不能用
`pip install deepseek-harness-sdk` 得到本次固定版本。本地已使用固定版本源码生成 Node dev runtime；官方单文件 EXE 封装在生产安装触发缺失 `lefthook` 时失败。该失败不影响源码 Node runtime 的本地集成 smoke，正式分发仍待解决。

## Python 与 DSH 的边界

Python 负责医学状态机、角色输入、JSON/业务校验、全局 deadline、预算和最终提交。DSH 只负责一次隔离角色执行、会话事件和 provider 通信。`dsh_smoke/role_smoke.py` 是可替换薄适配层；MedSearch-R1 的标签协议和检索循环不迁入 DSH。

`sdk-minimal` 默认仍注册持久 shell，并使用 `danger-full-access`。本项目通过
`dsh_smoke/no_tools.patch.yml` 禁用模型可调用的 Bash/PowerShell，同时关闭 DeepSeek session-log
上传。已用固定版本 `--dump-config` 导出 `.runtime/effective-config.yml`：`persistent-bash` 和 `persistent-pwsh` 均为 `disabled: true`，`session-log-deepseek` 为 `enabled: false`。

SDK 的 `request_timeout_seconds` 只限制 JSON-RPC 请求。`Session.run()` 等待
`session.status=idle` 时没有独立 deadline；本项目使用 Python 外层 deadline，到期调用 `close()`
并拒绝迟到结果。这只能证明本地生命周期有界，不能证明远端请求已经取消或停止计费。

## 状态

| 检查 | 状态 | 证据 |
|---|---|---|
| Python Planner JSON 与 usage 边界 | PASS | CMD 实测 4 tests passed（2026-09-16） |
| 外层 deadline 关闭 runtime | PASS | 50ms deadline 后调用 close，迟到结果未提交 |
| 固定 profile 最终配置无模型工具 | PASS | `.runtime/effective-config.yml`；两个 persistent shell `disabled: true`，session log `enabled: false` |
| 标准库假服务响应与抓包 | PASS | 本机回环请求返回固定 SSE、usage=17/9/26，并保存 messages |
| DSH → 假服务单次角色执行 | PASS | Windows Python SDK + 同 commit Node runtime，Planner SEARCH JSON，`turn/end`，usage=17/9/26；抓包一条 DSH 请求 |
| Planner/Verifier 实际 messages 隔离 | TODO | 待扩展假服务集成测试，不以最终文本代替抓包 |
| 不可重试/暂态/挂起行为 | TODO | 假服务 `error` / `transient` / `hang` 模式 |
| 真实 DeepSeek 短调用 | NOT RUN | 仅在以上项目通过且存在凭据后执行一次 |

## 已验收的集成结果

2026-09-16 本地假服务单次 Planner 运行成功：`finish_reason=completed`，返回 `route=SEARCH`、`search_query=aspirin evidence`，事件包含 `assistant/message` 与 `turn/end`。假服务抓包中真实 DSH 请求有一条，目标为本机 `127.0.0.1:8765`；请求包含测试系统提示和测试任务、`stream_options.include_usage=true`，没有真实 API key。此前独立 curl 探针也留有一条抓包，计数时已排除。

这证明角色执行、事件和 usage 边界，不证明不同角色会话隔离、retry 策略、远端取消或医学工作流效果。

## CMD 命令

```bat
cd /d "D:\code_list\some tricks\llmleanring\MedAgent"
scripts\run_dsh_smoke.cmd unit
```

集成测试需要两个 CMD 窗口：

```bat
cd /d "D:\code_list\some tricks\llmleanring\MedAgent"
scripts\run_dsh_smoke.cmd fake-server normal
```

```bat
cd /d "D:\code_list\some tricks\llmleanring\MedAgent"
scripts\run_dsh_smoke.cmd fake-client
```

## 采用标准

只有无 DSH 核心源码修改、无模型可调用 shell、会话隔离、结果/事件/usage 可提取、错误请求次数明确、超时不提交迟到结果全部通过后，才把 DSH 从候选底座升级为正式选型。
