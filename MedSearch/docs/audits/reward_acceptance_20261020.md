# 最终答案reward验收

日期：2026-10-20。结论：本次用户修改通过；P5另行验收。

## 实际修改及检查

- `rl/reward.py::compute_verifier_outcome_reward`从`extra_info["final_answer"]`读取答案，None/空字符串直接返回score=-1、acc=0、verifier_prob=0；POST中的answer使用该字段。原references、HTTP错误处理、阈值与reward映射保留。主对话未改写用户函数。
- `rl/medsearch_agent_loop.py`两个answer分支均记录内容，通过extra_fields返回；原生NaiveRewardManager负责tool_extra_fields→extra_info，辅助脚本使用naive，算法仍为GRPO。

## 运行验证

命令（CMD）：

```cmd
C:\Users\dell\anaconda3\python.exe tmp\docker_medsearch.py --log tmp\reward_acceptance_20261020.log python -m pytest MedSearch/tests/test_reward_wiring.py MedSearch/tests/test_agent_loop_smoke.py MedSearch/tests/test_teacher_trace_export.py -q
```

结果：17 passed，1条既有Ray弃用警告，exit 0。

新增验收使用原生veRL reward manager、动态装载的实际reward函数、临时本地aiohttp HTTP服务。None/空答案即使全文含正确answer标签也不发请求；非空答案精确发送`{"answer":"Beijing","references":["Beijing"]}`，不受全文最后一个Shanghai标签干扰；prob=0.9返回score=1、acc=1。原loop的0/1/2次搜索、检索标签隔离、token/mask/logprob与轨迹导出检查通过。

此测试不使用付费API，不把临时服务当作真实医学verifier效果测评，也不证明GPU训练或checkpoint验收已完成。

## 下一步P5

已进入一题两条rollout、单optimizer step、无教师、save_freq=1的真实WSL GRPO验收。首次启动在配置校验前失败：train_batch_size=1而ppo_mini_batch_size=2。源码trainer_base.py在更新时还会乘rollout.n，因此辅助脚本改为ppo_mini_batch_size=1，保持实际mini-batch为两条轨迹；未改用户核心代码。原失败日志tmp/grpo_p5_20261020.log，重跑日志tmp/grpo_p5_retry01_20261020.log，各运行目录独立保留。

后续retry01、retry02均在vLLM初始化时报告没有KV cache可用。辅助脚本增加max_num_seqs=2、max_num_batched_tokens=2048、enforce_eager=true；retry02仍未通过。retry03增加原生kv_cache_memory_bytes=536870912，但在该参数效果验收前，用户说明同卡脑电模型正在训练。为避免争抢，主对话仅向命令行包含本次唯一运行标记的GRPO driver PID43670发送SIGTERM，未向其他进程发送信号，未停止全局Ray集群或脑电任务。日志tmp/stop_own_grpo_20261020.log；driver退出134是本次主动终止产生的Ray abort，不能当作新reward错误。之后只读进程检查没有发现GRPO driver。

retry03曾实际加载基座及Top32 epoch8 adapter并建立FSDP；未完成rollout、optimizer更新或checkpoint保存。512MiB KV配置尚未通过真实初始化，不能称已解决缓存问题；需在GPU可用时继续验证。共享GPU状态下的失败不支持“模型一定放不下”的结论，GPU占用归因也不能只靠nvidia-smi总量。

P5只有在更新、checkpoint保存、重载及动作推理完成后才通过；之后再进入P6失败OPD接线。当前等待合适GPU资源，不启动额外GPU任务，也不自行安排定时重试。
