# QLoRA 加载环境与教学接口（2026-08-31）

## 配置发生在哪里

调用链为 Windows CMD → WSL → Docker 容器 `llm-pytorch` → 容器内 Python。
没有修改 Windows 或 WSL 主机的 Python。工具命令在后台运行，不会弹出独立终端。

已新增 `bitsandbytes==0.50.2`，位置为容器内
`/usr/local/lib/python3.12/dist-packages/bitsandbytes/`。安装仅写入该容器的可写层；
停止/启动同一容器仍保留，删除并重建容器后需要重装，不等于已修改 Docker 镜像。

主机 CMD 可直接核验：

```cmd
wsl.exe docker exec llm-pytorch python -m pip show bitsandbytes
wsl.exe bash MedSearch/docs/tests/check_qlora_environment.sh
```

## 安装与依赖边界

固定版本见 `MedSearch/requirements-qlora.txt`，这是现有训练容器的附加依赖，不是整套环境文件。
首次 pip dry-run 遇到容器 Python 的 PEP 668 系统包保护。本次经工具权限批准，仅在容器内
显式使用 `--break-system-packages --no-deps` 安装一个新增 wheel，没有升级/卸载已有依赖。
该保护开关不是建议用于主机 Python 的通用做法。

本次实际安装命令：

```cmd
wsl.exe docker exec llm-pytorch python -m pip install --break-system-packages --no-deps --only-binary=:all: --report /tmp/medsearch-bnb-install.json bitsandbytes==0.50.2
```

安装报告仅含 bitsandbytes，wheel SHA256：
`55348a9a4a21bfd99cf8c7b32fe67b4030ae5c2a05738e03c1747f65fa6ec283`。

核对了[官方安装说明](https://huggingface.co/docs/bitsandbytes/main/en/installation)中的 CUDA 构建支持；
最终判断以本机运算为准，不仅凭支持表声明环境可用。

| 项目 | 实测版本/设备 |
|---|---|
| Python | 3.12.3 |
| torch | 2.13.0，CUDA 13.0 |
| transformers | 5.10.4 |
| accelerate / peft | 1.14.0 / 0.20.0 |
| bitsandbytes | 0.50.2 |
| numpy / packaging | 2.3.5 / 25.0 |
| GPU | RTX 5090，compute capability 12.0 |

## 已验证与尚未验证

实测日志：[check.log](run_logs/qlora-env-2p8oGYHS/check.log)。微型 `Linear4bit(128, 64)`
在 GPU 上成功执行 NF4 + double quant + BF16 前向；权重打包 dtype 为 uint8，quant_state.nested
为 True，输出 `[2, 64]` 且全部有限。PyTorch 该微型检查峰值 allocated 为 41,984 bytes，
不包含 CUDA 上下文及整卡背景占用，更不是 3B 模型显存。

学生量化加载分支及真实 Qwen2.5-3B 4-bit 加载已独立通过验收：252 个 NF4 双重量化层，
所有参数位于 cuda:0，35-token 单次前向输出有限。命令：
`wsl.exe bash MedSearch/docs/tests/check_qlora_loading.sh`。
完整指标、源码哈希及范围见[加载报告](run_logs/qlora-load-jxVC6GU3/SUMMARY.md)。
后续真实低比特准备与单样本backward已通过，见下方；尚未验证QLoRA优化器更新和保存重载。

## 已验收的学生接口边界

学生已填写 `sft/train.py::load_base_model` 内 `quantization_config is not None` 分支。

- 新参数为仅限关键字的 `quantization_config`，默认 None；传入的是已经构造好的对象，
  不是 `use_qlora` 开关。已有 config 继续提供 model_path 与 dtype。
- 非量化路径已保留，旧调用不会进入量化分支；不改 main 和 YAML。
- 量化分支只负责本地加载：缺少 CUDA 时提前抛 RuntimeError，错误信息包含 CUDA；
  单个可见 GPU 的 device_map 为 `{"": 0}`，不是 `"auto"`。
- 复用已有解析后的 model_path/dtype，保持 local_files_only、low_cpu_mem_usage，
  将同一个 quantization_config 对象传给原生 from_pretrained 并返回模型。
- 不做 model.to、LoRA 挂载、k-bit 训练准备或 Trainer 训练。正式 main 编排仍归学生。

教师测试：`docs/tests/test_sft_loading.py`。非量化路径使用真实 tiny checkpoint 检查权重；
CPU 的量化分支测试仅截取外部加载调用，验证参数传递和无 CUDA 拒绝，不冒充真实量化验证。
GPU 加载验收由独立脚本完成，未挂 LoRA、未反向传播、未保存模型。

本轮回归：`40 passed, 16 warnings in 15.08s`，两项占位 xfail 已撤去。
16 条警告来自第三方 torch.jit.script_method（14 条）与 tqdm UTC 时间 API（2 条）的弃用提示，
未屏蔽，也未据此修改学生逻辑。安装另有 pip root 用户提示，原始工具输出保留。

学生已实现 `sft/train.py::prepare_qlora_model(model, use_gradient_checkpointing)`，通过CPU验收。
复用PEFT准备并使用non-reentrant检查点，接口不参与现有main；完整契约见README当前任务。
撤去占位标记后默认回归42 passed、16 warnings（11.11s），无xfail。
真实4-bit准备与LoRA梯度联检现已通过：252个NF4层保留、504个LoRA梯度有限，基座无梯度且
数值保留。108个bias存储dtype由库转换，已用数值精确比较区分该变化与参数更新。
报告见[GPU梯度联检](run_logs/qlora-backward-VKJlYdcw/SUMMARY.md)；命令为
`wsl.exe bash MedSearch/docs/tests/check_qlora_loading.sh --backward`。
学生主入口的Trainer单步现已通过：global_step1、8条Stage-A、loss1.1438109874725342，
LoRA更新且基座数值保留。命令 `wsl.exe bash MedSearch/docs/tests/run_sft_3b_smoke.sh --qlora`；
详见[单步报告](run_logs/3b-smoke-SQf36DFi/SUMMARY.md)。QLoRA保存重载尚未验收。
下一项整理正式入口，main编排不由教师代写。
