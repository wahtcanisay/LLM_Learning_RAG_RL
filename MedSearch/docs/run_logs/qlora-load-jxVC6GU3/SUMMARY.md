# Qwen2.5-3B 量化加载验收（2026-08-31）

结论：学生 load_base_model 的量化加载分支通过；未修改学生实现或正式 main。

命令（项目根目录 CMD）：

```cmd
wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh
wsl.exe bash MedSearch/docs/tests/check_qlora_loading.sh
```

- CPU：40 passed, 16 warnings in 15.08s，无 xfail；警告为已记录第三方弃用提示。
- GPU：Docker llm-pytorch，RTX5090，退出码0。原始证据见 [check.log](check.log)。
- 模型 models/Qwen2.5-3B-Instruct；seed20260831；NF4、double quant、BF16 compute。
- 252 个 Linear4bit 层，Params4bit 打包 uint8，全部 NF4/nested；所有模型参数在 cuda:0。
- 固定问题 `What does a thermometer measure?`，chat template 后35 tokens；
  单次 eval/no_grad/BF16 autocast 前向 logits 形状 [1,35,151936]，全部有限。
- 加载耗时27.84161577699706秒。
- PyTorch加载及单次前向峰值allocated2,149,335,040 bytes，reserved2,176,843,776 bytes。
  不含整卡背景及全部CUDA开销，不是训练显存，不可外推4096-token训练。
- 未挂LoRA、未backward、未训练、未保存模型。完成后nvidia-smi未显示运行中的GPU进程；
  仍有3149MiB背景占用，不将其归零或视为本检查训练占用。
- torch2.13.0/CUDA13.0、transformers5.10.4、bitsandbytes0.50.2、accelerate1.14.0、peft0.20.0。
- Git HEAD：8cccacd57afb66018aa0cc43725fea2706793dbf，工作树有未提交改动，以源码哈希补充：
  train.py SHA256 e8c890e5dbc1329d226d60a829bfc7926c751b8a91160618f7a5c8cb657a950d；
  检查脚本 SHA256 6d8ac4a56367056d649cdbb47e29ca2a8bb4955e3c03cf7c0dd92aa6ce7aa153。

首次检查在所有断言后因教师报告读取缺失的hf_device_map属性失败，保留于
[失败说明](../qlora-load-Gd9jRIwx/SUMMARY.md)。根据当前Transformers单设备加载源码，
改为报告实际参数device并保留CUDA断言后重新运行；不要求学生给模型补造属性。

下一关：低比特训练准备，随后才验证QLoRA挂载/反向传播与训练；正式main仍由学生编写。
