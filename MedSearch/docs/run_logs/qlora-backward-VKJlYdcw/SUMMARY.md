# 真实 QLoRA 准备与梯度联检（2026-08-31）

结论：学生的量化配置、加载、准备、attach_lora 与数据接口组合通过单样本反向传播检查。
没有修改学生实现、main或YAML；未调用优化器、Trainer.train或保存模型。

```cmd
wsl.exe bash MedSearch/docs/tests/check_qlora_loading.sh --backward
wsl.exe bash MedSearch/docs/tests/run_sft_checks.sh
```

GPU退出码0；[原始日志](check.log)含完整配置、版本、Git提交、源码和数据哈希。
CPU回归：42 passed, 16 warnings in 12.09s，无xfail；第三方弃用警告未屏蔽。

## 实测结果

- Docker llm-pytorch，RTX5090；Qwen2.5-3B-Instruct；seed20260831。
- 252个NF4双重量化层，打包权重uint8；准备前后字节保持不变，quant_state对象保留。
- LoRA r=8、alpha=16、dropout=0.05，七类target_modules，non-reentrant梯度检查点开启。
- 固定Stage-A第0条medqa-dialog_4790，batch[1,515]，max_length1024；移位后281个监督token。
  使用原Dataset/encoder/collator/labels，无数据改写或截断。
- loss=1.9581148624420166，logits全部有限；504个LoRA张量梯度均存在且有限，252个非零。
  其中A非零0、B非零252；B零初始化时首次A梯度为零是正常结果，不要求每个张量首步非零。
- 434个基座参数均被冻结且grad为None；108个q/k/v bias发生FP32→BF16存储转换，
  转回FP32后数值哈希精确一致；其他326个基座参数字节哈希不变。
  全部LoRA参数字节也不变：只求梯度，optimizer_steps=0。
- 加载耗时28.13029776199255秒；单次forward+backward0.9319077680120245秒。
- PyTorch峰值allocated4,657,840,640 bytes（约4.34GiB），reserved4,752,146,432 bytes。
  这是加载/准备/单样本反向传播探针的分配器峰值，不含全部CUDA/整卡背景开销；
  无optimizer状态，不能作为完整训练显存或外推4096-token训练。
- 日志parameter_counts复用现有describe_model，其total是打包参数存储numel，
  不等于解包后的原始模型参数总数；不能因此把3B模型称为1.7B模型。

## 首次失败及修正

[首次日志说明](../qlora-backward-29qWf9JQ/SUMMARY.md)：教师最初要求所有基座字节完全一致，
该断言失败。核对bitsandbytes0.50.2的nn/modules.py:630-635后，记录所有变化参数并实测确认
仅108个Linear4bit.bias发生dtype转换；新断言仍要求它们数值精确不变、其他参数字节不变。
未删除基座检查，也未改变学生代码或第三方库。

本结论不含optimizer更新、QLoRA adapter保存重载、Stage-B或医学效果评估。
下一唯一任务：由学生编排QLoRA调用路径，为Trainer单步更新验收做准备；正式main不由教师代写。
