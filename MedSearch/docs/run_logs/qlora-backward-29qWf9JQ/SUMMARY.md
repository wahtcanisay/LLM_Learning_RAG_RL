# 首次量化反向传播联检：严格字节哈希断言失败

命令：`wsl.exe bash MedSearch/docs/tests/check_qlora_loading.sh --backward`，退出码1。
原始日志见check.log；失败发生在Python第80行，即所有冻结参数的字节哈希一致性断言。
此前已通过量化准备、LoRA挂载、有限loss、反向传播、全部LoRA梯度有限、非零梯度存在、基座无梯度检查。

源码调查：容器bitsandbytes0.50.2的nn/modules.py:630-635在Linear4bit.forward中
会将bias.data转成计算dtype；这会使冻结bias的字节表示改变，不能直接等同参数优化更新。
后续探针将记录变化名称并严格要求：只有这些Linear4bit的bias允许FP32→BF16，
且转回FP32后数值哈希精确相等；其他所有基座参数仍要求字节一致。
尚须以重跑实测确认假设。未修改学生函数、未更新或保存模型权重。

后续重跑已确认：仅108个q/k/v bias符合上述存储转换，所有数值与其他参数字节均保留。
验收通过见[最终报告](../qlora-backward-VKJlYdcw/SUMMARY.md)。
