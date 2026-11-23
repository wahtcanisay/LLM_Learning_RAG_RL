# 2026-08-31：教师加载报告字段错误

运行 `wsl.exe bash MedSearch/docs/tests/check_qlora_loading.sh`，退出码1。
实际量化加载、Linear4bit/NF4/double-quant 检查、所有参数位于 CUDA0、单次有限前向的断言均已执行通过，
但教师在最后打印 JSON 时访问 `model.hf_device_map`，当前单卡模型没有该属性，导致报告未生成。

已核对 Transformers5.10.4 的 modeling_utils.py：单设备加载不走多设备 accelerate_dispatch 分支。
设备映射参数传入不保证该辅助属性始终存在，不能根据缺少属性判断加载失败。

教师只修改报告字段为真实参数设备集合，保留既有设备断言；学生 load_base_model 与 main 未改。
本目录 check.log 保留，重新验证使用全新日志目录。未训练或保存模型权重。
