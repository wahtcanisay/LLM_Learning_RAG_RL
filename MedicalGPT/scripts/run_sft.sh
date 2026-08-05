# 学习注释：这是官方的“双卡 SFT + LoRA”启动示例，不是我们最终的单卡 5090 配置。
# `torchrun --nproc_per_node 2` 会启动两个训练进程；以后做单卡最小实验时，应先改成
# `CUDA_VISIBLE_DEVICES=0 python training/supervised_finetuning.py`，其余变量一次只改一个。
#
# 阅读时可把下面的参数分成五组：
# 1. 模型与数据：model_name_or_path、train/validation_file_dir、model_max_length；
# 2. 训练规模：batch_size、gradient_accumulation_steps、epochs、max_*_samples；
# 3. 优化过程：learning_rate、warmup、weight_decay、gradient_checkpointing；
# 4. LoRA：use_peft、target_modules、rank、alpha、dropout；
# 5. 精度与记录：torch_dtype/bf16、logging/eval/save、output_dir。
# 当前脚本没有开启 QLoRA；QLoRA 还需要 `--qlora True --load_in_4bit True`。
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 training/supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen3.5-0.8B \
    --train_file_dir ./data/sft \
    --validation_file_dir ./data/sft \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --model_max_length 512 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_steps 5 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 13 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 4 \
    --output_dir outputs-sft-qwen-v1 \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --tool_format default \
    --cache_dir ./cache --flash_attn True
