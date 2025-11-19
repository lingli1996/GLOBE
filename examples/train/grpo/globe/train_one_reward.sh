output_dir=experiments/globe_acc_reward
mkdir -p $output_dir
base_model=Qwen/Qwen2.5-VL-7B-Instruct
dataset=data/mp16-reason-train


NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model $base_model \
    --external_plugins examples/train/grpo/geo/plugin.py \
    --custom_register_path examples/train/grpo/geo/dataset.py \
    --reward_funcs globe_accuracy  external_math_format \
    --reward_weights 1 1 \
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.85 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset $dataset \
    --max_length 8192 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --output_dir $output_dir \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 16 \
    --temperature 1.0 \
    --deepspeed zero3 \
    --log_completions true \
    --vllm_max_model_len 1024 \
    --num_iterations 1 \
    --num_infer_workers 2 \
    --async_generate true \
    --beta 0.001