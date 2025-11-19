output_dir=experiments/sft_geoloc_reason
mkdir -p $output_dir

MAX_PIXELS=1003520 \
NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --custom_register_path examples/train/grpo/geo/dataset.py \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'data/mp16-reason-train' \
    --max_length 8192 \
    --num_train_epochs 2 \
    --freeze_vit true \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 200 \
    --save_steps 200 \
    --max_length 2048 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --output_dir $output_dir \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4