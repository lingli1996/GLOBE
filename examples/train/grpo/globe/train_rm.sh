output_dir=experiments/globe_rm_model
mkdir -p $output_dir


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=1003520 \
NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --custom_register_path examples/train/grpo/geo/dataset.py \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset 'data/geo_loc_rm_20w' \
    --max_length 2048 \
    --num_train_epochs 2 \
    --freeze_vit true \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 200 \
    --save_steps 200 \
    --max_length 2048 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --output_dir $output_dir \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'