export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3

export PATH=/usr/local/cuda/bin:$PATH


python -m torch.distributed.run --nproc_per_node=4 train.py \
    --deepspeed config/dp_config_zero1.json \
    \
    --dataset_save_dir ${DATA_ROOT} \
    \
    --output_dir ${SAVE_ROOT} \
    --remove_unused_columns False \
    --seed 1 \
    --do_train True \
    --bf16  True \
    \
    --learning_rate 1e-4 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 1000 \
    \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 24 \
    --num_train_epochs 1 \
    \
    --whisper_model $whisper_path \
    --qwen_model $qwen_path \
    --unfreeze_adapter True \
    --loss_names response_kl \
    \
    --disable_tqdm True \
    \
    --logging_steps 20 \
    --save_steps 200 \
    --save_total_limit 1