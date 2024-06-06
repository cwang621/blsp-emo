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
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 200 \
    \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    \
    --blsp_model $blsp_path \
    --unfreeze_qwen True \
    --unfreeze_adapter True \
    --loss_names "input_er,response_ce" \
    \
    --disable_tqdm True \
    \
    --logging_steps 20 \
    --save_steps 500 \
    --save_total_limit 1