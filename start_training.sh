#!/bin/bash

cd /root/autodl-tmp/PartCrafter

# 设置环境
export PYTHONPATH="/root/autodl-tmp/PartCrafter:$PYTHONPATH"

# 启动训练
/root/miniconda3/envs/partcrafter/bin/python src/train_partcrafter.py \
    --config configs/my_training_contrastive.yaml \
    --tag full_training_20251007_232400 \
    --output_dir /root/autodl-tmp/output_training \
    --use_ema \
    --gradient_accumulation_steps 4 \
    --mixed_precision fp16 \
    --allow_tf32 \
    --pin_memory \
    --num_workers 8 \
    --max_grad_norm 1.0 \
    --offline_wandb

