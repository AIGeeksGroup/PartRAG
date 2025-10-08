#!/bin/bash

# ============================================================================
# 细粒度对比学习训练脚本
# ============================================================================
# 功能：训练带有细粒度对比学习的 PartCrafter 模型
# 
# 使用方法：
#   bash scripts/train_contrastive.sh
# 
# 或自定义参数：
#   bash scripts/train_contrastive.sh \
#     --config configs/mp8_nt512_contrastive.yaml \
#     --tag my_experiment
# ============================================================================

# 激活虚拟环境
echo "🚀 激活 partcrafter_env 虚拟环境..."
source /root/miniconda3/bin/activate partcrafter_env || conda activate partcrafter_env

# 检查环境
if [ $? -ne 0 ]; then
    echo "❌ 错误: 无法激活 partcrafter_env 环境"
    echo "请先创建环境: conda create -n partcrafter_env python=3.11"
    exit 1
fi

echo "✅ 环境激活成功"
python --version

# 设置 WandB (可选)
# export WANDB_API_KEY="your_wandb_api_key_here"
# export WANDB_BASE_URL=https://api.bandw.top  # 如果连接有问题

# 默认配置
DEFAULT_CONFIG="configs/mp8_nt512_contrastive.yaml"
DEFAULT_TAG="contrastive_$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT_DIR="output_partcrafter"

# 解析命令行参数
CONFIG=${1:-$DEFAULT_CONFIG}
shift
TAG=${1:-$DEFAULT_TAG}
shift

echo "================================================"
echo "📋 训练配置"
echo "================================================"
echo "配置文件: $CONFIG"
echo "实验标签: $TAG"
echo "输出目录: $DEFAULT_OUTPUT_DIR"
echo "================================================"
echo ""

# 训练命令
echo "🎯 开始训练..."
echo ""

# 使用 torchrun 进行分布式训练（多卡）
# 如果只有一张卡，可以直接用 python
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $NUM_GPUS 张GPU"

if [ $NUM_GPUS -gt 1 ]; then
    echo "使用多卡训练模式"
    torchrun --nproc_per_node=$NUM_GPUS \
        src/train_partcrafter.py \
        --config $CONFIG \
        --tag $TAG \
        --output_dir $DEFAULT_OUTPUT_DIR \
        --use_ema \
        --gradient_accumulation_steps 4 \
        --mixed_precision fp16 \
        --allow_tf32 \
        --pin_memory \
        --num_workers 8 \
        --max_grad_norm 1.0 \
        "$@"
else
    echo "使用单卡训练模式"
    python src/train_partcrafter.py \
        --config $CONFIG \
        --tag $TAG \
        --output_dir $DEFAULT_OUTPUT_DIR \
        --use_ema \
        --gradient_accumulation_steps 4 \
        --mixed_precision fp16 \
        --allow_tf32 \
        --pin_memory \
        --num_workers 8 \
        --max_grad_norm 1.0 \
        "$@"
fi

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✅ 训练完成！"
    echo "================================================"
    echo "结果保存在: $DEFAULT_OUTPUT_DIR/$TAG"
    echo "checkpoints: $DEFAULT_OUTPUT_DIR/$TAG/checkpoints"
    echo "evaluations: $DEFAULT_OUTPUT_DIR/$TAG/evaluations"
    echo "================================================"
else
    echo ""
    echo "================================================"
    echo "❌ 训练失败，请检查错误信息"
    echo "================================================"
    exit 1
fi

