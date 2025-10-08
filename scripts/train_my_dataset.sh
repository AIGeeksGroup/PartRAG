#!/bin/bash

# ============================================================================
# 使用你的数据集训练 PartCrafter
# ============================================================================
# 预训练模型: /root/autodl-tmp/PartCrafter/pretrained_weights/PartCrafter
# 训练数据: /root/autodl-tmp/datasets
# ============================================================================

set -e  # 遇到错误立即退出

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         PartCrafter 细粒度对比学习训练                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 切换到项目目录
cd /root/autodl-tmp/PartCrafter

# 设置Python路径
export PYTHONPATH="/root/autodl-tmp/PartCrafter:$PYTHONPATH"

# 激活环境
echo -e "${YELLOW}🔧 激活 partcrafter 环境...${NC}"
source /root/miniconda3/bin/activate partcrafter || conda activate partcrafter

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 无法激活环境！${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 环境激活成功${NC}"
python --version
echo ""

# 检查预训练模型
PRETRAINED_PATH="/root/autodl-tmp/PartCrafter/pretrained_weights/PartCrafter"
if [ ! -d "$PRETRAINED_PATH" ]; then
    echo -e "${RED}❌ 预训练模型未找到: $PRETRAINED_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 找到预训练模型${NC}"

# 检查数据集
DATA_CONFIG="/root/autodl-tmp/datasets/object_part_configs_400.json"
if [ ! -f "$DATA_CONFIG" ]; then
    echo -e "${RED}❌ 数据配置文件未找到: $DATA_CONFIG${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 找到数据配置文件${NC}"

# 检查数据集大小
NUM_SAMPLES=$(python -c "import json; print(len(json.load(open('$DATA_CONFIG'))))")
echo -e "${GREEN}✅ 数据集包含 $NUM_SAMPLES 个样本${NC}"
echo ""

# 解析参数
MODE=${1:-"test"}  # test 或 full
EXTRA_ARGS="${@:2}"

if [ "$MODE" = "test" ]; then
    echo "════════════════════════════════════════════════════════════"
    echo "📋 模式: 测试运行"
    echo "════════════════════════════════════════════════════════════"
    CONFIG="configs/my_training_test.yaml"
    TAG="test_$(date +%Y%m%d_%H%M%S)"
    MAX_STEPS=100
    echo "配置: $CONFIG"
    echo "标签: $TAG"
    echo "最大步数: $MAX_STEPS"
elif [ "$MODE" = "full" ]; then
    echo "════════════════════════════════════════════════════════════"
    echo "📋 模式: 完整训练"
    echo "════════════════════════════════════════════════════════════"
    CONFIG="configs/my_training_contrastive.yaml"
    TAG="full_training_$(date +%Y%m%d_%H%M%S)"
    MAX_STEPS=null
    echo "配置: $CONFIG"
    echo "标签: $TAG"
    echo -e "${YELLOW}⚠️  这将进行完整训练，可能需要很长时间！${NC}"
    echo "自动开始训练..."
else
    echo -e "${RED}❌ 未知模式: $MODE${NC}"
    echo "用法: $0 [test|full]"
    exit 1
fi
echo "════════════════════════════════════════════════════════════"
echo ""

# 检查GPU
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ $NUM_GPUS -eq 0 ]; then
    echo -e "${RED}❌ 未检测到GPU！${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 检测到 $NUM_GPUS 张GPU${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 创建输出目录
OUTPUT_DIR="/root/autodl-tmp/output_training"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✅ 输出目录: $OUTPUT_DIR/$TAG${NC}"
echo ""

# 开始训练
echo "════════════════════════════════════════════════════════════"
echo "🚀 开始训练..."
echo "════════════════════════════════════════════════════════════"
echo ""

if [ $NUM_GPUS -gt 1 ]; then
    echo "使用多卡训练模式 ($NUM_GPUS GPUs)"
    CMD="torchrun --nproc_per_node=$NUM_GPUS"
else
    echo "使用单卡训练模式"
    CMD="python"
fi

# 构建训练命令
TRAIN_CMD="$CMD src/train_partcrafter.py \
    --config $CONFIG \
    --tag $TAG \
    --output_dir $OUTPUT_DIR \
    --use_ema \
    --gradient_accumulation_steps 4 \
    --mixed_precision fp16 \
    --allow_tf32 \
    --pin_memory \
    --num_workers 8 \
    --max_grad_norm 1.0 \
    --max_val_steps 8 \
    --offline_wandb"

# 添加最大步数限制（仅测试模式）
if [ "$MODE" = "test" ]; then
    TRAIN_CMD="$TRAIN_CMD --max_train_steps $MAX_STEPS"
fi

# 添加额外参数
if [ -n "$EXTRA_ARGS" ]; then
    TRAIN_CMD="$TRAIN_CMD $EXTRA_ARGS"
fi

# 执行训练
echo "执行命令:"
echo "$TRAIN_CMD"
echo ""

eval $TRAIN_CMD

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo -e "${GREEN}✅ 训练完成！${NC}"
    echo "════════════════════════════════════════════════════════════"
    echo "结果位置: $OUTPUT_DIR/$TAG"
    echo "  - 检查点: $OUTPUT_DIR/$TAG/checkpoints/"
    echo "  - 评估结果: $OUTPUT_DIR/$TAG/evaluations/"
    echo "  - 日志文件: $OUTPUT_DIR/$TAG/log.txt"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    
    # 显示最后几行日志
    if [ -f "$OUTPUT_DIR/$TAG/log.txt" ]; then
        echo "最后10行日志:"
        tail -10 "$OUTPUT_DIR/$TAG/log.txt"
    fi
else
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo -e "${RED}❌ 训练失败！${NC}"
    echo "════════════════════════════════════════════════════════════"
    echo "请检查错误信息"
    if [ -f "$OUTPUT_DIR/$TAG/log.txt" ]; then
        echo ""
        echo "日志文件: $OUTPUT_DIR/$TAG/log.txt"
        echo "最后20行:"
        tail -20 "$OUTPUT_DIR/$TAG/log.txt"
    fi
    exit 1
fi

