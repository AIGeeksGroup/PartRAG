# PartCrafter 训练和推理

PartCrafter 用于 part-level 3D 物体生成，支持检索增强功能。

## 快速开始

### 训练

```bash
cd /root/autodl-tmp/PartCrafter
./start_training.sh
```

### 推理

**基础推理**：
```bash
python scripts/inference_partcrafter.py \
  --input_image input.png \
  --output_dir outputs \
  --num_parts 4
```

**检索增强推理**（idea2）：
```bash
# 1. 构建检索数据库
python scripts/build_retrieval_database.py \
  --images_dir reference_images \
  --output_path retrieval_db/database.pt

# 2. 运行检索增强生成
python scripts/inference_with_retrieval.py \
  --model_path /root/autodl-tmp/output_training/.../checkpoints/... \
  --input_image input.png \
  --output_dir outputs_retrieval \
  --num_parts 4 \
  --use_retrieval \
  --retrieval_db_path retrieval_db/database.pt \
  --retrieval_db_images_dir reference_images
```

## 目录结构

```
PartCrafter/
├── src/                    # 源代码
│   ├── retrieval/          # 检索增强模块（idea2）
│   ├── models/             # 模型定义
│   ├── pipelines/          # Pipeline（支持检索增强）
│   └── ...
├── scripts/                # 工具脚本
├── configs/                # 训练配置
├── pretrained_weights/     # 预训练模型
└── start_training.sh       # 训练启动脚本
```

## 核心功能

### 1. Part-Level 3D生成
基于图像生成多部件3D物体

### 2. 检索增强生成（idea2）
- 参考 ReMoMask 检索增强方法
- 使用 CLIP 检索相关参考图像
- Concatenate embeddings 通过 cross attention 指导生成

## 工具脚本

| 脚本 | 功能 |
|------|------|
| `inference_partcrafter.py` | 基础推理 |
| `inference_with_retrieval.py` | 检索增强推理 |
| `build_retrieval_database.py` | 构建检索数据库 |
| `check_and_select_model.py` | 模型选择工具 |
| `train_my_dataset.sh` | 自定义数据训练 |

## 模型路径

推荐使用训练后的 EMA checkpoint：
```
/root/autodl-tmp/output_training/full_training_*/checkpoints/000XXX/
```

## 参数说明

### 训练参数
- `--num_parts`: 部件数量（4-8）
- `--batch_size`: 批大小
- `--learning_rate`: 学习率

### 推理参数
- `--num_inference_steps`: 推理步数（30-100）
- `--guidance_scale`: 引导强度（5.0-10.0）
- `--seed`: 随机种子

### 检索增强参数
- `--use_retrieval`: 启用检索增强
- `--num_retrieved_images`: 检索图像数量（1-3）
- `--retrieval_query_text`: 文本查询（可选）

## 依赖

- PyTorch
- Diffusers
- Transformers
- CLIP（用于检索增强）

## VPN（如需下载模型）

```bash
source /etc/network_turbo
export HF_ENDPOINT=https://hf-mirror.com
```

## 参考

- **ReMoMask**: https://aigeeksgroup.github.io/ReMoMask/
- **检索增强方法**: Concatenate retrieved content with prompt via cross attention
