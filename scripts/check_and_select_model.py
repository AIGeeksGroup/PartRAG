"""
模型选择工具
判断应该使用哪个模型：预训练模型 vs 训练后的checkpoint
"""

import os
import json
from pathlib import Path
from datetime import datetime
import torch


def check_model_info(model_path):
    """检查模型信息"""
    info = {
        'path': model_path,
        'exists': os.path.exists(model_path),
        'is_pretrained': False,
        'is_checkpoint': False,
        'has_ema': False,
        'transformer_path': None,
        'config': None,
        'last_modified': None,
    }
    
    if not info['exists']:
        return info
    
    # 检查是否是预训练模型
    model_index_path = os.path.join(model_path, 'model_index.json')
    if os.path.exists(model_index_path):
        info['is_pretrained'] = True
        with open(model_index_path, 'r') as f:
            info['config'] = json.load(f)
        transformer_path = os.path.join(model_path, 'transformer')
        if os.path.exists(transformer_path):
            info['transformer_path'] = transformer_path
            # 获取最后修改时间
            safetensors_path = os.path.join(transformer_path, 'diffusion_pytorch_model.safetensors')
            if os.path.exists(safetensors_path):
                timestamp = os.path.getmtime(safetensors_path)
                info['last_modified'] = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    # 检查是否是训练checkpoint
    if 'checkpoints' in model_path or 'checkpoint-' in model_path:
        info['is_checkpoint'] = True
        # 查找transformer和transformer_ema
        transformer_path = os.path.join(model_path, 'transformer')
        transformer_ema_path = os.path.join(model_path, 'transformer_ema')
        
        if os.path.exists(transformer_ema_path):
            info['has_ema'] = True
            info['transformer_path'] = transformer_ema_path
            print(f"  ✓ Found EMA model (preferred)")
        elif os.path.exists(transformer_path):
            info['transformer_path'] = transformer_path
            print(f"  ✓ Found standard transformer")
        
        # 检查config
        if info['transformer_path']:
            config_path = os.path.join(info['transformer_path'], 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    info['config'] = json.load(f)
            
            # 获取最后修改时间
            safetensors_path = os.path.join(info['transformer_path'], 'diffusion_pytorch_model.safetensors')
            if os.path.exists(safetensors_path):
                timestamp = os.path.getmtime(safetensors_path)
                info['last_modified'] = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                
                # 获取文件大小
                file_size = os.path.getsize(safetensors_path) / (1024**3)  # GB
                print(f"  ✓ Model size: {file_size:.2f} GB")
    
    return info


def compare_models(pretrained_path, checkpoint_path):
    """比较预训练模型和checkpoint"""
    print("\n" + "="*80)
    print("模型比较")
    print("="*80)
    
    print("\n1. 预训练模型:")
    print(f"   路径: {pretrained_path}")
    pretrained_info = check_model_info(pretrained_path)
    if pretrained_info['exists']:
        print(f"   ✓ 存在")
        print(f"   类型: {'预训练模型' if pretrained_info['is_pretrained'] else '未知'}")
        if pretrained_info['last_modified']:
            print(f"   最后修改: {pretrained_info['last_modified']}")
    else:
        print(f"   ✗ 不存在")
    
    print("\n2. 训练Checkpoint:")
    print(f"   路径: {checkpoint_path}")
    checkpoint_info = check_model_info(checkpoint_path)
    if checkpoint_info['exists']:
        print(f"   ✓ 存在")
        print(f"   类型: {'训练Checkpoint' if checkpoint_info['is_checkpoint'] else '未知'}")
        if checkpoint_info['has_ema']:
            print(f"   EMA模型: ✓ (推荐使用)")
        if checkpoint_info['last_modified']:
            print(f"   最后修改: {checkpoint_info['last_modified']}")
    else:
        print(f"   ✗ 不存在")
    
    # 推荐
    print("\n" + "="*80)
    print("推荐:")
    print("="*80)
    
    if checkpoint_info['exists'] and checkpoint_info['has_ema']:
        print("✓ 推荐使用训练后的EMA checkpoint")
        print(f"  原因: EMA模型通常更稳定，性能更好")
        print(f"  路径: {checkpoint_path}")
        recommended = checkpoint_path
    elif checkpoint_info['exists']:
        print("✓ 推荐使用训练后的checkpoint")
        print(f"  原因: 已经过训练，可能性能更好")
        print(f"  路径: {checkpoint_path}")
        recommended = checkpoint_path
    elif pretrained_info['exists']:
        print("✓ 推荐使用预训练模型")
        print(f"  原因: checkpoint不存在")
        print(f"  路径: {pretrained_path}")
        recommended = pretrained_path
    else:
        print("✗ 两个模型都不存在！")
        recommended = None
    
    print("="*80 + "\n")
    
    return recommended, pretrained_info, checkpoint_info


def find_latest_checkpoint(training_output_dir):
    """查找最新的checkpoint"""
    if not os.path.exists(training_output_dir):
        return None
    
    # 查找所有checkpoint目录
    checkpoint_dirs = []
    for root, dirs, files in os.walk(training_output_dir):
        if 'transformer' in dirs or 'transformer_ema' in dirs:
            checkpoint_dirs.append(root)
    
    if not checkpoint_dirs:
        return None
    
    # 按修改时间排序
    checkpoint_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return checkpoint_dirs[0]


def load_model_for_inference(model_path, device='cuda', dtype=torch.float16):
    """
    加载模型用于推理
    
    Args:
        model_path: 模型路径
        device: 设备
        dtype: 数据类型
    
    Returns:
        pipeline: 加载好的pipeline
    """
    from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
    
    print(f"\n加载模型: {model_path}")
    print(f"设备: {device}")
    print(f"数据类型: {dtype}")
    
    try:
        pipe = PartCrafterPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(device)
        
        print(f"✓ 模型加载成功")
        
        # 打印一些模型信息
        if hasattr(pipe.transformer, 'config'):
            config = pipe.transformer.config
            print(f"\n模型配置:")
            print(f"  - 层数: {config.num_layers}")
            print(f"  - 注意力头数: {config.num_attention_heads}")
            print(f"  - 隐藏维度: {config.width}")
            print(f"  - 输入通道: {config.in_channels}")
            if hasattr(config, 'enable_part_embedding'):
                print(f"  - Part embedding: {config.enable_part_embedding}")
            if hasattr(config, 'global_attn_block_ids'):
                print(f"  - Global attention blocks: {config.global_attn_block_ids}")
        
        return pipe
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="检查并选择模型")
    parser.add_argument(
        "--pretrained_path", 
        type=str, 
        default="/root/autodl-tmp/PartCrafter/pretrained_weights/PartCrafter",
        help="预训练模型路径"
    )
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default="/root/autodl-tmp/output_training/full_training_20251008_003310/checkpoints/000120",
        help="训练checkpoint路径"
    )
    parser.add_argument(
        "--training_output_dir",
        type=str,
        default="/root/autodl-tmp/output_training",
        help="训练输出目录（自动查找最新checkpoint）"
    )
    parser.add_argument(
        "--auto_find_latest",
        action="store_true",
        help="自动查找最新的checkpoint"
    )
    parser.add_argument(
        "--test_load",
        action="store_true",
        help="测试加载推荐的模型"
    )
    
    args = parser.parse_args()
    
    # 如果启用自动查找
    if args.auto_find_latest:
        print(f"\n自动查找最新checkpoint...")
        print(f"搜索目录: {args.training_output_dir}")
        latest_checkpoint = find_latest_checkpoint(args.training_output_dir)
        if latest_checkpoint:
            print(f"✓ 找到最新checkpoint: {latest_checkpoint}")
            args.checkpoint_path = latest_checkpoint
        else:
            print(f"✗ 未找到checkpoint")
    
    # 比较模型
    recommended, pretrained_info, checkpoint_info = compare_models(
        args.pretrained_path, 
        args.checkpoint_path
    )
    
    # 测试加载
    if args.test_load and recommended:
        print(f"\n测试加载推荐的模型...")
        pipe = load_model_for_inference(recommended)
        if pipe:
            print(f"\n✓ 模型可以正常加载和使用")
        else:
            print(f"\n✗ 模型加载失败")
    
    # 输出使用建议
    print("\n" + "="*80)
    print("使用建议:")
    print("="*80)
    print("\n在推理脚本中使用:")
    print(f"  python scripts/inference_with_retrieval.py \\")
    print(f"    --model_path {recommended} \\")
    print(f"    --input_image <your_image.png> \\")
    print(f"    --use_retrieval \\")
    print(f"    --retrieval_db_path <database.pt> \\")
    print(f"    --retrieval_db_images_dir <images_dir>")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()



