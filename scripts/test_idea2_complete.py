"""
完整的 Idea 2 端到端测试脚本
测试检索增强生成的完整流程
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import trimesh
import time

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.retrieval.retrieval_module import RetrievalModule


def load_pipeline_with_checkpoint(pretrained_path, checkpoint_path, device="cuda"):
    """
    加载pipeline并用checkpoint替换transformer
    """
    print("\n" + "="*80)
    print("加载模型")
    print("="*80)
    
    # 1. 加载完整pipeline
    print(f"正在加载完整pipeline: {pretrained_path}")
    pipeline = PartCrafterPipeline.from_pretrained(
        pretrained_path,
        torch_dtype=torch.float16,
    ).to(device)
    print("✓ Pipeline 加载完成")
    
    # 2. 加载训练好的transformer
    if checkpoint_path:
        print(f"\n正在加载训练好的transformer: {checkpoint_path}")
        transformer_path = os.path.join(checkpoint_path, "transformer_ema")
        
        if os.path.exists(transformer_path):
            from src.models.transformers import PartCrafterDiTModel
            trained_transformer = PartCrafterDiTModel.from_pretrained(
                transformer_path,
                torch_dtype=torch.float16,
            ).to(device)
            
            # 替换pipeline中的transformer
            pipeline.transformer = trained_transformer
            print("✓ Transformer 替换完成（使用EMA权重）")
        else:
            print("⚠️  警告: 未找到transformer_ema，使用预训练权重")
    
    print("="*80 + "\n")
    return pipeline


def test_retrieval_augmented_generation(
    pipeline,
    retrieval_module,
    test_image_path,
    output_dir,
    num_parts=3,
    num_retrieved_images=1,
    num_inference_steps=50,
    guidance_scale=7.5,
):
    """
    测试检索增强生成
    """
    print("\n" + "="*80)
    print("运行检索增强生成")
    print("="*80)
    print(f"输入图像: {test_image_path}")
    print(f"部件数量: {num_parts}")
    print(f"检索图像数量: {num_retrieved_images}")
    print(f"推理步数: {num_inference_steps}")
    print(f"引导比例: {guidance_scale}")
    print("="*80 + "\n")
    
    # 加载测试图像
    image = Image.open(test_image_path).convert("RGB")
    
    # 1. 有检索增强的生成
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("测试 1: 使用检索增强")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    start_time = time.time()
    output_with_retrieval = pipeline(
        image=[image] * num_parts,
        attention_kwargs={"num_parts": num_parts},
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        use_retrieval=True,
        num_retrieved_images=num_retrieved_images,
    )
    time_with_retrieval = time.time() - start_time
    
    print(f"✓ 生成完成 (耗时: {time_with_retrieval:.2f}秒)")
    print(f"  生成的网格数: {len(output_with_retrieval.meshes)}")
    
    # 保存结果
    output_with_path = os.path.join(output_dir, "result_with_retrieval.obj")
    valid_meshes_with = []
    for i, mesh in enumerate(output_with_retrieval.meshes):
        if mesh is not None:
            part_path = os.path.join(output_dir, f"result_with_retrieval_part{i}.obj")
            mesh.export(part_path)
            valid_meshes_with.append(mesh)
        else:
            print(f"  ⚠️  Part {i} is None (decoding error)")
    
    # 合并所有有效parts
    if valid_meshes_with:
        combined_mesh = trimesh.util.concatenate(valid_meshes_with)
        combined_mesh.export(output_with_path)
        print(f"✓ 结果已保存到: {output_with_path}\n")
    else:
        print(f"⚠️  警告: 所有部件都是None，无法保存\n")
    
    # 2. 无检索增强的生成（对比）
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("测试 2: 不使用检索增强（对比基线）")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    start_time = time.time()
    output_without_retrieval = pipeline(
        image=[image] * num_parts,
        attention_kwargs={"num_parts": num_parts},
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        use_retrieval=False,
    )
    time_without_retrieval = time.time() - start_time
    
    print(f"✓ 生成完成 (耗时: {time_without_retrieval:.2f}秒)")
    print(f"  生成的网格数: {len(output_without_retrieval.meshes)}")
    
    # 保存结果
    output_without_path = os.path.join(output_dir, "result_without_retrieval.obj")
    valid_meshes_without = []
    for i, mesh in enumerate(output_without_retrieval.meshes):
        if mesh is not None:
            part_path = os.path.join(output_dir, f"result_without_retrieval_part{i}.obj")
            mesh.export(part_path)
            valid_meshes_without.append(mesh)
        else:
            print(f"  ⚠️  Part {i} is None (decoding error)")
    
    # 合并所有有效parts
    if valid_meshes_without:
        combined_mesh = trimesh.util.concatenate(valid_meshes_without)
        combined_mesh.export(output_without_path)
        print(f"✓ 结果已保存到: {output_without_path}\n")
    else:
        print(f"⚠️  警告: 所有部件都是None，无法保存\n")
    
    # 3. 统计对比
    print("="*80)
    print("性能对比")
    print("="*80)
    print(f"{'方法':<20} {'时间(秒)':<15} {'网格数':<10}")
    print("-"*80)
    print(f"{'有检索增强':<20} {time_with_retrieval:<15.2f} {len(output_with_retrieval.meshes):<10}")
    print(f"{'无检索增强':<20} {time_without_retrieval:<15.2f} {len(output_without_retrieval.meshes):<10}")
    print(f"{'时间差异':<20} {abs(time_with_retrieval - time_without_retrieval):<15.2f}")
    print("="*80 + "\n")
    
    # 保存输入图像副本
    image.save(os.path.join(output_dir, "input_image.png"))
    
    return {
        "with_retrieval": output_with_retrieval,
        "without_retrieval": output_without_retrieval,
        "time_with": time_with_retrieval,
        "time_without": time_without_retrieval,
    }


def main():
    parser = argparse.ArgumentParser(description="完整的 Idea 2 测试")
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
        "--retrieval_db_path",
        type=str,
        default="/root/autodl-tmp/retrieval_database/database.pt",
        help="检索数据库路径"
    )
    parser.add_argument(
        "--test_image",
        type=str,
        required=True,
        help="测试图像路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/idea2_test_results",
        help="输出目录"
    )
    parser.add_argument(
        "--num_parts",
        type=int,
        default=3,
        help="生成的部件数量"
    )
    parser.add_argument(
        "--num_retrieved_images",
        type=int,
        default=1,
        help="检索图像数量"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="推理步数"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="引导比例"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "Idea 2 完整测试" + " "*38 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # 1. 加载Pipeline
    pipeline = load_pipeline_with_checkpoint(
        args.pretrained_path,
        args.checkpoint_path,
        args.device
    )
    
    # 2. 加载检索模块和数据库
    print("="*80)
    print("加载检索模块")
    print("="*80)
    print(f"检索数据库: {args.retrieval_db_path}")
    
    retrieval_module = RetrievalModule(
        clip_model_name="openai/clip-vit-large-patch14",
        device=args.device
    )
    retrieval_module.load_database(args.retrieval_db_path)
    print(f"✓ 检索数据库加载完成")
    num_images = len(retrieval_module.database_images) if retrieval_module.database_images is not None else retrieval_module.database_embeddings.shape[0]
    print(f"  数据库大小: {num_images} 张图像")
    print(f"  Embedding维度: {retrieval_module.database_embeddings.shape[1]}")
    print("="*80 + "\n")
    
    # 3. 将检索模块注册到pipeline
    pipeline.retrieval_module = retrieval_module
    
    # 4. 运行测试
    results = test_retrieval_augmented_generation(
        pipeline=pipeline,
        retrieval_module=retrieval_module,
        test_image_path=args.test_image,
        output_dir=args.output_dir,
        num_parts=args.num_parts,
        num_retrieved_images=args.num_retrieved_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    
    # 5. 总结
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*30 + "测试完成！" + " "*37 + "║")
    print("╚" + "="*78 + "╝\n")
    print("结果保存在:", args.output_dir)
    print("  - result_with_retrieval.obj      (使用检索增强)")
    print("  - result_without_retrieval.obj   (不使用检索增强)")
    print("  - input_image.png                (输入图像)")
    print("\n您可以使用 Blender 或 MeshLab 等工具查看生成的3D模型。\n")


if __name__ == "__main__":
    main()

