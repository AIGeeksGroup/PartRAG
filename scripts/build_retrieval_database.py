"""
构建检索数据库
从图像目录构建CLIP embeddings数据库用于检索增强
"""

import argparse
import os
import sys
from pathlib import Path
from glob import glob
from PIL import Image
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval import RetrievalModule


def build_database(
    images_dir: str,
    output_path: str,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    batch_size: int = 32,
    image_extensions: list = None,
    device: str = "cuda",
):
    """
    从图像目录构建检索数据库
    
    Args:
        images_dir: 图像目录
        output_path: 输出数据库路径
        clip_model_name: CLIP模型名称
        batch_size: 批处理大小
        image_extensions: 图像扩展名列表
        device: 设备
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    print("\n" + "="*80)
    print("构建检索数据库")
    print("="*80)
    print(f"图像目录: {images_dir}")
    print(f"输出路径: {output_path}")
    print(f"CLIP模型: {clip_model_name}")
    print(f"批处理大小: {batch_size}")
    print(f"设备: {device}")
    print("="*80 + "\n")
    
    # 查找所有图像
    print("正在搜索图像文件...")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(images_dir, f"**/*{ext}"), recursive=True))
        image_paths.extend(glob(os.path.join(images_dir, f"**/*{ext.upper()}"), recursive=True))
    
    image_paths = sorted(list(set(image_paths)))
    print(f"✓ 找到 {len(image_paths)} 张图像\n")
    
    if len(image_paths) == 0:
        print("错误: 未找到任何图像文件")
        return
    
    # 加载检索模块
    print("正在加载CLIP模型...")
    retrieval_module = RetrievalModule(
        clip_model_name=clip_model_name,
        device=device,
    )
    print("✓ CLIP模型加载完成\n")
    
    # 加载所有图像
    print("正在加载图像...")
    images = []
    valid_image_paths = []
    
    for img_path in tqdm(image_paths, desc="加载图像"):
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            valid_image_paths.append(img_path)
        except Exception as e:
            print(f"警告: 无法加载 {img_path}: {e}")
    
    print(f"✓ 成功加载 {len(images)} 张图像\n")
    
    # 构建数据库
    print("正在构建embeddings数据库...")
    retrieval_module.build_database(images, batch_size=batch_size)
    print("✓ 数据库构建完成\n")
    
    # 保存数据库
    print(f"正在保存数据库到 {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    retrieval_module.save_database(output_path)
    print("✓ 数据库保存完成\n")
    
    # 保存图像路径列表
    image_list_path = output_path.replace('.pt', '_image_list.txt')
    with open(image_list_path, 'w') as f:
        for img_path in valid_image_paths:
            f.write(img_path + '\n')
    print(f"✓ 图像列表保存到 {image_list_path}\n")
    
    # 统计信息
    print("="*80)
    print("数据库统计:")
    print("="*80)
    print(f"总图像数: {len(images)}")
    print(f"Embedding维度: {retrieval_module.database_embeddings.shape[1]}")
    print(f"数据库文件: {output_path}")
    print(f"图像列表文件: {image_list_path}")
    print("="*80 + "\n")
    
    # 测试检索
    print("测试检索功能...")
    test_image = images[0]
    retrieved_images, scores = retrieval_module.retrieve_by_image(
        test_image, top_k=5, return_scores=True
    )
    print(f"✓ 检索测试成功")
    print(f"  查询图像: {valid_image_paths[0]}")
    print(f"  Top-5 相似度分数: {scores.tolist()}")
    print("\n" + "="*80)
    print("数据库构建完成！")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="构建检索数据库")
    parser.add_argument(
        "--images_dir", 
        type=str, 
        required=True,
        help="图像目录路径"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="输出数据库路径（.pt文件）"
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP模型名称"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="批处理大小"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备"
    )
    
    args = parser.parse_args()
    
    build_database(
        images_dir=args.images_dir,
        output_path=args.output_path,
        clip_model_name=args.clip_model_name,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()



