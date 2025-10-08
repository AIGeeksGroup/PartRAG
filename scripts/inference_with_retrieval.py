"""
检索增强的3D生成推理脚本（idea2实现）
参考 ReMoMask 的检索增强方法
"""

import argparse
import os
import sys
from pathlib import Path
from glob import glob
import time
from typing import Any, Union

import numpy as np
import torch
import trimesh
from PIL import Image
from accelerate.utils import set_seed

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_utils import get_colored_mesh_composition
from src.utils.render_utils import render_views_around_mesh, make_grid_for_images_or_videos, export_renderings
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG
from src.retrieval import RetrievalModule


@torch.no_grad()
def run_partcrafter_with_retrieval(
    pipe: Any,
    image_input: Union[str, Image.Image],
    num_parts: int,
    rmbg_net: Any = None,
    seed: int = 42,
    num_tokens: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    max_num_expanded_coords: int = 1e9,
    use_flash_decoder: bool = False,
    rmbg: bool = False,
    # 检索增强参数
    use_retrieval: bool = False,
    retrieval_query_text: str = None,
    num_retrieved_images: int = 1,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> trimesh.Scene:
    """
    使用检索增强运行PartCrafter生成
    
    Args:
        pipe: PartCrafterPipeline实例
        image_input: 输入图像路径或PIL图像
        num_parts: 部件数量
        rmbg_net: 背景移除网络
        seed: 随机种子
        num_tokens: token数量
        num_inference_steps: 推理步数
        guidance_scale: 引导尺度
        max_num_expanded_coords: 最大扩展坐标数
        use_flash_decoder: 是否使用flash decoder
        rmbg: 是否移除背景
        use_retrieval: 是否使用检索增强（idea2）
        retrieval_query_text: 检索查询文本
        num_retrieved_images: 检索图像数量
        dtype: 数据类型
        device: 设备
    """
    
    # 准备输入图像
    if rmbg and rmbg_net is not None:
        img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
    else:
        if isinstance(image_input, str):
            img_pil = Image.open(image_input).convert('RGB')
        else:
            img_pil = image_input
    
    print(f"{'='*80}")
    print(f"Running PartCrafter with Retrieval Augmentation (idea2)")
    print(f"{'='*80}")
    print(f"Input image: {image_input if isinstance(image_input, str) else 'PIL Image'}")
    print(f"Number of parts: {num_parts}")
    print(f"Use retrieval: {use_retrieval}")
    if use_retrieval:
        print(f"  - Retrieval query text: {retrieval_query_text if retrieval_query_text else 'Using input image'}")
        print(f"  - Number of retrieved images: {num_retrieved_images}")
    print(f"Seed: {seed}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    outputs = pipe(
        image=[img_pil] * num_parts,
        attention_kwargs={"num_parts": num_parts},
        num_tokens=num_tokens,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_num_expanded_coords=max_num_expanded_coords,
        use_flash_decoder=use_flash_decoder,
        # 检索增强参数
        use_retrieval=use_retrieval,
        retrieval_query_text=retrieval_query_text,
        num_retrieved_images=num_retrieved_images,
    ).meshes
    
    end_time = time.time()
    print(f"\n{'='*80}")
    print(f"Generation completed in {end_time - start_time:.2f} seconds")
    print(f"{'='*80}\n")
    
    # 处理None mesh
    for i in range(len(outputs)):
        if outputs[i] is None:
            print(f"Warning: Part {i} mesh is None, using dummy mesh")
            outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
    
    return outputs, img_pil


def load_pipeline_with_retrieval(
    model_path: str,
    retrieval_db_path: str = None,
    retrieval_db_images_dir: str = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """
    加载带有检索模块的PartCrafter pipeline
    
    Args:
        model_path: 模型路径
        retrieval_db_path: 检索数据库embeddings路径
        retrieval_db_images_dir: 检索数据库图像目录
        device: 设备
        dtype: 数据类型
    """
    print(f"\n{'='*80}")
    print(f"Loading PartCrafter Pipeline with Retrieval Module")
    print(f"{'='*80}")
    print(f"Model path: {model_path}")
    
    # 加载基础pipeline
    pipe = PartCrafterPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    ).to(device)
    
    print(f"✓ Base pipeline loaded")
    
    # 加载检索模块（如果提供了数据库）
    if retrieval_db_path and retrieval_db_images_dir:
        print(f"\nLoading retrieval module...")
        print(f"  - Database embeddings: {retrieval_db_path}")
        print(f"  - Database images dir: {retrieval_db_images_dir}")
        
        retrieval_module = RetrievalModule(device=device)
        
        # 加载数据库图像
        image_paths = sorted(glob(os.path.join(retrieval_db_images_dir, "*.[pj][np]g")))
        if len(image_paths) == 0:
            print(f"Warning: No images found in {retrieval_db_images_dir}")
            database_images = None
        else:
            print(f"  - Found {len(image_paths)} images")
            database_images = [Image.open(p).convert('RGB') for p in image_paths]
        
        # 加载或构建embeddings
        if os.path.exists(retrieval_db_path):
            retrieval_module.load_database(retrieval_db_path, database_images)
            print(f"✓ Retrieval database loaded")
        elif database_images is not None:
            print(f"  - Building database embeddings...")
            retrieval_module.build_database(database_images)
            retrieval_module.save_database(retrieval_db_path)
            print(f"✓ Database built and saved to {retrieval_db_path}")
        
        # 将检索模块添加到pipeline
        pipe.retrieval_module = retrieval_module
        print(f"✓ Retrieval module integrated")
    else:
        print(f"\nNo retrieval database provided, retrieval augmentation disabled")
    
    print(f"{'='*80}\n")
    
    return pipe


def main():
    parser = argparse.ArgumentParser(description="PartCrafter with Retrieval Augmentation (idea2)")
    
    # 基本参数
    parser.add_argument("--input_image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output_dir", type=str, default="outputs_retrieval", help="输出目录")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--num_parts", type=int, default=4, help="部件数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 生成参数
    parser.add_argument("--num_tokens", type=int, default=1024, help="Token数量")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="引导尺度")
    parser.add_argument("--rmbg", action="store_true", help="是否移除背景")
    
    # 检索增强参数（idea2）
    parser.add_argument("--use_retrieval", action="store_true", help="是否使用检索增强（idea2）")
    parser.add_argument("--retrieval_db_path", type=str, default=None, help="检索数据库embeddings路径")
    parser.add_argument("--retrieval_db_images_dir", type=str, default=None, help="检索数据库图像目录")
    parser.add_argument("--retrieval_query_text", type=str, default=None, help="检索查询文本")
    parser.add_argument("--num_retrieved_images", type=int, default=1, help="检索图像数量")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"], help="数据类型")
    
    args = parser.parse_args()
    
    # 设置数据类型
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载RMBG模型（如果需要）
    rmbg_net = None
    if args.rmbg:
        print("Loading RMBG model...")
        rmbg_net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
        rmbg_net = rmbg_net.to(args.device).to(dtype)
        rmbg_net.eval()
        print("✓ RMBG model loaded")
    
    # 加载pipeline
    pipe = load_pipeline_with_retrieval(
        model_path=args.model_path,
        retrieval_db_path=args.retrieval_db_path,
        retrieval_db_images_dir=args.retrieval_db_images_dir,
        device=args.device,
        dtype=dtype,
    )
    
    # 运行生成
    meshes, input_image = run_partcrafter_with_retrieval(
        pipe=pipe,
        image_input=args.input_image,
        num_parts=args.num_parts,
        rmbg_net=rmbg_net,
        seed=args.seed,
        num_tokens=args.num_tokens,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        use_retrieval=args.use_retrieval,
        retrieval_query_text=args.retrieval_query_text,
        num_retrieved_images=args.num_retrieved_images,
        dtype=dtype,
        device=args.device,
    )
    
    # 保存结果
    print("Saving results...")
    
    # 保存输入图像
    input_image_path = os.path.join(args.output_dir, "input_image.png")
    input_image.save(input_image_path)
    print(f"✓ Input image saved to {input_image_path}")
    
    # 组合mesh
    colored_mesh = get_colored_mesh_composition(meshes, args.num_parts)
    
    # 保存组合mesh
    mesh_output_path = os.path.join(args.output_dir, "output_mesh.obj")
    colored_mesh.export(mesh_output_path)
    print(f"✓ Combined mesh saved to {mesh_output_path}")
    
    # 保存各个部件
    parts_dir = os.path.join(args.output_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    for i, mesh in enumerate(meshes):
        if mesh is not None and len(mesh.vertices) > 1:
            part_path = os.path.join(parts_dir, f"part_{i:02d}.obj")
            mesh.export(part_path)
    print(f"✓ Individual parts saved to {parts_dir}")
    
    # 渲染视图
    print("Rendering views...")
    rendered_images = render_views_around_mesh(colored_mesh, num_views=8)
    grid_image = make_grid_for_images_or_videos(rendered_images, rows=2, cols=4)
    grid_path = os.path.join(args.output_dir, "rendered_views.png")
    grid_image.save(grid_path)
    print(f"✓ Rendered views saved to {grid_path}")
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()



