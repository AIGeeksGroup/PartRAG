#!/usr/bin/env python3
"""
:checkpointcheckpoint,
"""
import argparse
import os
import sys
import time
from typing import Any, Union

import numpy as np
import torch
import trimesh
from PIL import Image
from accelerate.utils import set_seed

# src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_utils import get_colored_mesh_composition
from src.utils.render_utils import (
    render_views_around_mesh,
    render_normal_views_around_mesh,
    make_grid_for_images_or_videos,
    export_renderings
)
from src.pipelines.pipeline_partrag import PartragPipeline
from src.retrieval.retrieval_module import RetrievalModule
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG
from src.utils.weights_utils import resolve_or_download_weights
from trimesh import repair as tm_repair
from trimesh.smoothing import filter_taubin

def _postprocess_mesh(mesh: trimesh.Trimesh, smooth_iters: int = 10, taubin_lamb: float = 0.5, taubin_nu: float = -0.53):
    try:
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        tm_repair.fix_normals(mesh)
        try:
            tm_repair.fill_holes(mesh)
        except Exception:
            pass
        if smooth_iters and smooth_iters > 0:
            filter_taubin(mesh, lamb=taubin_lamb, nu=taubin_nu, iterations=int(smooth_iters))
    except Exception as e:
        print(f"  : {e}")
    return mesh

@torch.no_grad()
def run_inference(
    pipe: Any,
    image_input: Union[str, Image.Image],
    num_parts: int,
    rmbg_net: Any,
    seed: int,
    num_tokens: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    max_num_expanded_coords: int = 1e9,
    use_flash_decoder: bool = False,
    rmbg: bool = False,
    use_retrieval: bool = False,
    num_retrieved_images: int = 0,
    postprocess: bool = False,
    smooth_iters: int = 10,
    taubin_lamb: float = 0.5,
    taubin_nu: float = -0.53,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> trimesh.Scene:
    """"""
    if rmbg and rmbg_net is not None:
        img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
    else:
        img_pil = Image.open(image_input)
    
    print(f"  {num_parts} ...")
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
        use_retrieval=use_retrieval,
        num_retrieved_images=num_retrieved_images,
    ).meshes
    end_time = time.time()
    print(f"  : {end_time - start_time:.2f} ")
    
    for i in range(len(outputs)):
        if outputs[i] is None:
            outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
        elif postprocess:
            outputs[i] = _postprocess_mesh(
                outputs[i], smooth_iters=smooth_iters, taubin_lamb=taubin_lamb, taubin_nu=taubin_nu
            )
    
    return outputs, img_pil

MAX_NUM_PARTS = 16

if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float32  # float32dtype

    parser = argparse.ArgumentParser(description="checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="")
    parser.add_argument("--num_parts", type=int, required=True, help="")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                       help="Checkpoint(transformertransformer_ema)")
    parser.add_argument("--use_ema", action="store_true", 
                       help="EMA(transformer_ema)(transformer)")
    parser.add_argument("--pretrained_model_path", type=str, 
                       default="/root/autodl-tmp/PartRAG/pretrained_weights/PartRAG",
                       help="(VAE, scheduler)")
    parser.add_argument("--output_dir", type=str, default="./inference_results")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_tokens", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--max_num_expanded_coords", type=int, default=1e9)
    parser.add_argument("--use_flash_decoder", action="store_true")
    parser.add_argument("--rmbg", action="store_true", help="")
    parser.add_argument("--render", action="store_true", help="")
    parser.add_argument("--use_retrieval", action="store_true", help="(RAG)")
    parser.add_argument("--database_path", type=str, default=None, help="")
    parser.add_argument("--num_retrieved_images", type=int, default=3, help="(top_k)")
    parser.add_argument("--postprocess", action="store_true", help="(//)")
    parser.add_argument("--smooth_iters", type=int, default=10, help="Taubin")
    parser.add_argument("--taubin_lamb", type=float, default=0.5, help="Taubin lambda")
    parser.add_argument("--taubin_nu", type=float, default=-0.53, help="Taubin nu")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    assert 1 <= args.num_parts <= MAX_NUM_PARTS, f"num_parts [1, {MAX_NUM_PARTS}] "

    # 
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f": {args.image_path}")
    
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint: {args.checkpoint_path}")
    
    args.pretrained_model_path = resolve_or_download_weights(
        preferred_dir=args.pretrained_model_path,
        repo_id="wgsxm/PartCrafter",
        legacy_dirs=(
            "/root/autodl-tmp/PartRAG/pretrained_weights/PartCrafter",
            "/root/autodl-tmp/PartCrafter/pretrained_weights/PartCrafter",
        ),
        required_subdirs=("vae", "transformer", "scheduler"),
        local_files_only=args.local_files_only,
    )

    # transformer
    transformer_subdir = "transformer_ema" if args.use_ema else "transformer"
    transformer_checkpoint_path = os.path.join(args.checkpoint_path, transformer_subdir)
    
    if not os.path.exists(transformer_checkpoint_path):
        raise FileNotFoundError(f"Transformer checkpoint: {transformer_checkpoint_path}")

    print("="*80)
    print(" PartRAG  - Checkpoint")
    print("="*80)
    print(f" : {args.image_path}")
    print(f" : {args.num_parts}")
    print(f" Checkpoint: {args.checkpoint_path}")
    print(f"   |- Transformer: {transformer_subdir}")
    print(f"  : {args.pretrained_model_path}")
    print(f"   |- (VAE, Scheduler, Image Encoder)")
    print(f" : {args.seed}")
    print("="*80)
    print()

    # RMBG()
    rmbg_weights_dir = resolve_or_download_weights(
        preferred_dir=os.path.join(os.path.dirname(args.pretrained_model_path), "RMBG-1.4"),
        repo_id="briaai/RMBG-1.4",
        legacy_dirs=("/root/autodl-tmp/PartCrafter/pretrained_weights/RMBG-1.4",),
        required_files=("config.json",),
        local_files_only=args.local_files_only,
    )
    if os.path.exists(rmbg_weights_dir):
        rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
        rmbg_net.eval()
        print(f" RMBG")
    else:
        rmbg_net = None
        print(f"  RMBG,")

    # pipeline
    print(f"\n ...")
    print(f"     ...")
    pipe: PartragPipeline = PartragPipeline.from_pretrained(
        args.pretrained_model_path
    ).to(device, dtype)
    
    print(f"     Transformercheckpoint...")
    # transformer
    from safetensors.torch import load_file
    transformer_weights_path = os.path.join(transformer_checkpoint_path, "diffusion_pytorch_model.safetensors")
    if os.path.exists(transformer_weights_path):
        state_dict = load_file(transformer_weights_path)
        pipe.transformer.load_state_dict(state_dict)
        # transformerdtype
        pipe.transformer = pipe.transformer.to(device=device, dtype=dtype)
        print(f"    Transformer: {transformer_weights_path}")
        print(f"    Transformer dtype: {pipe.transformer.dtype}")
    else:
        raise FileNotFoundError(f": {transformer_weights_path}")
    
    print(f" !\n")

    # :()
    if args.use_retrieval:
        if args.database_path is None or not os.path.exists(args.database_path):
            raise FileNotFoundError(" --use_retrieval  --database_path")
        print(f"     : {args.database_path}")
        retrieval_module = RetrievalModule(device=device)
        retrieval_module.load_database(args.database_path)
        pipe.retrieval_module = retrieval_module
        print(f"     (top_k={args.num_retrieved_images})")

    set_seed(args.seed)

    # 
    outputs, processed_image = run_inference(
        pipe,
        image_input=args.image_path,
        num_parts=args.num_parts,
        rmbg_net=rmbg_net,
        seed=args.seed,
        num_tokens=args.num_tokens,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_num_expanded_coords=args.max_num_expanded_coords,
        use_flash_decoder=args.use_flash_decoder,
        rmbg=args.rmbg,
        use_retrieval=args.use_retrieval,
        num_retrieved_images=args.num_retrieved_images,
        postprocess=args.postprocess,
        smooth_iters=args.smooth_iters,
        taubin_lamb=args.taubin_lamb,
        taubin_nu=args.taubin_nu,
        dtype=dtype,
        device=device,
    )

    # 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.tag is None:
        args.tag = time.strftime("%Y%m%d_%H%M%S")
    
    export_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(export_dir, exist_ok=True)

    # 
    print(f"\n : {export_dir}")
    total_size = 0
    for i, mesh in enumerate(outputs):
        part_path = os.path.join(export_dir, f"part_{i:02}.glb")
        mesh.export(part_path)
        size_mb = os.path.getsize(part_path) / (1024 * 1024)
        total_size += size_mb
        print(f"   Part {i:02}: {size_mb:.2f} MB ({mesh.vertices.shape[0]} , {mesh.faces.shape[0]} )")
    
    # 
    merged_mesh = get_colored_mesh_composition(outputs)
    merged_path = os.path.join(export_dir, "object.glb")
    merged_mesh.export(merged_path)
    merged_size = os.path.getsize(merged_path) / (1024 * 1024)
    print(f"   : {merged_size:.2f} MB")
    print(f"\n  {len(outputs)} ")
    print(f" : {total_size:.2f} MB () + {merged_size:.2f} MB () = {total_size + merged_size:.2f} MB")

    # 
    if args.render:
        print("\n ...")
        num_views = 36
        radius = 4
        fps = 18
        rendered_images = render_views_around_mesh(
            merged_mesh,
            num_views=num_views,
            radius=radius,
        )
        rendered_normals = render_normal_views_around_mesh(
            merged_mesh,
            num_views=num_views,
            radius=radius,
        )
        rendered_grids = make_grid_for_images_or_videos(
            [
                [processed_image] * num_views,
                rendered_images,
                rendered_normals,
            ], 
            nrow=3
        )
        export_renderings(
            rendered_images,
            os.path.join(export_dir, "rendering.gif"),
            fps=fps,
        )
        export_renderings(
            rendered_normals,
            os.path.join(export_dir, "rendering_normal.gif"),
            fps=fps,
        )
        export_renderings(
            rendered_grids,
            os.path.join(export_dir, "rendering_grid.gif"),
            fps=fps,
        )

        rendered_image, rendered_normal, rendered_grid = rendered_images[0], rendered_normals[0], rendered_grids[0]
        rendered_image.save(os.path.join(export_dir, "rendering.png"))
        rendered_normal.save(os.path.join(export_dir, "rendering_normal.png"))
        rendered_grid.save(os.path.join(export_dir, "rendering_grid.png"))
        print(" ")

    print("\n" + "="*80)
    print(" !")
    print("="*80)
