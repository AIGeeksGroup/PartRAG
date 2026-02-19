"""
checkpoint
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
import trimesh
from transformers import BitImageProcessor, Dinov2Model

from src.models.autoencoders import TripoSGVAEModel
from src.models.transformers import PartragDiTModel
from src.schedulers import RectifiedFlowScheduler
from src.pipelines.pipeline_partrag import PartragPipeline
from src.utils.data_utils import get_colored_mesh_composition
from src.utils.render_utils import render_views_around_mesh, export_renderings
from src.utils.weights_utils import resolve_or_download_weights


def main():
    parser = argparse.ArgumentParser(description="PartRAG checkpoint inference")
    
    # 
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Checkpoint (transformer_ema)")
    parser.add_argument("--pretrained_path", type=str,
                        default="/root/autodl-tmp/PartRAG/pretrained_weights/PartRAG",
                        help="Pretrained base path (VAE + image encoder)")
    
    # 
    parser.add_argument("--input_image", type=str, required=True,
                        help="")
    parser.add_argument("--output_dir", type=str, default="inference_outputs",
                        help="")
    parser.add_argument("--output_name", type=str, default=None,
                        help="()")
    
    # 
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="")
    parser.add_argument("--guidance_scale", type=float, default=7.0,
                        help="CFG")
    parser.add_argument("--num_parts", type=int, default=1,
                        help="")
    parser.add_argument("--seed", type=int, default=42,
                        help="")
    
    # 
    parser.add_argument("--render", action="store_true",
                        help="")
    parser.add_argument("--num_views", type=int, default=36,
                        help="")
    parser.add_argument("--local_files_only", action="store_true")
    
    args = parser.parse_args()

    args.pretrained_path = resolve_or_download_weights(
        preferred_dir=args.pretrained_path,
        repo_id="wgsxm/PartCrafter",
        legacy_dirs=(
            "/root/autodl-tmp/PartRAG/pretrained_weights/PartCrafter",
            "/root/autodl-tmp/PartCrafter/pretrained_weights/PartCrafter",
        ),
        required_subdirs=("vae", "transformer", "scheduler"),
        local_files_only=args.local_files_only,
    )
    
    # 
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Using device: {device}")
    
    # ========================================
    # 1. 
    # ========================================
    print(" Loading models...")
    
    # VAEimage encoder
    vae = TripoSGVAEModel.from_pretrained(
        args.pretrained_path,
        subfolder="vae"
    ).to(device)
    
    feature_extractor_dinov2 = BitImageProcessor.from_pretrained(
        args.pretrained_path,
        subfolder="feature_extractor_dinov2"
    )
    
    image_encoder_dinov2 = Dinov2Model.from_pretrained(
        args.pretrained_path,
        subfolder="image_encoder_dinov2"
    ).to(device)
    
    # Transformercheckpoint(EMA)
    transformer_path = os.path.join(args.checkpoint_path, "transformer_ema")
    if not os.path.exists(transformer_path):
        print(f" EMA transformer not found at {transformer_path}")
        print(f"   Trying non-EMA version...")
        transformer_path = os.path.join(args.checkpoint_path, "transformer")
    
    print(f"   Loading transformer from: {transformer_path}")
    transformer = PartragDiTModel.from_pretrained(transformer_path).to(device)
    
    # Scheduler
    scheduler = RectifiedFlowScheduler.from_pretrained(
        args.pretrained_path,
        subfolder="scheduler"
    )
    
    # 
    vae.eval()
    image_encoder_dinov2.eval()
    transformer.eval()
    
    print(" Models loaded successfully!")
    
    # ========================================
    # 2. Pipeline
    # ========================================
    pipeline = PartragPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        feature_extractor_dinov2=feature_extractor_dinov2,
        image_encoder_dinov2=image_encoder_dinov2,
    )
    pipeline.set_progress_bar_config(disable=False)
    
    # ========================================
    # 3. 
    # ========================================
    print(f" Loading input image: {args.input_image}")
    input_image = Image.open(args.input_image).convert("RGB")
    
    # 
    if args.output_name is None:
        args.output_name = Path(args.input_image).stem
    
    # ========================================
    # 4. 3D mesh
    # ========================================
    print(f" Generating 3D mesh...")
    print(f"   Num inference steps: {args.num_inference_steps}")
    print(f"   Guidance scale: {args.guidance_scale}")
    print(f"   Num parts: {args.num_parts}")
    print(f"   Seed: {args.seed}")
    
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    with torch.autocast("cuda", torch.float16):
        result = pipeline(
            input_image,
            num_inference_steps=args.num_inference_steps,
            num_tokens=1024,
            guidance_scale=args.guidance_scale,
            attention_kwargs={"num_parts": args.num_parts},
            generator=generator,
            max_num_expanded_coords=1e8,
            use_flash_decoder=False,
        )
    
    pred_meshes = result.meshes
    print(f" Generated {len(pred_meshes)} mesh part(s)")
    
    # ========================================
    # 5. 
    # ========================================
    print(f" Saving results to {args.output_dir}")
    
    # 
    input_image.save(os.path.join(args.output_dir, f"{args.output_name}_input.png"))
    
    # 
    for i, mesh in enumerate(pred_meshes):
        if mesh is not None:
            mesh_path = os.path.join(args.output_dir, f"{args.output_name}_part{i:02d}.glb")
            mesh.export(mesh_path)
            print(f"   Saved part {i}: {mesh_path}")
        else:
            print(f"     Part {i} is None (generation error)")
    
    # 
    if len(pred_meshes) > 1:
        combined_mesh = get_colored_mesh_composition([m for m in pred_meshes if m is not None])
    else:
        combined_mesh = pred_meshes[0] if pred_meshes[0] is not None else trimesh.Trimesh()
    
    combined_path = os.path.join(args.output_dir, f"{args.output_name}_combined.glb")
    combined_mesh.export(combined_path)
    print(f"   Saved combined mesh: {combined_path}")
    
    # ========================================
    # 6. ()
    # ========================================
    if args.render:
        print(f" Rendering {args.num_views} views...")
        
        rendered_images = render_views_around_mesh(
            combined_mesh,
            num_views=args.num_views,
            radius=4.0,
        )
        
        # GIF
        gif_path = os.path.join(args.output_dir, f"{args.output_name}_rendered.gif")
        export_renderings(rendered_images, gif_path, fps=18)
        print(f"   Saved rendered GIF: {gif_path}")
        
        # 
        preview_path = os.path.join(args.output_dir, f"{args.output_name}_preview.png")
        rendered_images[0].save(preview_path)
        print(f"   Saved preview: {preview_path}")
    
    print(" !")
    print(f"\n : {args.output_dir}")


if __name__ == "__main__":
    main()
