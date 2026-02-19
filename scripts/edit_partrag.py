import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch
import trimesh
from PIL import Image
from transformers import BitImageProcessor, Dinov2Model

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.autoencoders import TripoSGVAEModel
from src.models.transformers import PartragDiTModel
from src.pipelines.pipeline_partrag import PartragPipeline
from src.retrieval.retrieval_module import RetrievalModule
from src.schedulers import RectifiedFlowScheduler
from src.utils.data_utils import get_colored_mesh_composition
from src.utils.render_utils import export_renderings, render_views_around_mesh
from src.utils.weights_utils import resolve_or_download_weights


def parse_indices(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_pipeline(pretrained_path: str, checkpoint_path: str, device: str, retrieval_db: str = None):
    vae = TripoSGVAEModel.from_pretrained(pretrained_path, subfolder="vae").to(device)
    feature_extractor = BitImageProcessor.from_pretrained(pretrained_path, subfolder="feature_extractor_dinov2")
    image_encoder = Dinov2Model.from_pretrained(pretrained_path, subfolder="image_encoder_dinov2").to(device)
    scheduler = RectifiedFlowScheduler.from_pretrained(pretrained_path, subfolder="scheduler")

    transformer_path = os.path.join(checkpoint_path, "transformer_ema")
    if not os.path.exists(transformer_path):
        transformer_path = os.path.join(checkpoint_path, "transformer")
    transformer = PartragDiTModel.from_pretrained(transformer_path).to(device)

    retrieval_module = None
    if retrieval_db is not None:
        retrieval_module = RetrievalModule(device=device)
        retrieval_module.load_database(retrieval_db)

    pipe = PartragPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        image_encoder_dinov2=image_encoder,
        feature_extractor_dinov2=feature_extractor,
        retrieval_module=retrieval_module,
    )
    pipe.set_progress_bar_config(disable=False)
    return pipe


def main():
    parser = argparse.ArgumentParser(description="PartRAG masked part-level editing")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--pretrained_path", type=str, default="/root/autodl-tmp/PartRAG/pretrained_weights/PartRAG")
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="editing_outputs")

    parser.add_argument("--num_parts", type=int, default=4)
    parser.add_argument("--target_parts", type=str, required=True, help="Comma-separated indices, e.g. 1,3")
    parser.add_argument("--edit_text", type=str, default=None)
    parser.add_argument("--num_refinement_steps", type=int, default=20)
    parser.add_argument("--semantic_similarity_threshold", type=float, default=0.1)
    parser.add_argument("--num_retrieved_images", type=int, default=3)
    parser.add_argument("--retrieval_db", type=str, default=None)

    parser.add_argument("--latents_path", type=str, default=None, help="Optional .pt file with initial part latents")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--render", action="store_true")
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

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target_parts = parse_indices(args.target_parts)
    if len(target_parts) == 0:
        raise ValueError("target_parts cannot be empty")

    pipe = load_pipeline(
        pretrained_path=args.pretrained_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
        retrieval_db=args.retrieval_db,
    )
    image = Image.open(args.input_image).convert("RGB")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Prepare initial part latents
    if args.latents_path and os.path.exists(args.latents_path):
        loaded = torch.load(args.latents_path, map_location=device)
        part_transforms = None
        if isinstance(loaded, dict):
            part_latents = loaded.get("latents", None)
            part_transforms = loaded.get("part_transforms", None)
            if part_transforms is not None:
                part_transforms = part_transforms.to(device)
        else:
            part_latents = loaded
        if part_latents is None:
            raise ValueError(f"Cannot read latents from {args.latents_path}")
    else:
        init = pipe(
            image=image,
            num_inference_steps=args.num_inference_steps,
            num_tokens=1024,
            guidance_scale=args.guidance_scale,
            attention_kwargs={"num_parts": args.num_parts},
            generator=generator,
            return_latents=True,
        )
        part_latents = init.latents.to(device)
        part_transforms = init.part_transforms.to(device) if init.part_transforms is not None else None
        torch.save(
            {
                "latents": part_latents.cpu(),
                "part_transforms": part_transforms.cpu() if part_transforms is not None else None,
            },
            os.path.join(args.output_dir, "initial_latents.pt"),
        )

    # Edit selected parts
    edited = pipe.edit_parts(
        image=image,
        part_latents=part_latents,
        part_transforms=part_transforms,
        target_part_indices=target_parts,
        num_refinement_steps=args.num_refinement_steps,
        attention_kwargs={"num_parts": part_latents.shape[0]},
        edit_condition_text=args.edit_text,
        num_retrieved_images=args.num_retrieved_images,
        semantic_similarity_threshold=args.semantic_similarity_threshold,
        apply_boundary_smoothing=True,
        smoothing_iterations=2,
    )

    torch.save(
        {
            "latents": edited.latents,
            "part_transforms": edited.part_transforms,
        },
        os.path.join(args.output_dir, "edited_latents.pt"),
    )
    image.save(os.path.join(args.output_dir, "input.png"))

    for i, mesh in enumerate(edited.meshes):
        if mesh is None:
            continue
        mesh.export(os.path.join(args.output_dir, f"edited_part_{i:02d}.glb"))

    valid_meshes = [m for m in edited.meshes if m is not None]
    if len(valid_meshes) > 0:
        combined = get_colored_mesh_composition(valid_meshes)
    else:
        combined = trimesh.Trimesh()
    combined.export(os.path.join(args.output_dir, "edited_combined.glb"))

    if args.render and len(valid_meshes) > 0:
        rendered = render_views_around_mesh(combined, num_views=36, radius=4.0)
        export_renderings(rendered, os.path.join(args.output_dir, "edited.gif"), fps=18)

    print(f"Editing completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
