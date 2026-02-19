"""Build a PartRAG retrieval database with optional k-means subset selection and FAISS index."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import BitImageProcessor, CLIPModel, CLIPProcessor, Dinov2Model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import ObjaverseSimpleDataset
from src.utils.train_utils import get_configs


def _l2_normalize_np(x: np.ndarray) -> np.ndarray:
    eps = 1e-12
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def _to_pil_image(image_field) -> Image.Image:
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")

    if isinstance(image_field, list):
        if len(image_field) == 0:
            raise ValueError("Empty image list")
        image_field = image_field[0]

    if isinstance(image_field, torch.Tensor):
        img = image_field.detach().cpu()
        if img.dim() == 4:
            img = img[0]
        if img.dim() != 3:
            raise ValueError(f"Unsupported tensor image shape: {tuple(img.shape)}")
        if img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)
        img = img.numpy()
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        return Image.fromarray(img).convert("RGB")

    raise TypeError(f"Unsupported image field type: {type(image_field)}")


def _spherical_kmeans_select(
    clip_embeddings: np.ndarray,
    target_size: int,
    num_iters: int,
    seed: int,
) -> np.ndarray:
    """Select diverse exemplars by spherical k-means and cluster-medoid picking."""
    n_samples = clip_embeddings.shape[0]
    if target_size <= 0 or target_size >= n_samples:
        return np.arange(n_samples, dtype=np.int64)

    rng = np.random.default_rng(seed)
    features = _l2_normalize_np(clip_embeddings.astype(np.float32))

    num_clusters = min(target_size, n_samples)
    init_idx = rng.choice(n_samples, size=num_clusters, replace=False)
    centers = features[init_idx].copy()
    labels = np.zeros(n_samples, dtype=np.int64)

    for _ in range(num_iters):
        sims = features @ centers.T
        new_labels = sims.argmax(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        new_centers = np.zeros_like(centers)
        for cid in range(num_clusters):
            members = features[labels == cid]
            if len(members) == 0:
                new_centers[cid] = features[rng.integers(0, n_samples)]
            else:
                new_centers[cid] = members.mean(axis=0)
        centers = _l2_normalize_np(new_centers)

    selected = []
    used = set()
    for cid in range(num_clusters):
        members = np.where(labels == cid)[0]
        if members.size == 0:
            continue
        member_sims = features[members] @ centers[cid]
        best_idx = int(members[int(member_sims.argmax())])
        if best_idx not in used:
            used.add(best_idx)
            selected.append(best_idx)

    if len(selected) < target_size:
        remaining = np.array([i for i in range(n_samples) if i not in used], dtype=np.int64)
        fill_count = target_size - len(selected)
        if remaining.size > 0:
            fill = rng.choice(remaining, size=min(fill_count, remaining.size), replace=False).tolist()
            selected.extend(fill)

    selected = np.array(selected[:target_size], dtype=np.int64)
    selected.sort()
    return selected


def _build_faiss_index(embeddings: np.ndarray, output_dir: str, approximate: bool = True) -> Dict[str, str]:
    """Build and save FAISS index on CLIP embeddings (cosine via normalized inner product)."""
    try:
        import faiss  # type: ignore
    except Exception:
        print("[Warning] faiss is not installed; skipping FAISS index build")
        return {}

    emb = _l2_normalize_np(embeddings.astype(np.float32))
    dim = emb.shape[1]

    if approximate and emb.shape[0] >= 4096:
        nlist = min(4096, max(128, int(np.sqrt(emb.shape[0]) * 2)))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            index.train(emb)
        index.add(emb)
        nprobe = min(32, nlist)
        index.nprobe = nprobe
        index_type = f"IndexIVFFlat(nlist={nlist}, nprobe={nprobe})"
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        index_type = "IndexFlatIP"

    index_path = os.path.join(output_dir, "clip_faiss.index")
    alias_path = os.path.join(output_dir, "faiss.index")
    faiss.write_index(index, index_path)
    if os.path.abspath(index_path) != os.path.abspath(alias_path):
        shutil.copyfile(index_path, alias_path)

    print(f"Saved FAISS index: {index_path} ({index_type})")
    return {
        "faiss_index": "clip_faiss.index",
        "faiss_index_alias": "faiss.index",
        "faiss_type": index_type,
        "faiss_metric": "inner_product_on_l2_normalized_embeddings",
    }


@torch.no_grad()
def build_database(
    config_path: str,
    output_dir: str,
    device: str = "cuda",
    clip_model_name: str = "openai/clip-vit-large-patch14",
    subset_size: int = 1236,
    kmeans_iters: int = 30,
    kmeans_seed: int = 42,
    build_faiss: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Build PartRAG retrieval database (CLIP + DINOv2)")
    print("=" * 80)

    print("\n[1/6] Loading training config and dataset...")
    configs = get_configs(config_path)
    train_dataset = ObjaverseSimpleDataset(configs=configs, training=True)
    print(f"Training samples: {len(train_dataset)}")

    print("\n[2/6] Loading encoders...")
    pretrained_path = configs["model"]["pretrained_model_name_or_path"]

    dino_processor = BitImageProcessor.from_pretrained(
        pretrained_path,
        subfolder="feature_extractor_dinov2",
    )
    dino_encoder = Dinov2Model.from_pretrained(
        pretrained_path,
        subfolder="image_encoder_dinov2",
    ).to(device)
    dino_encoder.eval()

    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval()

    print("\n[3/6] Extracting embeddings...")
    image_seq_features: List[np.ndarray] = []
    dino_global_features: List[np.ndarray] = []
    clip_global_features: List[np.ndarray] = []
    fused_global_features: List[np.ndarray] = []

    image_paths: List[str] = []
    mesh_paths: List[str] = []
    uids: List[str] = []

    for idx in tqdm(range(len(train_dataset)), desc="Extract"):
        try:
            sample = train_dataset[idx]
            image = _to_pil_image(sample["images"])

            dino_inputs = dino_processor(images=image, return_tensors="pt")
            dino_pixel_values = dino_inputs["pixel_values"].to(device)
            dino_out = dino_encoder(dino_pixel_values).last_hidden_state
            dino_seq = F.normalize(dino_out.squeeze(0), dim=-1)
            dino_global = F.normalize(dino_seq.mean(dim=0), dim=-1)

            clip_inputs = clip_processor(images=image, return_tensors="pt")
            clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
            clip_global = clip_model.get_image_features(**clip_inputs).squeeze(0)
            clip_global = F.normalize(clip_global, dim=-1)

            fused_global = torch.cat([dino_global, clip_global], dim=-1)
            fused_global = F.normalize(fused_global, dim=-1)

            image_seq_features.append(dino_seq.cpu().numpy())
            dino_global_features.append(dino_global.cpu().numpy())
            clip_global_features.append(clip_global.cpu().numpy())
            fused_global_features.append(fused_global.cpu().numpy())

            data = train_dataset.data_list[idx]
            image_paths.append(data.get("image_path") or data.get("rendering_rmbg") or data.get("rendering"))
            mesh_paths.append(data.get("mesh_path") or data.get("surface_path"))
            uids.append(data.get("uid") or data.get("file") or f"sample_{idx}")

        except Exception as exc:
            print(f"[Warning] Skip sample {idx}: {exc}")

    if len(fused_global_features) == 0:
        raise RuntimeError("No valid sample was processed when building retrieval database")

    image_seq_array = np.stack(image_seq_features, axis=0)
    dino_global_array = np.stack(dino_global_features, axis=0)
    clip_global_array = np.stack(clip_global_features, axis=0)
    fused_global_array = np.stack(fused_global_features, axis=0)

    total_samples = int(fused_global_array.shape[0])

    print("\n[4/6] Selecting diverse subset...")
    selected_indices = _spherical_kmeans_select(
        clip_embeddings=clip_global_array,
        target_size=subset_size,
        num_iters=kmeans_iters,
        seed=kmeans_seed,
    )

    image_seq_array = image_seq_array[selected_indices]
    dino_global_array = dino_global_array[selected_indices]
    clip_global_array = clip_global_array[selected_indices]
    fused_global_array = fused_global_array[selected_indices]
    image_paths = [image_paths[i] for i in selected_indices.tolist()]
    mesh_paths = [mesh_paths[i] for i in selected_indices.tolist()]
    uids = [uids[i] for i in selected_indices.tolist()]

    print(f"Selected {len(selected_indices)} / {total_samples} samples (target={subset_size})")

    print("\n[5/6] Saving files...")
    np.save(os.path.join(output_dir, "image_embeddings.npy"), image_seq_array)
    np.save(os.path.join(output_dir, "dino_embeddings.npy"), dino_global_array)
    np.save(os.path.join(output_dir, "clip_embeddings.npy"), clip_global_array)
    np.save(os.path.join(output_dir, "fused_embeddings.npy"), fused_global_array)
    np.save(os.path.join(output_dir, "embeddings.npy"), fused_global_array)

    metadata: Dict = {
        "num_samples": int(fused_global_array.shape[0]),
        "num_samples_before_selection": total_samples,
        "selected_indices": selected_indices.tolist(),
        "selection_method": "spherical_kmeans_medoid",
        "selection_target_size": int(subset_size),
        "selection_kmeans_iters": int(kmeans_iters),
        "selection_seed": int(kmeans_seed),
        "dino_seq_dim": int(image_seq_array.shape[-1]),
        "dino_seq_len": int(image_seq_array.shape[1]),
        "dino_global_dim": int(dino_global_array.shape[-1]),
        "clip_global_dim": int(clip_global_array.shape[-1]),
        "fused_global_dim": int(fused_global_array.shape[-1]),
        "dino_encoder": "dinov2",
        "clip_encoder": clip_model_name,
        "fusion": "concat_l2norm",
        "image_paths": image_paths,
        "mesh_paths": mesh_paths,
        "uids": uids,
    }

    if build_faiss:
        faiss_meta = _build_faiss_index(clip_global_array, output_dir=output_dir)
        metadata.update(faiss_meta)

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[6/6] Done")
    print(f"Saved database to: {output_dir}")
    print(f"Samples after selection: {metadata['num_samples']}")
    print(f"Fused feature dim: {metadata['fused_global_dim']}")


def main():
    parser = argparse.ArgumentParser(description="Build PartRAG retrieval database")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "partrag_stage1.yaml"),
        help="Training config path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "retrieval_database_high_quality"),
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model used for retrieval indexing",
    )
    parser.add_argument("--subset_size", type=int, default=1236, help="Target retrieval DB size after k-means")
    parser.add_argument("--kmeans_iters", type=int, default=30)
    parser.add_argument("--kmeans_seed", type=int, default=42)
    parser.add_argument("--build_faiss", dest="build_faiss", action="store_true")
    parser.add_argument("--no_faiss", dest="build_faiss", action="store_false")
    parser.set_defaults(build_faiss=True)
    args = parser.parse_args()

    build_database(
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        clip_model_name=args.clip_model_name,
        subset_size=args.subset_size,
        kmeans_iters=args.kmeans_iters,
        kmeans_seed=args.kmeans_seed,
        build_faiss=args.build_faiss,
    )


if __name__ == "__main__":
    main()
