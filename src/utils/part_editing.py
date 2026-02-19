from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh


def prepare_part_transforms(
    part_transforms: Optional[torch.Tensor],
    num_parts: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Ensure part transforms are available in shape [num_parts, 4, 4].
    Falls back to identity transforms when no transform tensor is provided.
    """
    target_device = device if device is not None else (part_transforms.device if part_transforms is not None else None)
    target_dtype = dtype if dtype is not None else (part_transforms.dtype if part_transforms is not None else torch.float32)

    if part_transforms is None:
        eye = torch.eye(4, dtype=target_dtype, device=target_device)
        return eye.unsqueeze(0).repeat(num_parts, 1, 1)

    transforms = part_transforms
    if transforms.dim() == 2 and transforms.shape == (4, 4):
        transforms = transforms.unsqueeze(0).repeat(num_parts, 1, 1)

    if transforms.dim() != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(
            f"part_transforms must have shape [num_parts, 4, 4], got {tuple(transforms.shape)}"
        )
    if transforms.shape[0] != num_parts:
        raise ValueError(
            f"part_transforms count ({transforms.shape[0]}) must match num_parts ({num_parts})"
        )

    return transforms.to(device=target_device, dtype=target_dtype)


def _to_index_tensor(indices: Sequence[int], length: int, device: torch.device) -> torch.Tensor:
    if len(indices) == 0:
        raise ValueError("target_part_indices cannot be empty")
    idx = torch.tensor(sorted(set(int(i) for i in indices)), device=device, dtype=torch.long)
    if idx.min().item() < 0 or idx.max().item() >= length:
        raise ValueError(
            f"target_part_indices out of range [0, {length - 1}], got {idx.tolist()}"
        )
    return idx


@torch.no_grad()
def masked_flow_edit_latents(
    *,
    transformer: torch.nn.Module,
    scheduler,
    latents: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    target_part_indices: Sequence[int],
    num_refinement_steps: int = 20,
    attention_kwargs: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Masked flow matching update:
    - only update target part latents
    - keep non-target part latents frozen
    """
    if latents.dim() != 3:
        raise ValueError(f"Expected latents shape [num_parts, num_tokens, dim], got {tuple(latents.shape)}")

    device = latents.device
    edited = latents.clone()
    frozen_reference = latents.clone()

    target_idx = _to_index_tensor(target_part_indices, edited.shape[0], device)
    target_mask = torch.zeros(edited.shape[0], dtype=torch.bool, device=device)
    target_mask[target_idx] = True

    scheduler.set_timesteps(num_refinement_steps, device=device)
    step_timesteps = scheduler.timesteps

    attn_kwargs = dict(attention_kwargs or {})
    attn_kwargs.setdefault("num_parts", edited.shape[0])

    for t in step_timesteps:
        timestep = t.expand(edited.shape[0])
        noise_pred = transformer(
            edited,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            attention_kwargs=attn_kwargs,
            return_dict=False,
        )[0]
        updated = scheduler.step(noise_pred, t, edited, return_dict=False)[0]
        edited[target_mask] = updated[target_mask]
        edited[~target_mask] = frozen_reference[~target_mask]

    return edited, target_idx


def validate_edited_latents(
    edited_latents: torch.Tensor,
    original_latents: torch.Tensor,
    edited_indices: torch.Tensor,
    similarity_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Lightweight semantic safety check:
    reject updated target parts whose latent direction drifts too far from original latent manifold.
    """
    if edited_indices.numel() == 0:
        return edited_latents

    output = edited_latents.clone()
    for idx in edited_indices.tolist():
        old_feat = original_latents[idx].reshape(-1).float()
        new_feat = edited_latents[idx].reshape(-1).float()
        sim = F.cosine_similarity(old_feat.unsqueeze(0), new_feat.unsqueeze(0), dim=-1).item()
        if sim < similarity_threshold:
            output[idx] = original_latents[idx]
    return output


def _boundary_vertex_indices(mesh: trimesh.Trimesh) -> np.ndarray:
    if mesh.edges_boundary is None or len(mesh.edges_boundary) == 0:
        return np.array([], dtype=np.int64)
    return np.unique(mesh.edges_boundary.reshape(-1)).astype(np.int64)


def smooth_edited_boundaries(
    meshes: List[Optional[trimesh.Trimesh]],
    edited_indices: Iterable[int],
    iterations: int = 2,
    laplacian_lambda: float = 0.3,
    projection_alpha: float = 0.4,
) -> List[Optional[trimesh.Trimesh]]:
    """
    Boundary post-process for edited parts:
    1) laplacian smooth on edited part meshes
    2) project edited boundary vertices toward nearest points on frozen parts
    """
    out = list(meshes)
    edited_set = set(int(i) for i in edited_indices)

    frozen_meshes = [m for i, m in enumerate(meshes) if i not in edited_set and m is not None]
    frozen_union = None
    if len(frozen_meshes) > 0:
        try:
            frozen_union = trimesh.util.concatenate(frozen_meshes)
        except Exception:
            frozen_union = None

    for idx in edited_set:
        if idx < 0 or idx >= len(out):
            continue
        mesh = out[idx]
        if mesh is None:
            continue
        mesh = mesh.copy()

        # Laplacian smoothing only on edited mesh
        try:
            trimesh.smoothing.filter_laplacian(mesh, lamb=laplacian_lambda, iterations=iterations)
        except Exception:
            pass

        # Seam projection toward frozen geometry to reduce cracks
        if frozen_union is not None:
            bidx = _boundary_vertex_indices(mesh)
            if bidx.size > 0:
                try:
                    q = mesh.vertices[bidx]
                    nearest, _, _ = trimesh.proximity.closest_point(frozen_union, q)
                    mesh.vertices[bidx] = (1.0 - projection_alpha) * q + projection_alpha * nearest
                except Exception:
                    pass

        out[idx] = mesh

    return out


def apply_rigid_transforms_to_meshes(
    meshes: List[Optional[trimesh.Trimesh]],
    part_transforms: torch.Tensor,
) -> List[Optional[trimesh.Trimesh]]:
    """
    Apply per-part rigid transforms [num_parts, 4, 4] to decoded meshes.
    """
    if part_transforms.dim() != 3 or part_transforms.shape[1:] != (4, 4):
        raise ValueError(
            f"part_transforms must have shape [num_parts, 4, 4], got {tuple(part_transforms.shape)}"
        )

    out: List[Optional[trimesh.Trimesh]] = []
    transforms_np = part_transforms.detach().cpu().numpy()

    for idx, mesh in enumerate(meshes):
        if mesh is None or idx >= transforms_np.shape[0]:
            out.append(mesh)
            continue
        m = mesh.copy()
        try:
            m.apply_transform(transforms_np[idx])
        except Exception:
            pass
        out.append(m)
    return out
