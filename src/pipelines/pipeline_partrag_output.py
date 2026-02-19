from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import trimesh
from diffusers.utils import BaseOutput


@dataclass
class PartragPipelineOutput(BaseOutput):
    r"""
    Output class for ShapeDiff pipelines.
    """

    samples: torch.Tensor
    meshes: List[trimesh.Trimesh]
    latents: Optional[torch.Tensor] = None
    part_transforms: Optional[torch.Tensor] = None
    edited_indices: Optional[List[int]] = None
