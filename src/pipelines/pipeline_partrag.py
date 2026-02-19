import inspect
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import PIL.Image
import torch
import trimesh
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    BitImageProcessor,
    Dinov2Model,
)
from ..utils.inference_utils import hierarchical_extract_geometry
from ..utils.part_editing import (
    masked_flow_edit_latents,
    validate_edited_latents,
    smooth_edited_boundaries,
    prepare_part_transforms,
    apply_rigid_transforms_to_meshes,
)

from ..models.autoencoders import TripoSGVAEModel
from ..models.transformers import PartragDiTModel
from ..retrieval import RetrievalModule
from .pipeline_partrag_output import PartragPipelineOutput
from .pipeline_utils import TransformerDiffusionMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class PartragPipeline(DiffusionPipeline, TransformerDiffusionMixin):
    """
    Pipeline for image to 3D part-level object generation.       
    """

    def __init__(
        self,   
        vae: TripoSGVAEModel,
        transformer: PartragDiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_encoder_dinov2: Dinov2Model,
        feature_extractor_dinov2: BitImageProcessor,
        retrieval_module: Optional[RetrievalModule] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder_dinov2=image_encoder_dinov2,
            feature_extractor_dinov2=feature_extractor_dinov2,
        )
        
        # (,)
        self.retrieval_module = retrieval_module

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def decode_progressive(self):
        return self._decode_progressive

    def encode_image(
        self, 
        image, 
        device, 
        num_images_per_prompt,
        use_retrieval: bool = False,
        retrieval_query_text: Optional[str] = None,
        num_retrieved_images: int = 3,
    ):
        """
        ,
        
        Args:
            image: 
            device: 
            num_images_per_prompt: prompt
            use_retrieval: (idea2)
            retrieval_query_text: ()
            num_retrieved_images: 
        """
        dtype = next(self.image_encoder_dinov2.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor_dinov2(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder_dinov2(image).last_hidden_state
        
        # (idea2  - :+mesh)
        if use_retrieval and self.retrieval_module is not None:
            logger.info(f"Using retrieval augmentation with {num_retrieved_images} retrieved objects")
            
            # tensorPIL
            if isinstance(image, torch.Tensor):
                # PIL
                images_for_retrieval = []
                for img_tensor in image:
                    #  :BFloat16numpy,float32
                    # [-1, 1][0, 1]
                    img_np = img_tensor.cpu().float().numpy().transpose(1, 2, 0)
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
                    img_np = img_np.astype(np.uint8)
                    images_for_retrieval.append(PIL.Image.fromarray(img_np))
            
            # mesh
            retrieved_embeds_list = []
            batch_size = image_embeds.shape[0]
            
            for i in range(batch_size):
                # ()
                if retrieval_query_text is not None:
                    retrieved_data = self.retrieval_module.retrieve_by_text(
                        retrieval_query_text, top_k=num_retrieved_images, return_dict=True
                    )
                else:
                    retrieved_data = self.retrieval_module.retrieve_by_image(
                        images_for_retrieval[i], top_k=num_retrieved_images, return_dict=True
                    )
                
                if retrieved_data is not None:
                    retrieved_images = [
                        img for img in (retrieved_data.get("images", []) or [])
                        if isinstance(img, PIL.Image.Image)
                    ]
                    retrieved_mesh_paths = retrieved_data.get("mesh_paths", None)
                    
                    embeds_to_concat = []
                    
                    # 1. mesh()
                    if retrieved_mesh_paths is not None and len(retrieved_mesh_paths) > 0:
                        import trimesh
                        from src.utils.data_utils import mesh_to_surface
                        
                        mesh_embeds_list = []
                        for mesh_path in retrieved_mesh_paths:
                            if mesh_path and os.path.exists(mesh_path):
                                try:
                                    # mesh
                                    mesh = trimesh.load(mesh_path, process=False)
                                    
                                    # 
                                    points, normals = mesh_to_surface(mesh, num_pc=102400)  # 
                                    surface = torch.cat([
                                        torch.from_numpy(points).float(),
                                        torch.from_numpy(normals).float()
                                    ], dim=-1).unsqueeze(0)  # [1, P, 6]
                                    surface = surface.to(device=device, dtype=dtype)
                                    
                                    # VAEmesh
                                    with torch.no_grad():
                                        vae_output = self.vae.encode(surface, num_tokens=256)  # tokens
                                        mesh_tokens = vae_output.latent_dist.projected  # [1, 256, D]
                                        mesh_embeds_list.append(mesh_tokens)
                                    
                                    logger.info(f"Encoded retrieved mesh: {os.path.basename(mesh_path)}")
                                except Exception as e:
                                    logger.warning(f"Failed to encode mesh {mesh_path}: {e}")
                        
                        if len(mesh_embeds_list) > 0:
                            # Concatmesh embeddings
                            retrieved_mesh_embeds = torch.cat(mesh_embeds_list, dim=0)  # [K, 256, D]
                            if num_retrieved_images > 1:
                                retrieved_mesh_embeds = retrieved_mesh_embeds.reshape(1, -1, retrieved_mesh_embeds.shape[-1])
                            embeds_to_concat.append(retrieved_mesh_embeds)
                            logger.info(f"Retrieved mesh embeddings shape: {retrieved_mesh_embeds.shape}")
                    
                    # 2. 
                    if retrieved_images is not None and len(retrieved_images) > 0:
                        retrieved_pixel_values = self.feature_extractor_dinov2(
                            retrieved_images, return_tensors="pt"
                        ).pixel_values
                        retrieved_pixel_values = retrieved_pixel_values.to(device=device, dtype=dtype)
                        retrieved_image_embeds = self.image_encoder_dinov2(
                            retrieved_pixel_values
                        ).last_hidden_state  # [K, seq_len, D]
                        
                        if num_retrieved_images > 1:
                            retrieved_image_embeds = retrieved_image_embeds.reshape(1, -1, retrieved_image_embeds.shape[-1])
                        embeds_to_concat.append(retrieved_image_embeds)
                        logger.info(f"Retrieved image embeddings shape: {retrieved_image_embeds.shape}")
                    
                    # 3. promptembedding
                    prompt_embed = image_embeds[i:i+1]  # [1, seq_len, D]
                    embeds_to_concat.append(prompt_embed)
                    
                    # 4. Concatenateembeddings: [mesh, image, prompt]
                    # ReMoMask,[retrieved_motion, text]
                    if len(embeds_to_concat) > 1:
                        fused_embed = torch.cat(embeds_to_concat, dim=1)
                        # Shape: [1, (mesh_seq + image_seq + prompt_seq), D]
                        logger.info(f"Fused embedding shape: {fused_embed.shape}")
                    else:
                        fused_embed = prompt_embed
                    
                    retrieved_embeds_list.append(fused_embed)
                else:
                    # ,embedding
                    retrieved_embeds_list.append(image_embeds[i:i+1])
            
            # batchembeddings
            image_embeds = torch.cat(retrieved_embeds_list, dim=0)  # [B, augmented_seq_len, D]
            logger.info(f"Final augmented image embeddings shape: {image_embeds.shape}")
        
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (batch_size, num_tokens, num_channels_latents)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return noise

    @torch.no_grad()
    def _decode_latents_to_meshes(
        self,
        latents: torch.Tensor,
        bounds: Union[Tuple[float], List[float], float],
        dense_octree_depth: int,
        hierarchical_octree_depth: int,
        max_num_expanded_coords: int,
        use_flash_decoder: bool,
    ) -> Tuple[List[Optional[Tuple[np.ndarray, np.ndarray]]], List[Optional[trimesh.Trimesh]]]:
        output, meshes = [], []
        if use_flash_decoder:
            self.vae.set_flash_decoder()

        self.set_progress_bar_config(
            desc="Decoding",
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        with self.progress_bar(total=latents.shape[0]) as progress_bar:
            for i in range(latents.shape[0]):
                geometric_func = lambda x, i=i: self.vae.decode(latents[i].unsqueeze(0), sampled_points=x).sample
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        self._execution_device,
                        dtype=latents.dtype,
                        bounds=bounds,
                        dense_octree_depth=dense_octree_depth,
                        hierarchical_octree_depth=hierarchical_octree_depth,
                        max_num_expanded_coords=max_num_expanded_coords,
                    )
                    mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                except Exception:
                    mesh_v_f = None
                    mesh = None
                output.append(mesh_v_f)
                meshes.append(mesh)
                progress_bar.update()
        return output, meshes

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
        dense_octree_depth: int = 8, 
        hierarchical_octree_depth: int = 9,
        max_num_expanded_coords: int = 1e8,
        flash_octree_depth: int = 9,
        use_flash_decoder: bool = True,
        return_dict: bool = True,
        return_latents: bool = False,
        part_transforms: Optional[torch.FloatTensor] = None,
        apply_part_transforms_to_meshes: bool = False,
        # (idea2)
        use_retrieval: bool = False,
        retrieval_query_text: Optional[str] = None,
        num_retrieved_images: int = 3,
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = self._execution_device
        dtype = self.image_encoder_dinov2.dtype

        # 3. Encode condition
        image_embeds, negative_image_embeds = self.encode_image(
            image, 
            device, 
            num_images_per_prompt,
            use_retrieval=use_retrieval,
            retrieval_query_text=retrieval_query_text,
            num_retrieved_images=num_retrieved_images,
        )

        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_tokens,
            num_channels_latents,
            image_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        self.set_progress_bar_config(
            desc="Denoising", 
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0].to(dtype)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    image_embeds_1 = callback_outputs.pop(
                        "image_embeds_1", image_embeds_1
                    )
                    negative_image_embeds_1 = callback_outputs.pop(
                        "negative_image_embeds_1", negative_image_embeds_1
                    )
                    image_embeds_2 = callback_outputs.pop(
                        "image_embeds_2", image_embeds_2
                    )
                    negative_image_embeds_2 = callback_outputs.pop(
                        "negative_image_embeds_2", negative_image_embeds_2
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()


        # 7. decoder mesh
        output, meshes = self._decode_latents_to_meshes(
            latents=latents,
            bounds=bounds,
            dense_octree_depth=dense_octree_depth,
            hierarchical_octree_depth=hierarchical_octree_depth,
            max_num_expanded_coords=max_num_expanded_coords,
            use_flash_decoder=use_flash_decoder,
        )

        prepared_part_transforms = prepare_part_transforms(
            part_transforms=part_transforms,
            num_parts=latents.shape[0],
            device=latents.device,
            dtype=latents.dtype,
        )
        if apply_part_transforms_to_meshes:
            meshes = apply_rigid_transforms_to_meshes(
                meshes=meshes,
                part_transforms=prepared_part_transforms,
            )
       
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (output, meshes)

        return PartragPipelineOutput(
            samples=output,
            meshes=meshes,
            latents=latents.detach().cpu() if return_latents else None,
            part_transforms=prepared_part_transforms.detach().cpu(),
        )

    @torch.no_grad()
    def edit_parts(
        self,
        *,
        image: PipelineImageInput,
        part_latents: torch.FloatTensor,
        part_transforms: Optional[torch.FloatTensor] = None,
        target_part_indices: List[int],
        num_refinement_steps: int = 20,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        edit_condition_text: Optional[str] = None,
        num_retrieved_images: int = 3,
        semantic_similarity_threshold: float = 0.1,
        apply_boundary_smoothing: bool = True,
        apply_part_transforms_to_meshes: bool = True,
        smoothing_iterations: int = 2,
        bounds: Union[Tuple[float], List[float], float] = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 9,
        max_num_expanded_coords: int = 1e8,
        use_flash_decoder: bool = True,
        return_dict: bool = True,
    ):
        """
        Part-level editing with masked flow matching:
        - update only target parts
        - keep non-target parts frozen
        - optional retrieval-conditioned editing context
        """
        if part_latents.dim() != 3:
            raise ValueError(
                f"part_latents must be [num_parts, num_tokens, dim], got {tuple(part_latents.shape)}"
            )

        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype

        latents = part_latents.to(device=device, dtype=dtype)
        num_parts = latents.shape[0]
        prepared_part_transforms = prepare_part_transforms(
            part_transforms=part_transforms,
            num_parts=num_parts,
            device=device,
            dtype=latents.dtype,
        )

        use_retrieval = self.retrieval_module is not None and edit_condition_text is not None
        image_embeds, _ = self.encode_image(
            image=image,
            device=device,
            num_images_per_prompt=1,
            use_retrieval=use_retrieval,
            retrieval_query_text=edit_condition_text,
            num_retrieved_images=num_retrieved_images,
        )

        # Expand single condition token sequence to all part latents.
        if image_embeds.shape[0] == 1 and num_parts > 1:
            image_embeds = image_embeds.repeat(num_parts, 1, 1)
        elif image_embeds.shape[0] != num_parts:
            raise ValueError(
                f"image_embeds batch ({image_embeds.shape[0]}) must match num_parts ({num_parts}) for editing"
            )

        edit_attention_kwargs = dict(attention_kwargs or {})
        edit_attention_kwargs.setdefault("num_parts", num_parts)

        edited_latents, edited_idx = masked_flow_edit_latents(
            transformer=self.transformer,
            scheduler=self.scheduler,
            latents=latents,
            encoder_hidden_states=image_embeds,
            target_part_indices=target_part_indices,
            num_refinement_steps=num_refinement_steps,
            attention_kwargs=edit_attention_kwargs,
        )
        edited_latents = validate_edited_latents(
            edited_latents=edited_latents,
            original_latents=latents,
            edited_indices=edited_idx,
            similarity_threshold=semantic_similarity_threshold,
        )

        output, meshes = self._decode_latents_to_meshes(
            latents=edited_latents,
            bounds=bounds,
            dense_octree_depth=dense_octree_depth,
            hierarchical_octree_depth=hierarchical_octree_depth,
            max_num_expanded_coords=max_num_expanded_coords,
            use_flash_decoder=use_flash_decoder,
        )

        if apply_part_transforms_to_meshes:
            meshes = apply_rigid_transforms_to_meshes(
                meshes=meshes,
                part_transforms=prepared_part_transforms,
            )

        if apply_boundary_smoothing:
            meshes = smooth_edited_boundaries(
                meshes=meshes,
                edited_indices=edited_idx.tolist(),
                iterations=smoothing_iterations,
            )

        self.maybe_free_model_hooks()

        if not return_dict:
            return output, meshes

        return PartragPipelineOutput(
            samples=output,
            meshes=meshes,
            latents=edited_latents.detach().cpu(),
            edited_indices=edited_idx.tolist(),
            part_transforms=prepared_part_transforms.detach().cpu(),
        )
