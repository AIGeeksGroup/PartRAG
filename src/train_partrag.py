import warnings
import hashlib
warnings.filterwarnings("ignore")  # ignore all warnings
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()  # ignore diffusers warnings

from src.utils.typing_utils import *

import os
import argparse
import logging
import time
import math
import gc
import copy
from packaging import version

import trimesh
from PIL import Image
import numpy as np
import wandb
from tqdm import tqdm

import torch
import torch.nn.functional as tF
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from accelerate import DataLoaderConfiguration, DeepSpeedPlugin
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3
)

from transformers import (
    BitImageProcessor,
    Dinov2Model,
)
from src.schedulers import RectifiedFlowScheduler
from src.models.autoencoders import TripoSGVAEModel
from src.models.transformers import PartragDiTModel
from src.pipelines.pipeline_partrag import PartragPipeline

from src.datasets import (
    ObjaversePartDataset, 
    BatchedObjaversePartDataset,
    ObjaverseSimpleDataset,
    BatchedObjaverseSimpleDataset,
    MultiEpochsDataLoader, 
    yield_forever
)
from src.utils.data_utils import get_colored_mesh_composition
from src.utils.train_utils import (
    MyEMAModel, 
    get_configs,
    get_optimizer,
    get_lr_scheduler,
    save_experiment_params,
    save_model_architecture,
    ContrastiveLossHelper,
    FeatureQueue,
    momentum_update,
)
from src.utils.retrieval_helper import (
    RetrievalHelper,
    concat_retrieved_to_encoder_hidden_states,
)
from src.models.projection_layers import DualProjectionModule
from src.utils.render_utils import (
    render_views_around_mesh, 
    render_normal_views_around_mesh, 
    make_grid_for_images_or_videos,
    export_renderings
)
from src.utils.metric_utils import compute_cd_and_f_score_in_training
from src.utils.weights_utils import resolve_or_download_weights

def main():
    PROJECT_NAME = "PartRAG"

    parser = argparse.ArgumentParser(
        description="Train a diffusion model for 3D object generation",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--resume_from_iter",
        type=int,
        default=None,
        help="The iteration to load the checkpoint from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--offline_wandb",
        action="store_true",
        help="Use offline WandB for experiment tracking"
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="The max iteration step for training"
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        default=2,
        help="The max iteration step for validation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for the data loader"
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA model for training"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale lr with total batch size (base batch size: 256)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.,
        help="Max gradient norm for gradient clipping"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Type of mixed precision training"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs"
    )

    parser.add_argument(
        "--val_guidance_scales",
        type=list,
        nargs="+",
        default=[7.0],
        help="CFG scale used for validation"
    )

    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for training"
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=1,
        choices=[1, 2, 3],  # https://huggingface.co/docs/accelerate/usage_guides/deepspeed
        help="ZeRO stage type for DeepSpeed"
    )

    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Train from scratch"
    )
    parser.add_argument(
        "--load_pretrained_model",
        type=str,
        default=None,
        help="Tag of a pretrained PartragDiTModel in this project"
    )
    parser.add_argument(
        "--load_pretrained_model_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained PartragDiTModel checkpoint"
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Use only local checkpoints and disable auto-download of upstream weights"
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()
    # Parse the config file
    configs = get_configs(args.config, extras)  # change yaml configs by `extras`

    args.val_guidance_scales = [float(x[0]) if isinstance(x, list) else float(x) for x in args.val_guidance_scales]
    if args.max_val_steps > 0: 
        # If enable validation, the max_val_steps must be a multiple of nrow
        # Always keep validation batchsize 1
        divider = configs["val"]["nrow"]
        args.max_val_steps = max(args.max_val_steps, divider)
        if args.max_val_steps % divider != 0:
            args.max_val_steps = (args.max_val_steps // divider + 1) * divider

    # Create an experiment directory using the `tag`
    if args.tag is None:
        args.tag = time.strftime("%Y%m%d_%H_%M_%S")
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    eval_dir = os.path.join(exp_dir, "evaluations")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = get_accelerate_logger(__name__, log_level="INFO")
    file_handler = logging.FileHandler(os.path.join(exp_dir, "log.txt"))  # output to file
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    logger.logger.addHandler(file_handler)
    logger.logger.propagate = True  # propagate to the root logger (console)

    # Set DeepSpeed config
    if args.use_deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.max_grad_norm,
            zero_stage=int(args.zero_stage),
            offload_optimizer_device="cpu",  # hard-coded here, TODO: make it configurable
        )
    else:
        deepspeed_plugin = None

    # Initialize the accelerator
    accelerator = Accelerator(
        project_dir=exp_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        split_batches=False,  # batch size per GPU
        dataloader_config=DataLoaderConfiguration(non_blocking=args.pin_memory),
        deepspeed_plugin=deepspeed_plugin,
    )
    logger.info(f"Accelerator state:\n{accelerator.state}\n")

    # Set the random seed
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Resolve pretrained base weights:
    # 1) prefer configured PartRAG path, 2) fallback to local legacy PartCrafter paths,
    # 3) if still missing, download from the upstream PartCrafter repo for fine-tuning.
    configured_pretrained_path = configs["model"]["pretrained_model_name_or_path"]
    resolved_pretrained_path = resolve_or_download_weights(
        preferred_dir=configured_pretrained_path,
        repo_id="wgsxm/PartCrafter",
        legacy_dirs=(
            "/root/autodl-tmp/PartRAG/pretrained_weights/PartCrafter",
            "/root/autodl-tmp/PartCrafter/pretrained_weights/PartCrafter",
        ),
        required_subdirs=("vae", "transformer", "scheduler"),
        local_files_only=args.local_files_only,
    )
    if resolved_pretrained_path != configured_pretrained_path:
        logger.info(
            "Resolved pretrained backbone from [%s] to [%s]",
            configured_pretrained_path,
            resolved_pretrained_path,
        )
        configs["model"]["pretrained_model_name_or_path"] = resolved_pretrained_path

    if configs["model"].get("vae_model_name_or_path"):
        configured_vae_path = configs["model"]["vae_model_name_or_path"]
        resolved_vae_path = resolve_or_download_weights(
            preferred_dir=configured_vae_path,
            repo_id="wgsxm/PartCrafter",
            legacy_dirs=(resolved_pretrained_path,),
            required_subdirs=("vae", "scheduler"),
            local_files_only=args.local_files_only,
        )
        if resolved_vae_path != configured_vae_path:
            logger.info(
                "Resolved VAE/source weights from [%s] to [%s]",
                configured_vae_path,
                resolved_vae_path,
            )
            configs["model"]["vae_model_name_or_path"] = resolved_vae_path

    #  :
    # part-level,PartDataset
    use_part_dataset = configs["train"].get("use_part_dataset", True)
    min_num_parts = configs["dataset"].get("min_num_parts", 1)
    
    if use_part_dataset and min_num_parts > 1:
        # Part Dataset()
        logger.info(" Using PartDataset for part-level contrastive learning")
        train_dataset = BatchedObjaversePartDataset(
            configs=configs,
            batch_size=configs["train"]["batch_size_per_gpu"],
            training=True,
        )
        val_dataset = ObjaversePartDataset(
            configs=configs,
            training=False,
        )
    else:
        # Simple Dataset()
        logger.info("Using SimpleDataset for object-level generation")
        train_dataset = BatchedObjaverseSimpleDataset(
            configs=configs,
            training=True,
        )
        val_dataset = ObjaverseSimpleDataset(
            configs=configs,
            training=False,
        )
    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=configs["train"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        collate_fn=val_dataset.collate_fn if hasattr(val_dataset, 'collate_fn') else None,  #  :collate_fn
    )
    random_val_loader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size_per_gpu"],
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        collate_fn=val_dataset.collate_fn if hasattr(val_dataset, 'collate_fn') else None,  #  :collate_fn
    )

    logger.info(f"Loaded [{len(train_dataset)}] training samples and [{len(val_dataset)}] validation samples\n")

    # Compute the effective batch size and scale learning rate
    total_batch_size = configs["train"]["batch_size_per_gpu"] * \
        accelerator.num_processes * args.gradient_accumulation_steps
    configs["train"]["total_batch_size"] = total_batch_size
    if args.scale_lr:
        configs["optimizer"]["lr"] *= (total_batch_size / 256)
        configs["lr_scheduler"]["max_lr"] = configs["optimizer"]["lr"]
    
    # Initialize the model
    logger.info("Initializing the model...")
    #  VAEencodersvae_model_name_or_path(),pretrained_model_name_or_path
    vae_path = configs["model"].get("vae_model_name_or_path", configs["model"]["pretrained_model_name_or_path"])
    vae = TripoSGVAEModel.from_pretrained(
        vae_path,
        subfolder="vae"
    )
    feature_extractor_dinov2 = BitImageProcessor.from_pretrained(
        vae_path,
        subfolder="feature_extractor_dinov2"
    )
    image_encoder_dinov2 = Dinov2Model.from_pretrained(
        vae_path,
        subfolder="image_encoder_dinov2"
    )

    enable_part_embedding = configs["model"]["transformer"].get("enable_part_embedding", True)
    enable_local_cross_attn = configs["model"]["transformer"].get("enable_local_cross_attn", True)
    enable_global_cross_attn = configs["model"]["transformer"].get("enable_global_cross_attn", True)
    global_attn_block_ids = configs["model"]["transformer"].get("global_attn_block_ids", None)
    if global_attn_block_ids is not None:
        global_attn_block_ids = list(global_attn_block_ids)
    global_attn_block_id_range = configs["model"]["transformer"].get("global_attn_block_id_range", None)
    if global_attn_block_id_range is not None:
        global_attn_block_id_range = list(global_attn_block_id_range)
    if args.from_scratch:
        logger.info(f"Initialize PartragDiTModel from scratch\n")
        transformer = PartragDiTModel.from_config(
            os.path.join(
                configs["model"]["pretrained_model_name_or_path"],
                "transformer"
            ), 
            enable_part_embedding=enable_part_embedding,
            enable_local_cross_attn=enable_local_cross_attn,
            enable_global_cross_attn=enable_global_cross_attn,
            global_attn_block_ids=global_attn_block_ids,
            global_attn_block_id_range=global_attn_block_id_range,
        )
    elif args.load_pretrained_model is None:
        logger.info(f"Load pretrained TripoSGDiTModel to initialize PartragDiTModel from [{configs['model']['pretrained_model_name_or_path']}]\n")
        transformer, loading_info = PartragDiTModel.from_pretrained(
            configs["model"]["pretrained_model_name_or_path"],
            subfolder="transformer",
            low_cpu_mem_usage=False, 
            output_loading_info=True, 
            enable_part_embedding=enable_part_embedding,
            enable_local_cross_attn=enable_local_cross_attn,
            enable_global_cross_attn=enable_global_cross_attn,
            global_attn_block_ids=global_attn_block_ids,
            global_attn_block_id_range=global_attn_block_id_range,
        )
    else:
        logger.info(f"Load PartragDiTModel EMA checkpoint from [{args.load_pretrained_model}] iteration [{args.load_pretrained_model_ckpt:06d}]\n")
        path = os.path.join(
            args.output_dir,
            args.load_pretrained_model, 
            "checkpoints", 
            f"{args.load_pretrained_model_ckpt:06d}"
        )
        transformer, loading_info = PartragDiTModel.from_pretrained(
            path, 
            subfolder="transformer_ema",
            low_cpu_mem_usage=False, 
            output_loading_info=True, 
            enable_part_embedding=enable_part_embedding,
            enable_local_cross_attn=enable_local_cross_attn,
            enable_global_cross_attn=enable_global_cross_attn,
            global_attn_block_ids=global_attn_block_ids,
            global_attn_block_id_range=global_attn_block_id_range,
        )
    if not args.from_scratch:
        for v in loading_info.values():
            if v and len(v) > 0:
                logger.info(f"Loading info of PartragDiTModel: {loading_info}\n")
                break


    # ============ LoRA Configuration ============
    # LoRA
    if hasattr(args, 'use_lora') and args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        logger.info(" Applying LoRA to transformer...")
        
        # LoRA
        lora_config = LoraConfig(
            r=getattr(args, 'lora_rank', 16),  # LoRA rank
            lora_alpha=getattr(args, 'lora_alpha', 32),  # LoRA alpha
            target_modules=[
                "to_q", "to_k", "to_v", "to_out.0",  # attention layers
                "proj_in", "proj_out",  # projection layers  
                "ff.net.0.proj", "ff.net.2",  # feedforward layers
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=None,  # 
        )
        
        # LoRA
        transformer = get_peft_model(transformer, lora_config)
        transformer.print_trainable_parameters()
        logger.info(" LoRA applied successfully!\n")
    # ============ End LoRA Configuration ============

    #  Schedulervae_model_name_or_path(),pretrained_model_name_or_path
    scheduler_path = configs["model"].get("vae_model_name_or_path", configs["model"]["pretrained_model_name_or_path"])
    noise_scheduler = RectifiedFlowScheduler.from_pretrained(
        scheduler_path,
        subfolder="scheduler"
    )

    if args.use_ema:
        ema_transformer = MyEMAModel(
            transformer.parameters(),
            model_cls=PartragDiTModel,
            model_config=transformer.config,
            **configs["train"]["ema_kwargs"]
        )

    # Freeze VAE and image encoder
    vae.requires_grad_(False)
    image_encoder_dinov2.requires_grad_(False)
    vae.eval()
    image_encoder_dinov2.eval()

    # Initialize projection layers for contrastive learning
    # projectionencoder
    projection_dim = configs["train"].get("projection_dim", 1024)
    projection_hidden_dim = configs["train"].get("projection_hidden_dim", 2048)
    projection_layers = configs["train"].get("projection_layers", 2)
    
    contrastive_projection = DualProjectionModule(
        image_dim=projection_dim,
        mesh_dim=projection_dim,
        hidden_dim=projection_hidden_dim,
        output_dim=projection_dim,
        num_layers=projection_layers,
        use_part_label_embeddings=configs["train"].get("use_part_label_embeddings", True),
        num_part_labels=configs["train"].get("num_part_labels", 512),
    )
    contrastive_projection.train()  # 
    logger.info(f"Initialized contrastive projection layers: {sum(p.numel() for p in contrastive_projection.parameters()) / 1e6:.2f}M parameters\n")

    # Optional momentum encoder for bidirectional momentum queues (paper HCR setting)
    contrastive_enabled_cfg = configs["train"].get("enable_contrastive", True)
    use_momentum_queue = bool(configs["train"].get("use_momentum_queue", False) and contrastive_enabled_cfg)
    momentum_coefficient = float(configs["train"].get("momentum_coefficient", 0.999))
    momentum_queue_size = int(configs["train"].get("momentum_queue_size", 65536))
    momentum_projection = None
    if use_momentum_queue:
        momentum_projection = copy.deepcopy(contrastive_projection)
        momentum_projection.requires_grad_(False)
        momentum_projection.eval()
        logger.info(
            f"Enabled bidirectional momentum queue: queue_size={momentum_queue_size}, momentum={momentum_coefficient}"
        )

    #  
    freeze_pretrained_backbone = configs["train"].get("freeze_pretrained_backbone", False)
    freeze_modules_patterns = configs["train"].get("freeze_modules", [])
    trainable_modules_patterns = configs["train"].get("trainable_modules", [])
    
    if freeze_pretrained_backbone or freeze_modules_patterns:
        logger.info(" Applying parameter freezing strategy...")
        
        # Step 1: 
        transformer.requires_grad_(False)
        frozen_params = []
        trainable_params = []
        
        # Step 2: trainable_modules
        import fnmatch
        for name, param in transformer.named_parameters():
            should_train = False
            
            # trainable patterns
            for pattern in trainable_modules_patterns:
                if fnmatch.fnmatch(name, pattern) or pattern in name:
                    should_train = True
                    break
            
            # freeze patterns()
            for pattern in freeze_modules_patterns:
                if fnmatch.fnmatch(name, pattern) or pattern in name:
                    should_train = False
                    break
            
            param.requires_grad = should_train
            if should_train:
                trainable_params.append(name)
            else:
                frozen_params.append(name)
        
        logger.info(f" Frozen {len(frozen_params)} parameters")
        logger.info(f" Trainable {len(trainable_params)} parameters")
        logger.info(f"   Trainable params (first 10): {trainable_params[:10]}")
        logger.info(f"   Frozen params (first 10): {frozen_params[:10]}\n")
    else:
        # :
        trainable_modules = configs["train"].get("trainable_modules", None)
        if trainable_modules is None:
            transformer.requires_grad_(True)
            logger.info("All transformer parameters are trainable\n")
        else:
            trainable_module_names = []
            transformer.requires_grad_(False)
            for name, module in transformer.named_modules():
                for module_name in tuple(trainable_modules.split(",")):
                    if module_name in name:
                        for params in module.parameters():
                            params.requires_grad = True
                        trainable_module_names.append(name)
            logger.info(f"Trainable parameter names: {trainable_module_names}\n")

    # transformer.enable_xformers_memory_efficient_attention()  # use `tF.scaled_dot_product_attention` instead

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # Create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_transformer.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                for i, model in enumerate(models):
                    # Get the underlying model if wrapped by DDP
                    unwrapped_model = accelerator.unwrap_model(model)
                    model_class_name = unwrapped_model.__class__.__name__
                    
                    # Save to different directories based on model type
                    if "PartragDiTModel" in model_class_name or "Transformer" in model_class_name:
                        save_dir = os.path.join(output_dir, "transformer")
                    elif "DualProjectionModule" in model_class_name or "Projection" in model_class_name:
                        save_dir = os.path.join(output_dir, "projection")
                    else:
                        # Fallback for unknown model types
                        save_dir = os.path.join(output_dir, f"model_{i}")
                    
                    # Use save_pretrained if available, otherwise use torch.save
                    if hasattr(unwrapped_model, 'save_pretrained'):
                        unwrapped_model.save_pretrained(save_dir)
                    else:
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, "pytorch_model.pt"))

                    # Make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = MyEMAModel.from_pretrained(os.path.join(input_dir, "transformer_ema"), PartragDiTModel)
                ema_transformer.load_state_dict(load_model.state_dict())
                ema_transformer.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # Pop models so that they are not loaded again
                model = models.pop()
                unwrapped_model = accelerator.unwrap_model(model)
                model_class_name = unwrapped_model.__class__.__name__

                # Load from different directories based on model type
                if "PartragDiTModel" in model_class_name or "Transformer" in model_class_name:
                    load_model = PartragDiTModel.from_pretrained(input_dir, subfolder="transformer")
                    unwrapped_model.register_to_config(**load_model.config)
                    unwrapped_model.load_state_dict(load_model.state_dict())
                    del load_model
                elif "DualProjectionModule" in model_class_name or "Projection" in model_class_name:
                    from src.models.projection_layers import DualProjectionModule
                    load_dir = os.path.join(input_dir, "projection")
                    if os.path.exists(load_dir):
                        load_model = DualProjectionModule.from_pretrained(load_dir)
                        unwrapped_model.load_state_dict(load_model.state_dict())
                        del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if configs["train"]["grad_checkpoint"]:
        transformer.enable_gradient_checkpointing()

    # Initialize the optimizer and learning rate scheduler
    logger.info("Initializing the optimizer and learning rate scheduler...\n")
    
    #  
    use_differential_lr = configs["train"].get("use_differential_lr", False)
    contrastive_enabled = configs["train"].get("enable_contrastive", True)
    
    if use_differential_lr:
        logger.info(" Using differential learning rate strategy...")
        
        frozen_modules_lr = configs["train"].get("frozen_modules_lr", 0.0)
        pretrained_modules_lr = configs["train"].get("pretrained_modules_lr", 1e-6)
        new_modules_lr = configs["train"].get("new_modules_lr", configs["optimizer"]["lr"])
        projection_modules_lr = configs["train"].get("projection_modules_lr", configs["optimizer"]["lr"])
        
        # 
        frozen_params = []
        pretrained_params = []
        new_params = []
        
        import fnmatch
        for name, param in transformer.named_parameters():
            if not param.requires_grad:
                frozen_params.append(param)
            else:
                # ()
                is_new = any(keyword in name for keyword in ["attn2", "cross_attn", "global_attn"])
                if is_new:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
        
        logger.info(f"   Frozen params: {len(frozen_params)}, LR: {frozen_modules_lr}")
        logger.info(f"   Pretrained params: {len(pretrained_params)}, LR: {pretrained_modules_lr}")
        logger.info(f"   New params: {len(new_params)}, LR: {new_modules_lr}")
        
        optimizer_params = []
        if len(frozen_params) > 0 and frozen_modules_lr > 0:
            optimizer_params.append({"params": frozen_params, "lr": frozen_modules_lr})
        if len(pretrained_params) > 0:
            optimizer_params.append({"params": pretrained_params, "lr": pretrained_modules_lr})
        if len(new_params) > 0:
            optimizer_params.append({"params": new_params, "lr": new_modules_lr})
        
        # Add projection layers
        if contrastive_enabled:
            projection_params = list(contrastive_projection.parameters())
            optimizer_params.append({"params": projection_params, "lr": projection_modules_lr})
            logger.info(f"   Projection params: {len(projection_params)}, LR: {projection_modules_lr}\n")
        else:
            contrastive_projection.requires_grad_(False)
            logger.info("   Contrastive disabled: projection parameters frozen\n")
        
    else:
        # 
        name_lr_mult = configs["train"].get("name_lr_mult", None)
        lr_mult = configs["train"].get("lr_mult", 1.0)
        params, params_lr_mult, names_lr_mult = [], [], []
        for name, param in transformer.named_parameters():
            if name_lr_mult is not None:
                for k in name_lr_mult.split(","):
                    if k in name:
                        params_lr_mult.append(param)
                        names_lr_mult.append(name)
                if name not in names_lr_mult:
                    params.append(param)
            else:
                params.append(param)
        
        # Add projection layers parameters to optimizer
        if contrastive_enabled:
            projection_params = list(contrastive_projection.parameters())
            logger.info(f"Adding {len(projection_params)} projection parameters to optimizer\n")
        else:
            contrastive_projection.requires_grad_(False)
            projection_params = []
            logger.info("Contrastive disabled: projection parameters frozen\n")
        
        optimizer_params = [
            {"params": params, "lr": configs["optimizer"]["lr"]},
        ]
        if params_lr_mult:
            optimizer_params.append({"params": params_lr_mult, "lr": configs["optimizer"]["lr"] * lr_mult})
        if projection_params:
            optimizer_params.append({"params": projection_params, "lr": configs["optimizer"]["lr"]})

    optimizer = get_optimizer(
        params=optimizer_params,
        **configs["optimizer"]
    )
    if not use_differential_lr:
        name_lr_mult = configs["train"].get("name_lr_mult", None)
        if name_lr_mult is not None:
            lr_mult = configs["train"].get("lr_mult", 1.0)
            logger.info(f"Learning rate x [{lr_mult}] parameter names: {names_lr_mult}\n")

    configs["lr_scheduler"]["total_steps"] = configs["train"]["epochs"] * math.ceil(
        len(train_loader) // accelerator.num_processes / args.gradient_accumulation_steps)  # only account updated steps
    configs["lr_scheduler"]["total_steps"] *= accelerator.num_processes  # for lr scheduler setting
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] *= accelerator.num_processes  # for lr scheduler setting
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, **configs["lr_scheduler"])
    configs["lr_scheduler"]["total_steps"] //= accelerator.num_processes  # reset for multi-gpu
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] //= accelerator.num_processes  # reset for multi-gpu

    # Prepare everything with `accelerator`
    transformer, contrastive_projection, optimizer, lr_scheduler, train_loader, val_loader, random_val_loader = accelerator.prepare(
        transformer, contrastive_projection, optimizer, lr_scheduler, train_loader, val_loader, random_val_loader
    )
    # Set classes explicitly for everything
    transformer: DistributedDataParallel
    optimizer: AcceleratedOptimizer
    lr_scheduler: AcceleratedScheduler
    train_loader: DataLoaderShard
    val_loader: DataLoaderShard
    random_val_loader: DataLoaderShard

    if args.use_ema:
        ema_transformer.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move `vae` and `image_encoder_dinov2` to gpu and cast to `weight_dtype`
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder_dinov2.to(accelerator.device, dtype=weight_dtype)
    if momentum_projection is not None:
        momentum_projection.to(accelerator.device, dtype=torch.float32)

    # Training configs after distribution and accumulation setup
    updated_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updated_steps = configs["lr_scheduler"]["total_steps"]
    if args.max_train_steps is None:
        args.max_train_steps = total_updated_steps
    assert configs["train"]["epochs"] * updated_steps_per_epoch == total_updated_steps
    if accelerator.num_processes > 1 and accelerator.is_main_process:
        print()
    accelerator.wait_for_everyone()
    logger.info(f"Total batch size: [{total_batch_size}]")
    logger.info(f"Learning rate: [{configs['optimizer']['lr']}]")
    logger.info(f"Gradient Accumulation steps: [{args.gradient_accumulation_steps}]")
    logger.info(f"Total epochs: [{configs['train']['epochs']}]")
    logger.info(f"Total steps: [{total_updated_steps}]")
    logger.info(f"Steps for updating per epoch: [{updated_steps_per_epoch}]")
    logger.info(f"Steps for validation: [{len(val_loader)}]\n")

    # (Optional) Load checkpoint
    global_update_step = 0
    if args.resume_from_iter is not None:
        if args.resume_from_iter < 0:
            args.resume_from_iter = int(sorted(os.listdir(ckpt_dir))[-1])
        logger.info(f"Load checkpoint from iteration [{args.resume_from_iter}]\n")
        # Load everything
        if version.parse(torch.__version__) >= version.parse("2.4.0"):
            torch.serialization.add_safe_globals([
                int, list, dict, 
                defaultdict,
                Any,
                DictConfig, ListConfig, Metadata, ContainerMetadata, AnyNode
            ]) # avoid deserialization error when loading optimizer state
        accelerator.load_state(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}"))  # torch < 2.4.0 here for `weights_only=False`
        global_update_step = int(args.resume_from_iter)

    # Save all experimental parameters and model architecture of this run to a file (args and configs)
    if accelerator.is_main_process:
        exp_params = save_experiment_params(args, configs, exp_dir)
        save_model_architecture(accelerator.unwrap_model(transformer), exp_dir)

    # WandB logger
    if accelerator.is_main_process:
        # 
        os.environ["WANDB_MODE"] = "offline"
        if args.offline_wandb:
            os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project=PROJECT_NAME, name=args.tag,
            config=exp_params, dir=exp_dir,
            resume=True
        )
        # Wandb artifact for logging experiment information
        arti_exp_info = wandb.Artifact(args.tag, type="exp_info")
        arti_exp_info.add_file(os.path.join(exp_dir, "params.yaml"))
        arti_exp_info.add_file(os.path.join(exp_dir, "model.txt"))
        arti_exp_info.add_file(os.path.join(exp_dir, "log.txt"))  # only save the log before training
        wandb.log_artifact(arti_exp_info)

    def get_sigmas(timesteps: Tensor, n_dim: int, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(dtype=dtype, device=accelerator.device)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero()[0].item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    contrastive_loss = ContrastiveLossHelper(
        temperature=configs["train"].get("contrastive_temperature", 0.07)
    )
    # Note: contrastive_enabled is already defined above before optimizer initialization
    #  ,
    contrastive_part_weight_target = configs["train"].get("contrastive_part_weight", 0.0) if contrastive_enabled else 0.0
    contrastive_object_weight_target = configs["train"].get("contrastive_object_weight", 0.0) if contrastive_enabled else 0.0
    
    def get_progressive_contrastive_weight(current_step, target_weight, total_steps=5950, start_step=850, warmup_end=1500, rampup_end=3000):
        """
        (Curriculum Learning)
        
        1 (850-1500): 0.0050.5xtarget ()
        2 (1500-3000): 0.5xtarget0.9xtarget ()
        3 (3000-5950): 0.9xtarget1.0xtarget ()
        
        :
        1. 
        2. 
        3. 
        4. NaN,
        """
        if current_step < warmup_end:
            # : 0.0050.5xtarget
            progress = max(0, min(1, (current_step - start_step) / (warmup_end - start_step)))
            weight = 0.005 + progress * (0.5 * target_weight - 0.005)
        elif current_step < rampup_end:
            # : 0.5xtarget0.9xtarget
            progress = (current_step - warmup_end) / (rampup_end - warmup_end)
            weight = 0.5 * target_weight + progress * (0.9 * target_weight - 0.5 * target_weight)
        else:
            # : 0.9xtarget1.0xtarget
            progress = min(1, (current_step - rampup_end) / (total_steps - rampup_end))
            weight = 0.9 * target_weight + progress * (1.0 * target_weight - 0.9 * target_weight)
        
        return weight
    
    # EMAloss (clipping) - 
    ema_state = {
        "contrastive_part_loss": 0.05,
        "contrastive_object_loss": 0.05
    }

    # Optional bidirectional momentum queues (image<->mesh for part/object granularity)
    part_image_queue = part_mesh_queue = None
    object_image_queue = object_mesh_queue = None
    if use_momentum_queue:
        part_image_queue = FeatureQueue(projection_dim, momentum_queue_size, accelerator.device)
        part_mesh_queue = FeatureQueue(projection_dim, momentum_queue_size, accelerator.device)
        object_image_queue = FeatureQueue(projection_dim, momentum_queue_size, accelerator.device)
        object_mesh_queue = FeatureQueue(projection_dim, momentum_queue_size, accelerator.device)
        logger.info("Momentum feature queues initialized for part/object levels\n")

    # Initialize retrieval helper
    retrieval_config = configs.get("retrieval", {})
    retrieval_enabled = retrieval_config.get("enabled", False)
    if retrieval_enabled:
        logger.info(f"Initializing retrieval helper...")
        retrieval_helper = RetrievalHelper(
            database_path=retrieval_config.get("database_path"),
            device=accelerator.device,
            enabled=True,
            use_fused_embeddings=True,  # fused embeddings (image+mesh)
            use_sequence_broadcast=True,  # sequence broadcast
        )
        retrieval_top_k = retrieval_config.get("top_k", 3)
        logger.info(f"Retrieval enabled with top_k={retrieval_top_k}\n")
    else:
        retrieval_helper = None
        retrieval_top_k = 0
        logger.info(f"Retrieval disabled\n")

    #  
    if False:  # 
        logger.info(f" Running validation immediately after resuming from step {global_update_step}")
        
        # Use EMA parameters for evaluation
        if args.use_ema:
            ema_transformer.store(transformer.parameters())
            ema_transformer.copy_to(transformer.parameters())
        
        transformer.eval()
        
        log_validation(
            val_loader, random_val_loader,
            feature_extractor_dinov2, image_encoder_dinov2,
            vae, transformer,
            global_update_step, eval_dir,
            accelerator, logger,
            args, configs,
            retrieval_helper=retrieval_helper,
            retrieval_top_k=retrieval_top_k
        )
        
        if args.use_ema:
            ema_transformer.restore(transformer.parameters())
        
        transformer.train()
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f" Initial validation completed\n")

    # Start training
    if accelerator.is_main_process:
        print()
    logger.info(f"Start training into {exp_dir}\n")
    logger.logger.propagate = False  # not propagate to the root logger (console)
    progress_bar = tqdm(
        range(total_updated_steps),
        initial=global_update_step,
        desc="Training",
        ncols=125,
        disable=not accelerator.is_main_process
    )
    for batch in yield_forever(train_loader):

        if global_update_step == args.max_train_steps:
            progress_bar.close()
            logger.logger.propagate = True  # propagate to the root logger (console)
            if accelerator.is_main_process:
                wandb.finish()
            logger.info("Training finished!\n")
            return

        transformer.train()

        #  
        contrastive_part_weight = get_progressive_contrastive_weight(
            current_step=global_update_step,
            target_weight=contrastive_part_weight_target,
            total_steps=args.max_train_steps,
            start_step=args.resume_from_iter if args.resume_from_iter else 0
        ) if contrastive_enabled else 0.0
        
        contrastive_object_weight = get_progressive_contrastive_weight(
            current_step=global_update_step,
            target_weight=contrastive_object_weight_target,
            total_steps=args.max_train_steps,
            start_step=args.resume_from_iter if args.resume_from_iter else 0
        ) if contrastive_enabled else 0.0

        with accelerator.accumulate(transformer):
            
            images = batch["images"] # [N, H, W, 3]
            with torch.no_grad():
                if isinstance(images, torch.Tensor):
                    if images.dtype != torch.uint8:
                        images = (images.clamp(0, 1) * 255).to(torch.uint8)
                    # Convert CHW -> HWC numpy arrays for the processor
                    images_list = [img.permute(1, 2, 0).cpu().numpy() for img in images]
                else:
                    images_list = images
                # Process images and ensure correct batch dimension
                processed = feature_extractor_dinov2(images=images_list, return_tensors="pt")
                images = processed.pixel_values
                # Ensure images is [B, C, H, W]
                if images.ndim == 3:
                    images = images.unsqueeze(0)
            images = images.to(device=accelerator.device, dtype=weight_dtype, non_blocking=True)
            with torch.no_grad():
                # Ensure image_encoder_dinov2 is in eval mode and on correct device
                image_encoder_dinov2.eval()
                if not hasattr(image_encoder_dinov2, '_moved_to_device'):
                    image_encoder_dinov2 = image_encoder_dinov2.to(device=accelerator.device)
                    image_encoder_dinov2._moved_to_device = True
                image_embeds_original = image_encoder_dinov2(images).last_hidden_state  # [N, seq, D]
            
            # (detached)
            image_embeds_for_contrastive = image_embeds_original.detach()
            
            # Retrieval: concatcondition(DIT)
            image_embeds = image_embeds_original  # 
            if retrieval_helper is not None:
                with torch.no_grad():
                    retrieved_embeddings, retrieved_indices = retrieval_helper.retrieve(
                        query_embeddings=image_embeds_original,  # [N, seq, D]
                        top_k=retrieval_top_k,
                    )
                    # retrieved_embeddings: [N, top_k, D]
                    # Concat: [retrieved, prompt] -> DIT condition
                    image_embeds = concat_retrieved_to_encoder_hidden_states(
                        encoder_hidden_states=image_embeds_original,  # [N, seq, D]
                        retrieved_embeddings=retrieved_embeddings,  # [N, top_k, D]
                    )  # [N, top_k + seq, D]
            
            negative_image_embeds = torch.zeros_like(image_embeds)

            part_surfaces = batch["part_surfaces"] # [N, P, 6]
            part_surfaces = part_surfaces.to(device=accelerator.device, dtype=weight_dtype)

            num_parts = batch["num_parts"] # [M, ] The shape of num_parts is not fixed
            num_objects = num_parts.shape[0] # M
        
            # Convert part_labels to tensor indices; if contrastive
            part_labels_str = batch["part_labels"]
            if contrastive_enabled:
                # Stable hash ids for part labels across batches/steps
                num_part_label_buckets = int(configs["train"].get("num_part_labels", 1024))
                #  : part_labels_str
                if isinstance(part_labels_str, torch.Tensor):
                    part_labels_str = [str(x) for x in part_labels_str.cpu().tolist()]
                elif not isinstance(part_labels_str, list):
                    part_labels_str = list(part_labels_str)
                
                hashed_ids = [
                    int(hashlib.md5(str(lbl).encode("utf-8")).hexdigest(), 16) % num_part_label_buckets
                    for lbl in part_labels_str
                ]
                part_labels = torch.tensor(hashed_ids, device=accelerator.device, dtype=torch.long)
            else:
                part_labels = torch.zeros(len(part_labels_str), device=accelerator.device, dtype=torch.long)
            
            object_labels = torch.arange(num_objects, device=accelerator.device).repeat_interleave(num_parts)
            object_labels_global = torch.arange(num_objects, device=accelerator.device)

            with torch.no_grad():
                vae_outputs = vae.encode(
                    part_surfaces,
                    **configs["model"]["vae"]
                )
                posterior = vae_outputs.latent_dist
                latents = posterior.sample()
                mesh_tokens = posterior.projected
            mesh_part_features_detached = mesh_tokens.mean(dim=1).to(dtype=torch.float32)  # Detached
            mesh_global_detached = torch.stack([
                chunk.mean(dim=0) for chunk in torch.split(mesh_part_features_detached, num_parts.tolist())
            ], dim=0)

            # projection_module(DDP)
            projection_module = contrastive_projection.module if hasattr(contrastive_projection, 'module') else contrastive_projection
            
            # :Projection Layersdetached
            # gradientprojection,projection layers
            if contrastive_enabled:
                image_part_features_raw = projection_module.aggregate_image_part_features(
                    image_embeds_for_contrastive,
                    part_labels
                ).to(dtype=torch.float32)
                image_global_raw = torch.stack([
                    chunk.mean(dim=0) for chunk in torch.split(image_part_features_raw, num_parts.tolist())
                ], dim=0)  # [M, D]

                # Projection Layers(gradient)
                image_part_features = projection_module.forward_image(image_part_features_raw)
                image_global = projection_module.forward_image(image_global_raw)
                mesh_part_features = projection_module.forward_mesh(mesh_part_features_detached)
                mesh_global = projection_module.forward_mesh(mesh_global_detached)
            else:
                image_part_features = None
                image_global = None
                mesh_part_features = None
                mesh_global = None

            noise = torch.randn_like(latents)
            # For weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=configs["train"]["weighting_scheme"],
                batch_size=num_objects,
                logit_mean=configs["train"]["logit_mean"],
                logit_std=configs["train"]["logit_std"],
                mode_scale=configs["train"]["mode_scale"],
            )
            indices = (u * noise_scheduler.config.num_train_timesteps).long()
            timesteps = noise_scheduler.timesteps[indices].to(accelerator.device) # [M, ]
            # Repeat the timesteps for each part
            timesteps = timesteps.repeat_interleave(num_parts) # [N, ]

            sigmas = get_sigmas(timesteps, len(latents.shape), weight_dtype)
            latent_model_input = noisy_latents = (1. - sigmas) * latents + sigmas * noise

            if configs["train"]["cfg_dropout_prob"] > 0:
                # We use the same dropout mask for the same part
                dropout_mask = torch.rand(num_objects, device=accelerator.device) < configs["train"]["cfg_dropout_prob"] # [M, ]
                dropout_mask = dropout_mask.repeat_interleave(num_parts) # [N, ]
                if dropout_mask.any():
                    image_embeds[dropout_mask] = negative_image_embeds[dropout_mask]

            #  :num_parts(torch.Tensorint)
            if isinstance(num_parts, np.ndarray):
                num_parts_for_attn = torch.from_numpy(num_parts).long()
            elif isinstance(num_parts, (np.integer, np.int64, np.int32)):
                num_parts_for_attn = int(num_parts)
            else:
                num_parts_for_attn = num_parts
            
            model_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timesteps,
                encoder_hidden_states=image_embeds, 
                attention_kwargs={"num_parts": num_parts_for_attn}
            ).sample

            if configs["train"]["training_objective"] == "x0":  # Section 5 of https://arxiv.org/abs/2206.00364
                model_pred = model_pred * (-sigmas) + noisy_latents  # predicted x_0
                target = latents
            elif configs["train"]["training_objective"] == 'v':  # flow matching
                target = noise - latents
            elif configs["train"]["training_objective"] == '-v':  # reverse flow matching
                # The training objective for TripoSG is the reverse of the flow matching objective. 
                # It uses "different directions", i.e., the negative velocity. 
                # This is probably a mistake in engineering, not very harmful. 
                # In TripoSG's rectified flow scheduler, prev_sample = sample + (sigma - sigma_next) * model_output
                # See TripoSG's scheduler https://github.com/VAST-AI-Research/TripoSG/blob/main/triposg/schedulers/scheduling_rectified_flow.py#L296
                # While in diffusers's flow matching scheduler, prev_sample = sample + (sigma_next - sigma) * model_output
                # See https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L454
                target = latents - noise
            else:
                raise ValueError(f"Unknown training objective [{configs['train']['training_objective']}]")

            # For these weighting schemes use a uniform timestep sampling, so post-weight the loss
            weighting = compute_loss_weighting_for_sd3(
                configs["train"]["weighting_scheme"],
                sigmas
            )
            # : NaN/Inf  0, clamp  1.0
            weighting = torch.nan_to_num(weighting, nan=0.0, posinf=1.0, neginf=0.0)
            weighting = weighting.clamp(max=1.0)

            diffusion_loss = weighting * tF.mse_loss(model_pred.float(), target.float(), reduction="none")
            diffusion_loss = diffusion_loss.mean(dim=list(range(1, len(diffusion_loss.shape))))

            total_loss = diffusion_loss.mean()
            contrastive_logs = {}

            #  NaN-Safe 
            # batch
            if not hasattr(ema_state, 'contrastive_skipped_count'):
                ema_state['contrastive_skipped_count'] = 0
                ema_state['contrastive_total_count'] = 0
            
            contrastive_safe_to_use = True  # 
            
            # Pull queue features from previous steps (if enabled)
            part_image_q_feats, part_image_q_labels = (
                part_image_queue.get() if part_image_queue is not None else (None, None)
            )
            part_mesh_q_feats, part_mesh_q_labels = (
                part_mesh_queue.get() if part_mesh_queue is not None else (None, None)
            )
            object_image_q_feats, object_image_q_labels = (
                object_image_queue.get() if object_image_queue is not None else (None, None)
            )
            object_mesh_q_feats, object_mesh_q_labels = (
                object_mesh_queue.get() if object_mesh_queue is not None else (None, None)
            )

            part_contrastive_value = None
            if contrastive_part_weight > 0 and contrastive_enabled:
                try:
                    with torch.cuda.amp.autocast(enabled=False):
                        image_part_features_fp32 = image_part_features.float()
                        mesh_part_features_fp32 = mesh_part_features.float()

                        # image -> mesh (mesh queue as extra negatives)
                        key_mesh_features = mesh_part_features_fp32
                        key_mesh_labels = part_labels
                        if part_mesh_q_feats is not None and part_mesh_q_labels is not None:
                            key_mesh_features = torch.cat([key_mesh_features, part_mesh_q_feats], dim=0)
                            key_mesh_labels = torch.cat([key_mesh_labels, part_mesh_q_labels], dim=0)
                        part_i2m = contrastive_loss.compute(
                            image_part_features_fp32,
                            key_mesh_features,
                            labels_query=part_labels,
                            labels_key=key_mesh_labels,
                        )

                        # mesh -> image (image queue as extra negatives)
                        key_image_features = image_part_features_fp32
                        key_image_labels = part_labels
                        if part_image_q_feats is not None and part_image_q_labels is not None:
                            key_image_features = torch.cat([key_image_features, part_image_q_feats], dim=0)
                            key_image_labels = torch.cat([key_image_labels, part_image_q_labels], dim=0)
                        part_m2i = contrastive_loss.compute(
                            mesh_part_features_fp32,
                            key_image_features,
                            labels_query=part_labels,
                            labels_key=key_image_labels,
                        )

                        part_contrastive_value = 0.5 * (part_i2m + part_m2i)

                    if torch.isnan(part_contrastive_value) or torch.isinf(part_contrastive_value):
                        contrastive_safe_to_use = False
                        logger.warning(
                            f"  Step {global_update_step}: Part contrastive loss is NaN/Inf, skipping contrastive learning for this batch"
                        )
                    elif part_contrastive_value.item() > 0.8:
                        contrastive_safe_to_use = False
                        logger.warning(
                            f"  Step {global_update_step}: Part contrastive loss too large ({part_contrastive_value.item():.4f}), skipping"
                        )

                    if contrastive_safe_to_use:
                        current_loss = part_contrastive_value.item()
                        ema_state["contrastive_part_loss"] = 0.9 * ema_state["contrastive_part_loss"] + 0.1 * current_loss
                        adaptive_clip_part = ema_state["contrastive_part_loss"]

                        total_loss = total_loss + contrastive_part_weight * part_contrastive_value
                        contrastive_logs["loss/contrastive_part"] = part_contrastive_value
                        contrastive_logs["train/contrastive_part_weight"] = contrastive_part_weight
                        contrastive_logs["train/adaptive_clip_part"] = adaptive_clip_part
                except Exception as e:
                    contrastive_safe_to_use = False
                    logger.warning(f"  Step {global_update_step}: Part contrastive computation failed: {e}")

            object_contrastive_value = None
            if contrastive_object_weight > 0 and contrastive_enabled and contrastive_safe_to_use:
                try:
                    with torch.cuda.amp.autocast(enabled=False):
                        image_global_fp32 = image_global.float()
                        mesh_global_fp32 = mesh_global.float()

                        # image -> mesh (object queue)
                        key_mesh_features = mesh_global_fp32
                        key_mesh_labels = object_labels_global
                        if object_mesh_q_feats is not None and object_mesh_q_labels is not None:
                            key_mesh_features = torch.cat([key_mesh_features, object_mesh_q_feats], dim=0)
                            key_mesh_labels = torch.cat([key_mesh_labels, object_mesh_q_labels], dim=0)
                        obj_i2m = contrastive_loss.compute(
                            image_global_fp32,
                            key_mesh_features,
                            labels_query=object_labels_global,
                            labels_key=key_mesh_labels,
                        )

                        # mesh -> image (object queue)
                        key_image_features = image_global_fp32
                        key_image_labels = object_labels_global
                        if object_image_q_feats is not None and object_image_q_labels is not None:
                            key_image_features = torch.cat([key_image_features, object_image_q_feats], dim=0)
                            key_image_labels = torch.cat([key_image_labels, object_image_q_labels], dim=0)
                        obj_m2i = contrastive_loss.compute(
                            mesh_global_fp32,
                            key_image_features,
                            labels_query=object_labels_global,
                            labels_key=key_image_labels,
                        )

                        object_contrastive_value = 0.5 * (obj_i2m + obj_m2i)

                    if torch.isnan(object_contrastive_value) or torch.isinf(object_contrastive_value):
                        contrastive_safe_to_use = False
                        logger.warning(
                            f"  Step {global_update_step}: Object contrastive loss is NaN/Inf, skipping contrastive learning for this batch"
                        )
                    elif object_contrastive_value.item() > 0.8:
                        contrastive_safe_to_use = False
                        logger.warning(
                            f"  Step {global_update_step}: Object contrastive loss too large ({object_contrastive_value.item():.4f}), skipping"
                        )

                    if contrastive_safe_to_use:
                        current_loss = object_contrastive_value.item()
                        ema_state["contrastive_object_loss"] = 0.9 * ema_state["contrastive_object_loss"] + 0.1 * current_loss
                        adaptive_clip_object = ema_state["contrastive_object_loss"]

                        total_loss = total_loss + contrastive_object_weight * object_contrastive_value
                        contrastive_logs["loss/contrastive_object"] = object_contrastive_value
                        contrastive_logs["train/contrastive_object_weight"] = contrastive_object_weight
                        contrastive_logs["train/adaptive_clip_object"] = adaptive_clip_object
                except Exception as e:
                    contrastive_safe_to_use = False
                    logger.warning(f"  Step {global_update_step}: Object contrastive computation failed: {e}")

            # Momentum encoder update + queue enqueue after using current queues
            if use_momentum_queue and momentum_projection is not None and contrastive_enabled:
                with torch.no_grad():
                    momentum_update(projection_module, momentum_projection, momentum_coefficient)

                    mom_image_part_raw = momentum_projection.aggregate_image_part_features(
                        image_embeds_for_contrastive, part_labels
                    ).float()
                    mom_image_global_raw = torch.stack(
                        [chunk.mean(dim=0) for chunk in torch.split(mom_image_part_raw, num_parts.tolist())], dim=0
                    )
                    mom_mesh_part_raw = mesh_part_features_detached.float()
                    mom_mesh_global_raw = torch.stack(
                        [chunk.mean(dim=0) for chunk in torch.split(mom_mesh_part_raw, num_parts.tolist())], dim=0
                    )

                    mom_image_part = momentum_projection.forward_image(mom_image_part_raw).float()
                    mom_image_global = momentum_projection.forward_image(mom_image_global_raw).float()
                    mom_mesh_part = momentum_projection.forward_mesh(mom_mesh_part_raw).float()
                    mom_mesh_global = momentum_projection.forward_mesh(mom_mesh_global_raw).float()

                    part_image_queue.enqueue(mom_image_part, part_labels)
                    part_mesh_queue.enqueue(mom_mesh_part, part_labels)
                    object_image_queue.enqueue(mom_image_global, object_labels_global)
                    object_mesh_queue.enqueue(mom_mesh_global, object_labels_global)

                    contrastive_logs["train/part_queue_size"] = float(
                        momentum_queue_size if part_image_queue.full else part_image_queue.ptr
                    )
                    contrastive_logs["train/object_queue_size"] = float(
                        momentum_queue_size if object_image_queue.full else object_image_queue.ptr
                    )
            
            #  
            ema_state['contrastive_total_count'] += 1
            if not contrastive_safe_to_use:
                ema_state['contrastive_skipped_count'] += 1
            skip_rate = ema_state['contrastive_skipped_count'] / max(1, ema_state['contrastive_total_count'])
            contrastive_logs["train/contrastive_skip_rate"] = skip_rate

            # Backpropagate
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            # Gather the losses across all processes for logging (if we use distributed training)
            diffusion_loss_mean = accelerator.gather(diffusion_loss.detach()).mean()
            total_loss_mean = accelerator.gather(total_loss.detach()).mean()

            logs = {
                "loss": total_loss_mean.item(),
                "loss/diffusion": diffusion_loss_mean.item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            for key, value in contrastive_logs.items():
                # tensorfloat
                if isinstance(value, torch.Tensor):
                    logs[key] = accelerator.gather(value.detach()).mean().item()
                else:
                    logs[key] = float(value)
            if args.use_ema:
                ema_transformer.step(transformer.parameters())
                logs.update({"ema": ema_transformer.cur_decay_value})

            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
            global_update_step += 1

            logger.info(
                f"[{global_update_step:06d} / {total_updated_steps:06d}] " +
                f"loss: {logs['loss']:.4f}, lr: {logs['lr']:.2e}" +
                f", ema: {logs['ema']:.4f}" if args.use_ema else ""
            )

            # Log the training progress
            if (
                global_update_step % configs["train"]["log_freq"] == 0 
                or global_update_step == 1
                or global_update_step % updated_steps_per_epoch == 0 # last step of an epoch
            ):  
                if accelerator.is_main_process:
                    wandb.log({
                        "training/loss": logs["loss"],
                        "training/loss_diffusion": logs["loss/diffusion"],
                        "training/lr": logs["lr"],
                        **{f"training/{k.split('/')[-1]}": logs[k] for k in logs if k.startswith("loss/contrastive")},
                    }, step=global_update_step)
                    if args.use_ema:
                        wandb.log({
                            "training/ema": logs["ema"]
                        }, step=global_update_step)

            # Save checkpoint
            if (
                global_update_step % configs["train"]["save_freq"] == 0  # 1. every `save_freq` steps
                or global_update_step % (configs["train"]["save_freq_epoch"] * updated_steps_per_epoch) == 0  # 2. every `save_freq_epoch` epochs
                or global_update_step == total_updated_steps # 3. last step of an epoch
                # or global_update_step == 1 # 4. first step
            ): 

                gc.collect()
                
                # Save checkpoint to both locations
                checkpoint_name = f"{global_update_step:06d}"
                main_ckpt_path = os.path.join(ckpt_dir, checkpoint_name)
                
                logger.info(f"Saving current state to {main_ckpt_path}")
                # 1. Save to main checkpoint directory (for resuming)
                if accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
                    accelerator.save_state(main_ckpt_path)
                elif accelerator.is_main_process:
                    accelerator.save_state(main_ckpt_path)
                accelerator.wait_for_everyone()
                
                # checkpoint (3resume)
                if accelerator.is_main_process:
                    import shutil
                    checkpoints = sorted([d for d in os.listdir(ckpt_dir) if d.isdigit()])
                    if len(checkpoints) > 3:
                        for old_ckpt in checkpoints[:-3]:
                            old_ckpt_path = os.path.join(ckpt_dir, old_ckpt)
                            if os.path.isdir(old_ckpt_path):
                                shutil.rmtree(old_ckpt_path, ignore_errors=True)
                                logger.info(f"Cleaned up old checkpoint: {old_ckpt}")
                
                accelerator.wait_for_everyone()
                gc.collect()

            # Evaluate on the validation set
            #  ()
            if False:  #   

                # Use EMA parameters for evaluation
                if args.use_ema:
                    # Store the Transformer parameters temporarily and load the EMA parameters to perform inference
                    ema_transformer.store(transformer.parameters())
                    ema_transformer.copy_to(transformer.parameters())

                transformer.eval()

                log_validation(
                    val_loader, random_val_loader,
                    feature_extractor_dinov2, image_encoder_dinov2,
                    vae, transformer,
                    global_update_step, eval_dir,
                    accelerator, logger,
                    args, configs,
                    retrieval_helper=retrieval_helper,
                    retrieval_top_k=retrieval_top_k
                )

                if args.use_ema:
                    # Switch back to the original Transformer parameters
                    ema_transformer.restore(transformer.parameters())

                torch.cuda.empty_cache()
                gc.collect()

@torch.no_grad()
def log_validation(
    dataloader, random_dataloader,
    feature_extractor_dinov2, image_encoder_dinov2,
    vae, transformer, 
    global_step, eval_dir,
    accelerator, logger,  
    args, configs,
    retrieval_helper=None,
    retrieval_top_k=0
):  

    #  Validation schedulervae_model_name_or_path(),pretrained_model_name_or_path
    val_scheduler_path = configs["model"].get("vae_model_name_or_path", configs["model"]["pretrained_model_name_or_path"])
    val_noise_scheduler = RectifiedFlowScheduler.from_pretrained(
        val_scheduler_path,
        subfolder="scheduler"
    )

    #  unwrapped model()
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    
    #  
    if hasattr(unwrapped_transformer, 'enable_contrastive'):
        original_contrastive_state = unwrapped_transformer.enable_contrastive
        unwrapped_transformer.enable_contrastive = False
    else:
        original_contrastive_state = None
    
    #  part_embedding()
    if hasattr(unwrapped_transformer, 'enable_part_embedding'):
        original_part_emb_state = unwrapped_transformer.enable_part_embedding
        unwrapped_transformer.enable_part_embedding = False
    else:
        original_part_emb_state = None
    
    #  :RetrievalHelperRetrievalModule
    # retrieval(RetrievalHelperRetrievalModule)
    # TODO: retrieval
    pipeline = PartragPipeline(
        vae=vae,
        transformer=unwrapped_transformer,
        scheduler=val_noise_scheduler,
        feature_extractor_dinov2=feature_extractor_dinov2,
        image_encoder_dinov2=image_encoder_dinov2,
        retrieval_module=None,  # ,
    )

    pipeline.set_progress_bar_config(disable=True)
    # pipeline.enable_xformers_memory_efficient_attention()

    if args.seed >= 0:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    else:
        generator = None
        

    val_progress_bar = tqdm(
        range(len(dataloader)) if args.max_val_steps is None else range(args.max_val_steps),
        desc=f"Validation [{global_step:06d}]",
        ncols=125,
        disable=not accelerator.is_main_process
    )

    medias_dictlist, metrics_dictlist = defaultdict(list), defaultdict(list)

    #  :part_image_aggregatorpart_labels
    original_aggregator = None
    if hasattr(transformer, 'projection_module') and hasattr(transformer.projection_module, 'part_image_aggregator'):
        original_aggregator = transformer.projection_module.part_image_aggregator
        transformer.projection_module.part_image_aggregator = None  # 

    val_dataloder, random_val_dataloader = yield_forever(dataloader), yield_forever(random_dataloader)
    val_step = 0
    while val_step < args.max_val_steps:

        if val_step < args.max_val_steps // 2:
            # fix the first half
            batch = next(val_dataloder)
        else:
            # randomly sample the next batch
            batch = next(random_val_dataloader)

        #  :batchpart_labels,
        if "part_labels" in batch:
            del batch["part_labels"]

        images = batch["images"]
        
        # imagesPIL Image(collate_fn)
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], Image.Image):
            # PIL Image,
            pass
        else:
            # imagestensor,
            if len(images.shape) == 5:
                images = images[0] # (1, N, H, W, 3) -> (N, H, W, 3)
            
            # PIL (uint8shape [H, W, C])
            images_np = images.cpu().numpy()
            if images_np.ndim == 4:  # [B, C, H, W] -> [B, H, W, C]
                images_np = images_np.transpose(0, 2, 3, 1)
            if images_np.ndim == 3:  # [B, H, W]  -> [B, H, W, 1]
                images_np = images_np[..., None]
            # 0-255uint8
            if images_np.max() <= 1.0:
                images_np = (images_np * 255).clip(0, 255).astype('uint8')
            else:
                images_np = images_np.clip(0, 255).astype('uint8')
            # ,
            images = [Image.fromarray(img.squeeze() if img.shape[-1] == 1 else img) for img in images_np]
        part_surfaces = batch["part_surfaces"].cpu().numpy()
        if len(part_surfaces.shape) == 4:
            part_surfaces = part_surfaces[0] # (1, N, P, 6) -> (N, P, 6)

        #  :PartDatasetpart
        # PartDatasetcollate_fnpart,images=part
        # num_partspart
        if "num_parts" in batch:
            num_parts_per_object = batch["num_parts"].cpu().numpy()  # [M] M=
            num_objects = len(num_parts_per_object)  # 
            
            # images()
            object_images = []
            idx = 0
            for num_parts in num_parts_per_object:
                object_images.append(images[idx])  # 
                idx += num_parts  # 
            
            # part_surfaces,
            object_part_surfaces = []
            idx = 0
            for num_parts in num_parts_per_object:
                obj_surfaces = part_surfaces[idx:idx+num_parts]  # [num_parts, P, 6]
                object_part_surfaces.append(obj_surfaces)
                idx += num_parts
            
            # 
            images = object_images  # 
            part_surfaces = object_part_surfaces  # ,part surfaces
            N = num_objects  # 
            num_parts_val = num_parts_per_object  # part()
        else:
            # SimpleDataset:1part
            num_parts_val = 1
            N = len(images)  # number of objects in this validation batch
            # part_surfaces
            part_surfaces = [part_surfaces[i:i+1] for i in range(N)]

        val_progress_bar.set_postfix(
            {"num_objects": N}
        )

        #  BF16,FP16
        with torch.autocast("cuda", torch.bfloat16):
            for guidance_scale in sorted(args.val_guidance_scales):
                #  :mesh,part
                all_object_meshes = []  # part meshes
                
                for obj_idx in range(N):
                    #  CFG+num_parts:
                    # ,batchpart
                    # PartDataset,parts
                    # ,validationpart
                    obj_image = [images[obj_idx]]  # 
                    obj_num_parts = int(num_parts_val[obj_idx]) if isinstance(num_parts_val, np.ndarray) else 1
                    
                    # :num_parts,guidance_scale=1.0CFG
                    # num_parts,pipelinepart
                    #  :RetrievalHelperRetrievalModule,retrieval
                    obj_pred_part_meshes = pipeline(
                        obj_image, 
                        num_inference_steps=configs['val']['num_inference_steps'],
                        num_tokens=configs['model']['vae']['num_tokens'],
                        guidance_scale=1.0,  #  CFG
                        attention_kwargs={},  #  ,num_parts
                        generator=generator,
                        max_num_expanded_coords=configs['val']['max_num_expanded_coords'],
                        use_flash_decoder=configs['val']['use_flash_decoder'],
                        use_retrieval=False,  # ,
                        num_retrieved_images=0,
                    ).meshes
                    
                    all_object_meshes.append(obj_pred_part_meshes)
                    
                    #  :
                    torch.cuda.empty_cache()
                    gc.collect()

                # Save the generated meshes
                if accelerator.is_main_process:
                    local_eval_dir = os.path.join(eval_dir, f"{global_step:06d}", f"guidance_scale_{guidance_scale:.1f}")
                    os.makedirs(local_eval_dir, exist_ok=True)
                    rendered_images_list, rendered_normals_list = [], []
                    
                    # gt imagemesh
                    for obj_idx in range(N):
                        # 1. save the gt image
                        images[obj_idx].save(os.path.join(local_eval_dir, f"{val_step:04d}_{obj_idx:02d}_gt.png"))
                        
                        # 2. save the generated part meshes for this object
                        obj_pred_part_meshes = all_object_meshes[obj_idx]
                        for part_idx, part_mesh in enumerate(obj_pred_part_meshes):
                            if part_mesh is None:
                                # If the generated mesh is None (decoding error), use a dummy mesh
                                part_mesh = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                                obj_pred_part_meshes[part_idx] = part_mesh
                            part_mesh.export(os.path.join(local_eval_dir, f"{val_step:04d}_{obj_idx:02d}_part{part_idx:02d}.glb"))
                        
                        # 3. render the generated mesh and save the rendered images
                        pred_mesh = get_colored_mesh_composition(obj_pred_part_meshes)
                        rendered_images: List[Image.Image] = render_views_around_mesh(
                            pred_mesh, 
                            num_views=configs['val']['rendering']['num_views'],
                            radius=configs['val']['rendering']['radius'],
                        )
                        rendered_normals: List[Image.Image] = render_normal_views_around_mesh(
                            pred_mesh,
                            num_views=configs['val']['rendering']['num_views'],
                            radius=configs['val']['rendering']['radius'],
                        )
                        export_renderings(
                            rendered_images,
                            os.path.join(local_eval_dir, f"{val_step:04d}_{obj_idx:02d}.gif"),
                            fps=configs['val']['rendering']['fps']
                        )
                        export_renderings(
                            rendered_normals,
                            os.path.join(local_eval_dir, f"{val_step:04d}_{obj_idx:02d}_normals.gif"),
                            fps=configs['val']['rendering']['fps']
                        )
                        rendered_images_list.append(rendered_images)
                        rendered_normals_list.append(rendered_normals)
                        
                        # 
                        del pred_mesh, rendered_images, rendered_normals

                    # wandb
                    medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/gt_image"] += images  # List[Image.Image]
                    medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/pred_rendered_images"] += rendered_images_list # List[List[Image.Image]]
                    medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/pred_rendered_normals"] += rendered_normals_list # List[List[Image.Image]]
                    
                    #  
                    del rendered_images_list, rendered_normals_list
                
                #  :
                torch.cuda.empty_cache()
                gc.collect()

                ################################ Compute generation metrics ################################

                parts_chamfer_distances, parts_f_scores = [], []

                for obj_idx in range(N):
                    gt_obj_part_surfaces = part_surfaces[obj_idx]  # part surfaces [num_parts, P, 6]
                    pred_obj_part_meshes = all_object_meshes[obj_idx]  # part meshes
                    
                    # partsmetrics
                    for part_idx in range(len(pred_obj_part_meshes)):
                        gt_part_surface = gt_obj_part_surfaces[part_idx]  # [P, 6]
                        pred_part_mesh = pred_obj_part_meshes[part_idx]
                        
                        if pred_part_mesh is None:
                            # If the generated mesh is None (decoding error), use a dummy mesh
                            pred_part_mesh = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                            pred_obj_part_meshes[part_idx] = pred_part_mesh
                        
                        part_cd, part_f = compute_cd_and_f_score_in_training(
                            gt_part_surface, pred_part_mesh,
                            num_samples=configs['val']['metric']['cd_num_samples'],
                            threshold=configs['val']['metric']['f1_score_threshold'],
                            metric=configs['val']['metric']['cd_metric']
                        )
                        # avoid nan
                        part_cd = configs['val']['metric']['default_cd'] if np.isnan(part_cd) else part_cd
                        part_f = configs['val']['metric']['default_f1'] if np.isnan(part_f) else part_f
                        parts_chamfer_distances.append(part_cd)
                        parts_f_scores.append(part_f)

                parts_chamfer_distances = torch.tensor(parts_chamfer_distances, device=accelerator.device)
                parts_f_scores = torch.tensor(parts_f_scores, device=accelerator.device)

                metrics_dictlist[f"parts_chamfer_distance_cfg{guidance_scale:.1f}"].append(parts_chamfer_distances.mean())
                metrics_dictlist[f"parts_f_score_cfg{guidance_scale:.1f}"].append(parts_f_scores.mean())
                
                #  metric
                del all_object_meshes
                
                #  :metric
                torch.cuda.empty_cache()
                gc.collect()
            
        # Only log the last (biggest) cfg metrics in the progress bar
        val_logs = {
            "parts_chamfer_distance": parts_chamfer_distances.mean().item(),
            "parts_f_score": parts_f_scores.mean().item(),
        }
        val_progress_bar.set_postfix(**val_logs)
        logger.info(
            f"Validation [{val_step:02d}/{args.max_val_steps:02d}] " +
            f"parts_chamfer_distance: {val_logs['parts_chamfer_distance']:.4f}, parts_f_score: {val_logs['parts_f_score']:.4f}"
        )
        logger.info(
            f"parts_chamfer_distances: {[f'{x:.4f}' for x in parts_chamfer_distances.tolist()]}"
        )
        logger.info(
            f"parts_f_scores: {[f'{x:.4f}' for x in parts_f_scores.tolist()]}"
        )
        val_step += 1
        val_progress_bar.update(1)
        
        #  
        del images, part_surfaces, parts_chamfer_distances, parts_f_scores
        if 'batch' in locals():
            del batch
        
        #  OOM: GPU
        torch.cuda.empty_cache()
        gc.collect()
        
        #  GPU,
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    val_progress_bar.close()
    
    #  
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if accelerator.is_main_process:
        for key, value in medias_dictlist.items():
            if isinstance(value[0], Image.Image): # assuming gt_image
                image_grid = make_grid_for_images_or_videos(
                    value, 
                    nrow=configs['val']['nrow'],
                    return_type='pil', 
                )
                image_grid.save(os.path.join(eval_dir, f"{global_step:06d}", f"{key}.png"))
                wandb.log({f"validation/{key}": wandb.Image(image_grid)}, step=global_step)
            else: # assuming pred_rendered_images or pred_rendered_normals
                image_grids = make_grid_for_images_or_videos(
                    value, 
                    nrow=configs['val']['nrow'],
                    return_type='ndarray',
                )
                wandb.log({
                    f"validation/{key}": wandb.Video(
                        image_grids, 
                        fps=configs['val']['rendering']['fps'], 
                        format="gif"
                )}, step=global_step)
                image_grids = [Image.fromarray(image_grid.transpose(1, 2, 0)) for image_grid in image_grids]
                export_renderings(
                    image_grids, 
                    os.path.join(eval_dir, f"{global_step:06d}", f"{key}.gif"), 
                    fps=configs['val']['rendering']['fps']
                )

        for k, v in metrics_dictlist.items():
            wandb.log({f"validation/{k}": torch.tensor(v).mean().item()}, step=global_step)

if __name__ == "__main__":
    main()
