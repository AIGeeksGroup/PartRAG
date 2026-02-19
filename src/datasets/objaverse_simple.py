"""
Simple Objaverse Dataset for single objects (no part segmentation)
Treats each object as a single part
"""

from src.utils.typing_utils import *

import json
import os
import random

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

class ObjaverseSimpleDataset(torch.utils.data.Dataset):
    """
    Simple dataset for Objaverse objects without part segmentation.
    Each object is treated as a single part.
    """
    
    def __init__(
        self, 
        configs: DictConfig, 
        training: bool = True, 
    ):
        super().__init__()
        self.configs = configs
        self.training = training
        
        # Load data config
        config_paths = configs['dataset']['config']
        # Handle ListConfig from OmegaConf
        try:
            # Try to iterate (works for list and ListConfig)
            config_paths = list(config_paths)
        except TypeError:
            # If it's a single string
            config_paths = [config_paths]
        
        data_list = []
        for config_path in config_paths:
            with open(config_path, 'r') as f:
                data_list.extend(json.load(f))
        
        # Filter valid samples
        data_list = [d for d in data_list if d.get('valid', True)]
        
        # Split train/val
        training_ratio = configs['dataset']['training_ratio']
        split_idx = int(len(data_list) * training_ratio)
        
        if training:
            self.data_list = data_list[:split_idx]
        else:
            self.data_list = data_list[split_idx:]
        
        # Data augmentation
        self.rotating_ratio = configs['dataset'].get('rotating_ratio', 0.0)
        self.rotating_degree = configs['dataset'].get('ratating_degree', 10.0)  # typo in config
        self.transform = transforms.Compose([
            transforms.RandomRotation(
                degrees=(-self.rotating_degree, self.rotating_degree), 
                fill=(255, 255, 255)
            ),
        ])
        
        self.image_size = (512, 512)
        self.num_surface_samples = configs['model']['vae']['num_tokens']
        
        print(f"{'Train' if training else 'Val'} dataset: {len(self.data_list)} samples")
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.data_list[idx]
        
        # Load image
        image_path = data['image_path']  # Use image_path field from config
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Apply augmentation during training
        if self.training and random.random() < self.rotating_ratio:
            image = self.transform(image)
        
        # Convert to tensor [3, H, W]
        image = transforms.ToTensor()(image)
        
        # Load point cloud
        points_path = data['surface_path']  # Use surface_path field from config
        points_data = np.load(points_path, allow_pickle=True)
        
        # Extract object data (whole object, not parts)
        if points_data.dtype == object:
            points_dict = points_data.item()
            # Use the 'object' key which contains the full object
            if 'object' in points_dict:
                obj_data = points_dict['object']
                # Combine surface_points and surface_normals
                surface_points = np.array(obj_data['surface_points'])
                surface_normals = np.array(obj_data['surface_normals'])
                points = np.concatenate([surface_points, surface_normals], axis=1)  # [N, 6]
            else:
                raise ValueError(f"Unexpected points_data structure: {points_dict.keys()}")
        else:
            points = points_data
        
        # Sample points
        if len(points) > self.num_surface_samples:
            indices = np.random.choice(len(points), self.num_surface_samples, replace=False)
            points = points[indices]
        elif len(points) < self.num_surface_samples:
            # Pad with duplicates
            indices = np.random.choice(len(points), self.num_surface_samples, replace=True)
            points = points[indices]
        
        # Convert to tensor [num_tokens, 6] -> [1, num_tokens, 6] (1 part)
        surface = torch.from_numpy(points).float()
        surface = surface.unsqueeze(0)  # Add part dimension
        
        # Create part labels (single part = "object")
        part_labels = ["object"]
        
        # Extract uid from file field (remove .glb extension)
        uid = data['file'].replace('.glb', '') if 'file' in data else 'unknown'
        
        return {
            'images': image.unsqueeze(0),  # [1, 3, H, W]
            'part_surfaces': surface,  # [1, num_tokens, 6]
            'part_labels': part_labels,  # ["object"]
            'uid': uid,
        }
    
    def collate_fn(self, batch):
        """Collate function for DataLoader"""
        # Filter out empty items
        batch = [data for data in batch if len(data) > 0]
        if len(batch) == 0:
            return {}
        
        #  :List[PIL.Image]tensor
        images = []
        for data in batch:
            # data['images']  tensor [1, 3, H, W]  PIL Image
            img = data['images']
            if isinstance(img, torch.Tensor):
                # PIL Image
                if img.shape[0] == 1:
                    img = img[0]  # [3, H, W]
                img = img.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
                img = (img * 255).astype(np.uint8)
                from PIL import Image
                img = Image.fromarray(img)
            images.append(img)
        
        # Concatenate all data
        surfaces = torch.cat([data['part_surfaces'] for data in batch], dim=0)  # [B, num_tokens, 6]
        part_labels = []
        num_parts = []
        for data in batch:
            part_labels.extend(data['part_labels'])
            num_parts.append(len(data['part_labels']))
        
        return {
            'images': images,  # List[PIL.Image] 
            'part_surfaces': surfaces,
            'num_parts': torch.LongTensor(num_parts),
            'part_labels': part_labels,
        }


# Batched version
class BatchedObjaverseSimpleDataset(ObjaverseSimpleDataset):
    """
    Batched version that returns multiple parts by repeating the single object
    to match the expected training format.
    """
    
    def __init__(self, configs: DictConfig, training: bool = True):
        super().__init__(configs, training)
        
        self.min_num_parts = configs['dataset']['min_num_parts']
        self.max_num_parts = configs['dataset']['max_num_parts']
        self.object_ratio = configs['dataset'].get('object_ratio', 0.3)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = super().__getitem__(idx)
        
        # Randomly decide number of "parts" (actually we duplicate the object)
        # With object_ratio chance, use 1 part (object mode)
        # Otherwise, use random number of parts
        if random.random() < self.object_ratio:
            num_parts = 1
        else:
            num_parts = random.randint(self.min_num_parts, self.max_num_parts)
        
        # Duplicate the surface to create multiple "parts"
        surface = data['part_surfaces']  # [1, num_tokens, 6]
        images = data['images']  # [1, 3, H, W]
        
        if num_parts > 1:
            # Repeat the surface
            surface = surface.repeat(num_parts, 1, 1)  # [num_parts, num_tokens, 6]
            # Add small noise to make parts slightly different
            noise = torch.randn_like(surface) * 0.01
            surface = surface + noise
            
            # Also repeat the image for each part
            images = images.repeat(num_parts, 1, 1, 1)  # [num_parts, 3, H, W]
            
            part_labels = [f"part_{i}" for i in range(num_parts)]
        else:
            part_labels = ["object"]
        
        data['images'] = images
        data['part_surfaces'] = surface
        data['part_labels'] = part_labels
        
        return data

