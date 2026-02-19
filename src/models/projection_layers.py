"""
Projection Layers for Contrastive Learning
encoder
"""

import math
import os

from typing import Optional

import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    """
    Projection MLP
    encoder
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 2048,
        output_dim: int = 1024,
        num_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        
        # Output layer
        if num_layers > 1:
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, D] or [B, N, D]
        Returns:
            projected features with same shape
        """
        return self.mlp(x)


class PartAwareImageAggregator(nn.Module):
    """
    token,
    """

    def __init__(
        self,
        embed_dim: int,
        num_part_labels: int,
    ):
        super().__init__()

        self.num_part_labels = num_part_labels

        self.label_embedding = nn.Embedding(num_part_labels, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, image_tokens: torch.Tensor, part_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tokens: [N, seq, D] token
            part_labels: [N] 
        Returns:
            part_features: [N, D]
        """
        if image_tokens.dim() != 3:
            raise ValueError("image_tokens  [N, seq, D] ")

        # dtype
        original_dtype = image_tokens.dtype
        # float32Linear
        image_tokens = image_tokens.float()

        #  :part_labels
        # 1. long
        part_labels = part_labels.long()
        # 2. abs,
        labels = torch.abs(part_labels) % self.num_part_labels
        # 3. clamp[0, num_part_labels-1]
        labels = torch.clamp(labels, 0, self.num_part_labels - 1)
        
        queries = self.query_proj(self.label_embedding(labels)).unsqueeze(1)  # [N, 1, D]

        keys = self.key_proj(image_tokens)  # [N, seq, D]
        values = self.value_proj(image_tokens)  # [N, seq, D]

        attn_scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(keys.size(-1))  # [N, 1, seq]
        attn_weights = torch.softmax(attn_scores, dim=-1)

        aggregated = torch.matmul(attn_weights, values).squeeze(1)  # [N, D]
        aggregated = self.out_proj(aggregated)
        aggregated = self.norm(aggregated)
        
        # dtype
        return aggregated.to(original_dtype)


class DualProjectionModule(nn.Module):
    """
    :imagemeshprojection
    
    """
    
    def __init__(
        self,
        image_dim: int = 1024,
        mesh_dim: int = 1024,
        hidden_dim: int = 2048,
        output_dim: int = 1024,
        num_layers: int = 2,
        use_part_label_embeddings: bool = False,
        num_part_labels: int = 512,
    ):
        super().__init__()
        
        self.image_projection = FeatureProjection(
            input_dim=image_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )
        
        self.mesh_projection = FeatureProjection(
            input_dim=mesh_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )

        self.use_part_label_embeddings = use_part_label_embeddings
        if use_part_label_embeddings:
            self.part_image_aggregator = PartAwareImageAggregator(
                embed_dim=image_dim,
                num_part_labels=num_part_labels,
            )
        else:
            self.part_image_aggregator = None
    
    def aggregate_image_part_features(
        self,
        image_tokens: torch.Tensor,
        part_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.part_image_aggregator is not None and part_labels is not None:
            try:
                return self.part_image_aggregator(image_tokens, part_labels)
            except (IndexError, RuntimeError) as e:
                #  ,(fallback)
                print(f"  Warning: part_image_aggregator failed ({e}), using mean pooling")
                return image_tokens.mean(dim=1)
        # 
        return image_tokens.mean(dim=1)
    
    def forward_image(self, image_features):
        """Project image features"""
        return self.image_projection(image_features)
    
    def forward_mesh(self, mesh_features):
        """Project mesh features"""
        return self.mesh_projection(mesh_features)
    
    def forward(self, image_features, mesh_features):
        """
        Project both image and mesh features
        
        Args:
            image_features: image embeddings
            mesh_features: mesh embeddings
            
        Returns:
            projected_image, projected_mesh
        """
        projected_image = self.forward_image(image_features)
        projected_mesh = self.forward_mesh(mesh_features)
        return projected_image, projected_mesh
    
    def save_pretrained(self, output_dir):
        """
        Save the projection module to a directory (HuggingFace compatible)
        
        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save state dict
        model_path = os.path.join(output_dir, "pytorch_model.pt")
        torch.save(self.state_dict(), model_path)
        
        # Save config for reconstruction
        config = {
            'image_dim': self.image_projection.mlp[0].in_features,
            'mesh_dim': self.mesh_projection.mlp[0].in_features,
            'output_dim': self.image_projection.mlp[-1].out_features if hasattr(self.image_projection.mlp[-1], 'out_features') else self.image_projection.mlp[0].in_features,
            'use_part_label_embeddings': self.use_part_label_embeddings,
        }
        
        if self.use_part_label_embeddings and self.part_image_aggregator is not None:
            config['num_part_labels'] = self.part_image_aggregator.num_part_labels
        
        config_path = os.path.join(output_dir, "config.pt")
        torch.save(config, config_path)
        
        print(f" DualProjectionModule saved to {output_dir}")
    
    @classmethod
    def from_pretrained(cls, pretrained_path, **override_kwargs):
        """
        Load the projection module from a directory (HuggingFace compatible)
        
        Args:
            pretrained_path: Directory containing the saved model
            **override_kwargs: Optional config overrides
            
        Returns:
            DualProjectionModule instance
        """
        config_path = os.path.join(pretrained_path, "config.pt")
        model_path = os.path.join(pretrained_path, "pytorch_model.pt")
        
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find config.pt or pytorch_model.pt in {pretrained_path}")
        
        # Load config
        config = torch.load(config_path)
        config.update(override_kwargs)  # Allow overriding
        
        # Create model
        model = cls(**config)
        
        # Load state dict
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        
        print(f" DualProjectionModule loaded from {pretrained_path}")
        return model


if __name__ == "__main__":
    # Test
    batch_size = 16
    seq_len = 257
    dim = 1024
    
    # Test FeatureProjection
    proj = FeatureProjection(input_dim=dim, output_dim=dim)
    x = torch.randn(batch_size, seq_len, dim)
    out = proj(x)
    print(f"FeatureProjection: {x.shape} -> {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in proj.parameters()) / 1e6:.2f}M")
    
    # Test DualProjectionModule
    dual_proj = DualProjectionModule()
    img_feat = torch.randn(batch_size, dim)
    mesh_feat = torch.randn(batch_size, dim)
    proj_img, proj_mesh = dual_proj(img_feat, mesh_feat)
    print(f"\nDualProjection:")
    print(f"  Image: {img_feat.shape} -> {proj_img.shape}")
    print(f"  Mesh: {mesh_feat.shape} -> {proj_mesh.shape}")
    print(f"  Total Parameters: {sum(p.numel() for p in dual_proj.parameters()) / 1e6:.2f}M")

