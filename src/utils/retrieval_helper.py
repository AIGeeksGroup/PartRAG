"""
Retrieval Helper for Training - FIXED VERSION
:
- Database: mean-pooled features [N, D] 
- : full sequence features [N, seq, D] cross-attention
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple


class RetrievalHelper:
    """
    Retrieval - :
    1. Databaseglobal feature()
    2. indicesimage/meshsequence features
    3. global feature broadcastsequence
    """
    
    def __init__(
        self,
        database_path: str,
        device: str = "cuda",
        enabled: bool = True,
        use_fused_embeddings: bool = True,
        use_sequence_broadcast: bool = True,  # :sequence
    ):
        """
        Args:
            database_path: database
            device: 
            enabled: retrieval
            use_fused_embeddings: fused embeddings(image+mesh)
            use_sequence_broadcast: global featuressequence
        """
        self.enabled = enabled
        self.device = device
        self.use_sequence_broadcast = use_sequence_broadcast
        self.database_path = database_path
        
        if not enabled:
            return
        
        # embeddings
        # fused (global) 
        embedding_file = "fused_embeddings.npy" if use_fused_embeddings else "image_embeddings.npy"
        embeddings_path = os.path.join(database_path, embedding_file)
        
        # database(embeddings.npy)
        if not os.path.exists(embeddings_path):
            embeddings_path = os.path.join(database_path, "embeddings.npy")
            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(f"Cannot find embeddings in {database_path}")
        
        self.database_embeddings = torch.from_numpy(
            np.load(embeddings_path)
        ).to(device).float()  # [N, D] or [N, seq, D]

        # Optional DINO global fallback for dimension-safe retrieval when fused dims differ.
        dino_global_path = os.path.join(database_path, "dino_embeddings.npy")
        if os.path.exists(dino_global_path):
            self.dino_global_embeddings = torch.from_numpy(
                np.load(dino_global_path)
            ).to(device).float()
            self.dino_global_embeddings = F.normalize(self.dino_global_embeddings, p=2, dim=-1)
        else:
            self.dino_global_embeddings = None
        
        # sequence embeddings
        image_seq_path = os.path.join(database_path, "image_embeddings.npy")
        self.has_sequence_embeddings = os.path.exists(image_seq_path)
        if self.has_sequence_embeddings:
            # sequence
            test_arr = np.load(image_seq_path, mmap_mode='r')
            self.has_sequence_embeddings = (test_arr.ndim == 3)
        
        # 
        if self.database_embeddings.dim() == 2:
            self.database_embeddings = F.normalize(self.database_embeddings, p=2, dim=-1)
        else:
            # 3D,
            self.database_embeddings = F.normalize(self.database_embeddings, p=2, dim=-1)
        
        # metadata
        metadata_path = os.path.join(database_path, "metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.database_image_paths = self.metadata["image_paths"]
        self.database_mesh_paths = self.metadata["mesh_paths"]
        self.database_uids = self.metadata["uids"]
        
        print(f" Retrieval database loaded: {len(self.database_embeddings)} samples")
        print(f"   Embedding file: {embedding_file}")
        print(f"   Shape: {self.database_embeddings.shape}")
        print(f"   Has sequence embeddings: {self.has_sequence_embeddings}")
        print(f"   Use sequence broadcast: {use_sequence_broadcast}")
    
    def retrieve(
        self,
        query_embeddings: torch.Tensor,  # [B, D] or [B, seq, D]
        top_k: int = 3,
        target_seq_len: int = 256,  # sequence(DINOv2256)
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        
        
        Args:
            query_embeddings: embeddings [B, D] or [B, seq, D]
            top_k: top-k
            target_seq_len: sequence
            
        Returns:
            retrieved_embeddings: [B, top_k, seq_len, D] if broadcast, else [B, top_k, D]
            retrieved_indices: List[List[int]] - querytop-k indices
        """
        if not self.enabled:
            return None, None
        
        # ==================== Step 1: query ====================
        # [B, seq, D],mean(spatial)
        if query_embeddings.dim() == 3:
            query_global = query_embeddings.mean(dim=1)  # [B, D]
        else:
            query_global = query_embeddings  # [B, D]
        
        # dtypedatabase
        query_global = query_global.to(dtype=self.database_embeddings.dtype, device=self.device)
        
        # 
        query_global = F.normalize(query_global, p=2, dim=-1)
        
        # ==================== Step 2:  ====================
        # databasesequence,mean
        if self.database_embeddings.dim() == 3:
            db_for_search = self.database_embeddings.mean(dim=1)  # [N, D]
        else:
            db_for_search = self.database_embeddings  # [N, D]

        # If fused retrieval dim != query dim, fall back to DINO/global-compatible features.
        if db_for_search.shape[-1] != query_global.shape[-1]:
            if self.dino_global_embeddings is not None and self.dino_global_embeddings.shape[-1] == query_global.shape[-1]:
                db_for_search = self.dino_global_embeddings
            elif self.has_sequence_embeddings:
                image_seq_path = os.path.join(self.database_path, "image_embeddings.npy")
                image_seq_db = torch.from_numpy(np.load(image_seq_path)).to(self.device).float()
                seq_global = F.normalize(image_seq_db.mean(dim=1), p=2, dim=-1)
                if seq_global.shape[-1] == query_global.shape[-1]:
                    db_for_search = seq_global
                else:
                    raise ValueError(
                        f"Retrieval feature dim mismatch: query={query_global.shape[-1]}, database={db_for_search.shape[-1]}"
                    )
            else:
                raise ValueError(
                    f"Retrieval feature dim mismatch: query={query_global.shape[-1]}, database={db_for_search.shape[-1]}"
                )
        
        #  [B, N]
        similarities = torch.matmul(
            query_global,  # [B, D]
            db_for_search.t()  # [D, N]
        )  # [B, N]
        
        # top-k indices
        topk_values, topk_indices = torch.topk(similarities, k=top_k, dim=-1)  # [B, top_k]
        
        # ==================== Step 3: retrieved embeddings ====================
        batch_size = query_global.shape[0]
        retrieved_embeddings = []
        retrieved_indices_list = []
        
        for b in range(batch_size):
            indices = topk_indices[b].tolist()  # [top_k]
            retrieved_indices_list.append(indices)
            
            # ==================== Step 4: sequence embeddings ====================
            if self.has_sequence_embeddings:
                # A: sequence embeddings()
                image_seq_path = os.path.join(self.database_path, "image_embeddings.npy")
                image_seq_db = torch.from_numpy(np.load(image_seq_path)).to(self.device).float()
                batch_retrieved_seq = image_seq_db[indices]  # [top_k, seq, D]
                batch_retrieved_seq = F.normalize(batch_retrieved_seq, p=2, dim=-1)
                retrieved_embeddings.append(batch_retrieved_seq)
            else:
                # B: broadcast(sequence)
                batch_retrieved_global = db_for_search[indices]  # [top_k, D]
                if self.use_sequence_broadcast:
                    batch_retrieved_seq = batch_retrieved_global.unsqueeze(1).expand(
                        -1, target_seq_len, -1
                    )  # [top_k, seq_len, D]
                    retrieved_embeddings.append(batch_retrieved_seq)
                else:
                    retrieved_embeddings.append(batch_retrieved_global)
        
        if self.has_sequence_embeddings or self.use_sequence_broadcast:
            retrieved_embeddings = torch.stack(retrieved_embeddings, dim=0)  # [B, top_k, seq_len, D]
        else:
            retrieved_embeddings = torch.stack(retrieved_embeddings, dim=0)  # [B, top_k, D]
        
        return retrieved_embeddings, retrieved_indices_list
    
    def get_retrieved_images(self, indices: List[int]) -> List[Image.Image]:
        """
        indices
        
        Args:
            indices: List of indices
            
        Returns:
            List of PIL Images
        """
        images = []
        for idx in indices:
            image_path = self.database_image_paths[idx]
            if os.path.exists(image_path):
                img = Image.open(image_path).convert('RGB')
                images.append(img)
        return images
    
    def get_retrieved_mesh_paths(self, indices: List[int]) -> List[str]:
        """
        indicesmesh
        
        Args:
            indices: List of indices
            
        Returns:
            List of mesh paths
        """
        return [self.database_mesh_paths[idx] for idx in indices]


def concat_retrieved_to_encoder_hidden_states(
    encoder_hidden_states: torch.Tensor,  # [B, seq, D]
    retrieved_embeddings: Optional[torch.Tensor],  # [B, top_k, seq, D] or [B, top_k, D]
) -> torch.Tensor:
    """
    retrieved embeddings concatencoder_hidden_states - 
    
    Args:
        encoder_hidden_states: image embeddings [B, seq, D]
        retrieved_embeddings: embeddings 
                            [B, top_k, seq, D] (if sequence broadcast)
                            or [B, top_k, D] (if global)
        
    Returns:
        concatenated embeddings [B, seq + top_k*seq, D] or [B, seq + top_k, D]
    """
    if retrieved_embeddings is None:
        return encoder_hidden_states
    
    batch_size = encoder_hidden_states.shape[0]
    
    if retrieved_embeddings.dim() == 4:
        # Case 1: Retrievedsequence features [B, top_k, seq, D]
        # reshape: [B, top_k, seq, D] -> [B, top_k*seq, D]
        top_k, seq_len, embed_dim = retrieved_embeddings.shape[1:]
        retrieved_flat = retrieved_embeddings.reshape(batch_size, top_k * seq_len, embed_dim)
        
        # Concatenate: [retrieved_seq, prompt_seq]
        concatenated = torch.cat([
            retrieved_flat,  # [B, top_k*seq, D]
            encoder_hidden_states  # [B, seq, D]
        ], dim=1)  # [B, top_k*seq + seq, D]
        
    elif retrieved_embeddings.dim() == 3:
        # Case 2: Retrievedglobal features [B, top_k, D]
        # concat
        concatenated = torch.cat([
            retrieved_embeddings,  # [B, top_k, D]
            encoder_hidden_states   # [B, seq, D]
        ], dim=1)  # [B, top_k + seq, D]
    
    else:
        raise ValueError(f"Unexpected retrieved_embeddings dim: {retrieved_embeddings.dim()}")
    
    return concatenated
