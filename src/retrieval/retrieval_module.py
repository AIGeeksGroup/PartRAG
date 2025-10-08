"""
检索模块 - 参考 ReMoMask 的检索增强方法
用于从数据库中检索相关的物体/图像来指导3D生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple
from PIL import Image
import numpy as np
from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor, CLIPTextModel, CLIPVisionModel
import logging

logger = logging.getLogger(__name__)


class RetrievalModule(nn.Module):
    """
    检索模块，用于检索相关的图像来增强生成过程
    参考 ReMoMask 的双向动量对比学习方法
    """
    
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        database_embeddings: Optional[torch.Tensor] = None,
        database_images: Optional[List] = None,
        device: str = "cuda",
    ):
        """
        Args:
            clip_model_name: CLIP模型名称
            database_embeddings: 预计算的数据库图像embeddings [N, D]
            database_images: 数据库图像列表
            device: 设备
        """
        super().__init__()
        
        self.device = device
        
        # 加载CLIP模型用于检索
        logger.info(f"Loading CLIP model: {clip_model_name}")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        
        # 数据库
        self.database_embeddings = database_embeddings  # [N, D]
        self.database_images = database_images  # List of images
        
        if database_embeddings is not None:
            self.database_embeddings = database_embeddings.to(device)
            logger.info(f"Loaded database with {len(database_embeddings)} embeddings")
    
    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        使用CLIP编码图像
        
        Args:
            images: PIL图像或图像列表
            
        Returns:
            image_embeddings: [B, D]
        """
        if not isinstance(images, list):
            images = [images]
        
        with torch.no_grad():
            inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self.clip_model.get_image_features(**inputs)
            # 归一化
            image_features = F.normalize(image_features, p=2, dim=-1)
        
        return image_features
    
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        使用CLIP编码文本
        
        Args:
            texts: 文本或文本列表
            
        Returns:
            text_embeddings: [B, D]
        """
        if not isinstance(texts, list):
            texts = [texts]
        
        with torch.no_grad():
            inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.clip_model.get_text_features(**inputs)
            # 归一化
            text_features = F.normalize(text_features, p=2, dim=-1)
        
        return text_features
    
    def retrieve_by_image(
        self, 
        query_image: Union[Image.Image, torch.Tensor],
        top_k: int = 1,
        return_scores: bool = False,
    ) -> Union[List[Image.Image], Tuple[List[Image.Image], torch.Tensor]]:
        """
        根据查询图像检索最相关的数据库图像
        
        Args:
            query_image: 查询图像
            top_k: 返回top-k个最相关的图像
            return_scores: 是否返回相似度分数
            
        Returns:
            retrieved_images: 检索到的图像列表
            scores (optional): 相似度分数
        """
        if self.database_embeddings is None or self.database_images is None:
            logger.warning("Database not loaded, returning None")
            return None
        
        # 编码查询图像
        if isinstance(query_image, torch.Tensor):
            query_embedding = query_image
        else:
            query_embedding = self.encode_image(query_image)  # [1, D]
        
        # 计算相似度
        # query_embedding: [1, D], database_embeddings: [N, D]
        similarities = torch.matmul(query_embedding, self.database_embeddings.T)  # [1, N]
        
        # 获取top-k
        top_k = min(top_k, len(self.database_images))
        scores, indices = torch.topk(similarities, k=top_k, dim=-1)  # [1, top_k]
        
        # 获取检索到的图像
        retrieved_images = []
        for idx in indices[0]:
            retrieved_images.append(self.database_images[idx.item()])
        
        if return_scores:
            return retrieved_images, scores[0]
        else:
            return retrieved_images
    
    def retrieve_by_text(
        self,
        query_text: str,
        top_k: int = 1,
        return_scores: bool = False,
    ) -> Union[List[Image.Image], Tuple[List[Image.Image], torch.Tensor]]:
        """
        根据查询文本检索最相关的数据库图像
        
        Args:
            query_text: 查询文本
            top_k: 返回top-k个最相关的图像
            return_scores: 是否返回相似度分数
            
        Returns:
            retrieved_images: 检索到的图像列表
            scores (optional): 相似度分数
        """
        if self.database_embeddings is None or self.database_images is None:
            logger.warning("Database not loaded, returning None")
            return None
        
        # 编码查询文本
        query_embedding = self.encode_text(query_text)  # [1, D]
        
        # 计算相似度
        similarities = torch.matmul(query_embedding, self.database_embeddings.T)  # [1, N]
        
        # 获取top-k
        top_k = min(top_k, len(self.database_images))
        scores, indices = torch.topk(similarities, k=top_k, dim=-1)  # [1, top_k]
        
        # 获取检索到的图像
        retrieved_images = []
        for idx in indices[0]:
            retrieved_images.append(self.database_images[idx.item()])
        
        if return_scores:
            return retrieved_images, scores[0]
        else:
            return retrieved_images
    
    def build_database(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
    ):
        """
        构建图像数据库的embeddings
        
        Args:
            images: 图像列表
            batch_size: 批处理大小
        """
        logger.info(f"Building database with {len(images)} images...")
        
        self.database_images = images
        embeddings_list = []
        
        # 批量处理
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_embeddings = self.encode_image(batch_images)
            embeddings_list.append(batch_embeddings)
        
        # 合并所有embeddings
        self.database_embeddings = torch.cat(embeddings_list, dim=0)
        logger.info(f"Database built with shape: {self.database_embeddings.shape}")
    
    def save_database(self, save_path: str):
        """保存数据库embeddings"""
        if self.database_embeddings is not None:
            torch.save({
                'embeddings': self.database_embeddings.cpu(),
                'num_images': len(self.database_images) if self.database_images else 0,
            }, save_path)
            logger.info(f"Database saved to {save_path}")
    
    def load_database(self, load_path: str, images: Optional[List[Image.Image]] = None):
        """
        加载数据库embeddings
        
        Args:
            load_path: embeddings文件路径
            images: 对应的图像列表（可选）
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        self.database_embeddings = checkpoint['embeddings'].to(self.device)
        
        if images is not None:
            self.database_images = images
            assert len(images) == checkpoint['num_images'], \
                f"Number of images ({len(images)}) doesn't match saved embeddings ({checkpoint['num_images']})"
        
        logger.info(f"Database loaded from {load_path} with {len(self.database_embeddings)} embeddings")


class RetrievalAugmentedEncoder(nn.Module):
    """
    检索增强编码器
    将检索到的图像与prompt图像进行融合
    """
    
    def __init__(
        self,
        image_encoder,
        feature_extractor,
        retrieval_module: Optional[RetrievalModule] = None,
        num_retrieved: int = 1,
    ):
        """
        Args:
            image_encoder: 图像编码器（如 DINOv2）
            feature_extractor: 特征提取器
            retrieval_module: 检索模块
            num_retrieved: 检索图像数量
        """
        super().__init__()
        
        self.image_encoder = image_encoder
        self.feature_extractor = feature_extractor
        self.retrieval_module = retrieval_module
        self.num_retrieved = num_retrieved
    
    def encode_with_retrieval(
        self,
        prompt_images: Union[Image.Image, List[Image.Image]],
        query_text: Optional[str] = None,
        use_retrieval: bool = True,
        device: str = "cuda",
        num_images_per_prompt: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码图像并进行检索增强
        
        Args:
            prompt_images: 输入的prompt图像
            query_text: 可选的查询文本
            use_retrieval: 是否使用检索增强
            device: 设备
            num_images_per_prompt: 每个prompt的图像数量
            
        Returns:
            image_embeds: 增强后的图像embeddings
            uncond_image_embeds: 无条件图像embeddings
        """
        dtype = next(self.image_encoder.parameters()).dtype
        
        # 处理输入图像
        if not isinstance(prompt_images, list):
            prompt_images = [prompt_images]
        
        # 1. 编码prompt图像
        if not isinstance(prompt_images[0], torch.Tensor):
            prompt_pixel_values = self.feature_extractor(prompt_images, return_tensors="pt").pixel_values
        else:
            prompt_pixel_values = torch.stack(prompt_images)
        
        prompt_pixel_values = prompt_pixel_values.to(device=device, dtype=dtype)
        prompt_embeds = self.image_encoder(prompt_pixel_values).last_hidden_state
        
        # 2. 检索相关图像（如果启用）
        if use_retrieval and self.retrieval_module is not None:
            retrieved_embeds_list = []
            
            for prompt_image in prompt_images:
                # 检索图像
                if query_text is not None:
                    retrieved_images = self.retrieval_module.retrieve_by_text(
                        query_text, top_k=self.num_retrieved
                    )
                else:
                    retrieved_images = self.retrieval_module.retrieve_by_image(
                        prompt_image, top_k=self.num_retrieved
                    )
                
                if retrieved_images is not None:
                    # 编码检索到的图像
                    retrieved_pixel_values = self.feature_extractor(
                        retrieved_images, return_tensors="pt"
                    ).pixel_values
                    retrieved_pixel_values = retrieved_pixel_values.to(device=device, dtype=dtype)
                    retrieved_embeds = self.image_encoder(retrieved_pixel_values).last_hidden_state
                    retrieved_embeds_list.append(retrieved_embeds)
            
            if len(retrieved_embeds_list) > 0:
                # 将所有检索到的图像embeddings concatenate起来
                retrieved_embeds = torch.cat(retrieved_embeds_list, dim=0)  # [B*num_retrieved, seq_len, D]
                
                # 3. 融合prompt embeddings和retrieved embeddings
                # 方法1: 直接concatenate在序列维度上（类似ReMoMask）
                # prompt_embeds: [B, seq_len, D]
                # retrieved_embeds: [B*num_retrieved, seq_len, D]
                
                # 重复prompt embeddings以匹配检索图像的数量
                batch_size = len(prompt_images)
                if self.num_retrieved > 0:
                    # 对每个prompt，将其embedding与对应的retrieved embeddings concatenate
                    fused_embeds_list = []
                    for i in range(batch_size):
                        p_embed = prompt_embeds[i:i+1]  # [1, seq_len, D]
                        r_embeds = retrieved_embeds[i*self.num_retrieved:(i+1)*self.num_retrieved]  # [num_retrieved, seq_len, D]
                        # Concatenate: [retrieved_embeds, prompt_embeds]
                        fused = torch.cat([r_embeds, p_embed], dim=1)  # [num_retrieved, seq_len*2, D]
                        # 取平均或者concatenate
                        fused = fused.mean(dim=0, keepdim=True)  # [1, seq_len*2, D]
                        fused_embeds_list.append(fused)
                    
                    image_embeds = torch.cat(fused_embeds_list, dim=0)  # [B, seq_len*2, D]
                else:
                    image_embeds = prompt_embeds
            else:
                image_embeds = prompt_embeds
        else:
            image_embeds = prompt_embeds
        
        # 重复以匹配num_images_per_prompt
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)
        
        return image_embeds, uncond_image_embeds



