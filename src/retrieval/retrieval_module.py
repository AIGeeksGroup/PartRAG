"""Retrieval module for PartRAG with optional FAISS acceleration."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover - optional dependency
    faiss = None
    _HAS_FAISS = False


def _l2_normalize_np(x: np.ndarray) -> np.ndarray:
    eps = 1e-12
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _as_image(x: Union[Image.Image, torch.Tensor, np.ndarray]) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().float()
        if arr.ndim == 4:
            if arr.shape[0] != 1:
                raise ValueError(f"Expected single image tensor, got shape {tuple(arr.shape)}")
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"Cannot convert tensor with shape {tuple(arr.shape)} to image")
        if arr.shape[0] in (1, 3):
            arr = arr.permute(1, 2, 0)
        arr = arr.numpy()
    else:
        arr = np.asarray(x)

    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    if arr.ndim != 3:
        raise ValueError(f"Cannot convert array with shape {arr.shape} to image")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr).convert("RGB")


class RetrievalModule(nn.Module):
    """CLIP-based retrieval over an offline PartRAG database."""

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        database_embeddings: Optional[torch.Tensor] = None,
        database_images: Optional[List[Image.Image]] = None,
        database_mesh_paths: Optional[List[str]] = None,
        device: str = "cuda",
        use_faiss: bool = True,
        load_images_eagerly: bool = False,
    ):
        super().__init__()

        self.device = device
        self.use_faiss = use_faiss and _HAS_FAISS
        self.load_images_eagerly = load_images_eagerly

        logger.info("Loading CLIP model: %s", clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        self.database_embeddings: Optional[torch.Tensor] = None
        self.database_images: Optional[List[Optional[Image.Image]]] = database_images
        self.database_image_paths: List[str] = []
        self.database_mesh_paths: List[str] = database_mesh_paths or []
        self.database_uids: List[str] = []
        self.database_dir: Optional[str] = None

        self._faiss_index = None

        if database_embeddings is not None:
            self._set_database_embeddings(database_embeddings)
            self._maybe_build_faiss_index()

    def _set_database_embeddings(self, embeddings: Union[torch.Tensor, np.ndarray]) -> None:
        if isinstance(embeddings, np.ndarray):
            emb = torch.from_numpy(embeddings)
        else:
            emb = embeddings

        if emb.ndim == 3:
            emb = emb.mean(dim=1)
        if emb.ndim != 2:
            raise ValueError(f"database_embeddings must be 2D or 3D, got shape {tuple(emb.shape)}")

        emb = emb.to(self.device).float()
        emb = F.normalize(emb, p=2, dim=-1)
        self.database_embeddings = emb

    def _maybe_build_faiss_index(self) -> None:
        if not self.use_faiss or self.database_embeddings is None:
            return

        db_np = self.database_embeddings.detach().cpu().float().numpy()
        dim = db_np.shape[1]

        if db_np.shape[0] >= 4096:
            nlist = min(4096, max(128, int(np.sqrt(db_np.shape[0]) * 2)))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            if not index.is_trained:
                index.train(db_np)
            index.add(db_np)
            index.nprobe = min(32, nlist)
        else:
            index = faiss.IndexFlatIP(dim)
            index.add(db_np)

        self._faiss_index = index

    def _search(self, query_embeddings: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.database_embeddings is None:
            raise RuntimeError("Retrieval database is not loaded")

        num_db = self.database_embeddings.shape[0]
        top_k = max(1, min(int(top_k), num_db))
        query = F.normalize(query_embeddings.float(), p=2, dim=-1)

        if self._faiss_index is not None:
            q = query.detach().cpu().numpy().astype(np.float32)
            scores_np, indices_np = self._faiss_index.search(q, top_k)
            scores = torch.from_numpy(scores_np).to(query.device)
            indices = torch.from_numpy(indices_np).to(query.device)
            return scores, indices

        scores = torch.matmul(query, self.database_embeddings.T)
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
        return top_scores, top_indices

    @torch.no_grad()
    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]
        images = [_as_image(x) for x in images]

        inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_features = self.clip_model.get_image_features(**inputs)
        return F.normalize(image_features, p=2, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if not isinstance(texts, list):
            texts = [texts]

        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_features = self.clip_model.get_text_features(**inputs)
        return F.normalize(text_features, p=2, dim=-1)

    def _load_images_for_indices(self, indices: List[int]) -> List[Optional[Image.Image]]:
        if self.database_images is not None and len(self.database_images) >= max(indices) + 1:
            return [self.database_images[i] for i in indices]

        images: List[Optional[Image.Image]] = []
        for idx in indices:
            img = None
            if idx < len(self.database_image_paths):
                image_path = self.database_image_paths[idx]
                if image_path and os.path.exists(image_path):
                    try:
                        img = Image.open(image_path).convert("RGB")
                    except Exception as exc:  # pragma: no cover - IO failure
                        logger.warning("Failed to load image %s: %s", image_path, exc)
            images.append(img)
        return images

    def _format_result(
        self,
        scores: torch.Tensor,
        indices: torch.Tensor,
        return_scores: bool,
        return_dict: bool,
    ):
        first_indices = indices[0].tolist()
        retrieved_images = self._load_images_for_indices(first_indices)
        retrieved_mesh_paths = [
            self.database_mesh_paths[i] if i < len(self.database_mesh_paths) else None
            for i in first_indices
        ]
        retrieved_uids = [
            self.database_uids[i] if i < len(self.database_uids) else None
            for i in first_indices
        ]

        if return_dict:
            out: Dict[str, Union[List, torch.Tensor]] = {
                "indices": first_indices,
                "images": retrieved_images,
                "mesh_paths": retrieved_mesh_paths if len(self.database_mesh_paths) > 0 else None,
                "uids": retrieved_uids if len(self.database_uids) > 0 else None,
            }
            if return_scores:
                out["scores"] = scores[0]
            return out

        if return_scores:
            return retrieved_images, scores[0]
        return retrieved_images

    @torch.no_grad()
    def retrieve_by_image(
        self,
        query_image: Union[Image.Image, torch.Tensor],
        top_k: int = 3,
        return_scores: bool = False,
        return_dict: bool = True,
    ):
        if self.database_embeddings is None:
            logger.warning("Database not loaded, returning None")
            return None

        if isinstance(query_image, torch.Tensor):
            if query_image.ndim == 2 and query_image.shape[-1] == self.database_embeddings.shape[-1]:
                query_embedding = F.normalize(query_image.to(self.device).float(), p=2, dim=-1)
            elif query_image.ndim == 1 and query_image.shape[0] == self.database_embeddings.shape[-1]:
                query_embedding = F.normalize(query_image.unsqueeze(0).to(self.device).float(), p=2, dim=-1)
            else:
                query_embedding = self.encode_image(_as_image(query_image))
        else:
            query_embedding = self.encode_image(query_image)

        scores, indices = self._search(query_embedding, top_k=top_k)
        return self._format_result(scores, indices, return_scores=return_scores, return_dict=return_dict)

    @torch.no_grad()
    def retrieve_by_text(
        self,
        query_text: str,
        top_k: int = 3,
        return_scores: bool = False,
        return_dict: bool = True,
    ):
        if self.database_embeddings is None:
            logger.warning("Database not loaded, returning None")
            return None

        query_embedding = self.encode_text(query_text)
        scores, indices = self._search(query_embedding, top_k=top_k)
        return self._format_result(scores, indices, return_scores=return_scores, return_dict=return_dict)

    def build_database(self, images: List[Image.Image], batch_size: int = 32):
        logger.info("Building retrieval database from %d in-memory images", len(images))
        self.database_images = images
        self.database_image_paths = []
        self.database_mesh_paths = []
        self.database_uids = []

        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            all_embeddings.append(self.encode_image(batch))

        self.database_embeddings = torch.cat(all_embeddings, dim=0)
        self._maybe_build_faiss_index()
        logger.info("Database built with shape %s", tuple(self.database_embeddings.shape))

    def save_database(self, save_path: str):
        if self.database_embeddings is None:
            raise RuntimeError("No database embeddings to save")

        payload = {
            "embeddings": self.database_embeddings.detach().cpu(),
            "image_paths": self.database_image_paths,
            "mesh_paths": self.database_mesh_paths,
            "uids": self.database_uids,
            "num_images": len(self.database_image_paths) if self.database_image_paths else len(self.database_embeddings),
        }
        torch.save(payload, save_path)
        logger.info("Database checkpoint saved to %s", save_path)

    def _load_database_directory(
        self,
        database_dir: str,
        images: Optional[List[Image.Image]] = None,
    ) -> None:
        self.database_dir = database_dir

        metadata_path = os.path.join(database_dir, "metadata.json")
        metadata: Dict = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        self.database_image_paths = metadata.get("image_paths", [])
        self.database_mesh_paths = metadata.get("mesh_paths", [])
        self.database_uids = metadata.get("uids", [])

        candidate_files = [
            "clip_embeddings.npy",  # paper default: CLIP index for inference retrieval
            "embeddings.npy",
            "fused_embeddings.npy",
            "dino_embeddings.npy",
            "image_embeddings.npy",
        ]

        embedding_path = None
        embedding_name = None
        for name in candidate_files:
            path = os.path.join(database_dir, name)
            if os.path.exists(path):
                arr = np.load(path, mmap_mode="r")
                if arr.ndim in (2, 3):
                    embedding_path = path
                    embedding_name = name
                    break

        if embedding_path is None:
            raise FileNotFoundError(f"No usable embedding file found in {database_dir}")

        embeddings_np = np.load(embedding_path)
        if embeddings_np.ndim == 3:
            embeddings_np = embeddings_np.mean(axis=1)
        embeddings_np = _l2_normalize_np(embeddings_np.astype(np.float32))
        self._set_database_embeddings(embeddings_np)

        if images is not None:
            self.database_images = [img.convert("RGB") for img in images]
        elif self.load_images_eagerly and len(self.database_image_paths) > 0:
            loaded_images: List[Optional[Image.Image]] = []
            for image_path in self.database_image_paths:
                if image_path and os.path.exists(image_path):
                    try:
                        loaded_images.append(Image.open(image_path).convert("RGB"))
                    except Exception:
                        loaded_images.append(None)
                else:
                    loaded_images.append(None)
            self.database_images = loaded_images

        index_candidates = []
        if metadata.get("faiss_index"):
            index_candidates.append(os.path.join(database_dir, metadata["faiss_index"]))
        index_candidates.extend(
            [
                os.path.join(database_dir, "faiss.index"),
                os.path.join(database_dir, "clip_faiss.index"),
            ]
        )

        self._faiss_index = None
        if self.use_faiss:
            for index_path in index_candidates:
                if os.path.exists(index_path):
                    try:
                        self._faiss_index = faiss.read_index(index_path)
                        logger.info("Loaded FAISS index: %s", index_path)
                        break
                    except Exception as exc:  # pragma: no cover - faiss runtime mismatch
                        logger.warning("Failed to load FAISS index %s: %s", index_path, exc)

            if self._faiss_index is None:
                self._maybe_build_faiss_index()

        logger.info(
            "Database loaded from %s (%d entries, embedding file=%s, faiss=%s)",
            database_dir,
            int(self.database_embeddings.shape[0]) if self.database_embeddings is not None else 0,
            embedding_name,
            self._faiss_index is not None,
        )

    def load_database(self, load_path: str, images: Optional[List[Image.Image]] = None):
        if os.path.isdir(load_path):
            self._load_database_directory(load_path, images=images)
            return

        checkpoint = torch.load(load_path, map_location="cpu")

        if "embeddings" in checkpoint:
            self._set_database_embeddings(checkpoint["embeddings"])
        elif "clip_embeddings" in checkpoint:
            self._set_database_embeddings(checkpoint["clip_embeddings"])
        else:
            raise KeyError(f"No embeddings found in {load_path}")

        self.database_image_paths = checkpoint.get("image_paths", [])
        self.database_mesh_paths = checkpoint.get("mesh_paths", [])
        self.database_uids = checkpoint.get("uids", [])

        if images is not None:
            self.database_images = [img.convert("RGB") for img in images]
        else:
            self.database_images = None

        self._maybe_build_faiss_index()

        logger.info(
            "Database checkpoint loaded from %s with %d embeddings",
            load_path,
            int(self.database_embeddings.shape[0]),
        )


class RetrievalAugmentedEncoder(nn.Module):
    """Helper wrapper that fuses retrieved image tokens with prompt image tokens."""

    def __init__(
        self,
        image_encoder,
        feature_extractor,
        retrieval_module: Optional[RetrievalModule] = None,
        num_retrieved: int = 3,
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.feature_extractor = feature_extractor
        self.retrieval_module = retrieval_module
        self.num_retrieved = num_retrieved

    @torch.no_grad()
    def encode_with_retrieval(
        self,
        prompt_images: Union[Image.Image, List[Image.Image]],
        query_text: Optional[str] = None,
        use_retrieval: bool = True,
        device: str = "cuda",
        num_images_per_prompt: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(prompt_images, list):
            prompt_images = [prompt_images]
        prompt_images = [_as_image(img) for img in prompt_images]

        prompt_pixel_values = self.feature_extractor(prompt_images, return_tensors="pt").pixel_values
        prompt_pixel_values = prompt_pixel_values.to(device=device, dtype=dtype)
        prompt_embeds = self.image_encoder(prompt_pixel_values).last_hidden_state

        if use_retrieval and self.retrieval_module is not None:
            fused_list: List[torch.Tensor] = []
            for idx, prompt_image in enumerate(prompt_images):
                if query_text is not None:
                    retrieved = self.retrieval_module.retrieve_by_text(
                        query_text,
                        top_k=self.num_retrieved,
                        return_dict=True,
                    )
                else:
                    retrieved = self.retrieval_module.retrieve_by_image(
                        prompt_image,
                        top_k=self.num_retrieved,
                        return_dict=True,
                    )

                retrieved_images = [] if retrieved is None else (retrieved.get("images", []) or [])
                retrieved_images = [img for img in retrieved_images if isinstance(img, Image.Image)]

                prompt_embed = prompt_embeds[idx : idx + 1]
                if len(retrieved_images) == 0:
                    fused_list.append(prompt_embed)
                    continue

                retrieved_pixels = self.feature_extractor(retrieved_images, return_tensors="pt").pixel_values
                retrieved_pixels = retrieved_pixels.to(device=device, dtype=dtype)
                retrieved_embeds = self.image_encoder(retrieved_pixels).last_hidden_state
                retrieved_embeds = retrieved_embeds.reshape(1, -1, retrieved_embeds.shape[-1])

                fused = torch.cat([retrieved_embeds, prompt_embed], dim=1)
                fused_list.append(fused)

            image_embeds = torch.cat(fused_list, dim=0)
        else:
            image_embeds = prompt_embeds

        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)
        return image_embeds, uncond_image_embeds
