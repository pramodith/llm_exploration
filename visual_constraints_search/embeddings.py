"""
Handles embedding images and queries using multimodal models.
"""
from typing import List, Tuple
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

import config

class ImageEmbedder:
    """
    Abstract base class for image embedding models.
    """
    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        """Embed a list of image file paths and return a numpy array of embeddings."""
        raise NotImplementedError

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text queries and return a numpy array of embeddings."""
        raise NotImplementedError

class SigLipEmbedder(ImageEmbedder):
    """
    SigLip-based image and text embedder using OpenAI's CLIP model via transformers.
    """
    def __init__(self, model_name: str = config.EMBEDDING_MODEL, batch_size: int = 8):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True,
        )
        import torch
        self.torch = torch
        self.device = self.model.device
        self.batch_size = batch_size

    def embed_images(self, images: list) -> np.ndarray:
        """
        Embed images in batches.
        """
        embs = []
        for i in tqdm(range(0, len(images), self.batch_size), desc="Embedding images"):
            batch = images[i:i+self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                emb = self.model.get_image_features(**inputs)
            embs.append(emb.float().cpu().numpy())
        return np.vstack(embs)

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts in batches.
        """
        embs = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding texts"):
            batch = texts[i:i+self.batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                emb = self.model.get_text_features(**inputs)
            embs.append(emb.float().cpu().numpy())
        return np.vstack(embs)

def get_embedder(model_name: str) -> ImageEmbedder:
    """
    Factory to get the appropriate embedder.
    """
    if model_name == config.EMBEDDING_MODEL:
        return SigLipEmbedder(model_name=model_name)
    else:
        raise ValueError(f"Unsupported embedding model: {model_name}")
