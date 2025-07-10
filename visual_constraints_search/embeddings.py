"""
Handles embedding images and queries using multimodal models.
"""
from typing import List, Tuple
import numpy as np

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

class CLIPEmbedder(ImageEmbedder):
    """
    CLIP-based image and text embedder using OpenAI's CLIP model via transformers.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        from transformers import CLIPProcessor, CLIPModel
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.torch = torch

    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        from PIL import Image
        embs = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                emb = self.model.get_image_features(**inputs)
            embs.append(emb.cpu().numpy())
        return np.vstack(embs)

    def embed_text(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        return emb.cpu().numpy()

def get_embedder(model_name: str) -> ImageEmbedder:
    """
    Factory to get the appropriate embedder.
    """
    if model_name == "clip":
        return CLIPEmbedder()
    else:
        raise NotImplementedError(f"Model {model_name} not supported yet.")
