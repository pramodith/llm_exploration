"""
Simple vector store for storing and retrieving image embeddings.
"""
from typing import List, Tuple
import numpy as np

class VectorStore:
    """
    Stores image embeddings and allows similarity search.
    """
    def __init__(self, embeddings: np.ndarray, image_paths: List[str]):
        self.embeddings = embeddings
        self.image_paths = image_paths
        self.norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    def search(self, query_emb: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Returns top_k (image_path, similarity) tuples for the query embedding.
        """
        sims = (self.embeddings @ query_emb.T) / (self.norms * np.linalg.norm(query_emb))
        sims = sims.squeeze()
        idxs = np.argsort(-sims)[:top_k]
        return [(self.image_paths[i], float(sims[i])) for i in idxs]
