"""
Simple vector store for storing and retrieving image embeddings.
"""
from typing import List, Tuple
import numpy as np

class VectorStore:
    """
    Stores image embeddings and allows similarity search.
    """
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        self.norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    def simple_rank_search(self, query_emb: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Returns top_k (image, similarity) tuples for the query embedding.
        """
        sims = (self.embeddings @ query_emb.T) / (self.norms * np.linalg.norm(query_emb))
        sims = sims.squeeze()
        idxs = np.argsort(-sims)[:top_k].tolist()
        return [(i, float(sims[i])) for i in idxs]

    def negative_query_rank_search(
        self, query_emb: np.ndarray, neg_emb: np.ndarray, top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Returns top_k (image, similarity) tuples for the negative query embedding.
        """
        query_emb -= neg_emb
        sims = (self.embeddings @ query_emb.T) / (self.norms * np.linalg.norm(query_emb))
        sims = sims.squeeze()
        idxs = np.argsort(-sims)[:top_k].tolist()
        return [(i, float(sims[i])) for i in idxs]
