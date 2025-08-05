"""
Orchestrates the full negation search experiment.
"""
import numpy as np
import config
from data import get_image_paths
from embeddings import get_embedder
from vector_store import VectorStore
from query_generation import generate_negation_queries
from judge import judge_images
from report import generate_report
import json
import os

from datasets import load_dataset

def run_experiment():
    # 1. Download images
    image_dataset = load_dataset(config.DATASET_NAME, split="train[:1000]")
    
    # 2. Embed images
    embedder = get_embedder(config.EMBEDDING_MODEL)
    if os.path.exists(config.EMBEDDINGS_PATH):
        embeddings = np.load(config.EMBEDDINGS_PATH)
    else:
        embeddings = embedder.embed_images(image_paths)
        np.save(config.EMBEDDINGS_PATH, embeddings)

    # 3. Generate negation queries
    queries = generate_negation_queries(config.N_QUERIES, config.QUERY_MODEL)

    # 4. Build vector store
    store = VectorStore(embeddings, image_paths)

    # 5. For each query, embed and search
    results = []
    for query in queries:
        query_emb = embedder.embed_text([query])
        topk = store.search(query_emb, top_k=config.TOP_K)
        topk_paths = [p for p, _ in topk]
        # 6. Judge
        fit_bools = judge_images(query, topk_paths, config.JUDGE_MODEL)
        precision = sum(fit_bools) / len(fit_bools)
        recall = sum(fit_bools) / config.TOP_K  # Approximate recall@k
        results.append({
            "query": query,
            "precision": precision,
            "recall": recall,
            "topk_paths": topk_paths,
            "fit_bools": fit_bools,
        })

    # 7. Report
    generate_report(results, config.REPORT_PATH)

if __name__ == "__main__":
    run_experiment()
