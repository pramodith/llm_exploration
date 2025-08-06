"""
Orchestrates the full negation search experiment.
"""
import numpy as np
import config
from embeddings import get_embedder
from vector_store import VectorStore
from query_generation import generate_negation_queries
from judge import judge_images
from report import generate_report
import os

from datasets import load_dataset
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()  # Load environment variables from .env file

def plot_topk_images(images, query, max_cols=5, save_path=None):
    """
    Plot the top-k images with the query as the title.
    If save_path is provided, save the plot to that path.
    """
    k = len(images)
    cols = min(max_cols, k)
    rows = (k + cols - 1) // cols
    plt.figure(figsize=(3 * cols, 3 * rows))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if hasattr(img, "convert"):
            plt.imshow(img.convert("RGB"))
        else:
            plt.imshow(img)
        plt.axis("off")
    plt.suptitle(f"Top-{k} results for query: '{query}'", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
    plt.show()

def run_experiment():
    # 1. Download images
    image_dataset = load_dataset(config.DATASET_NAME, split="test[:1000]")
    image_dataset = image_dataset["image"]
    
    # 2. Embed images
    embedder = get_embedder(config.EMBEDDING_MODEL)
    if os.path.exists(config.EMBEDDINGS_PATH):
        embeddings = np.load(config.EMBEDDINGS_PATH)
    else:
        embeddings = embedder.embed_images(image_dataset)
        np.save(config.EMBEDDINGS_PATH, embeddings)

    # 3. Generate negation queries (load if already saved)
    if hasattr(config, "QUERIES_PATH") and os.path.exists(config.QUERIES_PATH):
        with open(config.QUERIES_PATH, "r") as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        queries = generate_negation_queries(config.N_QUERIES, config.QUERY_MODEL)
        if hasattr(config, "QUERIES_PATH"):
            with open(config.QUERIES_PATH, "w") as f:
                for q in queries:
                    f.write(q + "\n")

    # 4. Build vector store
    store = VectorStore(embeddings, image_dataset)

    # 5. For each query, embed and search
    results = []
    for idx, query in enumerate(queries[:10]):
        query = "dogs"
        query_emb = embedder.embed_text([query])
        topk = store.search(query_emb, top_k=config.TOP_K)
        topk_images = [p for p, _ in topk]
        
        # Plot and save the top-k images for this query
        save_path = f"debug_images/topk_query_{idx+1}.png"
        plot_topk_images(topk_images, query, save_path=save_path)
        break
        
        
        # 6. Judge
        # fit_bools = judge_images(query, topk_paths, config.JUDGE_MODEL)
        # precision = sum(fit_bools) / len(fit_bools)
        # recall = sum(fit_bools) / config.TOP_K  # Approximate recall@k
        # results.append({
        #     "query": query,
        #     "precision": precision,
        #     "recall": recall,
        #     "topk_paths": topk_paths,
        #     "fit_bools": fit_bools,
        # })

    # 7. Report
    generate_report(results, config.REPORT_PATH)

if __name__ == "__main__":
    run_experiment()
