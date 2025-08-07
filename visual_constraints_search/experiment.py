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
from sklearn.feature_extraction.text import CountVectorizer

import os

from datasets import load_dataset
from dotenv import load_dotenv

import matplotlib.pyplot as plt
from bertopic import BERTopic
from bertopic.representation import OpenAI
import hdbscan  # Add this import
import tiktoken
import openai  # Ensure you have the OpenAI library installed

load_dotenv()  # Load environment variables from .env file

def extract_topic_keywords(image_captions, n_top_topics=1000):
    """
    Extracts topic keywords from image captions using BERTopic.
    Returns a set of keywords/phrases representing the main topics.
    """
    print("Extracting topics from captions using BERTopic...")
    cluster_model = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))       
    # Tokenizer
    tokenizer = tiktoken.encoding_for_model("gpt-4o")

    # Create your representation model
    client = openai.OpenAI()
    representation_model = OpenAI(
        client,
        model="gpt-4.1-mini",
        delay_in_seconds=2,
        chat=True,
        nr_docs=4,
        doc_length=2048,
        tokenizer=tokenizer
    )

    topic_model = BERTopic(
        verbose=True, hdbscan_model=cluster_model, vectorizer_model=vectorizer_model, representation_model=representation_model
    )
    
    
    topics, _ = topic_model.fit_transform(image_captions)
    topic_info = topic_model.get_topic_info()
    print("\nTop topics in captions:")
    print(topic_info.head(n_top_topics).to_string(index=False))
    # Extract keywords for each topic (skip -1, which is usually 'outlier')
    keywords = set()
    for topic_id in topic_info['Topic'].head(n_top_topics):
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        # words is a list of (word, score) tuples
        for word, _ in words:
            keywords.add(word)
    return keywords

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
    image_dataset = load_dataset(config.DATASET_NAME, split="test")
    images = image_dataset["image"]
    
    # 2. Load or extract topic keywords
    keywords_path = getattr(config, "KEYWORDS_PATH", "topic_keywords.txt")
    if os.path.exists(keywords_path):
        with open(keywords_path, "r") as f:
            topic_keywords = set(line.strip() for line in f if line.strip())
        print("\nLoaded topic keywords from file:")
        print(topic_keywords)
    else:
        image_captions = image_dataset["caption"]
        for caption in image_captions:
            max_len_caption_ind = len(caption)
            max_len = 0
            for i in range(len(caption)):
                if len(caption[i]) > max_len:
                    max_len = len(caption[i])
                    max_len_caption_ind = i
        image_captions = [caption[max_len_caption_ind] for caption in image_captions]

        # --- BERTopic: Analyze topics in captions and extract keywords ---
        topic_keywords = extract_topic_keywords(image_captions)
        print("\nExtracted topic keywords:")
        print(topic_keywords)
        with open(keywords_path, "w") as f:
            for kw in topic_keywords:
                f.write(kw + "\n")

    # 3. Generate negation queries (load if already saved)
    if hasattr(config, "QUERIES_PATH") and os.path.exists(config.QUERIES_PATH):
        with open(config.QUERIES_PATH, "r") as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        queries = generate_negation_queries(config.N_QUERIES, topic_keywords)
        if hasattr(config, "QUERIES_PATH"):
            with open(config.QUERIES_PATH, "w") as f:
                for q in queries:
                    f.write(q + "\n")
    
    # 3. Embed images
    embedder = get_embedder(config.EMBEDDING_MODEL)
    if os.path.exists(config.EMBEDDINGS_PATH):
        embeddings = np.load(config.EMBEDDINGS_PATH)
    else:
        embeddings = embedder.embed_images(images[:1000])
        np.save(config.EMBEDDINGS_PATH, embeddings)


    # 4. Build vector store
    store = VectorStore(embeddings, images[:1000])

    # 5. For each query, embed and search
    results = []
    for idx, query in enumerate(queries[:10]):
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
