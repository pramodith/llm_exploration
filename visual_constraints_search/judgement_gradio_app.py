import gradio as gr
import numpy as np
import os
from PIL import Image
import json

import pickle
# Load results
RESULTS_PATH = "results/judgement_results.pkl"
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "rb") as f:
        results = pickle.load(f)
else:
    results = []

# Helper to load images and judgements for a query
def get_images_and_judgements(query):
    for r in results:
        if r["query"] == query:
            # Each judgement contains 'is_relevant' and topk_images is a list of np arrays
            return r["topk_images"], r["fit_bools"]
    return [], []

def display_judgements(query):
    images, judgements = get_images_and_judgements(query)
    if not images:
        return [[], []]
    img_blocks = []
    captions = []
    for img_arr, judgement in zip(images, judgements):
        # img_arr is a numpy array (H, W, 3)
        is_relevant = json.loads(judgement)["is_relevant"]
        border_color = (0,255,0) if is_relevant else (255,0,0)  # green or red
        # Add border
        border_width = 10
        img = np.copy(img_arr)
        # Top
        img[:border_width,:,:] = border_color
        # Bottom
        img[-border_width:,:,:] = border_color
        # Left
        img[:,:border_width,:] = border_color
        # Right
        img[:,-border_width:,:] = border_color
        img_blocks.append(img)
        captions.append(f"Relevant" if is_relevant else "Irrelevant")
    return [img_blocks, captions]

# Gradio UI
def gradio_app():
    queries = [r["query"] for r in results]
    with gr.Blocks() as demo:
        gr.Markdown("# Query Judgement Viewer")
        query = gr.Dropdown(queries, label="Select Query")
        gallery = gr.Gallery(label="Judged Images").style(grid=[5], height="auto")
        captions = gr.Textbox(label="Captions", lines=2)
        query.change(fn=display_judgements, inputs=query, outputs=[gallery, captions])
    return demo

if __name__ == "__main__":
    gradio_app().launch()
