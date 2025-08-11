import gradio as gr
import numpy as np
import os
from PIL import Image
import json
import config

from judge import JudgeResponseFormat
from datasets import load_dataset
import pickle

image_dataset = load_dataset(config.DATASET_NAME, split="test")    
images = image_dataset["image"]# Load results

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
            # topk_indices is a list of indices into images
            return r["topk_indices"], r["fit_bools"]
    return [], []

def display_judgements(query):
    image_indices, judgements = get_images_and_judgements(query)
    if not image_indices:
        return [[], []]
    img_blocks = []
    captions = []
    for idx, judgement in zip(image_indices, judgements):
        # idx is an index into images (PIL Image)
        pil_img = images[int(idx)]
        img_arr = np.array(pil_img)
        # judgement is a JSON object to be parsed
        judgement = JudgeResponseFormat.model_validate_json(judgement)
        is_relevant = judgement.is_relevant
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
        captions.append(f"Relevant: {judgement.reason}" if is_relevant else f"Irrelevant: {judgement.reason}")
    return [img_blocks, captions]

def get_selected_caption(evt: gr.SelectData, captions_list):
    if captions_list and evt.index < len(captions_list):
        return captions_list[evt.index]
    return ""

# Gradio UI
def gradio_app():
    queries = [r["query"] for r in results]
    with gr.Blocks() as demo:
        gr.Markdown("# Query Judgement Viewer")
        query = gr.Dropdown(queries, label="Select Query")
        gallery = gr.Gallery(label="Judged Images", columns=5, height="auto")
        captions = gr.Markdown(label="Selected Image Caption")
        
        # State to store captions list
        captions_state = gr.State([])
        
        def update_gallery_and_captions(query):
            img_blocks, captions_list = display_judgements(query)
            return img_blocks, captions_list, "**Select an image to view its caption**"
        
        query.change(fn=update_gallery_and_captions, inputs=query, outputs=[gallery, captions_state, captions])
        gallery.select(fn=get_selected_caption, inputs=captions_state, outputs=captions)
    return demo

if __name__ == "__main__":
    import signal
    import sys
    
    demo = gradio_app()
    
    def signal_handler(sig, frame):
        print('\nShutting down Gradio app...')
        demo.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        demo.launch()
    except KeyboardInterrupt:
        print('\nGradio app stopped.')
