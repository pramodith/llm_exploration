"""
Handles downloading and preprocessing of the image dataset for the experiment.
"""
import os
from typing import List
from pathlib import Path

def download_coco_images(max_images: int, data_dir: str) -> List[str]:
    """
    Download up to `max_images` images from the MS COCO 2017 validation set.
    Returns a list of local image file paths.
    """
    import requests
    import zipfile
    import shutil

    coco_url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = os.path.join(data_dir, "val2017.zip")
    extract_dir = os.path.join(data_dir, "val2017")
    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(extract_dir):
        print("Downloading COCO val2017 images...")
        with requests.get(coco_url, stream=True) as r:
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        print("Extracting images...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)

    image_files = sorted(list(Path(extract_dir).glob("*.jpg")))[:max_images]
    return [str(p) for p in image_files]

def get_image_paths(dataset: str, max_images: int, data_dir: str) -> List[str]:
    """
    Returns a list of image file paths for the specified dataset.
    """
    if dataset == "coco":
        return download_coco_images(max_images, data_dir)
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported yet.")
