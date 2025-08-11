"""
Uses an LLM to judge if retrieved images fit the negation query constraints.
"""
from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm
from typing import List

import json
import time
import litellm
import io
import base64


class JudgeResponseFormat(BaseModel):
    is_relevant: bool = Field(..., description="Whether the image fits the query and constraints.")
    reason: str = Field(..., description="Reasoning for the judgment.")


def judge_images(query: str, images: List[Image], model: str = "gpt-4o") -> List[JudgeResponseFormat]:
    """
    Use an LLM to judge if each image fits the negation query.
    Returns a list of bools (True if image fits the query, False otherwise).
    """
    # For transparency, we could also pass captions/annotations if available
    judgements = []
    for image in tqdm(images, desc="Judging images"):
        prompt = (
            f"For the following query: '{query}', determine for each image if it fits the constraints. "
            "Return a list of True/False for each image."
        )
        # In practice, you might want to pass image URLs or captions, but here we just pass file names
        images_str = pil_image_to_base64(image)
        full_prompt = f"{prompt}\n{images_str}"
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": "data:image/jpeg;base64," + images_str}
                ]
                },
            ],
            temperature=0.2,
            response_format=JudgeResponseFormat
        )
        # Parse list of bools from LLM response
        content = response.choices[0].message.content
        content = JudgeResponseFormat.model_validate_json(content)
        judgements.append(content)
    return judgements


def pil_image_to_base64(img: Image.Image, format: str = "jpeg", image_resoultion: tuple = (480, 480)) -> str:
    """
    Converts a PIL Image to a base64-encoded string.
    """
    buffered = io.BytesIO()
    # resize image if needed
    if img.size != image_resoultion:
        img = img.resize(image_resoultion, Image.ANTIALIAS)
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

