"""
Uses an LLM to judge if retrieved images fit the negation query constraints.
"""
from PIL import Image
from pydantic import BaseModel, Field
from typing import List

import json
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
    for image in images:
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
                {"role": "user", "content": {"text": full_prompt, "image_url": 
                    {"url": "data:image/jpeg;base64," + images_str}
            }}],
            temperature=0.2,
        )
        # Parse list of bools from LLM response
        content = response.choices[0].message.content
        content = JudgeResponseFormat(json.loads(content))
        judgements.append(content)
    return judgements


def pil_image_to_base64(img: Image.Image, format: str = "jpeg") -> str:
    """
    Converts a PIL Image to a base64-encoded string.
    """
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

