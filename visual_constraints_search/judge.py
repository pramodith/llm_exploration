"""
Uses an LLM to judge if retrieved images fit the negation query constraints.
"""
from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm
from typing import List
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

import json
import time
import litellm
import io
import base64


class JudgeResponseFormat(BaseModel):
    is_relevant: bool = Field(..., description="Whether the image fits the query and constraints.")
    reason: str = Field(..., description="Reasoning for the judgment.")

def get_external_model_output(model: str, messages: list, images_str: str):
    return litellm.completion(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format=JudgeResponseFormat
        )


def get_local_model_output(llm: LLM, messages: list):
    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=512,
        guided_decoding=GuidedDecodingParams(json=JudgeResponseFormat.model_json_schema()),
    )
    llm_response = llm.generate(
        messages=messages,
        sampling_params=sampling_params
    )
    return llm_response

def judge_images(query: str, images: List[Image], model: str, is_local: bool = True) -> List[JudgeResponseFormat]:
    """
    Use an LLM to judge if each image fits the negation query.
    Returns a list of bools (True if image fits the query, False otherwise).
    """
    if is_local:
        llm = LLM(model=model, trust_remote_code=True, enforce_eager=True)
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
        messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": "data:image/jpeg;base64," + images_str}
                ]
                },
        ]
        if is_local:
            llm_response = get_local_model_output(llm, full_prompt, images_str)
            response = llm_response.outputs[0]
        else:
            response = get_external_model_output(model, messages, images_str)
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
        img = img.resize(image_resoultion, Image.Resampling.LANCZOS)
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

