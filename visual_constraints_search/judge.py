"""
Uses an LLM to judge if retrieved images fit the negation query constraints.
"""
from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm
from typing import List, Optional
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoProcessor

import litellm
import io
import base64


class JudgeResponseFormat(BaseModel):
    is_relevant: bool = Field(..., description="Whether the image fits the query and constraints.")
    reason: str = Field(..., description="Reasoning for the judgment.")

def get_external_model_output(model: str, full_prompt: str, images_str: str):
    messages=[
            {"role": "user", "content": [
                {"type": "text", "text": full_prompt},
                {"type": "image_url", "image_url": "data:image/jpeg;base64," + images_str}
            ]
            },
    ]

    return litellm.completion(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format=JudgeResponseFormat
        )


def get_local_model_output(llm: LLM, full_prompt: str, image: Image):
    messages=[
            {"role": "user", "content": [
                {"type": "text", "text": full_prompt},
                {"type": "image_pil", "image_pil": image}
            ]
            },
    ]

    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=512,
        guided_decoding=GuidedDecodingParams(json=JudgeResponseFormat.model_json_schema()),
    )
    llm_response = llm.chat(
        messages=messages,
        sampling_params=sampling_params
    )
    return llm_response[0]

def judge_images(
    query: str, 
    images: List[Image], 
    model: str, 
    is_local: bool = True, 
    image_resolution: tuple = (480, 480),
    llm: Optional[LLM] = None
) -> List[JudgeResponseFormat]:
    """
    Use an LLM to judge if each image fits the negation query.
    Returns a list of bools (True if image fits the query, False otherwise).
    """
    # For transparency, we could also pass captions/annotations if available
    judgements = []
    for image in tqdm(images, desc="Judging images"):
        if image.size != image_resolution:
            image = image.resize(image_resolution, Image.Resampling.LANCZOS)
        prompt = (
            f"For the following query: '{query}', determine for each image if it fits the constraints. "
            "Return a list of True/False for each image."
        )
        images_str = pil_image_to_base64(image)
        try:
            if is_local:
                llm_response = get_local_model_output(llm, prompt, image)
                content = llm_response.outputs[0].text
            else:
                full_prompt = f"{prompt}\n{images_str}"
                response = get_external_model_output(model, full_prompt, images_str)
                # Parse list of bools from LLM response
                content = response.choices[0].message.content
            content = JudgeResponseFormat.model_validate_json(content)
        except Exception as e:
            print(f"Error judging image: {e}")
            content = JudgeResponseFormat(is_relevant=True, reason="Error processing image.")
        judgements.append(content)
    return judgements


def pil_image_to_base64(img: Image.Image, format: str = "jpeg") -> str:
    """
    Converts a PIL Image to a base64-encoded string.
    """
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

if __name__ == "__main__":
    query = "Images of cats, without any dogs."
    # Create a dummy image for testing
    img = Image.new("RGB", (480, 480), color=(255, 0, 0))  # Red square
    images = [img] * 5  # Duplicate the image for testing
    model = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"  # Example model, replace with your actual model
    is_local = True
    judgements = judge_images(query, images, model, is_local)