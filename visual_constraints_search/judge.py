"""
Uses an LLM to judge if retrieved images fit the negation query constraints.
"""

from typing import List
import litellm

def judge_images(query: str, image_paths: List[str], model: str = "gpt-4o") -> List[bool]:
    """
    Use an LLM to judge if each image fits the negation query.
    Returns a list of bools (True if image fits the query, False otherwise).
    """
    # For transparency, we could also pass captions/annotations if available
    prompt = (
        f"For the following query: '{query}', determine for each image if it fits the constraints. "
        "Return a list of True/False for each image."
    )
    # In practice, you might want to pass image URLs or captions, but here we just pass file names
    images_str = "\n".join([f"Image: {p}" for p in image_paths])
    full_prompt = f"{prompt}\n{images_str}"
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.2,
    )
    # Parse list of bools from LLM response
    content = response.choices[0].message.content
    lines = content.split("\n")
    bools = ["true" in line.lower() for line in lines if line.strip()]
    return bools
