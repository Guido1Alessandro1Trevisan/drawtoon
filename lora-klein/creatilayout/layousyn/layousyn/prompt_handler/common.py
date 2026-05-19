from typing import Optional

import torch
import transformers

LABEL_SET_GENERATION_PROMPT = """You are a creative scene designer who predicts a scene from a natural language prompt. A scene is a JSON object containing a list of noun phrases with their counts {"phrase1": count1, "phrase2": count2, ...}. The noun phrases contain **ONLY** common nouns. You strictly follow the below process for predicting plausible scenes: 

    Step 1: Extract noun phrases from the prompt. For example, "happy people", "car engine", "brown dog", "parking lot", etc.
    Step 2: Limit noun phrases to common nouns and convert the noun phrase to its singular form. For example, "happy people" to "person", "tall women" to "woman", "group of old people" to "person", "children" to "child", "brown dog" to "dog", "parking lot" remains "parking lot", etc.
    Step 3: Predict the count of each noun phrase and ensure consistency with the count of other objects in the scene. If a particular object does not have any explicit count mentioned in the prompt, use your creativity to assign a count to make the overall scene plausible but not too cluttered. For example, if the prompt is "a group of young kids playing with their dogs," the count of "kid" can be 3, and the count of "dog" should be the same as the count of "kid".
    Step 4: Output the final scene as a JSON object, only including physical objects and phrases without referring to actions or activities.

    Complete example:

    Prompt: Three white sheep and few women walking down a town road.
    Steps:
    Step 1: noun phrases: white sheep, women, town road
    Step 2: noun phrase in singular form: sheep, woman, town road
    Step 3: Since the count of women is not mentioned, we will assign a count of 2 to make the scene plausible. The count of "sheep" is 3 and the count of "town road" is 1.
    Step 4: {"sheep": 3, "woman": 2, "town road": 1}
    Plausible scene: {"sheep": 3, "woman": 2, "town road": 1}

    Other examples with skipped step-by-step process: 

    Prompt: A desk and office chair in the cubicle 
    Plausible scene: {"office desk": 1, "office chair": 1, "cubicle": 1} 

    Prompt: A pizza is in a box on a corner desk table.
    Plausible scene: {"pizza": 1, "box": 1, "desk table": 1}

    Note: Print **ONLY** the final scene as a JSON object.
"""


@torch.no_grad()
def get_llama_gen_pipeline(
    model_id: str,
    batch_size: int,
    token: Optional[str] = None,
    device: int = 0,
) -> transformers.Pipeline:
    """
    Initialize a transformers pipeline with the given model_id and token.

    Args:
        model_id (str): The model id to use.
        batch_size (int): The batch size to use.
        token (str): The token to use.
        device_id (int): The device id to use.
        device_map (Optional[str]): The device map to use.
    Returns:
        transformers.Pipeline: The initialized pipeline.
    """
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
        },
        token=token,
        batch_size=batch_size,
        device=device,
    )

    # pad to max length
    pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token  # type: ignore
    pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id  # type: ignore
    pipeline.tokenizer.padding_side = "left"  # type: ignore

    return pipeline
