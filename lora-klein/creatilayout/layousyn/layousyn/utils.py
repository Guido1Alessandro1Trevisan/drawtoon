import os
import torch
import hashlib
import json
from typing import List, Tuple

import numpy as np

EPS = 1e-3


def assert_is_normalized(bbox: List[float]) -> bool:
    """
    Assert that the bounding box is normalized.

    Args:
        bbox (List[float]): A bounding box in the layout_format (x1, y1, x2, y2).
    """
    assert len(bbox) == 4, "bbox must be of length 4"
    assert (
        -1.0 - EPS <= bbox[0] <= 1.0 + EPS
        and -1.0 - EPS <= bbox[1] <= 1.0 + EPS
        and (-1.0 - EPS <= bbox[2] <= 1.0 + EPS and -1.0 - EPS <= bbox[3] <= 1.0 + EPS)
    ), f"bbox must be normalized. Got {bbox}"
    return True


def assert_is_unscaled(bbox: List[float]) -> bool:
    """
    Assert that the bounding box is scaled i.e in the range [0, 1].

    Args:
        bbox (List[float]): A bounding box in the layout_format (x1, y1, x2, y2).
    """
    assert len(bbox) == 4, "bbox must be of length 4"
    assert (
        0.0 - EPS <= bbox[0] <= 1.0 + EPS
        and 0.0 - EPS <= bbox[1] <= 1.0 + EPS
        and (0.0 - EPS <= bbox[2] <= 1.0 + EPS and 0.0 - EPS <= bbox[3] <= 1.0 + EPS)
    ), f"bbox must be unscaled. Got {bbox}"
    return True


def scale_bbox(bbox: List[float], w: int, h: int) -> List[float]:
    """
    Scale bounding box coordinates.

    Args:
        bbox (List[float]): A bounding box in the layout_format (x1, y1, x2, y2).
        w (int): The width of the image.
        h (int): The height of the image.

    Returns:
        List[float]: The scaled bounding box in the layout_format (x1, y1, x2, y2).
    """
    assert_is_unscaled(bbox)
    return [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]


def unscale_bbox(bbox: List[float], w: int, h: int) -> List[float]:
    """
    Unscale bounding box coordinates.

    Args:
        bbox (List[float]): A bounding box in the layout_format (x1, y1, x2, y2).
        w (int): The width of the image.
        h (int): The height of the image.

    Returns:
        List[float]: The unscaled bounding box in the layout_format (x1, y1, x2, y2).
    """
    return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]


def norm_bbox(bbox: List[float]) -> List[float]:
    """
    Normalize bounding box coordinates by scaling them to the range [-1, 1] from [0, 1].

    Args:
        bbox (List[float]): A bounding box in the layout_format (x1, y1, x2, y2)
        w (int): The width of the image.
        h (int): The height of the image.

    Returns:
        List[float]: The normalized bounding box in the layout_format (x1, y1, x2, y2).
    """
    assert_is_unscaled(bbox)
    return [
        bbox[0] * 2.0 - 1.0,
        bbox[1] * 2.0 - 1.0,
        bbox[2] * 2.0 - 1.0,
        bbox[3] * 2.0 - 1.0,
    ]


def unnorm_bbox(bbox: List[float]) -> List[float]:
    """
    Unnormalize bounding box coordinates.

    Args:
        bbox (List[float]): A bounding box in the layout_format (x1, y1, x2, y2)
        w (float): The width of the image.
        h (float): The height of the image.

    Returns:
        List[float]: The unnormalized bounding box in the layout_format (x1, y1, x2, y2).
    """
    assert_is_normalized(bbox)
    return [
        (bbox[0] + 1.0) / 2.0,
        (bbox[1] + 1.0) / 2.0,
        (bbox[2] + 1.0) / 2.0,
        (bbox[3] + 1.0) / 2.0,
    ]
    
    
def clip_bbox_to_image(bbox: List[float], w: int, h: int) -> List[float]:
    """
    Clip the bounding box to the image size.

    Args:
        bbox (List[float]): A bounding box in the layout_format (x1, y1, x2, y2).
        w (int): The width of the image.
        h (int): The height of the image.

    Returns:
        List[float]: The bounded bounding box in the layout_format (x1, y1, x2, y2).
    """
    return [
        max(0.0, min(w, bbox[0])),
        max(0.0, min(h, bbox[1])),
        max(0.0, min(w, bbox[2])),
        max(0.0, min(h, bbox[3])),
    ]

def pad_to_max_bboxs(bboxs, cats, max_bbox, no_box_location, no_box_label):
    # max bbox check
    if max_bbox:
        # truncate to max_bbox
        bboxs = bboxs[:max_bbox]
        cats = cats[:max_bbox]

        # pad to max_bbox
        if len(bboxs) < max_bbox:
            bboxs += [no_box_location] * (max_bbox - len(bboxs))
            cats += [no_box_label] * (max_bbox - len(cats))

    return bboxs, cats


def get_padding_mask(cats_padded, original_length):
    # get padded length
    padded_length = len(cats_padded)

    # get padding mask
    concept_embeddings_mask = [True] * padded_length # set all concepts to padding
    concept_count = min(padded_length, original_length) # get the minimum between the number of concepts and the number of cats
    concept_embeddings_mask[:concept_count] = [False] * concept_count # set the first concept_count to not padding
    concept_embeddings_mask = np.array(concept_embeddings_mask)

    # return concept embeddings mask
    return concept_embeddings_mask


def shuffle_bboxs(
    bboxs: List[List[float]], cats: List[str]
) -> Tuple[List[List[float]], List[str]]:
    """Shuffle the bounding boxes and categories.

    Args:
        bboxs (List[List[float]]): List of bounding boxes.
        cats (List[str]): List of categories.

    Returns:
        Tuple[List[List[float]], List[str]]: Tuple containing shuffled bounding boxes and categories.
    """
    shuffle_idx = np.random.permutation(len(bboxs))
    return [bboxs[i] for i in shuffle_idx], [cats[i] for i in shuffle_idx]


def str_to_sha256(s):
    return hashlib.sha256(s.encode()).hexdigest()


def bound_xyxy_bbox(bbox: List[float]) -> List[float]:
    """
    Ensure that the bottom right corner of the bounding box is greater than the top left corner.

    Args:
        bbox (List[float]): A bounding box in the layout_format (x1, y1, x2, y2).
    """
    assert_is_normalized(bbox)
    return [
        bbox[0],
        bbox[1],
        max(bbox[0], bbox[2]),
        max(bbox[1], bbox[3]),
    ]


def filter_caption_ids(caption_ids, ignore_caption_file):
    # load json list of caption ids to ignore
    ignore_caption_list = []
    with open(ignore_caption_file, "r") as f:
        ignore_caption_list = json.load(f)

    # filter caption ids
    filtered_caption_ids = [
        caption_id
        for caption_id in caption_ids
        if caption_id not in ignore_caption_list
    ]

    # return filtered caption ids
    return filtered_caption_ids


def get_crop_boundry(bboxs):
    # initialize crop bboxs
    assert assert_is_unscaled(bboxs[0]), "Bounding box must be unscaled"
    x1_min, y1_min, x2_max, y2_max = bboxs[0]

    for bbox in bboxs[1:]:
        assert assert_is_unscaled(bbox), "Bounding box must be unscaled"
        x1, y1, x2, y2 = bbox
        x1_min = min(x1_min, x1)
        y1_min = min(y1_min, y1)
        x2_max = max(x2_max, x2)
        y2_max = max(y2_max, y2)

    return [x1_min, y1_min, x2_max, y2_max]


def crop_layouts(bboxs, width, height):
    # get the crop boundry
    x1_min, y1_min, x2_max, y2_max = get_crop_boundry(bboxs)

    # for x1_min and y1_min, get a random value between 0 and x1_min and 0 and y1_min
    x1_min_rand = np.random.uniform(0, x1_min)
    y1_min_rand = np.random.uniform(0, y1_min)

    # for x2_max and y2_max, get a random value between x2_max and 1 and y2_max and 1
    x2_max_rand = np.random.uniform(x2_max, 1)
    y2_max_rand = np.random.uniform(y2_max, 1)

    # get the crop bbox
    w_prime = x2_max_rand - x1_min_rand
    h_prime = y2_max_rand - y1_min_rand

    # update bboxs
    bboxs = [
        [
            np.clip((bbox[0] - x1_min_rand) / w_prime, 0.0, 1.0),
            np.clip((bbox[1] - y1_min_rand) / h_prime, 0.0, 1.0),
            np.clip((bbox[2] - x1_min_rand) / w_prime, 0.0, 1.0),
            np.clip((bbox[3] - y1_min_rand) / h_prime, 0.0, 1.0),
        ]
        for bbox in bboxs
    ]

    return bboxs, width * w_prime, height * h_prime


def bbox_horizontal_flip(bboxs):
    # flip the bounding box
    bboxs = [
        [
            1.0 - bbox[2],
            bbox[1],
            1.0 - bbox[0],
            bbox[3],
        ]
        for bbox in bboxs
    ]

    return bboxs

def cleanup_prompt(prompt):
    # remove newline and tab characters
    prompt = prompt.replace("\n", " ")
    prompt = prompt.replace("\t", " ")
    return prompt

def find_model(model_name):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, weights_only=False)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint