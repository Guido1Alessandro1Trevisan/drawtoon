# ----------------------------------------------------------------------------
# Code based on pytorch COCO dataset: https://pytorch.org/vision/main/_modules/torchvision/datasets/coco.html#CocoDetection
# ----------------------------------------------------------------------------

import json
import os
import random
from logging import Logger
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from layout_evaluation import Layout

from .utils import (
    cleanup_prompt,
    crop_layouts,
    filter_caption_ids,
    shuffle_bboxs,
    unscale_bbox,
)

def get_bbox_for_img(
    bbox_obj: COCO,
    img_id: int,
) -> Tuple[List[List[float]], List[str], int, int]:
    """
    Load annotations for a given image id from a COCO bounding box object.

    Args:
        bbox_obj (COCO): A COCO bounding box object.
        img_id (int): The id of the image for which to load annotations.
    Returns:
        List[Any]: A list of bounding boxes and their corresponding categories.
    """
    # Load annotations for the given image id
    annotations = bbox_obj.loadAnns(bbox_obj.getAnnIds(img_id))

    bboxs = []  # List to store bounding boxes
    cats = []  # List to store categories

    # Iterate over all annotations
    img_metadata = bbox_obj.loadImgs(img_id)[0]
    w, h = img_metadata["width"], img_metadata["height"]
    for ann in annotations:
        # Skip annotations marked as 'crowd'
        if ann["iscrowd"] == 1:
            continue

        # Get bounding box and category
        bbox_loc = ann["bbox"]
        bbox_cat = bbox_obj.loadCats(ann["category_id"])
        assert len(bbox_cat) == 1
        bbox_cat = bbox_cat[0]["name"]

        # setup bounding box in the required layout format
        bbox_loc = [
            np.clip(bbox_loc[0], 0, w),
            np.clip(bbox_loc[1], 0, h),
            np.clip(bbox_loc[0] + bbox_loc[2], 0, w),
            np.clip(bbox_loc[1] + bbox_loc[3], 0, h),
        ]

        bbox_loc = unscale_bbox(bbox_loc, w, h)

        # Append bounding box and category to their respective lists
        bboxs.append(bbox_loc)
        cats.append(bbox_cat)

    # Return lists of bounding boxes and categories
    return bboxs, cats, w, h


def get_image_for_img_id(root, split, bbox_obj, img_id: int) -> Image.Image:
    """Get the PIL image for the given index.

    Args:
        index (int): The index of the image to get.

    Returns:
        Image: The PIL image.
    """
    split_name = split
    if split_name == "val":
        split_name = "validation"

    img_metadata = bbox_obj.loadImgs(img_id)[0]
    img_file_path = os.path.join(f"{root}/{split_name}/data", img_metadata["file_name"])
    return Image.open(img_file_path).convert("RGB")  # type: ignore


class COCODetection(Dataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory where data are downloaded to.

    Example:
    >>> from layousyn.datasets.coco import CocoDetection
    >>> z = CocoDetection("datasets/COCO17/raw/instances_train2017.json",  "datasets/COCO17/raw/captions_train2017.json", 100)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        load_image: bool = False,
        logger: Optional[Logger] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.load_image = load_image
        self.split = split

        # load annotations
        annFile = f"{root}/raw/instances_{split}2017.json"
        self.annotations = COCO(annFile)
        self.img_ids = list(sorted(self.annotations.imgs.keys()))

        # remove image ids with no bounding boxes
        self.img_ids = [
            img_id
            for img_id in self.img_ids
            if len(self.annotations.getAnnIds(img_id)) > 0
        ]

        if logger:
            logger.info(
                f"Initialized COCODetection dataset with root: {root}, split: {split}, load_image: {load_image}"
            )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        raise NotImplementedError("Debug before using this dataset")
        result_dict = {}
        img_id = self.img_ids[index]
        bbox, cats, w, h = get_bbox_for_img(self.annotations, img_id)

        result_dict["image_id"] = img_id
        result_dict["bboxs"] = bbox
        result_dict["cats"] = cats
        result_dict["caption"] = ""
        result_dict["width"] = w
        result_dict["height"] = h

        if self.load_image:
            result_dict["image"] = get_image_for_img_id(
                self.root, self.split, self.annotations, img_id
            )

        return result_dict

    def __len__(self) -> int:
        return len(self.img_ids)
    
    def get_categories(self):
        category_names = self.annotations.loadCats(self.annotations.getCatIds())
        return [category["name"] for category in category_names]

class COCOCaptioning(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        load_image: bool = False,
        transforms: Optional[Callable] = None,
        shuffle_bbox: bool = False,
        ignore_caption_id_file: Optional[str] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.load_image = load_image
        self.transforms = transforms
        self.shuffle_bbox = shuffle_bbox

        # load COCO objects
        self.bbox_obj = COCO(f"{root}/raw/instances_{split}2017.json")
        self.caption_obj = COCO(f"{root}/raw/captions_{split}2017.json")

        # get caption ids
        self.caption_ids = list(sorted(self.caption_obj.anns.keys()))

        # filter captions ids with no bounding boxes
        self.caption_ids = [
            caption_id
            for caption_id in self.caption_ids
            if len(
                self.bbox_obj.getAnnIds(
                    self.caption_obj.loadAnns([caption_id])[0]["image_id"]
                )
            )
            > 0
        ]

        # Ignore caption ids
        if ignore_caption_id_file:
            self.caption_ids = filter_caption_ids(
                self.caption_ids, ignore_caption_id_file
            )
            if logger:
                logger.info(
                    f"Ignoring captions in {ignore_caption_id_file}. New length: {len(self.caption_ids)}"
                )

        if logger:
            logger.info(
                f"Initialized COCOCaptioning dataset with root: {root}, split: {split}, load_image: {load_image}, transforms: {transforms}, shuffle_bbox: {shuffle_bbox}"
            )

    def get_image_for_caption_id(self, caption_id: int) -> Image.Image:
        """Get the PIL image for the given index.

        Args:
            index (int): The index of the image to get.

        Returns:
            Image: The PIL image.
        """
        img_id = self.caption_obj.loadAnns([caption_id])[0]["image_id"]
        return get_image_for_img_id(self.root, self.split, self.bbox_obj, img_id)

    def getitem_for_caption_id(self, caption_id: int) -> Any:
        # get caption annotation
        caption_anns = self.caption_obj.loadAnns([caption_id])
        assert len(caption_anns) == 1
        caption_ann = caption_anns[0]
        caption = caption_ann["caption"]  # type: ignore

        # clean caption of newline characters
        caption = cleanup_prompt(caption)

        # get bounding box information
        img_id = caption_ann["image_id"]
        bboxs, cats, w, h = get_bbox_for_img(
            self.bbox_obj,
            img_id,
        )

        # shuffle the bboxs and concept_embeddings
        if self.shuffle_bbox:
            bboxs, cats = shuffle_bboxs(bboxs, cats)

        # create result dictionary
        result_dict = {}
        result_dict["caption_id"] = caption_ann["id"]
        result_dict["caption"] = caption
        result_dict["image_id"] = img_id
        result_dict["bboxs"] = bboxs
        result_dict["cats"] = cats
        result_dict["width"] = w
        result_dict["height"] = h

        # get images
        if self.load_image:
            img = get_image_for_img_id(self.root, self.split, self.bbox_obj, img_id)
            img = self.transforms(img) if self.transforms else img
            result_dict["image"] = img

        return result_dict

    def __getitem__(self, index) -> Any:
        caption_id = self.caption_ids[index]
        return self.getitem_for_caption_id(caption_id)

    def __len__(self) -> int:
        return len(self.caption_ids)

    def get_captions(self, idx):
        caption = self.__getitem__(idx)["caption"]
        return [caption]


class COCOCaptionGrounded(COCOCaptioning):
    def __init__(
        self,
        root: str,
        grounded_dir: str,
        split: str = "train",
        load_image: bool = False,
        transforms: Optional[Callable] = None,
        shuffle_bbox: bool = False,
        crop_augment: bool = False,
        caption_augment: bool = False,
        ignore_caption_id_file: Optional[str] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            load_image=load_image,
            transforms=transforms,
            logger=logger,
        )
        self.shuffle_bbox = shuffle_bbox
        self.crop_augment = crop_augment
        self.caption_augment = caption_augment

        # Load grounded data
        data_file = f"{grounded_dir}/{split}.json"
        self.caption_id_data_map = {}
        metadatas = json.load(open(data_file, "r"))
        for metadata in metadatas:
            caption_id = metadata["caption_id"]
            if len(metadata["annotations"]) > 0:
                self.caption_id_data_map[caption_id] = metadata
        self.caption_ids = list(sorted(self.caption_id_data_map.keys()))

        # Ignore caption ids
        if ignore_caption_id_file:
            self.caption_ids = filter_caption_ids(
                self.caption_ids, ignore_caption_id_file
            )
            if logger:
                logger.info(
                    f"Ignoring captions in {ignore_caption_id_file}. New length: {len(self.caption_ids)}"
                )

        # if caption_augment:
        # load file grounded_dir/caption_augment/train.json
        if self.caption_augment:
            augment_data_file = f"datasets/COCOExtendedCaptions/{split}.json"
            augment_metadatas = json.load(open(augment_data_file, "r"))
            self.caption_id_captions_map = {}
            for metadata in augment_metadatas:
                caption_id = metadata["caption_id"]
                prompts = metadata["prompts"]
                if len(prompts) > 0:
                    self.caption_id_captions_map[caption_id] = [
                        cleanup_prompt(prompt) for prompt in prompts
                    ]

        # Log parameters
        if logger:
            logger.info(
                f"Initialized COCOCaptionGrounded dataset with grounded_dir: {grounded_dir}, shuffle_bbox: {shuffle_bbox}, crop_augment: {crop_augment}"
            )

    def __getitem__(self, index) -> Any:
        result_dict = super().__getitem__(index).copy()
        caption_id = result_dict["caption_id"]
        result_dict["gt_bboxs"] = result_dict["bboxs"]
        result_dict["gt_cats"] = result_dict["cats"]
        result_dict["caption"] = self.caption_id_data_map[caption_id]["caption"]

        # load grounded bounding boxes and categories
        grounded_bboxs = []
        grounded_cats = []
        for annotation in self.caption_id_data_map[caption_id]["annotations"]:
            bbox = annotation["box"]
            bbox = [
                np.clip(bbox[0], 0, result_dict["width"]),
                np.clip(bbox[1], 0, result_dict["height"]),
                np.clip(bbox[2], 0, result_dict["width"]),
                np.clip(bbox[3], 0, result_dict["height"]),
            ]
            obj = annotation["label"]

            # normalize bbox
            bbox = unscale_bbox(bbox, result_dict["width"], result_dict["height"])
            grounded_bboxs.append(bbox)
            grounded_cats.append(obj)

        # shuffle the bboxs and concept_embeddings
        if self.shuffle_bbox:
            grounded_bboxs, grounded_cats = shuffle_bboxs(grounded_bboxs, grounded_cats)

        if self.caption_augment:
            result_dict["caption"] = random.choice(self.get_captions(index))

        # data augment
        if self.crop_augment:
            grounded_bboxs, width, height = crop_layouts(
                bboxs=grounded_bboxs,
                width=result_dict["width"],
                height=result_dict["height"],
            )
            result_dict["width"] = int(width)
            result_dict["height"] = int(height)

        # update result_dict
        result_dict["layout"] = Layout(
            bboxs=grounded_bboxs,
            labels=grounded_cats,
            width=result_dict["width"],
            height=result_dict["height"],
        )

        return result_dict

    def get_captions(self, idx):
        data = super().__getitem__(idx).copy()
        caption_id = data["caption_id"]
        captions = [self.caption_id_data_map[caption_id]["caption"]]
        if self.caption_augment:
            captions.extend(self.caption_id_captions_map[data["caption_id"]][-3:])
        return captions
