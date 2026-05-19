import json
import random
import re
from logging import Logger
from typing import Callable, List, Optional

import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from layousyn.data.utils import bbox_horizontal_flip, cleanup_prompt, crop_layouts

from .coco import get_bbox_for_img, get_image_for_img_id
from layout_evaluation import Layout


class NSR1KSpatial(Dataset):
    # Data is present in a json files with contains items as list of dictionaries
    # Example:
    # {
    #     "id":1,
    #     "image_id":142589,
    #     "obj1":[
    #         "hot dog",
    #         [
    #             0.5768,
    #             0.16113999999999998,
    #             0.42319999999999997,
    #             0.6626599999999999
    #         ]
    #     ],
    #     "obj2":[
    #         "bowl",
    #         [
    #             0.0,
    #             0.18309999999999998,
    #             0.52588,
    #             0.673
    #         ]
    #     ],
    #     "prompt":"a hot dog to the right of a bowl",
    #     "relation":"right",
    #     "type":"template"
    # },
    def __init__(
        self,
        coco_root,
        nsr_root,
        split="train",
        load_image: bool = False,
        transforms: Optional[Callable] = None,
        shuffle_bbox=False,
        crop_augment=False,
        data_augmentation=False,
        use_gt_bboxs=False,
        logger: Optional[Logger] = None,
    ):
        self.coco_root = coco_root
        self.nsr_root = nsr_root
        self.split = split
        self.shuffle_bbox = shuffle_bbox
        self.crop_augment = crop_augment
        self.data_augmentation = data_augmentation
        self.load_image = load_image
        self.transforms = transforms
        self.use_gt_bboxs = use_gt_bboxs  # Note: GT bboxs are normalized and in format (top-left, width, height)

        file_name = f"{nsr_root}/spatial/spatial.{split}.json"
        self.bbox_obj_train = COCO(f"{coco_root}/raw/instances_train2017.json")
        self.bbox_obj_val = None
        if split == "val":
            self.bbox_obj_val = COCO(f"{coco_root}/raw/instances_val2017.json")
        self.data = json.load(open(file_name))

        if logger:
            logger.info(
                f"Loaded NSR1K spatial data coco_root: {coco_root}, nsr_root: {nsr_root}, split: {split}, shuffle_bbox: {shuffle_bbox}, load_image: {load_image}, transforms: {transforms}, horizontal_flip: {data_augmentation}, use_gt_bboxs: {use_gt_bboxs}, crop_augment: {crop_augment}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.get_clean_data(idx)

        # data augmentation
        if self.data_augmentation:
            if random.random() > 0.5:
                data["relation"], data["caption"], data["bboxs"] = self.augment_data(
                    data["type"], data["relation"], data["caption"], data["bboxs"]
                )

        # Shuffle bboxs
        if self.shuffle_bbox:
            if random.random() > 0.5:
                data["bboxs"] = data["bboxs"][::-1]
                data["cats"] = data["cats"][::-1]
                
        data["layout"] = Layout(
            bboxs=data["bboxs"],
            labels=data["cats"],
            width=data["width"],
            height=data["height"],
        )

        return data

    def get_clean_data(self, idx):
        data = self.data[idx].copy()
        data["caption"] = data["prompt"]
        data["bboxs"] = data["gt_bboxs"] = [data["obj1"][1], data["obj2"][1]]
        data["cats"] = data["gt_cats"] = [data["obj1"][0], data["obj2"][0]]
        data["image"] = None

        image_id = data["image_id"]
        if image_id == -1:
            print("No image found for idx", idx)
            return data

        # Load image and bboxs
        # Note: NSR 1K has samples from both COCO train and val
        # So we need to check both train and val for image_ids
        try:
            (
                data["bboxs"],
                data["cats"],
                data["width"],
                data["height"],
            ) = get_bbox_for_img(
                self.bbox_obj_train,
                data["image_id"],
            )
            image_split = "train"
            image_bbox_obj = self.bbox_obj_train
        except Exception as e:
            print(f"Error in getting bbox for {data['image_id']}, trying val")
            (
                data["bboxs"],
                data["cats"],
                data["width"],
                data["height"],
            ) = get_bbox_for_img(self.bbox_obj_val, data["image_id"])
            image_split = "val"
            image_bbox_obj = self.bbox_obj_val

        if self.use_gt_bboxs:
            # gt bboxs are normalized and in format (top-left, width, height)
            data["bboxs"] = data["gt_bboxs"]
            data["cats"] = data["gt_cats"]

            # Convert to xyxy format
            data["bboxs"] = [
                    [
                        np.clip(bbox[0], 0.0, 1.0),
                        np.clip(bbox[1], 0.0, 1.0),
                        np.clip(bbox[0] + bbox[2], 0.0, 1.0),
                        np.clip(bbox[1] + bbox[3], 0.0, 1.0),
                    ]
                for bbox in data["bboxs"]
            ]

        assert (
            len(data["bboxs"]) == 2
        ), f"Expected 2 bboxs, got {len(data["bboxs"])}"  # NSR 1K spatial only contains images with 2 objects

        # Load image
        if self.load_image:
            img = get_image_for_img_id(
                root=self.coco_root,
                split=image_split,
                bbox_obj=image_bbox_obj,
                img_id=data["image_id"],
            )
            img = self.transforms(img) if self.transforms else img
            data["image"] = img

        # data augment
        if self.crop_augment:
            data["bboxs"], width, height = crop_layouts(
                bboxs=data["bboxs"], width=data["width"], height=data["height"]
            )
            data["width"] = int(width)
            data["height"] = int(height)

        return data

    @staticmethod
    def augment_data(type_: str, relation: str, caption: str, bboxs: List[List[float]]):
        new_relation = relation
        new_caption = caption
        new_bboxs = bboxs
        if type_ == "template" and relation in ["left", "right"]:
            new_relation = "left" if relation == "right" else "right"
            new_caption = caption.replace(relation, new_relation)
            new_bboxs = bbox_horizontal_flip(bboxs)
        elif type_ == "template" and relation in ["bottom", "top"]:
            new_relation = "bottom" if relation == "top" else "top"

            # We want to change the relations, for example
            # A dog under a bed -> A bed on top of a dog
            regex = re.compile(
                rf"[a|A] (.+?) {NSR1KSpatial.get_prompt_command(relation)} [a|A] (.*)"
            )
            match = regex.match(caption)
            if match:
                obj1, obj2 = match.groups()
                new_caption = (
                    f"a {obj2} {NSR1KSpatial.get_prompt_command(new_relation)} a {obj1}"
                )
            else:
                print("Error in regex match", caption)

                # reset
                new_relation = relation
        elif relation in ["next to"]:
            # simply horizontal flip the bbox
            # both left and right count as next to
            new_bboxs = bbox_horizontal_flip(bboxs)

        return new_relation, new_caption, new_bboxs

    def get_captions(self, idx):
        captions = []
        data = self.get_clean_data(idx)

        # get augmented caption
        _, caption_aug, _ = self.augment_data(
            data["type"], data["relation"], data["caption"], data["bboxs"]
        )

        # append all captions
        captions.append(data["caption"])
        captions.append(caption_aug)

        return captions

    @staticmethod
    def get_prompt_command(relation):
        if relation == "left":
            return "to the left of"
        elif relation == "right":
            return "to the right of"
        elif relation == "top":
            return "on top of"
        elif relation == "bottom":
            return "under"
        else:
            return "near"


class NSR1KNumerical(Dataset):
    # Data is present in a json files with contains items as list of dictionaries
    # Example:
    # {
    #     "id":38699,
    #     "image_id":79657,
    #     "num_object":[
    #         [
    #             "clock",
    #             1
    #         ]
    #     ],
    #     "object_list":[
    #         [
    #             "clock",
    #             [
    #                 0.0,
    #                 0.175046875,
    #                 1.0,
    #                 0.645140625
    #             ]
    #         ]
    #     ],
    #     "prompt":"there is one clock in the image",
    #     "sub-type":"single-category",
    #     "type":"template"
    # },
    def __init__(
        self,
        coco_root,
        nsr_root,
        split="train",
        load_image: bool = False,
        transforms: Optional[Callable] = None,
        crop_augment=False,
        logger: Optional[Logger] = None,
    ):
        self.coco_root = coco_root
        self.nsr_root = nsr_root
        self.split = split
        self.crop_augment = crop_augment
        self.load_image = load_image
        self.transforms = transforms

        # load data
        file_name = f"{nsr_root}/counting/counting.{split}.json"
        self.bbox_obj_train = COCO(f"{coco_root}/raw/instances_train2017.json")
        self.bbox_obj_val = None
        if split == "val":
            self.bbox_obj_val = COCO(f"{coco_root}/raw/instances_val2017.json")
        self.data = json.load(open(file_name))

        if logger:
            logger.info(
                f"Loaded NSR1K numerical data coco_root: {coco_root}, nsr_root: {nsr_root}, split: {split}, load_image: {load_image}, transforms: {transforms}, crop_augment: {crop_augment}"
            )

    def get_clean_data(self, idx):
        data = self.data[idx].copy()
        data["caption"] = cleanup_prompt(data["prompt"])
        data["gt_bboxs"] = [item[1] for item in data["object_list"]]
        data["gt_cats"] = [item[0] for item in data["object_list"]]
        data["image"] = None

        # Load image and bboxs
        # Note: NSR 1K has samples from both COCO train and val
        # So we need to check both train and val for image_ids
        try:
            (
                data["bboxs"],
                data["cats"],
                data["width"],
                data["height"],
            ) = get_bbox_for_img(
                self.bbox_obj_train,
                data["image_id"],
            )
            image_split = "train"
            image_bbox_obj = self.bbox_obj_train
        except Exception as e:
            print(f"Error in getting bbox for {data['image_id']}, trying val")
            (
                data["bboxs"],
                data["cats"],
                data["width"],
                data["height"],
            ) = get_bbox_for_img(self.bbox_obj_val, data["image_id"])
            image_split = "val"
            image_bbox_obj = self.bbox_obj_val

        # Load image
        if self.load_image:
            img = get_image_for_img_id(
                root=self.coco_root,
                split=image_split,
                bbox_obj=image_bbox_obj,
                img_id=data["image_id"],
            )
            img = self.transforms(img) if self.transforms else img
            data["image"] = img

        # data augment
        if self.crop_augment:
            data["bboxs"], width, height = crop_layouts(
                bboxs=data["bboxs"], width=data["width"], height=data["height"]
            )
            data["width"] = int(width)
            data["height"] = int(height)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.get_clean_data(idx)
        data["layout"] = Layout(
            bboxs=data["bboxs"],
            labels=data["cats"],
            width=data["width"],
            height=data["height"],
        )

        return data
    
    def get_captions(self, idx):
        data = self.get_clean_data(idx)
        return [data["caption"]]