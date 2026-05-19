from logging import Logger
from typing import Any, Dict, List

from torch.utils.data import Dataset

from .coco import COCOCaptionGrounded
from .concat_dataset import ConcatDataset
from .embedded_dataset import EmbeddedDataset
from .nsr1k import NSR1KSpatial, NSR1KNumerical
from .scale_dataset import ScaleDataset
from .utils import bound_xyxy_bbox, unnorm_bbox
from .grit_dataset import GriT
# drawtoon patch: in-tree MangaLayout dataset
from .manga_layout import MangaLayout

__all__ = [
    "COCOCaptionGrounded",
    "ConcatDataset",
    "EmbeddedDataset",
    "MangaLayout",
    "NSR1KSpatial",
    "ScaleDataset",
    "bound_xyxy_bbox",
    "unnorm_bbox",
]


def get_datasets(
    datasets_dict: Dict[str, Dict[str, Any]], logger: Logger
) -> List[Dataset]:
    """
    Get datasets from a dictionary of dataset configurations.
    """
    datasets = []
    for dataset_name, dataset_config in datasets_dict.items():
        dataset_config = dataset_config.copy()
        dataset_config["logger"] = logger

        # remove key scale_factor from dataset_config
        scale_factor = dataset_config.pop("scale_factor", 1)
        if dataset_name == "COCOCaptionGrounded":
            dataset = COCOCaptionGrounded(**dataset_config)
        elif (
            dataset_name == "NSR1KSpatial"
            or dataset_name == "COCOCaptionGroundedSpatial"
        ):
            dataset = NSR1KSpatial(**dataset_config)
        elif (
            dataset_name == "NSR1KNumerical"
        ):
            dataset = NSR1KNumerical(**dataset_config)
        elif dataset_name == "GRIT":
            dataset = GriT(**dataset_config)
        # drawtoon patch: MangaLayout dataset
        elif dataset_name == "MangaLayout":
            dataset = MangaLayout(**dataset_config)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # scale dataset if necessary
        if scale_factor > 1:
            dataset = ScaleDataset(dataset, scale_factor, logger)

        # append dataset to list
        datasets.append(dataset)
    return datasets
