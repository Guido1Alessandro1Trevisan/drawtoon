import json
from dataclasses import asdict, dataclass
from enum import Enum
import logging
from typing import Dict, Union

from layout_evaluation import LayoutType

@dataclass
class Config:
    # one line summary of the experiment
    overview: str

    # Arguments for saving/loading data
    save_embed: bool  # whether to save embeddings
    embed_dir: str  # path to save embeddings
    result_dir: str  # path to save results

    # Arguments for datasets
    datasets: Dict[str, Dict[str, str]]  # dataset name to dataset config

    # Arguments for embedders
    concept_embedder_batch_size: int  # batch size for concept embedder
    caption_embedder_batch_size: int  # batch size for caption embedder

    # Arguments for model
    model: str  # model name
    in_channel: int  # number of channels for input
    concept_in_channel: int  # number of channels for concepts
    y_in_channel: int  # number of channels for y
    max_in_len: int  # maximum length of layout
    max_y_len: int  # maximum length of caption
    scale: float # scale for the diffusion model
    noise_schedule: str
    layout_type:LayoutType

    # Arguments for training
    diffusion_steps: int  # number of diffusion steps
    epochs: int  # number of epochs to train
    global_batch_size: int  # global batch size
    global_seed: int  # global seed for reproducibility
    num_workers: int  # number of workers for dataloader
    log_every: int  # log every n steps
    ckpt_every: int  # checkpoint every n steps
    lr: float  # learning rate

    # Arguments for evaluation
    t5_size: str  # size of T5 model

    # drawtoon patch: optional path to a val JSONL (same MangaLayout schema).
    # When set, train.py runs a val-loss pass at the end of each epoch.
    val_jsonl_path: Union[str, None] = None

    @classmethod
    def from_json(cls, file_path: str):
        logger = logging.getLogger(__name__)
        with open(file_path, "r") as f:
            data = json.load(f)

        if "scale" not in data:
            logger.warning("WARNING: scale not found in config, setting to 1.0")
            data["scale"] = 1.0

        if "noise_schedule" not in data:
            logger.warning("WARNING: noise_schedule not found in config, setting to linear")
            data["noise_schedule"] = "linear"

        # check if diffusion steps is present
        if "diffusion_steps" not in data:
            logger.warning("WARNING: diffusion_steps not found in config, setting to 1000")
            data["diffusion_steps"] = 1000

        if "y_in_channel" not in data:
            logger.warning("WARNING: y_in_channel not found in config, setting to None")
            data["y_in_channel"] = None

        if "max_y_len" not in data:
            logger.warning("WARNING: max_y_len not found in config, setting to None")
            data["max_y_len"] = None

        if "caption_embedder_batch_size" not in data:
            logger.warning("WARNING: caption_embedder_batch_size not found in config, setting to None")
            data["caption_embedder_batch_size"] = None
        
        if "t5_size" not in data:
            logger.warning("WARNING: t5_size not found in config, setting to None")
            data["t5_size"] = None

        if "embed_dir" not in data:
            logger.warning("WARNING: embed_dir not found in config, setting to None")
            data["embed_dir"] = None

        if "save_embed" not in data:
            logger.warning("WARNING: save_embed not found in config, setting to None")
            data["save_embed"] = None
            
        if "lr" not in data:
            logger.warning("WARNING: lr not found in config, setting to 1e-4")
            data["lr"] = 1e-4
        
        if "val_jsonl_path" not in data:
            data["val_jsonl_path"] = None

        if "layout_type" not in data:
            logger.warning("WARNING: layout_type not found in config, setting to xyxy")
            data["layout_type"] = LayoutType.XYXY
        elif data["layout_type"] == "xyxy":
            data["layout_type"] = LayoutType.XYXY
        elif data["layout_type"] == "cxcywh":
            data["layout_type"] = LayoutType.CXCYWH
        else:
            raise ValueError(f"Invalid layout type: {data['layout_type']}. Must be either 'xyxy' or 'cxcywh'.")

        return cls(**data)

    def to_json(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    def to_dict(self):
        return asdict(self)
