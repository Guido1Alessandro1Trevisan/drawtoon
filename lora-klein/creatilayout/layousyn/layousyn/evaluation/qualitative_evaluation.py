# Example: python -m evaluation.qualitative_evaluation --ckpt results/DiT-XS-1721691583-8807976/checkpoints/0650000.pt --ckpt-config results/DiT-XS-1721691583-8807976/config.json

"""
Sample new images from a pre-trained DiT.
"""
import argparse
import json
import os
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from layout_evaluation import LayoutPlot, LayoutType
from layousyn.config import Config
from layousyn.diffusion import create_diffusion
from layousyn.diffusion.gaussian_diffusion import GaussianDiffusion
from layousyn.model.preprocessor import Preprocessor
from layousyn.model.t5_google import T5EmbedderGoogle
from scripts.compare_images import process_folders

from .common import (
    load_model,
    sample_with_cfg,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


EXAMPLES = [
    {
        "id": 0,
        "caption": "A fire hydrant beside a house",
        "labels": ["fire hydrant", "house"],
    },
    {
        "id": 1,
        "caption": "A bus under a traffic light",
        "labels": ["bus", "traffic light"],
    },
    {
        "id": 2,
        "caption": "Two teddy bears and a stuffed snowman wearing hats",
        "labels": ["teddy bear", "teddy bear", "snowman", "hat", "hat", "hat"],
    },
    {
        "id": 3,
        "caption": "Three elephants standing beside a pool of water.",
        "labels": ["elephant", "elephant", "elephant", "water pool"],
    },
    {
        "id": 4,
        "caption": "A person holding a yellow umbrella",
        "labels": ["person", "umbrella"],
    },
    {
        "id": 5,
        "caption": "men posing on street with cars and trees infront of mountain range and clouds",
        "labels": [
            "man",
            "man",
            "tree",
            "tree",
            "car",
            "car",
            "street",
            "mountain",
            "cloud",
            "cloud",
        ],
    },
    {
        "id": 6,
        "caption": "A man riding a horse on the street",
        "labels": ["man", "horse", "street"],
    },
]


def evaluate_qualitative(
    config: Config,
    diffusion: Any,
    model: nn.Module,
    preprocessor: Preprocessor,
    label_encoder: Any,
    caption_encoder: Any,
    output_dir: str,
    sample_fn="sample_with_cfg",
    ar_int: float = 1.0,
    height: int = 256,
    cfg_scale: float = 8.0,
    sampling_type="ddim",
    device: Union[str, int] = "cuda",
) -> None:
    # create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/layouts", exist_ok=True)

    # setup data
    data_ids = [example["id"] for example in EXAMPLES]
    data_captions = [example["caption"] for example in EXAMPLES]
    data_labels = [example["labels"] for example in EXAMPLES]

    # sample layouts
    if sample_fn == "sample_with_cfg":
        output_bboxs = sample_with_cfg(
            captions=data_captions,
            labels_set=data_labels,
            config=config,
            diffusion=diffusion,
            model=model,
            label_encoder=label_encoder,
            caption_encoder=caption_encoder,
            cfg_scale=cfg_scale,
            ar_int=ar_int,
            type=sampling_type,
            device=device,
        )
    else:
        raise ValueError(f"Unknown sample function: {sample_fn}")

    # plotting function
    layout_plotter = LayoutPlot()

    # postprocessing
    layouts = preprocessor.to_layout(output_bboxs, data_labels)
    layouts = [layout.to(LayoutType.XYXY) for layout in layouts]

    # plot layout and save to sample_layout.png
    height = height
    width = int(height * ar_int)
    output_json: List[Dict[str, Any]] = []
    for id, caption, layout in zip(data_ids, data_captions, layouts):
        # plot layouts
        _ = layout_plotter.plot_bbox_on_img(
            layout,
            width=width,
            height=height,
            save_path=f"{output_dir}/layouts/{id}_0.png",
            add_label_text=True,
        )

        # write data to output file
        objects_list: List[List[Union[str, List[float]]]] = []
        for bbox, label in zip(layout.bboxs, layout.labels):
            objects_list.append([label, bbox])
        output_json.append(
            {
                "iter": 0,
                "object_list": objects_list,
                "prompt": caption,
                "query_id": id,
            }
        )

    # save the layout to a json file
    with open(f"{output_dir}/layouts.json", "w") as f:
        json.dump(output_json, f)


def run_qualitative_evaluation(
    config: Config,
    model: Any,
    preprocessor: Preprocessor,
    diffusion: GaussianDiffusion,
    label_encoder: Any,
    caption_encoder: Any,
    output_dir: str,
    cfg_scales: List[float] = [1.0, 2.0, 4.0, 8.0],
    ar: float = 1.0,
    height: int = 256,
    sampling_type: str = "ddim",
    device: Union[str, int] = "cuda",
) -> None:

    # initialize results dictionary
    result_folders = []

    # run cfg evaluation
    for cfg_scale in cfg_scales:
        cfg_out_dir = f"{output_dir}/CFG_{cfg_scale}"
        evaluate_qualitative(
            config,
            sample_fn="sample_with_cfg",
            diffusion=diffusion,
            model=model,
            preprocessor=preprocessor,
            label_encoder=label_encoder,
            caption_encoder=caption_encoder,
            output_dir=cfg_out_dir,
            ar_int=ar,
            height=height,
            cfg_scale=cfg_scale,
            sampling_type=sampling_type,
            device=device,
        )
        result_folders.append(cfg_out_dir)

    # add a combined result for comparison purposes
    result_folders = [f"{dir}/layouts" for dir in result_folders]
    process_folders(result_folders, f"{output_dir}/combined")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="evaluation_output/quantitative")

    # arguments to control layout generation quality
    parser.add_argument(
        "--height", type=int, default=256, help="Height of the generated image"
    )
    parser.add_argument(
        "--ar",
        type=float,
        default=1.0,
        help="aspect ratio i.e width/height (default: portrait=0.64072xxx)",
    )
    parser.add_argument(
        "--cfg-scales", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0]
    )
    parser.add_argument("--sampling-type", type=str, default="ddim")
    parser.add_argument("--num-sampling-steps", type=int, default=1000)

    # arguments to load layout generation trained model
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to a LDiT checkpoint",
    )
    parser.add_argument(
        "--ckpt-config",
        type=str,
        required=True,
        help="Path to a LDiT checkpoint",
    )

    # load arguments
    args = parser.parse_args()
    config = Config.from_json(args.ckpt_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # disable gradient computation
    torch.set_grad_enabled(False)

    # setup model
    model = load_model(args.ckpt, config, device=device)

    # setup encoders
    label_encoder = SentenceTransformer(
        f"sentence-transformers/sentence-t5-{config.t5_size}",
        device=device,  # type: ignore
    )
    caption_encoder = T5EmbedderGoogle(
        dir_or_name=f"t5-v1_1-{config.t5_size}",
        device=device,
        model_max_length=config.max_y_len,
    )

    # setup a diffusion class
    diffusion = create_diffusion(
        str(args.num_sampling_steps),
        alpha_scale=config.scale,
        noise_schedule=config.noise_schedule,
        diffusion_steps=config.diffusion_steps,
    )

    # Preprocessor
    preprocessor = Preprocessor(config.layout_type).to(device)

    # ensure output directory exists
    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

    # run evaluation
    run_qualitative_evaluation(
        config=config,
        model=model,
        preprocessor=preprocessor,
        diffusion=diffusion,
        label_encoder=label_encoder,
        caption_encoder=caption_encoder,
        output_dir=output_dir,
        cfg_scales=args.cfg_scales,
        ar=args.ar,
        height=args.height,
        sampling_type=args.sampling_type,
        device=device,
    )


if __name__ == "__main__":
    main()
