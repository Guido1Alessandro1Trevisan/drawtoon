import argparse
import json
import os
from typing import Any, Dict, List, Union

import torch
import tqdm
from sentence_transformers import SentenceTransformer

from layout_evaluation import LayoutPlot, evaluate_coco_grounded_fid, LayoutType
from layousyn.config import Config
from layousyn.diffusion import create_diffusion
from layousyn.diffusion.gaussian_diffusion import GaussianDiffusion
from layousyn.model.preprocessor import Preprocessor
from layousyn.model.t5_google import T5EmbedderGoogle
from scripts.compare_images import process_folders

from .common import (
    extract_concepts_from_prompts,
    load_json,
    load_model,
    sample_with_cfg,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def batch(
    config: Config,
    data_ids: List[str],
    data_captions: List[str],
    data_labels: List[List[str]],
    diffusion: GaussianDiffusion,
    model: Any,
    preprocessor: Preprocessor,
    label_encoder: SentenceTransformer,
    caption_encoder: T5EmbedderGoogle,
    output_dir: str,
    sample_fn: str = "sample_with_cfg",
    ar_int: float = 1.0,
    height: int = 256,
    cfg_scale: float = 8.0,
    sampling_type: str = "ddim",
    device: Union[str, int] = "cuda",
) -> List[Dict[str, Any]]:
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
    output_json = []
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
        assert layout.type == LayoutType.XYXY, "Layout type should be XYXY"
        objects_list: List[List[Union[str, List[float]]]] = []
        for bbox, label in zip(layout.bboxs, layout.labels):
            objects_list.append([label, bbox])
        output_json.append(
            {"object_list": objects_list, "prompt": caption, "query_id": id, "iter": 0}
        )

    return output_json

def evaluate_coco_grounded(
    config: Config,
    diffusion: GaussianDiffusion,
    model: Any,
    preprocessor: Preprocessor,
    label_encoder: SentenceTransformer,
    caption_encoder: T5EmbedderGoogle,
    val_file: str,
    output_dir: str,
    partial_file: bool = False,
    sample_fn: str = "sample_with_cfg",
    ar_int: float = 1.0,
    height: int = 256,
    cfg_scale: float = 8.0,
    sampling_type: str = "ddim",
    batch_size: int = 3200,
    device: Union[str, int] = "cuda",
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
) -> Dict[str, Any]:

    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/layouts", exist_ok=True)

    # load dataset
    input_file = load_json(val_file)
    if partial_file:
        input_file = input_file[:8705]
        print(f"Using only {len(input_file)} samples for evaluation")

    # setup data
    data_ids = [x["id"] for x in input_file]
    data_captions = [x["prompt"] for x in input_file]

    # extract labels
    data_labels = extract_concepts_from_prompts(data_captions, model_id=model_id)
    data_labels = [data_label[: config.max_in_len] for data_label in data_labels]

    # run in batch with tqdm
    batch_size = batch_size
    output_json = []
    for i in tqdm.tqdm(range(0, len(data_ids), batch_size)):
        batch_data_ids = data_ids[i : i + batch_size]
        batch_data_captions = data_captions[i : i + batch_size]
        batch_data_labels = data_labels[i : i + batch_size]

        output_json += batch(
            config,
            batch_data_ids,
            batch_data_captions,
            batch_data_labels,
            diffusion,
            model,
            preprocessor,
            label_encoder,
            caption_encoder,
            output_dir,
            sample_fn=sample_fn,
            ar_int=ar_int,
            height=height,
            cfg_scale=cfg_scale,
            sampling_type=sampling_type,
            device=device,
        )

    # save output json containing layout information
    output_file_path = f"{output_dir}/output.json"
    with open(output_file_path, "w") as f:
        json.dump(output_json, f)

    # evaluate the generated layouts
    result_dict: Dict[str, Any] = evaluate_coco_grounded_fid(
        layout_file=output_file_path,
        evaluation_dir=output_dir,
        device=device,  # type: ignore
    )
    return result_dict


def run_coco_grounded_evaluation(
    config: Config,
    model: Any,
    preprocessor: Preprocessor,
    diffusion: GaussianDiffusion,
    label_encoder: SentenceTransformer,
    caption_encoder: T5EmbedderGoogle,
    val_file: str,
    output_dir: str,
    partial_file: bool = False,
    cfg_scales: List[float] = [1.0, 2.0, 4.0, 8.0],
    ar: float = 1.0,
    height: int = 256,
    sampling_type: str = "ddim",
    batch_size: int = 3200,
    device: Union[str, int] = "cuda",
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
) -> Dict[str, Any]:

    # initialize results dictionary
    results = {}
    result_folders = []

    # run cfg evaluation
    for cfg_scale in cfg_scales:
        cfg_out_dir = f"{output_dir}/CFG_{cfg_scale}"
        result_dict = evaluate_coco_grounded(
            config,
            sample_fn="sample_with_cfg",
            diffusion=diffusion,
            model=model,
            preprocessor=preprocessor,
            label_encoder=label_encoder,
            caption_encoder=caption_encoder,
            val_file=val_file,
            output_dir=cfg_out_dir,
            partial_file=partial_file,
            ar_int=ar,
            height=height,
            cfg_scale=cfg_scale,
            sampling_type=sampling_type,
            batch_size=batch_size,
            device=device,
            model_id=model_id,
        )
        results[f"CFG_{cfg_scale}"] = result_dict
        result_folders.append(cfg_out_dir)

    # add a combined result for comparison purposes
    result_folders = [f"{dir}/layouts" for dir in result_folders]
    process_folders(result_folders, f"{output_dir}/combined")

    # write results dict to JSON file
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=str, default="evaluation_output/coco_grounded_fid")

    # arguments to control the layout
    parser.add_argument(
        "--val-file",
        type=str,
        default="layout_evaluation/coco_grounded/coco_grounded.val.json",
        help="Path to a JSON file to evaluate",
    )
    parser.add_argument(
        "--partial-file",
        action="store_true",
        help="Use only a subset of the validation file for evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2400,
        help="Batch size for layout generation",
    )

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
        "--cfg-scales", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0]
    )
    parser.add_argument("--sampling-type", type=str, default="ddim")
    parser.add_argument("--num-sampling-steps", type=str, default="")
    parser.add_argument(
        "--model-id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )

    # arguments to load layout generation trained model
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to a model checkpoint",
    )
    parser.add_argument(
        "--ckpt-config",
        type=str,
        required=True,
        help="Path to a model checkpoint",
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
        "sentence-transformers/sentence-t5-base",
        device=device,
    )
    caption_encoder = T5EmbedderGoogle(
        dir_or_name=f"t5-v1_1-{config.t5_size}",
        device=device,
        model_max_length=config.max_y_len,
    )

    # setup a diffusion class
    diffusion = create_diffusion(
        args.num_sampling_steps,
        alpha_scale=config.scale,
        noise_schedule=config.noise_schedule,
        diffusion_steps=config.diffusion_steps,
    )

    # Preprocessor
    preprocessor = Preprocessor(config.layout_type).to(device)

    # run evaluation
    run_coco_grounded_evaluation(
        config=config,
        model=model,
        preprocessor=preprocessor,
        diffusion=diffusion,
        label_encoder=label_encoder,
        caption_encoder=caption_encoder,
        val_file=args.val_file,
        output_dir=args.eval_dir,
        partial_file=args.partial_file,
        cfg_scales=args.cfg_scales,
        ar=args.ar,
        height=args.height,
        sampling_type=args.sampling_type,
        batch_size=args.batch_size,
        device=device,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()
