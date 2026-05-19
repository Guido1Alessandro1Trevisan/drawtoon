import argparse
import os
from typing import Dict, Any

from .coco_grounded.evaluate import evaluate_coco_grounded_fid
from .nsr.eval_counting_layout import evaluate_counting
from .nsr.eval_spatial_layout import evaluate_spatial

def main(
    evaluation_name: str, 
    layout_file: str, 
    evaluation_dir: str, 
    generate_images: bool = False, 
    device: str = "cpu",
    gligen_batch_size: int = 8
) -> Dict[str, Any]:
    """
    Main function to evaluate different metrics for the given input file.

    Args:
        evaluation_name (str): The name of the evaluation metric to use.
        layout_file (str): Path to the layout file.
        evaluation_dir (str): Path to the directory containing the evaluation files.
        generate_images (bool): Whether to generate images for the evaluation.
        device (str): Device to use for evaluation (e.g., 'cpu', 'cuda').
        gligen_batch_size (int): Batch size for GLIGEN image generation.

    Returns:
        dict: Results of the evaluation.
    """
    
    image_dir = None
    if generate_images:
        # Generate images from layout if requested
        image_dir = os.path.join(evaluation_dir, "gligen_images")
        args = argparse.Namespace(
            folder=image_dir,
            batch_size=gligen_batch_size,
            no_plms=False,
            guidance_scale=7.5,
            file=layout_file,
            alpha1=0.3,
        )
        generate_image_from_layout(args)
        image_dir = os.path.join(image_dir, "outputs", "clean")
    
    
    # Evaluate based on the specified evaluation name
    if evaluation_name == "nsr_spatial":
        results = evaluate_spatial(
            fname=layout_file,
            ref_file_path=os.path.join(
                os.path.dirname(__file__), "nsr/spatial.val.json"
            ),
        )
    elif evaluation_name == "nsr_counting":
        results = evaluate_counting(
            fname=layout_file,
            ref_file_path=os.path.join(
                os.path.dirname(__file__), "nsr/counting.val.json"
            ),
        )
    elif evaluation_name == "coco_grounded_lfid":
        results = evaluate_coco_grounded_fid(
            layout_file, evaluation_dir=evaluation_dir, device=device
        )
    else:
        raise ValueError(f"Unsupported evaluation function: {evaluation_name}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate different metrics for the given input file."
    )
    parser.add_argument(
        "--evaluation_name",
        choices=["nsr_spatial", "nsr_counting", "coco_grounded_lfid"],
        help="The name of the evaluation metric to use.",
    )
    parser.add_argument(
        "--layout_file",
        type=str,
        default="datasets/NSR-1K/spatial/spatial.val.json",
        help="Path to the layout file.",
    )
    parser.add_argument(
        "--evaluation_dir",
        type=str,
        help="Path to the directory containing the evaluation files with file names matching the layout file.",
    )
    parser.add_argument(
        "--generate_images",
        action="store_true",
        help="Whether to generate images for the evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--gligen_batch_size",
        type=int,
        default=8,
        help="Batch size for GLIGEN image generation.",
    )

    args = parser.parse_args()
    results = main(
        args.evaluation_name,
        args.layout_file,
        args.evaluation_dir,
        args.generate_images,
        args.device,
        args.gligen_batch_size,
    )
    print(results)
