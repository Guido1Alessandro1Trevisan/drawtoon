import argparse
import json
import os
from typing import Any, Dict, List

from pytorch_fid.fid_score import calculate_fid_given_paths, save_fid_stats
from tqdm import tqdm

from layout_evaluation.layout import Layout, LayoutPlot


def load_layout_file(layout_file: str) -> List[Dict[str, Any]]:
    """
    Load layout data from a JSON file.

    Args:
        layout_file (str): Path to the layout JSON file.

    Returns:
        List[Dict[str, Any]]: List of layout data dictionaries.
    """
    with open(layout_file, "r") as f:
        layout_data = json.load(f)
    return layout_data


def generate_images_from_layout(
    layout_file: str,
    layout_plotter: LayoutPlot,
    save_dir: str,
    ID_KEY: str = "query_id",
) -> None:
    """
    Generate images from layout data and save them to disk.

    Args:
        layout_file (str): Path to the layout JSON file.
        layout_plotter (LayoutPlot): Instance of LayoutPlot to plot the layouts.
    """
    layouts = load_layout_file(layout_file)
    for layout_data in tqdm(layouts, desc="Generating Layouts as Images"):
        layout_id = layout_data[ID_KEY]
        bboxs, label = [], []
        for item in layout_data["object_list"]:
            label.append(item[0])
            bboxs.append([float(coord) for coord in item[1]])
        
        # normalize between 0 and 1
        bboxs = [[max(0.0, min(1.0, coord)) for coord in bbox] for bbox in bboxs]
        
        # generate layout
        layout = Layout(bboxs, label)
        if "iter" in layout_data:
            layout_id = f"{layout_id}_{layout_data['iter']}"
        layout_plotter.plot_bbox_on_img(
            layout,
            save_path=os.path.join(save_dir, f"{layout_id}.png"),
            width=512,
            height=512,
            fill_color=True,
        )


def evaluate_coco_grounded_fid(
    layout_file: str,
    evaluation_dir: str,
    device: str,
    cache_dir: str = os.path.expanduser("~/.cache/layout_evaluation/coco_grounded"),
    num_workers: int = 8,
) -> Dict[str, float]:
    """
    Calculate the FID score between real and synthetic images generated from layout data.

    Args:
        layout_file (str): Path to the layout JSON file.
        evaluation_dir (str): Directory to save generated synthetic images.
        device (str): Device to use for FID calculation (e.g., 'cpu', 'cuda').
        cache_dir (str): Directory to cache real images.
        num_workers (int): Number of workers to use for FID calculation.

    Returns:
        Dict[str, float]: Dictionary containing the FID score.
    """
    
    # generate a warning
    print(
        "Warning: Set export OMP_NUM_THREADS=1 to avoid potential issues with PyTorch and OpenMP. "
    )
    
    # layout plotting tool
    layout_plotter = LayoutPlot(
        color_map_path=os.path.join(
            os.path.dirname(__file__), "color_map_coco_grounded.json"
        ),
        device=device,
    )

    # Check if real images directory exists, if not create it
    real_images_dir = f"{cache_dir}/real_images"
    real_images_npz_path = f"{cache_dir}/real_images.npz"
    if not os.path.exists(real_images_npz_path):
        os.makedirs(real_images_dir)
        print(f"Generating real images at {real_images_dir}")
        generate_images_from_layout(
            os.path.join(os.path.dirname(__file__), "coco_grounded.val.json"),
            layout_plotter,
            real_images_dir,
            ID_KEY="id",
        )

        save_fid_stats(
            [real_images_dir, real_images_npz_path],
            batch_size=256,
            device=device,
            dims=2048,
            num_workers=num_workers,
        )

    # Check if synthetic images directory exists, if not create it
    synthetic_images_dir = f"{evaluation_dir}/coco_grounded/synthetic_images"
    synthetic_images_npz_path = (
        f"{evaluation_dir}/coco_grounded/synthetic_images.npz"
    )
    print(f"Generating synthetic images at {synthetic_images_dir}")
    if not os.path.exists(synthetic_images_dir):
        os.makedirs(synthetic_images_dir)
    generate_images_from_layout(layout_file, layout_plotter, synthetic_images_dir)
    save_fid_stats(
        [synthetic_images_dir, synthetic_images_npz_path],
        batch_size=256,
        device=device,
        dims=2048,
        num_workers=num_workers,
    )

    # Calculate FID score
    fid_value = calculate_fid_given_paths(
        [real_images_npz_path, synthetic_images_npz_path],
        batch_size=256,
        device=device,
        dims=2048,
        num_workers=num_workers,
    )

    # return the FID score
    return {"lfid": float(fid_value)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate layout file and calculate FID."
    )
    parser.add_argument("--layout_file", type=str, help="Path to the layout file")
    parser.add_argument(
        "--evaluation_dir",
        type=str,
        default="evaluation_output",
        help="Path to the temporary directory for generated images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for FID calculation (e.g., 'cpu', 'cuda')",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.expanduser("~/.cache/layout_evaluation/coco_grounded"),
        help="Directory to cache real images",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use for FID calculation",
    )

    args = parser.parse_args()
    results = evaluate_coco_grounded_fid(
        args.layout_file,
        args.evaluation_dir,
        args.device,
        args.cache_dir,
        args.num_workers,
    )
    print(results)
