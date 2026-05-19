import json
from enum import Enum
from typing import Dict, List, Optional, Union

import clip
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

TOL = 1e-6

# Enum for layout types
class LayoutType(str, Enum):
    XYXY = "xyxy"
    CXCYWH = "cxcywh"

# Class representing a layout with bounding boxes and labels
class Layout:
    @property
    def bboxs(self) -> Union[List[List[float]], np.ndarray]:
        return self._bboxs

    def __init__(
        self,
        bboxs: Union[List[List[float]], np.ndarray],
        labels: List[str],
        width: int = 1,
        height: int = 1,
        type: LayoutType = LayoutType.XYXY,
    ):
        """
        Initialize a Layout object.

        Args:
            bboxs (Union[List[List[float]], np.ndarray]): Bounding boxes.
            labels (List[str]): Labels corresponding to the bounding boxes.
            width (Optional[int]): Width of the layout.
            height (Optional[int]): Height of the layout.
            type (LayoutType): Type of the layout.
        """
        # Validate input types and lengths
        assert isinstance(bboxs, list) or isinstance(
            bboxs, np.ndarray
        ), "bboxs must be a list or numpy array"
        assert len(bboxs) == len(labels), "bboxs and labels must have the same length"
        if isinstance(bboxs, np.ndarray):
            assert bboxs.shape[1] == 4, "bboxs must have 4 elements"
            bboxs = [[float(x) for x in bbox] for bbox in bboxs]
            
        # assert no labels are empty
        for label in labels:
            assert label != "" or label is not None, "label cannot be empty or None"

        # Ensure all bbox coordinates are in [0, 1]
        for bbox in bboxs:
            assert all(
                [0.0 - TOL <= x <= 1.0 + TOL for x in bbox]
            ), f"bbox {bbox} is not normalized to [0, 1]"
        
        self._bboxs = bboxs
        self.labels = labels
        self.width = width
        self.height = height
        self.type = type

    def to(self, type: LayoutType) -> 'Layout':
        """
        Convert the layout to a different type.

        Args:
            type (LayoutType): The target layout type.

        Returns:
            Layout: The converted layout.
        """
        if type == self.type:
            return self

        # Convert between layout types
        if self.type == LayoutType.XYXY and type == LayoutType.CXCYWH:
            bboxs = [
                [
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2,
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]
                for bbox in self.bboxs
            ]
        elif self.type == LayoutType.CXCYWH and type == LayoutType.XYXY:
            bboxs = [
                [
                    bbox[0] - bbox[2] / 2,
                    bbox[1] - bbox[3] / 2,
                    bbox[0] + bbox[2] / 2,
                    bbox[1] + bbox[3] / 2,
                ]
                for bbox in self.bboxs
            ]
        else:
            raise ValueError(f"Unknown layout type {type}")

        # Clip bboxs to [0, 1]
        bboxs = [[max(0.0, min(1.0, x)) for x in bbox] for bbox in bboxs]

        # Return new layout
        return Layout(bboxs, self.labels, self.width, self.height, type)
    
    def __repr__(self) -> str:
        """
        Get a string representation of the layout.

        Returns:
            str: The string representation of the layout.
        """
        return f"Layout(bboxs={self.bboxs}, labels={self.labels}, width={self.width}, height={self.height}, type={self.type})"
    
    def __len__(self) -> int:
        """
        Get the number of bounding boxes in the layout.

        Returns:
            int: The number of bounding boxes in the layout.
        """
        return len(self.bboxs)

# Class for mapping labels to colors using CLIP embeddings
class LabelColorMap:
    def __init__(self, color_map: Dict[str, List[int]], batch_size: int = 256, device: str = "cuda"):
        """
        Initialize a LabelColorMap object.

        Args:
            color_map (Dict[str, List[int]]): A dictionary mapping category names to colors.
            batch_size (int): Batch size for computing CLIP embeddings.
            device (str): Device to use for computation.
        """
        self.color_map = color_map
        self.objects = list(color_map.keys())
        self.device = device

        # Get CLIP embeddings for all objects
        self.model, _ = clip.load("ViT-L/14", device=device)
        tokenized_text = clip.tokenize(self.objects).to(device)
        outputs = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(self.objects), batch_size),
                desc="Computing CLIP embeddings",
            ):
                text_features = (
                    self.model.encode_text(tokenized_text[i : i + batch_size])
                    .detach()
                    .cpu()
                    .numpy()
                ).astype(np.float32)
                outputs.append(text_features)
        outputs = np.concatenate(outputs, axis=0)
        outputs /= np.linalg.norm(outputs, axis=1, keepdims=True)
        self.object_embeddings = outputs

    def __getitem__(self, cat: str) -> List[int]:
        """
        Get the color for a given category.

        Args:
            cat (str): The category name.

        Returns:
            List[int]: The color corresponding to the category.
        """
        if cat in self.color_map:
            return self.color_map[cat]

        # Find the most similar object in the color map
        tokenized_text = clip.tokenize([cat]).to(self.device)
        with torch.no_grad():
            text_features = (
                self.model.encode_text(tokenized_text).detach().cpu().numpy()
            )
            text_features /= np.linalg.norm(text_features, axis=1, keepdims=True)
        similarities = np.dot(text_features, self.object_embeddings.T)
        most_similar_idx = np.argmax(similarities)
        most_similar_cat = self.objects[most_similar_idx]
        # print(f"Mapping {cat} to {most_similar_cat}")
        return self.color_map[most_similar_cat]

# Class for plotting layouts with bounding boxes on images
class LayoutPlot:
    def __init__(self, color_map_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize a LayoutPlot object.

        Args:
            color_map_path (str): Path to the color map JSON file.
            device (str): Device to use for computation.
        """
        # Load color map
        self.color_map = None
        self.device = device
        if color_map_path:
            self.color_map = LabelColorMap(
                self.load_color_map(color_map_path), device=device
            )
        self.cat_color_cache = {}

    @staticmethod
    def load_color_map(color_map_path: str) -> Dict[str, List[int]]:
        """
        Load a color map from a JSON file.

        Args:
            color_map_path (str): The path to the color map JSON file.

        Returns:
            Dict[str, List[int]]: A dictionary mapping category names to colors.
        """
        with open(color_map_path, "r") as f:
            color_map: Dict[str, List[int]] = json.load(f)
        return color_map

    def plot_bbox_on_img(
        self,
        layout: Layout,
        image: Optional[Image.Image] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        add_label_text: bool = False,
        fill_color: bool = False,
        save_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Plot bounding boxes on an image.

        Args:
            layout (Layout): The layout containing bounding boxes and labels.
            image (Optional[Image.Image]): The image to plot the bounding boxes on. If None, a white image of size 512x512 will be created.
            width (int): The width of the image to create if image is None.
            height (int): The height of the image to create if image is None.
            add_label_text (bool): Whether to add the category text to the bounding boxes.
            fill_color (bool): Whether to fill the bounding boxes with color.
            save_path (Optional[str]): The path to save the image with bounding boxes.

        Returns:
            Image.Image: The image with the bounding boxes plotted on it.
        """
        # Copy the image to avoid modifying the original image
        if image is None:
            # Draw a white area of size 512x512
            assert (
                width is not None and height is not None
            ), "width and height must be provided if img is None"
            image = Image.new("RGB", (width, height), (255, 255, 255))
        else:
            image = image.copy()

        # Ensure that bottom right corner is greater than top left corner
        width, height = image.size
        layout = layout.to(LayoutType.XYXY)
        
        # IMPORTANT: sort by cat to ensure consistent ordering of colors
        sorted_indices = np.argsort(layout.labels)
        bboxs = [layout.bboxs[i] for i in sorted_indices]
        labels = [layout.labels[i] for i in sorted_indices]
        
        # Iterate over all bounding boxes
        for bbox, cat in zip(bboxs, labels):
            color = self.color_map[cat] if self.color_map else [255, 0, 0]
            bbox_scaled = [
                bbox[0] * width,
                bbox[1] * height,
                bbox[2] * width,
                bbox[3] * height,
            ]

            # Draw bounding box on image
            draw = ImageDraw.Draw(image, "RGBA")
            outline_color = tuple(color)
            fill_color_rgba = tuple(color) + (100,)
            
            # Draw rectangle
            if fill_color:
                draw.rectangle(
                    bbox_scaled,
                    outline=outline_color,
                    fill=fill_color_rgba,
                )
            else:
                draw.rectangle(bbox_scaled, outline=outline_color)

            # Add label text
            if add_label_text:
                draw.text((bbox_scaled[0], bbox_scaled[1]), cat, fill="red")
                
        # Save image if save_path is provided
        if save_path is not None:
            image.save(save_path)

        # Return image with bounding boxes plotted on it
        return image
