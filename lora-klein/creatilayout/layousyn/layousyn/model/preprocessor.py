from typing import List

import torch
import torch.nn as nn

from layout_evaluation import Layout, LayoutType
from layousyn.utils import bound_xyxy_bbox


class Preprocessor(nn.Module):
    def __init__(self, layout_type: LayoutType):
        """
        Initializes the Preprocessor with a given layout type.

        Args:
            layout_type (LayoutType): The type of layout to be processed.
        """
        super().__init__()
        self.layout_type = layout_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, 4) where N is the number of boxes.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Normalize the input tensor to the range [-1, 1]
        x = x * 2.0 - 1.0
        return x

    def to_layout(self, x: torch.Tensor, batched_cats: List[List[str]]) -> List[Layout]:
        """
        Converts the normalized tensor back to layout objects.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, 4) where N is the number of boxes.
            batched_cats (List[List[str]]): List of lists containing category labels for each batch.

        Returns:
            List[Layout]: List of Layout objects.
        """
        # Denormalize the tensor back to the original range
        x = (x + 1.0) / 2.0

        layouts = []
        for batch_idx in range(x.shape[0]):
            # Get the corresponding category labels
            cats = batched_cats[batch_idx]

            # Convert tensor to list of bounding boxes
            bboxs = x[batch_idx].tolist()[: len(cats)]

            if self.layout_type == LayoutType.XYXY:
                # ensure that bottom right corner is greater than top left corner
                bboxs = [bound_xyxy_bbox(bbox) for bbox in bboxs]
            elif self.layout_type == LayoutType.CXCYWH:
                # ensure that all coordinates are between 0 and 1
                bboxs = [
                    [
                        max(0, min(1.0, bbox[0])),
                        max(0, min(1.0, bbox[1])),
                        max(0, min(1.0, bbox[2])),
                        max(0, min(1.0, bbox[3])),
                    ]
                    for bbox in bboxs
                ]

            # Create a Layout object
            layout = Layout(bboxs=bboxs, labels=cats, type=self.layout_type)

            # Append the Layout object to the list
            layouts.append(layout)

        return layouts
