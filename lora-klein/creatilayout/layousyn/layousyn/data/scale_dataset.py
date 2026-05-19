from logging import Logger
from typing import Optional

from torch.utils.data import Dataset


class ScaleDataset(Dataset):
    def __init__(self, dataset: Dataset, scale: int, logger: Optional[Logger] = None):
        """
        Scale a dataset by repeating it multiple times.
        """

        super().__init__()
        self.dataset = dataset

        assert type(scale) == int and scale > 0, "scale must be a positive integer"
        self.scale = scale

        if logger is not None:
            logger.info(f"Scaling dataset {dataset.__class__.__name__} by {scale}")

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]

    def __len__(self):
        return self.scale * len(self.dataset)
    
    def get_captions(self, index):
        return self.dataset.get_captions(index % len(self.dataset))
