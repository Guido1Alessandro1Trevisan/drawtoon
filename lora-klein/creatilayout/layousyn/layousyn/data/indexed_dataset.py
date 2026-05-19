from torch.utils.data import Dataset

class IndexDataset(Dataset):
    """
    A PyTorch Dataset that includes the index of each item in the dataset.

    Attributes:
        dataset: The original dataset.
    """

    def __init__(self, dataset):
        """
        Initialize the IndexDataset.

        Args:
            dataset: The original dataset.
        """
        self.dataset = dataset  # Original dataset

    def __getitem__(self, index):
        """
        Get the item at the given index along with its index.

        Args:
            index: The index of the item.

        Returns:
            A tuple containing the index and the data at the index.
        """
        data = self.dataset[index]  # Get the data at the given index
        return index, *data  # Return the index and the data

    def __len__(self):
        """
        Get the number of items in the dataset.

        Returns:
            The number of items in the dataset.
        """
        return len(self.dataset)  # Return the number of items in the dataset
