from torch.utils.data import Dataset

class ApplyfnDataset(Dataset):
    """
    A PyTorch Dataset that applies a function to a specific dimension of the data.

    Attributes:
        dataset: The original dataset.
        dim: The dimension to which the function will be applied.
        fn: The function to apply.
    """

    def __init__(self, dataset, dim, fn):
        """
        Initialize the ApplyfnDataset.

        Args:
            dataset: The original dataset.
            dim: The dimension to which the function will be applied.
            fn: The function to apply.
        """
        self.dataset = dataset  # Original dataset
        self.dim = dim  # Dimension to which the function will be applied
        self.fn = fn  # Function to apply

    def __getitem__(self, index):
        """
        Get the item at the given index after applying the function.

        Args:
            index: The index of the item.

        Returns:
            A tuple containing the data before the dimension, the transformed data at the dimension, and the data after the dimension.
        """
        data = self.dataset[index]  # Get the data at the given index
        # Apply the function to the data at the dimension and return the data before the dimension, the transformed data, and the data after the dimension
        return *data[:self.dim], self.fn(data[self.dim]), *data[self.dim+1:]

    def __len__(self):
        """
        Get the number of items in the dataset.

        Returns:
            The number of items in the dataset.
        """
        return len(self.dataset)  # Return the number of items in the dataset