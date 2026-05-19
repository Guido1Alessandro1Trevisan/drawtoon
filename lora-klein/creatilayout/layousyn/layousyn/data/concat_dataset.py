from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

        # print name and length of each dataset
        for dataset in self.datasets:
            print(f"{dataset.__class__.__name__} has {len(dataset )} samples")

    def __getitem__(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            else:
                index -= len(dataset)
        raise IndexError(f"Index {index} out of range")

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def get_captions(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset.get_captions(index)
            else:
                index -= len(dataset)
        raise IndexError(f"Index {index} out of range")

    def get_idx_info(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return {"identifier": dataset.__class__.__name__, "index": index}
            else:
                index -= len(dataset)
        raise IndexError(f"Index {index} out of range")
