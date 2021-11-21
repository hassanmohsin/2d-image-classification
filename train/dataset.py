from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return self.subset.indices[idx], x, y # first element is the list of indices from the original dataset

    def __len__(self):
        return len(self.subset)
