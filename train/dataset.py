import numpy as np
import pandas as pd
import torch
from PIL import Image as Im
from torch.utils.data import Dataset

np.random.seed(42)


class CustomDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return self.subset.indices[idx], x, y  # first element is the list of indices from the original dataset

    def __len__(self):
        return len(self.subset)


class CombinedDataset(Dataset):
    """
    pos_samples: csv file containing positive sample files
    neg_samples: csv file containing negative sample files
    n_samples: number of samples to contain in the dataset
    label_ratio: positive/negative sample ratio
    data_ratio: probabilistic distribution for choosing one of the subsets, i.e., 40P, 50P, 60P or 80P
    """

    def __init__(self,
                 pos_samples,
                 neg_samples,
                 n_samples=None,
                 label_ratio=0.5,
                 data_ratio=[0.25, 0.25, 0.25, 0.25],
                 transform=None):
        self.pos_samples = pd.read_csv(pos_samples, nrows=n_samples)
        self.neg_samples = pd.read_csv(neg_samples, nrows=n_samples)
        self.label_ratio = label_ratio
        self.data_ratio = data_ratio
        self.transform = transform

    def __getitem__(self, idx):
        label = True if np.random.rand() < self.label_ratio else False
        subset = np.random.choice(['40P', '50P', '60P', '80P'], 1, self.data_ratio)[0]
        img_file = self.pos_samples.loc[idx, subset] if label else self.neg_samples.loc[idx, subset]
        image = Im.open(img_file).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return min(len(self.pos_samples), len(self.neg_samples))
