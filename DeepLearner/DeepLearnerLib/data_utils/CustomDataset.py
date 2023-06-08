import torch
import numpy as np


class GeomCnnDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        super(GeomCnnDataset, self).__init__()
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        print(f"Loading data ...")
        input = np.concatenate(self.transforms(self.image_files[idx]), axis=0)
        print(f"Loaded !!!")
        return input, self.labels[idx]
