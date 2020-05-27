import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np


class DataSet(Dataset):
    def __init__(self, X, Rate=16000):
        self.X = X
        self.Rate = Rate

    def __getitem__(self, index):
        x = self.X[index]
        x = np.load(x)
        x = np.expand_dims(x, 0)
        x = np.concatenate((x, x, x), 0)
        return x

    def __len__(self):
        return len(self.X)
