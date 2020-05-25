import torch
from torch.utils.data import Dataset, DataLoader
import librosa


class DataSet(Dataset):
    def __init__(self, X, Rate):
        self.X = X
        self.Rate = Rate

    def __getitem__(self, index):
        x = self.X[index]
        audioData, _ = librosa.load(x, self.Rate)
        return audioData

    def __len__(self):
        return len(self.X)
