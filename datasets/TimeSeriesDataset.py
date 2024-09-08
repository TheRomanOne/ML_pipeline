import torch
import numpy as np
from torch.utils.data import Dataset
from utils.utils import get_sequential_data
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path, seq_length, column):
        self.csv_path = csv_path
        self.seq_length = seq_length
        self.column = column
        self._load_csv()

    def _load_csv(self):
        self.df = pd.read_csv(self.csv_path)
        df = self.df[self.column]
        df = (df - df.min()) / (df.max() - df.min())
        X, y = get_sequential_data(df, self.seq_length)

        self.X_gt = torch.tensor([x.to_numpy() for x in X])
        self.y_gt = torch.tensor(y)


    def __len__(self):
        return len(self.X_gt)

    def __getitem__(self, idx):
        return self.X_gt[idx], self.y_gt[idx]
