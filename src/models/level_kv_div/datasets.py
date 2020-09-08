import torch
from torch.utils.data import TensorDataset
import pandas as pd


def get_dataset(path):
    t = torch.from_numpy(pd.read_csv(path, header=None).to_numpy())
    return TensorDataset(t)
