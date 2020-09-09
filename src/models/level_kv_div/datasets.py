import torch
from torch.utils.data import TensorDataset
import pandas as pd


def get_dataset(raw_path, sim_path):
    t = torch.from_numpy(pd.read_csv(raw_path, header=None).to_numpy())
    sim = torch.load(sim_path, 'cpu')
    return TensorDataset(t, sim)


def get_raw_tensor(path):
    return torch.from_numpy(pd.read_csv(path, header=None).to_numpy())
