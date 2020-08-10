import torch
from torch import nn
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset



def far_func(sorted_dist: torch.tensor, indices: torch.tensor):
    return sorted_dist[:, -1], indices[:, -1]


def calculate_distance(x, far_fn):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x_device = x.to(device)
    # TODO: we first assume memory fits in memory. Later, we can process the data in batches.
    dist = torch.cdist(x1=x_device, x2=x_device, p=2)  # (n, n)
    sorted_dist, indices = torch.sort(dist, dim=1, descending=False)
    sorted_dist, indices = sorted_dist.cpu(), indices.cpu()
    anchor_idx = torch.arange(x.shape[0])  # (n,)
    # the 0-th column is the distance to oneself
    close_distance, close_idx = sorted_dist[:, 1], indices[:, 1]  # (n,)
    far_distance, far_idx = far_fn(sorted_dist, indices)  # (n, r)
    return anchor_idx, close_idx, far_idx, close_distance, far_distance


def make_triplets(x, far_fn):
    anchor_idx, close_idx, far_idx, close_distance, far_distance = calculate_distance(x, far_fn)
    n, r = far_idx.shape
    anchor_idx = anchor_idx.view(-1, 1).expand(-1, r).view(-1)  # (n, r)
    close_idx = close_idx.view(-1, 1).expand(-1, r).view(-1)  # (n, r)
    far_idx = far_idx.view(-1)
    return anchor_idx, close_idx, far_idx


class TripletDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    @classmethod
    def from_df(cls, path_to_dataframe):
        x = torch.from_numpy(pd.read_csv(path_to_dataframe, header=None).to_numpy()).to(torch.float32)
        pairs, labels = make_pairs(x, far_func)
        return cls(pairs, labels)

    @classmethod
    def from_dataset(cls, path_to_tensor):
        return torch.load(path_to_tensor)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TripletNet(nn.Module):
    def __init__(self, encoder):
        super(Tripletnet, self).__init__()
        self.encoder = encoder

    def forward(self, x, y, z):
        embedded_x = self.encoder(x)
        embedded_close = self.encoder(y)
        embedded_far = self.encoder(z)
        dist_close = F.pairwise_distance(embedded_x, embedded_close, 2)
        dist_far = F.pairwise_distance(embedded_x, embedded_far, 2)
        return dist_close, dist_far, embedded_x, embedded_close, embedded_far

