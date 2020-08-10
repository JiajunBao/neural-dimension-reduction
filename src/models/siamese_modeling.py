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


def make_pairs(x, far_fn):
    anchor_idx, close_idx, far_idx, close_distance, far_distance = calculate_distance(x, far_fn)
    n, r = far_idx.shape
    anchor_idx = anchor_idx.view(-1, 1)  # (n, 1)
    close_idx = close_idx.view(-1, 1)  # (n, 1)
    positive_pairs = torch.cat((anchor_idx, close_idx), dim=1)  # (n, 2)
    positive_labels = torch.ones(n, dtype=torch.int64)  # (n, )
    far_idx = far_idx.view(-1)  # (n * r, )
    anchor_idx_flatten = anchor_idx.expand(-1, r).view(-1)  # (n * r, )
    negative_pairs = torch.cat((anchor_idx_flatten, far_idx), dim=1)  # (n * r, 2)
    negative_labels = torch.zeros(n * r, dtype=torch.int64)  # (n * r, )
    pairs = torch.cat((positive_pairs, negative_pairs), dim=0)
    labels = torch.cat((positive_labels, negative_labels), dim=0)
    return pairs, labels


class SiameseDataSet(Dataset):
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


class SiameseNet(nn.Module):
    def __init__(self, dim_in, cin=1, cout=8):
        super(SiameseNet, self).__init__()
        self.encoder = nn.Sequential(
            OrderedDict([
                ('bn1', nn.BatchNorm1d(cin)),
                ('relu1', nn.ReLU()),
                ('conv1', nn.Conv1d(cin, cout // 4, kernel_size=1, stride=1)),
                ('bn2', nn.BatchNorm1d(cout // 4)),
                ('relu2', nn.ReLU()),
                ('conv2', nn.Conv1d(cout // 4, cout // 4, kernel_size=3, stride=1, padding=1)),
                ('bn3', nn.BatchNorm1d(cout // 4)),
                ('relu3', nn.ReLU()),
                ('conv3', nn.Conv1d(cout // 4, cout, kernel_size=1)),
                ('flatten', nn.Flatten()),
                ('linear', nn.Linear(cout * dim_in, dim_in)),
                ('sigmoid', nn.Sigmoid())
            ]))
        self.out = nn.Linear(dim_in, 1)

    def forward(self, x1, x2, labels=None):
        out1 = self.encoder(x1)
        out2 = self.encoder(x2)
        dis = torch.abs(out1 - out2)
        logits = self.out(dis)
        if labels:
            loss_fn = nn.BCEWithLogitsLoss(size_average=True)
            loss = loss_fn(logits, labels)
            return logits, loss
        return logits
