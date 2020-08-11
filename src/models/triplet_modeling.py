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
    def __init__(self, data, anchor_idx, close_idx, far_idx):
        self.data = data
        self.anchor_idx = anchor_idx
        self.close_idx = close_idx
        self.far_idx = far_idx

    @classmethod
    def from_df(cls, path_to_dataframe):
        data = torch.from_numpy(pd.read_csv(path_to_dataframe, header=None).to_numpy()).to(torch.float32)
        anchor_idx, close_idx, far_idx = make_triplets(data, far_func)
        return cls(data, anchor_idx, close_idx, far_idx)

    @classmethod
    def from_dataset(cls, path_to_dataset):
        return torch.load(path_to_dataset)

    def __len__(self):
        return len(self.anchor_idx)

    def __getitem__(self, idx):
        anchor, pos, neg = self.anchor_idx[idx], self.close_idx[idx], self.far_idx[idx]
        return self.data[anchor], self.data[pos], self.data[neg]


class TripletNet(nn.Module):
    def __init__(self, encoder):
        super(TripletNet, self).__init__()
        self.encoder = encoder

    def encode_batch(self, x):
        return self.encoder(x)

    @staticmethod
    def decode_batch(out1, out2):
        return (out2 - out1).pow(2).sum(dim=1).sqrt()  # distances (after squared root)

    def forward(self, x, y, z, labels=None):
        embedded_anchor = self.encoder(x)
        embedded_pos = self.encoder(y)
        embedded_neg = self.encoder(z)
        if labels is not None:
            loss_fn = TripletLoss(1.)
            distance_pos, distance_neg, loss = loss_fn(embedded_anchor, embedded_pos, embedded_neg)
            return distance_pos, distance_neg, loss
            # the distance function in loss_fn should be the same as that of self.decode_batch
        else:
            distance_pos = self.decode_batch(embedded_anchor, embedded_pos)
            distance_neg = self.decode_batch(embedded_anchor, embedded_neg)
            return distance_pos, distance_neg


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_pos = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_neg = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = torch.clamp(distance_pos - distance_neg + self.margin, min=0)
        return distance_pos, distance_neg, losses.mean() if size_average else losses.sum()

