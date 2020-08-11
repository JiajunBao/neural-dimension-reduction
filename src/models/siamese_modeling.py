from collections import OrderedDict

import torch
from torch import nn
from tqdm.auto import tqdm
import pandas as pd
from torch.utils.data import Dataset


def far_func(sorted_dist: torch.tensor, indices: torch.tensor):
    return sorted_dist[:, -1].view(-1, 1), indices[:, -1].view(-1, 1)


def calculate_distance(x, far_fn):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 512
    x_device = x.to(device)
    if x.shape[0] * x.shape[1] < batch_size * 200:  # direct computes the whole matrix
        # TODO: we first assume memory fits in memory. Later, we can process the data in batches.
        dist = torch.cdist(x1=x_device, x2=x_device, p=2)  # (n, n)
        sorted_dist, indices = torch.sort(dist, dim=1, descending=False)
        sorted_dist, indices = sorted_dist.cpu(), indices.cpu()
        anchor_idx = torch.arange(x.shape[0])  # (n,)
        # the 0-th column is the distance to oneself
        close_distance, close_idx = sorted_dist[:, 1], indices[:, 1]  # (n,)
        far_distance, far_idx = far_fn(sorted_dist, indices)  # (n, r)
    else:
        num_iter = x.shape[0] // batch_size + 1
        anchor_idx_list, close_idx_list, far_idx_list = list(), list(), list()
        close_distance_list, far_distance_list = list(), list()
        for i in tqdm(torch.arange(num_iter), desc='create triplets'):
            batch_x = x[i * batch_size: (i + 1) * batch_size, :].to(device)

            dist = torch.cdist(x1=batch_x, x2=x_device, p=2)  # (n, n)
            sorted_dist, indices = torch.sort(dist, dim=1, descending=False)
            sorted_dist, indices = sorted_dist, indices
            anchor_idx = torch.arange(batch_x.shape[0])  # (n,)
            # the 0-th column is the distance to oneself
            close_distance, close_idx = sorted_dist[:, 1], indices[:, 1]  # (n,)
            far_distance, far_idx = far_fn(sorted_dist, indices)  # (n, r)
            anchor_idx_list.append(anchor_idx.cpu())
            close_idx_list.append(close_idx.cpu())
            far_idx_list.append(far_idx.cpu())
            close_distance_list.append(close_distance.cpu())
            far_distance_list.append(far_distance.cpu())
        anchor_idx = torch.cat(anchor_idx_list, dim=0)
        close_idx = torch.cat(close_idx_list, dim=0)
        far_idx = torch.cat(far_idx_list, dim=0)
        close_distance = torch.cat(close_distance_list, dim=0)
        far_distance = torch.cat(far_distance_list, dim=0)
    return anchor_idx, close_idx, far_idx, close_distance, far_distance


def make_pairs(x, far_fn):
    anchor_idx, close_idx, far_idx, close_distance, far_distance = calculate_distance(x, far_fn)
    n, r = far_idx.shape
    anchor_idx = anchor_idx.view(-1, 1)  # (n, 1)
    close_idx = close_idx.view(-1, 1)  # (n, 1)
    positive_pairs = torch.cat((anchor_idx, close_idx), dim=1)  # (n, 2)
    positive_labels = torch.ones(n, dtype=torch.int64)  # (n, )
    far_idx = far_idx.view(-1, 1)  # (n * r, )
    anchor_idx_flatten = anchor_idx.expand(-1, r).view(-1, 1)  # (n * r, )
    negative_pairs = torch.cat((anchor_idx_flatten, far_idx), dim=1)  # (n * r, 2)
    negative_labels = torch.zeros(n * r, dtype=torch.int64)  # (n * r, )
    pairs = torch.cat((positive_pairs, negative_pairs), dim=0)
    labels = torch.cat((positive_labels, negative_labels), dim=0)
    return pairs, labels, close_distance, far_distance


class SiameseDataSet(Dataset):
    def __init__(self, data, pairs, labels):
        self.data = data
        self.pairs = pairs
        self.labels = labels

    @classmethod
    def from_df(cls, path_to_dataframe):
        data = torch.from_numpy(pd.read_csv(path_to_dataframe, header=None).to_numpy()).to(torch.float32)
        pairs, labels, close_distance, far_distance = make_pairs(data, far_func)
        return cls(data, pairs, labels)

    @classmethod
    def from_dataset(cls, path_to_tensor):
        return torch.load(path_to_tensor)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        indexes = self.pairs[idx]
        left = self.data[indexes[0]]
        right = self.data[indexes[1]]
        return left, right, self.labels[idx]


class SiameseNetwork(nn.Module):
    def __init__(self, dim_in=200, dim_out=20):
        super(SiameseNetwork, self).__init__()
        self.encoder = nn.Sequential(
            OrderedDict([
                ('bn0', nn.BatchNorm1d(dim_in)),
                ('relu0', nn.ReLU(inplace=True)),
                ('fc0', nn.Linear(dim_in, 500)),
                ('bn1', nn.BatchNorm1d(500)),
                ('relu1', nn.ReLU(inplace=True)),
                ('fc1', nn.Linear(500, 100)),
                ('bn2', nn.BatchNorm1d(100)),
                ('relu2', nn.ReLU(inplace=True)),
                ('fc2', nn.Linear(100, 20)),
                ('bn3', nn.BatchNorm1d(20)),
                ('relu3', nn.ReLU(inplace=True)),
                ('fc3', nn.Linear(20, 20)),
                ('bn4', nn.BatchNorm1d(20)),
                ('relu4', nn.ReLU(inplace=True)),
                ('fc4', nn.Linear(20, 20)),
                ('bn5', nn.BatchNorm1d(20)),
                ('relu5', nn.ReLU(inplace=True)),
                ('fc5', nn.Linear(20, dim_out)),
                ('logistic', nn.Sigmoid())
            ])
        )
        self.decoder = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(dim_out, 1)),
            ]))

    def encode_batch(self, x):
        return self.encoder(x)

    def decode_batch(self, out1, out2):
        return (out2 - out1).pow(2).sum(dim=1).sqrt()  # distances (after squared root)

    def forward(self, x1, x2, labels=None):
        out1 = self.encode_batch(x1)
        out2 = self.encode_batch(x2)
        dist = self.decode_batch(out1, out2)
        if labels is not None:
            loss_fn = ContrastiveLoss(1.)
            loss = loss_fn(dist, labels)
            return dist, loss
        return dist


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin, size_average=True):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, distances, target):
        losses = 0.5 * (target.float() * distances +
                        (1 - target).float() * torch.clamp(self.margin - distances, min=0).pow(2))
        return losses.mean() if self.size_average else losses.sum()
