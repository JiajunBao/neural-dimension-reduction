from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader

from tqdm.auto import tqdm

STABLE_FACTOR = 1e-8


def far_func(sorted_dist: torch.tensor, indices: torch.tensor):
    return sorted_dist[:, 1].view(-1, 1), indices[:, 1].view(-1, 1)


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


class SurveyorDataSet(Dataset):
    def __init__(self, data, pairs, labels, q):
        self.data = data
        self.pairs = pairs
        self.labels = labels
        self.q = q

    @classmethod
    def from_df(cls, path_to_dataframe, func=far_func):
        data = torch.from_numpy(pd.read_csv(path_to_dataframe, header=None).to_numpy()).to(torch.float32)
        pairs, labels, close_distance, far_distance = make_pairs(data, func)
        q = thesis_input_inverse_similarity(data[pairs[:, 0]],
                                            data[pairs[:, 1]],
                                            close_distance[pairs[:, 0]],
                                            close_distance[pairs[:, 1]])
        return cls(data, pairs, labels, q)

    @classmethod
    def from_dataset(cls, path_to_tensor):
        return torch.load(path_to_tensor)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        indexs = self.pairs[idx]
        left = self.data[indexs[0]]
        right = self.data[indexs[1]]
        return left, right, self.labels[idx], self.q[idx]


class Surveyor(nn.Module):
    def __init__(self, dim_in=200, dim_out=20):
        super(Surveyor, self).__init__()
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
            ])
        )
        self.decoder = nn.Sequential(
            OrderedDict([
                ('bn1', nn.BatchNorm1d(2 * dim_out)),
                ('relu1', nn.ReLU()),
                ('fc1', nn.Linear(2 * 20, 20)),
                ('bn2', nn.BatchNorm1d(20)),
                ('relu2', nn.ReLU()),
                ('fc2', nn.Linear(20, 2)),
            ]))

    def encode_batch(self, x):
        return self.encoder(x)

    def decode_batch(self, out1, out2):
        p = thesis_output_inverse_similarity(out1, out2)
        x = torch.cat((out1, out2), dim=1)
        out = self.decoder(x)
        logits = F.softmax(out, dim=1)
        return logits, p

    def forward(self, x1, x2, q=None, labels=None, lam=1):
        out1 = self.encode_batch(x1)
        out2 = self.encode_batch(x2)
        logits, p = self.decode_batch(out1, out2)
        if labels is not None and q is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels) + lam * thesis_kl_div_add_mse_loss(p, q)
            return logits, p, out1, out2, loss
        return logits, p, out1, out2


def thesis_output_inverse_similarity(y1, y2):
    dout = torch.sum((y1 - y2) ** 2, dim=1)
    return 1 / (dout + 1)


def thesis_input_inverse_similarity(x1, x2, x1_min_dist, x2_min_dist):
    din = torch.sum((x1 - x2) ** 2, dim=1)
    q1 = 1 / ((din / (x1_min_dist ** 2)) + STABLE_FACTOR)
    q2 = 1 / ((din / (x2_min_dist ** 2)) + STABLE_FACTOR)
    return (q1 + q2) / 2


def thesis_kl_div_add_mse_loss(p, q, lam=1):
    """
    calculate the sum of kl divergence and mse loss
    :param p: p in the formula (P20-2) output similarities
    :param q: q in the formula (P20-2) input similarities
    :param lam: the constant that balances the influence of two losses
    :return: torch.tensor of the shape (,)
    """
    return torch.mean(p * torch.log(p / q)) + lam * torch.sum((p - q) ** 2)


class RetrieveSystem(object):
    def __init__(self, distance_measure):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        distance_measure = distance_measure.to(self.device)
        self.distance_measure = distance_measure

    def retrieve_query(self, query, ignore_idx, x_embedded, x_idx, topk=20):
        query_device = query.view(1, -1).to(self.device)
        cls_distances = list()
        p_distances = list()
        with torch.no_grad():
            for i, x in zip(x_idx, x_embedded):
                if ignore_idx is not None and i == ignore_idx:
                    continue
                x_device = x.view(1, -1).to(self.device)
                logits, p = self.distance_measure.decode_batch(query_device, x_device)
                cls_distances.append(logits[:, 1].item())
                p_distances.append(p.item())
        cls_distances = torch.tensor(cls_distances)
        p_distances = torch.tensor(p_distances)
        _, cls_nn_idx = cls_distances.sort()
        _, p_nn_idx = p_distances.sort()
        return cls_nn_idx[:topk], p_nn_idx[:topk]

    def retrieve_corpus(self, corpus, block_list, database):
        cls_pred_nn_top, p_distances_nn_top = list(), list()
        x_idx = range(database.shape[0])
        for ignore_idx, query in tqdm(zip(block_list, corpus), total=len(block_list), desc='retrieve each query'):
            cls_distances, p_distances = self.retrieve_query(query, ignore_idx, database, x_idx, 20)
            cls_pred_nn_top.append(cls_distances.view(1, -1))
            p_distances_nn_top.append(p_distances.view(1, -1))
        cls_pred_nn_top = torch.cat(cls_pred_nn_top, dim=0)
        p_distances_nn_top = torch.cat(p_distances_nn_top, dim=0)
        return cls_pred_nn_top, p_distances_nn_top

    def recall(self, pred, gold, at_n=None):
        results = dict()
        if at_n is None:
            at_n = [1, 5, 10, 20]
        for n in at_n:
            recall = float((pred[:, :n] == gold.view(-1, 1)).sum().item()) / len(gold)
            results[f'recall@{n}'] = recall
        return results
