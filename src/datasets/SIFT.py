import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import copy
import random
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from collections import Counter
import faiss
from pathlib import Path


random.seed(35)
np.random.seed(35)
torch.manual_seed(35)


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def level_grading(sorted_indexes: torch.tensor, k: int):
    # the first column of sorted_indexes is oneself.
    res = torch.ones_like(sorted_indexes)
    n = sorted_indexes.shape[0]
    res[:, 0] = -1
    for i in tqdm(range(n)):
        single_idxs = list()
        mutual_idxs = list()
        for j in range(1, k + 1):
            if i in sorted_indexes[sorted_indexes[i][j]][1:k + 1]:
                mutual_idxs.append(j)
            else:
                single_idxs.append(j)
        res[i][mutual_idxs] = -1
        res[i][single_idxs] = 0
    original_ordered_res = torch.ones_like(res)
    for i, tmp in enumerate(sorted_indexes):
        original_ordered_res[i, tmp] = res[i]
    return original_ordered_res


class LargeBaseDataset(Dataset):
    def __init__(self, x, k, balanced, random_neg):
        # compute the actual nearest neighbor
        self.k = k
        self.balanced, self.random_neg = balanced, random_neg
        n, d = x.shape
        index = faiss.IndexFlatL2(d)  # build the index
        index.add(x)  # add vectors to the index
        sorted_distance, sorted_index = index.search(x, x.shape[0])  # actual search
        self.x = torch.from_numpy(x)
        # compute pairwise level grades
        grades = level_grading(torch.from_numpy(sorted_index), k)
        n, m = grades.shape

        if not random_neg:
            raise NotImplementedError

        self.data = list()
        for i in tqdm(range(n)):
            pos, mid, neg = list(), list(), list()
            for j in range(n):
                if grades[i][j].item() == -1:
                    pos.append((i, j, -1))
                elif grades[i][j].item() == 0:
                    mid.append((i, j, 0))
                elif grades[i][j].item() == 1:
                    neg.append((i, j, 1))
                else:
                    print(grades[i][j].item())
                    raise NotImplementedError
            pos_or_mid = pos + mid
            if balanced and random_neg:
                size = min(len(pos_or_mid), len(neg))
                random.shuffle(pos_or_mid)
                random.shuffle(neg)
                pos_or_mid = pos_or_mid[:size]
                neg = neg[:size]
            self.data += pos_or_mid + neg

        _, _, label = zip(*self.data)
        label_counts = Counter(label)
        print(f'{x.shape[0]} points, {len(self.data)} pairs')
        print(f"mutual nn: {label_counts[-1]} one-direction nn: {label_counts[0]} not nn: {label_counts[1]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        i, j, label = self.data[idx]
        return self.x[i], self.x[j], label


class QueryDataset(Dataset):
    def __init__(self, query_vecs, base_vecs, k):
        self.k = k
        self.query_vecs = query_vecs
        print(f'shape of base vectors: {base_vecs.shape}, shape of query_vectors: {base_vecs.shape}')
        n, d = base_vecs.shape
        index = faiss.IndexFlatL2(d)  # build the index
        index.add(base_vecs)  # add vectors to the index
        print(f'search top {k} neighbors')
        sorted_distance, sorted_index = index.search(query_vecs, k)  # actual search
        self.ground_true_nn = sorted_index
        print(f'ground neighbors shape {sorted_index.shape}')

    def __len__(self):
        return self.query_vecs.shape[0]

    def __getitem__(self, idx):
        return self.query_vecs[idx], self.ground_true_nn[idx]

class PairingDataset(Dataset):
    def __init__(self, x):
        self.x = x
        n, d = x.shape
        self.pair_idxs = list()
        for i in range(n):
            for j in range(n):
                self.pair_idxs.append((i, j,))

    def __len__(self):
        return len(self.pair_idxs)

    def __getitem__(self, idx):
        i, j = self.pair_idxs[idx]
        return self.x[i], self.x[j]


def get_datasets(input_dir: Path=Path('/home/jiajunb/neural-dimension-reduction/data/sift/siftsmall'), model_type='SiameseNet'):
    dev_path = input_dir / f'sift.dev.query.{model_type}.dataset.pt'
    train_path = input_dir / f'sift.train.learn.{model_type}.dataset.pt'
    base_path = input_dir / f'sift.train.query.{model_type}.dataset.pt'
    if dev_path.is_file() and train_path.is_file() and base_path.is_file():
        print(f"loading dataset from {input_dir}")
        train_dataset = torch.load(train_path)
        base_dataset = torch.load(base_path)
        dev_dataset = torch.load(dev_path)
        return train_dataset, base_dataset, dev_dataset

    print(f"building dataset")
    x = fvecs_read(input_dir / "siftsmall_learn.fvecs")
    np.random.shuffle(x)
    train_frac = 0.8
    k = 100
    split_idx = int(x.shape[0] * train_frac)

    train_x, dev_x = x[:split_idx, :], x[split_idx:, :]
    print(f'training set: {train_x.shape[0]} points; dev set: {dev_x.shape[0]} points.')
    if model_type == 'ReconstructSiameseNet':
        train_dataset = PairingDataset(train_x)
        torch.save(train_dataset, train_path)
    else:
        train_dataset = LargeBaseDataset(train_x, k, True, True)
        torch.save(train_dataset, train_path)

    base_dataset = QueryDataset(query_vecs=train_x, base_vecs=train_x, k=k)
    dev_dataset = QueryDataset(query_vecs=dev_x, base_vecs=train_x, k=k)
    torch.save(base_dataset, base_path)
    torch.save(dev_dataset, dev_path)
    return train_dataset, base_dataset, dev_dataset
