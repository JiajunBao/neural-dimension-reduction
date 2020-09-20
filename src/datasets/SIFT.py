import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import copy
import random
import numpy as np
from tqdm.auto import tqdm
import pandas as pd

import faiss
from pathlib import  Path


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


class LargeSparseDataset(Dataset):
    def __init__(self, x, k, balanced, random_neg):
        # compute the actual nearest neighbor
        self.k = k
        self.balanced, self.random_neg = balanced, random_neg
        n, d = x.shape
        index = faiss.IndexFlatL2(d)  # build the index
        index.add(x)  # add vectors to the index
        sorted_distance, sorted_index = index.search(x, x.shape[0])  # actual search
        # compute pairwise level grades
        grades = level_grading(sorted_index, k)
        n, m = grades.shape
        mutual_nn, onedirection_nn, not_nn = list(), list(), list()
        for i in range(n):
            for j in range(k):
                if grades[i][j].item() == -1:
                    mutual_nn.append((i, j, -1))
                elif grades[i][j].item() == 0:
                    onedirection_nn.append((i, j, 0))
                else:
                    raise NotImplementedError
        if balanced and random_neg:
            neg = grades[:, k:]
            sample_idx = torch.multinomial(neg, k)
            nidx = torch.arange(n).view(-1, 1)
            neg_sample = neg[nidx, sample_idx]
            for i in range(n):
                for j in range(k):
                    not_nn.append((i, j + k, neg_sample[i][j].item(),))
        elif balanced:
            for i in range(n):
                for j in range(k, 2 * k):
                    not_nn.append((i, j, grades[i][j].item(),))
        else:  # everything
            for i in range(n):
                for j in range(m):
                    not_nn.append((i, j, grades[i][j].item(),))
        self.data = mutual_nn + onedirection_nn + not_nn
        self.x = x
        print(f'{x.shape[0]} points, {len(self.data)} pairs')
        print(f"mutual nn: {len(mutual_nn)} one-direction nn: {len(onedirection_nn)} not nn: {len(not_nn)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        i, j, label = self.data[idx]
        return self.x[i], self.x[j], label


def get_datasets(input_dir: Path=Path('/home/jiajunb/neural-dimension-reduction/data/sift/siftsmall')):
    if (input_dir / 'sift.dev.dataset.pt').is_file() and (input_dir / 'sift.train.dataset.pt').is_file():
        train_dataset = torch.load(input_dir / 'sift.train.dataset.pt')
        dev_dataset = torch.load(input_dir / 'sift.dev.dataset.pt')
        return train_dataset, dev_dataset

    x = torch.from_numpy(fvecs_read(input_dir / "siftsmall_learn.fvecs"))
    np.random.shuffle(x)
    train_frac = 0.8
    k = 100
    split_idx = int(x.shape[0] * train_frac)

    train_x, dev_x = x[:split_idx, :], x[split_idx:, :]
    print(f'training set: {train_x.shape[0]} points; dev set: {dev_x.shape[0]} points.')
    train_dataset = LargeSparseDataset(train_x, k, True, True)
    dev_dataset = LargeSparseDataset(dev_x, k, False, False)
    torch.save(train_dataset, input_dir / 'sift.train.dataset.pt')
    torch.save(dev_dataset, input_dir / 'sift.dev.dataset.pt')
    return train_dataset, dev_dataset
