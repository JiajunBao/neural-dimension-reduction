import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import copy
import random
import src.toolkit.network as network
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import faiss
import time

random.seed(35)


def get_distance(output1: torch.tensor, output2: torch.tensor):
    return torch.sum((output1 - output2) ** 2, dim=1)


class PowerMarginLoss:
    def __init__(self, margin, reduction):
        assert margin > 0, f'margin should be positive, not {margin}'
        self.margin = margin
        # <0> (pos) <m1> (margin) <m2> (mid) <m3> (margin) <m4> (neg)
        self.m1, self.m2, self.m3, self.m4 = 2 * margin, 4 * margin, 8 * margin, 16 * margin
        assert reduction in {'mean', 'sum'}
        self.reduction = reduction

    def forward(self, output1: torch.tensor, output2: torch.tensor, y: torch.tensor or int) -> torch.tensor:
        dist = get_distance(output1, output2)
        l1 = y * (y - 1) * (y + 1.5) * torch.clamp(dist - self.m1, min=0)
        l2 = (y - 1) * (y - 1) * (y + 1) * torch.max(dist - self.m2, self.m3 - dist)
        l3 = (y + 1) * (y - 0.5) * torch.clamp(self.m4 - dist, min=0)
        loss = l1 + l2 + l3

        pred = torch.ones_like(dist)
        pred[dist < self.m1] = -1
        pred[(dist >= self.m1) & (dist <= self.m4)] = 0
        pred[dist > self.m4] = 1
        if self.reduction == 'sum':
            print(self.reduction)
            return loss.sum(), pred, dist
        return loss.mean(), pred, dist


def train_one_epoch(train_loader, model, optimizer, criterion, verbose, device):
    model = model.to(device)
    model.train()
    train_margin_loss = 0.
    pred_list = list()
    label_list = list()
    dist_list = list()
    train_correct_pred = 0
    for i, batch in enumerate(train_loader):
        if isinstance(model, network.ReconstructSiameseNet):
            x1, x2 = batch
            x1_device, x2_device = x1.to(device), x2.to(device)
            loss = model(x1_device, x2_device)
        elif isinstance(model, network.SiameseNet):
            x1, x2, label = batch
            x1_device, x2_device = x1.to(device), x2.to(device)
            output1, output2 = model(x1_device, x2_device)
            loss, dist, pred = criterion.forward(output1, output2, label.to(device))
            pred_list.append(pred.cpu())
            label_list.append(label.cpu())
            dist_list.append(dist.cpu())
            train_correct_pred += (pred == label.to(device)).sum().item()
        else:
            raise NotImplementedError
        model.zero_grad()  # reset gradient
        loss.backward()
        optimizer.step()
        train_margin_loss += loss.item()
        if verbose and i % 20 == 0:
            print(f'training loss: {train_margin_loss / (i + 1):.6f}')

    log = (None, None, None, None,)
    # demon = len(train_loader) if criterion.reduction == 'mean' else len(train_loader.dataset)
    demon = len(train_loader.dataset)
    train_margin_loss /= demon

    if isinstance(model, network.SiameseNet):
        pred = torch.cat(pred_list, dim=0)
        gold = torch.cat(label_list, dim=0)
        dist = torch.cat(dist_list, dim=0)
        log = (train_correct_pred / demon, pred, gold, dist)
    elif isinstance(model, network.ReconstructSiameseNet):
        pass
    return train_margin_loss, log


def eval_with_query(base_loader, query_loader, model, device):
    start = time.time()
    model.eval()
    embedded_queries, embedded_base = list(), list()
    model = model.to(device)
    with torch.no_grad():
        for i, batch in enumerate(base_loader):
            # infer embeddings for all base vectors
            query_vecs, _ = batch
            embedded_batch = model.get_embedding(query_vecs.to(device))
            print(embedded_batch)
            embedded_base.append(embedded_batch)
        print('\n\n\n\n')
        print('step2\n')
        for i, batch in enumerate(query_loader):
            # infer embeddings for query loader
            query_vecs, _ = batch
            embedded_batch = model.get_embedding(query_vecs.to(device))
            print(embedded_batch)
            embedded_queries.append(embedded_batch)

    embedded_queries = torch.cat(embedded_queries, dim=0).numpy()
    embedded_base = torch.cat(embedded_base, dim=0).numpy()
    n, d = embedded_base.shape
    index = faiss.IndexFlatL2(d)  # build the index
    index.add(embedded_base)  # add vectors to the index

    k: int = 2
    base_neighbor_distance, base_neighbor_index = index.search(embedded_base, k)  # actual search
    query_neighbor_distance, query_neighbor_index = index.search(embedded_queries, k)  # actual search
    recall_on_base_set = get_recall(base_loader.dataset.ground_true_nn, base_neighbor_index[:, 1])
    recall_on_query_set = get_recall(query_loader.dataset.ground_true_nn, query_neighbor_index[:, 1])
    end = time.time()
    return recall_on_base_set, recall_on_query_set, base_neighbor_index, query_neighbor_index, end - start


def get_recall(gold: np.array, pred: np.array):
    # assert gold.shape[0] == pred.shape[0], f'inconsistent shape: {gold.shape} vs {pred.shape}'
    if len(pred.shape) == 1:
        pred = pred.reshape(-1, 1)

    # the first column is the point itself which is not interesting
    #
    top_k_gold = 20
    gold = gold[:, 1: 1 + top_k_gold]
    print(f'retrieve {pred.shape[1]} data points for {gold.shape[1]} ground truth.')
    tp = 0
    gold_list, pred_list = gold.tolist(), pred.tolist()
    for i in range(gold.shape[0]):
        tp += len(set(gold_list[i]) & set(pred_list[i]))
    tp_and_fn = gold.shape[0] * gold.shape[1]
    return tp / tp_and_fn


def train_with_eval(train_loader, base_loader, eval_query_loader, criterion, model, optimizer, num_epoches, log_epoch,
                    verbose, device, trial=None):
    start = time.time()
    best_model = None
    best_recall_query_set = 0
    its_recall_on_base_set = 0
    for epoch_idx in range(1, num_epoches + 1):
        avg_train_loss, (train_accuracy, train_pred, train_gold, dist) = train_one_epoch(train_loader, model, optimizer,
                                                                                         criterion, verbose, device)
        recall_on_base_set, recall_on_query_set, base_neighbor_index, \
            query_neighbor_index, elapse_time = eval_with_query(base_loader, eval_query_loader, model, device)
        if best_recall_query_set < recall_on_query_set:
            best_recall_query_set = recall_on_query_set
            its_recall_on_base_set = recall_on_base_set
            best_model = copy.deepcopy(model.cpu())
        if verbose and epoch_idx % log_epoch == 0:
            print(f'epoch [{epoch_idx}]/[{num_epoches}] training loss: {avg_train_loss:.6f} '
                  f'recall on query set: {best_recall_query_set:.2f} '
                  f'recall on base set: {its_recall_on_base_set: .2f}')
        if trial:
            trial.report(best_recall_query_set, epoch_idx)
    end = time.time()
    print(f'elapse time: {end - start}')
    return best_recall_query_set, its_recall_on_base_set, best_model, model
