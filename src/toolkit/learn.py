import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import copy
import random
from tqdm.auto import tqdm
import pandas as pd
import faiss

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
        x1, x2, label = batch
        x1_device, x2_device = x1.to(device), x2.to(device)
        output1, output2 = model(x1_device, x2_device)
        loss, dist, pred = criterion.forward(output1, output2, label.to(device))

        pred_list.append(pred.cpu())
        label_list.append(label.cpu())
        dist_list.append(dist.cpu())
        train_correct_pred += (pred == label.to(device)).sum().item()

        model.zero_grad()  # reset gradient
        loss.backward()
        optimizer.step()
        train_margin_loss += loss.item()
        if verbose and i % 20 == 0:
            print(f'training loss: {train_margin_loss / (i + 1):.6f}')
    pred = torch.cat(pred_list, dim=0)
    gold = torch.cat(label_list, dim=0)
    dist = torch.cat(dist_list, dim=0)
    return train_margin_loss / len(train_loader.dataset), (
        train_correct_pred / len(train_loader.dataset), pred, gold, dist)


def eval_with_query(base_loader, query_loader, model, device):
    model.eval()
    embedded_queries, embedded_base = list(), list()
    model = model.to(device)
    with torch.no_grad():
        for i, batch in enumerate(base_loader):
            # infer embeddings for all base vectors
            query_vecs, _ = batch
            embedded_batch = model.get_embedding(query_vecs.to(device)).cpu()
            embedded_base.append(embedded_batch)

        for i, batch in enumerate(query_loader):
            # infer embeddings for query loader
            query_vecs, _ = batch
            embedded_batch = model.get_embedding(query_vecs.to(device)).cpu()
            embedded_queries.append(embedded_batch)

    embedded_queries = torch.cat(embedded_queries, dim=0)
    embedded_base = torch.cat(embedded_base, dim=0)
    n, d = embedded_base.shape
    index = faiss.IndexFlatL2(d)  # build the index
    index.add(embedded_base)  # add vectors to the index

    base_neighbor_distance, base_neighbor_index = index.search(embedded_base, base_loader.k)  # actual search
    query_neighbor_distance, query_neighbor_index = index.search(embedded_queries, query_loader.k)  # actual search

    assert base_neighbor_index.shape[0] == query_neighbor_index.shape[0], 'inconsistent number of rows'
    assert base_neighbor_index.shape[1] == query_neighbor_index.shape[1], 'inconsistent number of neighbors retrieved'

    tp = 0
    for i in range(base_neighbor_index.shape[0]):
        tp += len(set(base_neighbor_index.tolist()) & set(query_neighbor_index.tolist()))
    tp_and_fn = query_neighbor_index.shape[0] * query_neighbor_index.shape[1]
    recall = tp / tp_and_fn
    print(f'recall @{query_neighbor_index.shape[1]}: {recall}')
    return recall, base_neighbor_index, query_neighbor_index


def val_one_epoch(val_loader, criterion, model, device):
    model.eval()
    val_margin_loss = 0.
    pred_list = list()
    label_list = list()
    dist_list = list()
    val_correct_pred = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x1, x2, label = batch
            x1_device, x2_device = x1.to(device), x2.to(device)
            output1, output2 = model(x1_device, x2_device)
            loss, dist, pred = criterion.forward(output1, output2, label.to(device))

            val_correct_pred += (pred == label.to(device)).sum().item()
            pred_list.append(pred.cpu())
            label_list.append(label.cpu())
            dist_list.append(dist.cpu())
            val_margin_loss += loss.item()
    #             if i % 20 == 0:
    #                 print(f'batch mean val loss: {val_margin_loss / (i + 1):.4f}')
    pred = torch.cat(pred_list, dim=0)
    gold = torch.cat(label_list, dim=0)
    dist = torch.cat(dist_list, dim=0)
    return val_margin_loss / len(val_loader.dataset), (val_correct_pred / len(val_loader.dataset), pred, gold, dist)


def train_with_eval(train_loader, val_loader, model, optimizer, num_epoches, log_epoch, verbose, device):
    best_model = None
    best_avg_val_margin_loss = float('inf')
    for epoch_idx in range(1, num_epoches + 1):
        avg_train_loss, (train_accuracy, train_pred, train_gold, dist) = train_one_epoch(train_loader, model, optimizer,
                                                                                         False, device)
        avg_val_margin_loss, (val_accuracy, val_pred, val_gold, dist) = val_one_epoch(val_loader, model, device)
        if avg_val_margin_loss < best_avg_val_margin_loss:
            best_avg_val_margin_loss = avg_val_margin_loss
            best_model = copy.deepcopy(model.cpu())
        if verbose and epoch_idx % log_epoch == 0:
            print(f'epoch [{epoch_idx}]/[{num_epoches}] training loss: {avg_train_loss:.6f} '
                  f'avg_val_margin_loss: {avg_val_margin_loss:.4f} '
                  f'train_accuracy: {train_accuracy: .2f} '
                  f'val_accuracy: {val_accuracy: .2f} ')
    return best_avg_val_margin_loss, best_model, model


def eval_in_train_one_epoch(train_loader, val_loader, criterion, model, optimizer, device):
    model = model.to(device)
    model.train()
    train_margin_loss = 0.
    pred_list = list()
    label_list = list()
    dist_list = list()
    train_correct_pred = 0
    best_val_loss, best_val_accuracy = float('inf'), 0
    best_model = None
    for i, batch in enumerate(train_loader):
        x1, x2, label = batch
        x1_device, x2_device = x1.to(device), x2.to(device)
        output1, output2 = model(x1_device, x2_device)
        loss, dist, pred = criterion.forward(output1, output2, label.to(device))

        pred_list.append(pred.cpu())
        label_list.append(label.cpu())
        dist_list.append(dist.cpu())
        train_correct_pred += (pred == label.to(device)).sum().item()

        model.zero_grad()  # reset gradient
        loss.backward()
        optimizer.step()
        train_margin_loss += loss.item()
        if i % (len(train_loader) // 10 + 1) == 0:
            val_loss, val_accuracy = val_one_epoch(val_loader, criterion, model, device)
            print(f'val_loss: {val_loss} val_accuracy: {val_accuracy}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_model = copy.deepcopy(model)
    pred = torch.cat(pred_list, dim=0)
    gold = torch.cat(label_list, dim=0)
    dist = torch.cat(dist_list, dim=0)
    return train_margin_loss / len(train_loader.dataset), \
        (train_correct_pred / len(train_loader.dataset), pred, gold, dist), \
        (best_val_loss, best_val_accuracy, best_model)


def eval_in_train(train_loader, val_loader, model, optimizer, num_epoches, log_epoch, verbose, device):
    best_model = None
    best_avg_val_margin_loss = float('inf')
    for epoch_idx in range(1, num_epoches + 1):
        avg_train_loss, (train_accuracy, train_pred, train_gold, dist), (val_loss, val_acc, epoch_best_model) = \
            eval_in_train_one_epoch(train_loader, val_loader, model, optimizer, device)

        if val_loss < best_avg_val_margin_loss:
            best_avg_val_margin_loss = val_loss
            best_model = epoch_best_model
        if verbose and epoch_idx % log_epoch == 0:
            print(f'epoch [{epoch_idx}]/[{num_epoches}] training loss: {avg_train_loss:.6f} '
                  f'avg_val_margin_loss: {val_loss:.4f} '
                  f'train_accuracy: {train_accuracy: .2f} '
                  f'val_accuracy: {val_acc: .2f} ')
    return best_avg_val_margin_loss, best_model, model
