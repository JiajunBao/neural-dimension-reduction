import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import copy
import random
from tqdm.auto import tqdm
import pandas as pd

random.seed(35)


def compare(x, embedded_x, loss_param, cache_dist, device, batch_size, k):
    m1, m2, m3, m4 = loss_param
    data_loader = DataLoader(TensorDataset(x, embedded_x), batch_size=batch_size, pin_memory=True, shuffle=False)
    margin_measure_confusion = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    linear_search_confusion = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    if cache_dist:
        pred_dist_list = list()
    else:
        pred_dist_list = None
    for batch_x, batch_embedded_x in tqdm(data_loader):
        gold_dist = torch.cdist(batch_x.to(device), x.to(device), p=2).cpu()
        gold_sorted_dist, gold_indices = torch.sort(gold_dist, dim=1, descending=False)
        pred_dist = torch.cdist(batch_embedded_x.to(device), embedded_x.to(device), p=2).cpu()
        pred_sorted_dist, pred_indices = torch.sort(pred_dist, dim=1, descending=False)
        # retrieve by linear-search
        binary_gold = torch.ones_like(gold_indices)
        idx = torch.arange(gold_indices.shape[0]).view(-1, 1)
        binary_gold[idx, gold_indices[:, :k + 1]] = 1  # is neighbor
        binary_gold[idx, gold_indices[:, k + 1:]] = 0  # is not neighbor

        linear_search_pred = torch.ones_like(gold_indices)
        linear_search_pred[idx, pred_indices[:, :k + 1]] = 1
        linear_search_pred[idx, pred_indices[:, k + 1:]] = 0

        # retrieve by margin
        margin_measure_pred = torch.ones_like(gold_indices)
        margin_measure_pred[pred_dist < m2] = 1  # is neighbor
        margin_measure_pred[pred_dist >= m2] = 0  # is not neighbor

        linear_search_confusion['tp'] += (binary_gold[linear_search_pred == 1] == 1).sum().item()
        linear_search_confusion['tn'] += (binary_gold[linear_search_pred == 0] == 0).sum().item()
        linear_search_confusion['fp'] += (binary_gold[linear_search_pred == 1] == 0).sum().item()
        linear_search_confusion['fn'] += (binary_gold[linear_search_pred == 0] == 1).sum().item()

        margin_measure_confusion['tp'] += (binary_gold[margin_measure_pred == 1] == 1).sum().item()
        margin_measure_confusion['tn'] += (binary_gold[margin_measure_pred == 0] == 0).sum().item()
        margin_measure_confusion['fp'] += (binary_gold[margin_measure_pred == 1] == 0).sum().item()
        margin_measure_confusion['fn'] += (binary_gold[margin_measure_pred == 0] == 1).sum().item()

        if cache_dist:
            pred_dist_list.append(pred_dist)

    assert margin_measure_confusion['tp'] + margin_measure_confusion['fn'] \
        == linear_search_confusion['tp'] + linear_search_confusion['fn'], 'there is a bug'
    assert margin_measure_confusion['tn'] + margin_measure_confusion['fp'] \
        == linear_search_confusion['tn'] + linear_search_confusion['fp'], 'there is a bug'

    def get_scores(conf):
        tp, tn, fp, fn = conf['tp'], conf['tn'], conf['fp'], conf['fn']
        res = dict()
        res['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        res['recall'] = tp / (tp + fn)
        res['precision'] = tp / (tp + fp)
        res['f1-score'] = 2 * res['recall'] * res['precision'] / (res['recall'] + res['precision'])
        return res

    margin_res = get_scores(margin_measure_confusion)
    linear_search_res = get_scores(linear_search_confusion)
    if cache_dist:
        pred_dist_list = torch.cat(pred_dist_list, dim=0)
        print('pred_dist is cached!')
    return (margin_res, margin_measure_confusion), (linear_search_res, linear_search_confusion), (pred_dist_list,)


def evaluate_results(x, model, k, loss_param, cache_dist):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 512
    model = model.to(device)
    # did inference
    data_loader = DataLoader(TensorDataset(x), batch_size=batch_size, pin_memory=True, shuffle=False)

    embedded_x_list = list()
    with torch.no_grad():
        for batch_x in data_loader:
            batch_x = batch_x[0].to(device)
            embedded_x = model.get_embedding(batch_x)
            embedded_x_list.append(embedded_x.cpu())
    embedded_x = torch.cat(embedded_x_list, dim=0)

    m1, m2, m3, m4 = loss_param
    data_loader = DataLoader(TensorDataset(x, embedded_x), batch_size=batch_size, pin_memory=True, shuffle=False)
    margin_measure_confusion = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    linear_search_confusion = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    if cache_dist:
        pred_dist_list = list()
    else:
        pred_dist_list = None
    for batch_x, batch_embedded_x in tqdm(data_loader):
        gold_dist = torch.cdist(batch_x.to(device), x.to(device), p=2).cpu()
        gold_sorted_dist, gold_indices = torch.sort(gold_dist, dim=1, descending=False)
        pred_dist = torch.cdist(batch_embedded_x.to(device), embedded_x.to(device), p=2).cpu()
        pred_sorted_dist, pred_indices = torch.sort(pred_dist, dim=1, descending=False)
        # retrieve by linear-search
        binary_gold = torch.ones_like(gold_indices)
        idx = torch.arange(gold_indices.shape[0]).view(-1, 1)
        binary_gold[idx, gold_indices[:, :k + 1]] = 1  # is neighbor
        binary_gold[idx, gold_indices[:, k + 1:]] = 0  # is not neighbor

        linear_search_pred = torch.ones_like(gold_indices)
        linear_search_pred[idx, pred_indices[:, :k + 1]] = 1
        linear_search_pred[idx, pred_indices[:, k + 1:]] = 0

        # retrieve by margin
        margin_measure_pred = torch.ones_like(gold_indices)
        margin_measure_pred[pred_dist < m2] = 1  # is neighbor
        margin_measure_pred[pred_dist >= m2] = 0  # is not neighbor

        linear_search_confusion['tp'] += (binary_gold[linear_search_pred == 1] == 1).sum().item()
        linear_search_confusion['tn'] += (binary_gold[linear_search_pred == 0] == 0).sum().item()
        linear_search_confusion['fp'] += (binary_gold[linear_search_pred == 1] == 0).sum().item()
        linear_search_confusion['fn'] += (binary_gold[linear_search_pred == 0] == 1).sum().item()

        margin_measure_confusion['tp'] += (binary_gold[margin_measure_pred == 1] == 1).sum().item()
        margin_measure_confusion['tn'] += (binary_gold[margin_measure_pred == 0] == 0).sum().item()
        margin_measure_confusion['fp'] += (binary_gold[margin_measure_pred == 1] == 0).sum().item()
        margin_measure_confusion['fn'] += (binary_gold[margin_measure_pred == 0] == 1).sum().item()

        if cache_dist:
            pred_dist_list.append(pred_dist)

    assert margin_measure_confusion['tp'] + margin_measure_confusion['fn'] \
        == linear_search_confusion['tp'] + linear_search_confusion['fn'], 'there is a bug'
    assert margin_measure_confusion['tn'] + margin_measure_confusion['fp'] \
        == linear_search_confusion['tn'] + linear_search_confusion['fp'], 'there is a bug'

    def get_scores(conf):
        tp, tn, fp, fn = conf['tp'], conf['tn'], conf['fp'], conf['fn']
        res = dict()
        res['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        res['recall'] = tp / (tp + fn)
        res['precision'] = tp / (tp + fp)
        res['f1-score'] = 2 * res['recall'] * res['precision'] / (res['recall'] + res['precision'])
        return res

    margin_res = get_scores(margin_measure_confusion)
    linear_search_res = get_scores(linear_search_confusion)
    if cache_dist:
        pred_dist_list = torch.cat(pred_dist_list, dim=0)
        print('pred_dist is cached!')
    return (margin_res, margin_measure_confusion), (linear_search_res, linear_search_confusion), (pred_dist_list,)

