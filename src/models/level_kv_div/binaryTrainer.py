import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import copy
# from torch.nn import KLDivLoss
import random
from tqdm.auto import tqdm
import pandas as pd

random.seed(35)


class LargeSparseDataset(Dataset):
    def __init__(self, x_path, k, balanced, random_neg):
        x = torch.from_numpy(pd.read_csv(x_path, header=None).to_numpy()).float()
        data = list()
        posn1_count, pos0_count, neg_count = 0, 0, 0
        for i in tqdm(range(x.shape[0])):
            dist = torch.cdist(x1=x[i].view(1, -1), x2=x, p=2)[0]  # (n, n)
            sorted_dist, indices = torch.sort(dist, descending=False)
            posn1_list = list()
            pos0_list = list()
            neg_list = list()
            for j in range(1, k + 1):
                dist_other = torch.cdist(x1=x[indices[j]].view(1, -1), x2=x, p=2)[0]  # (n, n)
                sorted_dist_other, indices_other = torch.sort(dist_other, descending=False)
                if i in indices_other[1:k + 1]:
                    posn1_list.append((i, indices[j], -1))
                else:
                    pos0_list.append((i, indices[j], 0))
            for j in range(k + 2, x.shape[0]):
                neg_list.append((i, indices[j], 1))
            if balanced:
                if random_neg:
                    random.shuffle(neg_list)
                data += neg_list[:len(posn1_list) + len(pos0_list)] + pos0_list + posn1_list
                neg_count += len(posn1_list) + len(pos0_list)
            else:
                data += neg_list + pos0_list + posn1_list
                neg_count += len(neg_list)
            posn1_count += len(posn1_list)
            pos0_count += len(pos0_list)
        self.data = data
        self.x = x
        print(f'{x.shape[0]} points, {len(data)} pairs')
        print(f'mutual neighbors: {posn1_count} one-direction neighbors {pos0_count} not neighbors {neg_count}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        i, j, label = self.data[idx]
        return self.x[i], self.x[j], label


class SparseDataset(Dataset):
    def __init__(self, x_path, label_path, balanced, random_neg):
        x = torch.from_numpy(pd.read_csv(x_path, header=None).to_numpy()).float()
        label_matrix = torch.load(label_path, 'cpu').float()
        assert x.shape[0] == label_matrix.shape[0], f'inconsistent size {x.shape[0]} vs {label_matrix.shape[0]} .'
        data = list()
        for i, vec in enumerate(label_matrix):
            tmp_pos_data, tmp_neg_data = list(), list()
            for j, label in enumerate(vec):
                if label.item() == 1.:
                    tmp_neg_data.append((i, j, label.item(),))
                elif label.item() == 0.:
                    tmp_pos_data.append((i, j, label.item(),))
                elif label.item() == -1.:
                    tmp_pos_data.append((i, j, label.item(),))
                else:
                    raise NotImplemented
            if random_neg:
                random.shuffle(tmp_neg_data)
            if balanced:
                data += tmp_neg_data[:len(tmp_pos_data)] + tmp_pos_data
            else:
                data += tmp_neg_data + tmp_pos_data
        self.data = data
        self.x = x
        print(f'{label_matrix.shape[0]} points {len(self.data)} data')
        count1, count0, countn1 = 0, 0, 0
        for _, _, l in data:
            if l == 1.:
                count1 += 1
            elif l == 0.:
                count0 += 1
            elif l == -1.:
                countn1 += 1
        print(f'mutual negihbors: {countn1}, one-direction neighbor: {count0}, not neighbor: {count1}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id1, id2, label = self.data[idx]
        return self.x[id1], self.x[id2], label


def evaluate_results(x, model, k, loss_param):
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
    return (margin_res, margin_measure_confusion), (linear_search_res, linear_search_confusion), (pred_dist)


class TriMarginLoss:
    def __init__(self, m1, m2, m3, m4, reduction):
        self.m1, self.m2, self.m3, self.m4 = m1, m2, m3, m4
        assert reduction in {'mean', 'sum'}
        self.reduction = reduction

    def forward(self, output1: torch.tensor, output2: torch.tensor, y: torch.tensor or int) -> torch.tensor:
        dist = self.get_distance(output1, output2)
        l1 = y * (y - 1) * (y + 1.5) * torch.clamp(dist - self.m1, min=0)
        l2 = (y - 1) * (y - 1) * (y + 1) * torch.max(dist - self.m2, self.m3 - dist)
        l3 = (y + 1) * (y - 0.5) * torch.clamp(self.m4 - dist, min=0)
        loss = l1 + l2 + l3
        if self.reduction == 'sum':
            print(self.reduction)
            return loss.sum(), dist
        return loss.mean(), dist

    def get_distance(self, output1: torch.tensor, output2: torch.tensor):
        return torch.sum((output1 - output2) ** 2, dim=1)


def train_one_epoch(train_loader, model, optimizer, verbose, device):
    model = model.to(device)
    model.train()
    criterion = TriMarginLoss(1, 3, 4, 6, 'mean')
    train_margin_loss = 0.
    pred_list = list()
    label_list = list()
    dist_list = list()
    train_correct_pred = 0
    for i, batch in enumerate(train_loader):
        x1, x2, label = batch
        x1_device, x2_device = x1.to(device), x2.to(device)
        output1, output2 = model(x1_device, x2_device)
        loss, dist = criterion.forward(output1, output2, label.to(device))

        pred = torch.ones_like(dist)
        pred[dist <= 1] = -1
        pred[(dist >= 3) & (dist <= 4)] = 0
        pred[dist >= 6] = 1
        pred_list.append(pred.cpu())
        label_list.append(label.cpu())
        dist_list.append(dist.cpu())
        train_correct_pred += (pred == label.to(device)).sum().item()

        model.zero_grad()  # reset gradient
        loss.backward()
        optimizer.step()
        train_margin_loss += loss.item()
        if verbose and i % 20 == 0:
            print(f'training loss: {train_margin_loss / (i + 1):.4f}')
    pred = torch.cat(pred_list, dim=0)
    gold = torch.cat(label_list, dim=0)
    dist = torch.cat(dist_list, dim=0)
    return train_margin_loss / len(train_loader.dataset), (
    train_correct_pred / len(train_loader.dataset), pred, gold, dist)


def val_one_epoch(val_loader, model, device):
    model.eval()
    criterion = TriMarginLoss(1, 3, 4, 6, 'mean')
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
            loss, dist = criterion.forward(output1, output2, label.to(device))

            pred = torch.ones_like(dist)
            pred[dist <= 1] = -1
            pred[(dist >= 3) & (dist <= 4)] = 0
            pred[dist >= 6] = 1
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
            print(f'epoch [{epoch_idx}]/[{num_epoches}] training loss: {avg_train_loss:.4f} '
                  f'avg_val_margin_loss: {avg_val_margin_loss:.4f} '
                  f'train_accuracy: {train_accuracy: .2f} '
                  f'val_accuracy: {val_accuracy: .2f} ')
    return best_avg_val_margin_loss, best_model, model
