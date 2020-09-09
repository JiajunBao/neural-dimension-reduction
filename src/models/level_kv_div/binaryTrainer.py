import torch
from torch import nn
from torch.utils.data import Dataset
import copy
# from torch.nn import KLDivLoss

import pandas as pd


def get_dataset(x_path, label_path):
    return SparseDataset(x_path, label_path)


class SparseDataset(Dataset):
    def __init__(self, x_path, label_path):
        x = torch.from_numpy(pd.read_csv(x_path, header=None).to_numpy()).float()
        label_matrix = torch.load(label_path, 'cpu').float()
        assert x.shape[0] == label_matrix.shape[0], f'inconsistent size {x.shape[0]} vs {label_matrix.shape[0]} .'
        data = list()
        for i, vec in enumerate(label_matrix):
            for j, label in enumerate(vec):
                data.append((i, j, label))
        self.data = data
        self.x = x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id1, id2, label = self.data[idx]
        return self.x[id1], self.x[id2], label


class TriMarginLoss:
    def __init__(self, m1, m2, m3, m4, reduction='mean'):
        self.m1, self.m2, self.m3, self.m4 = m1, m2, m3, m4
        assert reduction in {'mean', 'sum'}
        self.reduction = reduction

    def forward(self, output1: torch.tensor, output2: torch.tensor, y: torch.tensor or int) -> torch.tensor:
        dist = self.get_distance(output1, output2)
        l1 = y * (y - 1) * (y + 1.5) * torch.clamp(dist - self.m1, min=0)
        l2 = (y - 1) * (y - 1) * (y + 1) * torch.max(dist - self.m2, self.m3 - dist)
        l3 = (y + 1) * (y - 0.5) * torch.clamp(self.m4 - dist, min=0)
        loss = l1 + l2 + l3
        if self.reduction == 'mean':
            return torch.mean(loss)
        return loss, dist

    def get_distance(self, output1: torch.tensor, output2: torch.tensor):
        return torch.sum((output1 - output2) ** 2, dim=1)


def train_one_epoch(train_loader, model, optimizer, verbose, device):
    model = model.to(device)
    model.train()
    criterion = TriMarginLoss(1, 3, 4, 6, 'mean')
    train_margin_loss = 0.
    # pred_list = list()
    train_correct_pred = 0
    for i, batch in enumerate(train_loader):
        x1, x2, label = batch
        x1_device, x2_device = x1.to(device), x2.to(device)
        output1, output2 = model(x1_device, x2_device)
        loss, dist = criterion.forward(output1, output2, label.to(device))

        dist[dist <= 1] = -1
        dist[dist >= 3 & dist <= 4] = 0
        dist[dist >= 6] = 1
        # pred_list.append(dist.cpu())
        train_correct_pred += (dist == label.to(device)).sum().item()

        model.zero_grad()  # reset gradient
        loss.backward()
        optimizer.step()
        train_margin_loss += loss.item()
        if verbose and i % 20 == 0:
            print(f'training loss: {train_margin_loss / (i + 1):.4f}')
    # pred = torch.cat(pred_list, dim=0)
    return train_margin_loss / len(train_loader), train_correct_pred / len(train_loader)


def val_one_epoch(val_loader, model, device):
    model.eval()
    criterion = TriMarginLoss(1, 3, 4, 6, 'mean')
    val_margin_loss = 0.
    # pred_list = list()
    val_correct_pred = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x1, x2, label = batch
            x1_device, x2_device = x1.to(device), x2.to(device)
            output1, output2 = model(x1_device, x2_device)
            loss, dist = criterion.forward(output1, output2, label.to(device))
            dist[dist <= 1] = -1
            dist[dist >= 3 & dist <= 4] = 0
            dist[dist >= 6] = 1
            val_correct_pred += (dist == label.to(device)).sum().item()
            # pred_list.append(dist.cpu())
            val_margin_loss += loss.item()
    # pred = torch.cat(pred_list, dim=0)
    return val_margin_loss / len(val_loader), val_correct_pred / len(val_loader)


def train_with_eval(train_loader, val_loader, model, optimizer, num_epoches, log_epoch, verbose, device):
    best_model = None
    best_avg_val_margin_loss = float('inf')
    for epoch_idx in range(1, num_epoches + 1):
        avg_train_loss, train_accuracy = train_one_epoch(train_loader, model, optimizer, False, device)
        avg_val_margin_loss, val_accuracy = val_one_epoch(val_loader, model, device)
        if avg_val_margin_loss > best_avg_val_margin_loss:
            best_avg_val_margin_loss = avg_val_margin_loss
            best_model = copy.deepcopy(model.cpu())
        if verbose and epoch_idx % log_epoch == 0:
            print(f'epoch [{epoch_idx}]/[{num_epoches}] training loss: {avg_train_loss:.4f} '
                  f'avg_val_margin_loss: {avg_val_margin_loss:.4f} '
                  f'train_accuracy: {train_accuracy: .2f} '
                  f'val_accuracy: {val_accuracy: .2f} ')
    return best_avg_val_margin_loss, best_model, model

