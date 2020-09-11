import torch
from torch import nn
from torch.utils.data import Dataset
import copy
# from torch.nn import KLDivLoss
import random
from tqdm.auto import tqdm
import pandas as pd
random.seed(35)


def get_dataset(x_path, label_path):
    return SparseDataset(x_path, label_path)


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
    def __init__(self, x_path, label_path):
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
            random.shuffle(tmp_neg_data)
            data += tmp_neg_data[:len(tmp_pos_data)] + tmp_pos_data
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
    return train_margin_loss / len(train_loader.dataset), (train_correct_pred / len(train_loader.dataset), pred, gold, dist)


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
        avg_train_loss, (train_accuracy, train_pred, train_gold, dist) = train_one_epoch(train_loader, model, optimizer, False, device)
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

