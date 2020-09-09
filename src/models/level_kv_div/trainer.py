import torch
from torch import nn
import copy
from torch.nn import KLDivLoss


def train_one_epoch(train_loader, model, optimizer, verbose, device):
    model = model.to(device)
    model.train()
    criterion = KLDivLoss('mean')
    loss_sum = 0.
    for i, batch in enumerate(train_loader):
        x, sim = batch
        x_device, sim_device = x.to(device), sim.to(device)
        embedded_x = model.get_embedding(x_device)
        dist = torch.cdist(x1=embedded_x, x2=embedded_x, p=2)  # (n, n)
        sorted_dist, indices = torch.sort(dist, dim=1, descending=False)
        loss = criterion(sorted_dist, sim_device)

        model.zero_grad()  # reset gradient
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        if verbose and i % 20 == 0:
            print(f'training loss: {loss_sum / (i + 1):.4f}')
    return loss_sum / len(train_loader)


def val_one_epoch(val_loader, model, device):
    model.eval()
    criterion = KLDivLoss('mean')
    val_kl_loss = 0.
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x, sim = batch
            x_device, sim_device = x.to(device), sim.to(device)
            embedded_x = model.get_embedding(x_device)
            dist = torch.cdist(x1=embedded_x, x2=embedded_x, p=2)  # (n, n)
            sorted_dist, indices = torch.sort(dist, dim=1, descending=False)
            loss = criterion(sorted_dist, sim_device)
            val_kl_loss += loss.item()
    return val_kl_loss / len(val_loader)


def train_with_eval(train_loader, val_loader, model, optimizer, num_epoches, verbose, device):
    best_model = None
    best_avg_val_kl_loss = float('inf')
    for epoch_idx in range(1, num_epoches + 1):
        avg_loss = train_one_epoch(train_loader, model, optimizer, False, device)
        avg_val_kl_loss = val_one_epoch(val_loader, model, device)
        if avg_val_kl_loss > best_avg_val_kl_loss:
            best_avg_val_kl_loss = avg_val_kl_loss
            best_model = copy.deepcopy(model.cpu())
        if verbose and (epoch_idx) % 40 == 0:
            print(f'epoch [{epoch_idx}]/[{num_epoches}] training loss: {avg_loss:.4f} '
                  f'avg_val_kl_loss: {avg_val_kl_loss:.4f} ')
    return best_avg_val_kl_loss, best_model, model

