import torch
from torch import nn
from torch.nn import functional as F

STABLE_FACTOR = 1e-8


def kl_div_add_mse_loss(logit, target, alpha, reduction='sum'):
    """
    calculate the sum of kl divergence and mse loss
    :param logit: p in the formula (P20-2)
    :param target: q in the formula (P20-2)
    :param alpha: the constant that balances the influence of two losses
    :param reduction: "sum", "mean" or "batchmean": same as https://pytorch.org/docs/stable/nn.functional.html#kl-div
    :return: torch.tensor of the shape (,)
    """
    mse_loss = nn.MSELoss(reduction=reduction)
    return F.kl_div(input=logit, target=target, reduction=reduction) + alpha * mse_loss(input=logit, target=target)


def input_inverse_similarity(x, anchor_idx, min_dist_square, approximate_min_dist=True):
    """
    calculate inverse similarity for inputs:
        1 / ((d_{in}(x_i, x_j))^2 / d_i^2 + eps)
    :param x:  torch.tensor of the shape (n, d1)
                n rows, each of which represents an input vector (x_i in the formula)
                with dimension d1. (n can be the batch size)
    :param anchor_idx: torch.tensor of the shape (n, m)
                each row (x_j in the formula) in x has m anchor points, anchor_idx contains their indexes.
                ! anchor_idx should follows the i != j condition in P19-1
    :param min_dist_square: (torch.tensor)
                (n,) the **square** of minimum distance of a point from x_i
    :param approximate_min_dist: (bool)
            if the min_dist_square needs to be updated
    :return: q: qij in formula (P20-3) torch.tensor of the shape (n, m)
    """
    y = x[anchor_idx, :]  # (n, m, d)
    din = (x.squeeze(dim=1) - y).square().sum(dim=2)   # (n, m)
    dmin_x = min_dist_square.squeeze(dim=1)  # (n, )
    dmin_y = min_dist_square[anchor_idx]  # (n, m)
    sigma_xy = 1 / (din / dmin_x + STABLE_FACTOR)
    sigma_yx = 1 / (din / dmin_y + STABLE_FACTOR)
    return (sigma_xy + sigma_yx) / 2


def output_inverse_similarity(y, anchor_idx):
    """

    :param y: torch.tensor (n, d2),
            the forwarded results of input vectors through the networl
    :param anchor_idx: torch.tensor (n, m)
    :return:
    """
    y = y.squeeze(dim=1)  # (n, 1, d2)
    anchors = y[anchor_idx, :]  # (n, m, d2)
    return 1 / (1 + torch.sum((y - anchors).square(dim=2)))