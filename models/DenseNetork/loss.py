import torch
from scipy.spatial import distance_matrix
import numpy
from tqdm.auto import tqdm
from models.DenseNetork import ANCHOR_SIZE

STABLE_FACTOR = 1e-7


def nearest_neighbors(x, top_k):
    """
    calculate the nearest neighbors of x, return the
    :param x: for matrix to calculate nearest neighbor
    :param top_k: number of the nearest neighbor to be returned
    :return:
            ground_min_dist_square: torch.tensor (n, ) distance to the nearest neighbor
            topk_neighbors: torch.tensor (n, top_k) the index of the top-k nearest neighbors;
    """
    if x.shape[0] * x.shape[1] < 90000:  # direct computes the whole matrix
        dist = torch.cdist(x1=x, x2=x, p=2)  # (n, n)
        sorted_dist, indices = torch.sort(dist, dim=1, descending=False)
        ground_min_dist_square = sorted_dist[:, 1]  # the 0-th column is the distance to oneself
        topk_neighbors = indices[:, 1:1 + top_k]
    else:  # calculate the nearest neighbors in batches
        batch_size = 15000
        num_iter = x.shape[0] // batch_size + 1
        topk_neighbors_list = list()
        ground_min_dist_square_list = list()
        for i in tqdm(torch.arange(num_iter), desc='computing nearest neighbors in batches'):
            batch_x = x[i * batch_size: (i + 1) * batch_size, :]
            dist = torch.cdist(x1=batch_x, x2=x, p=2)  # (n, n)
            sorted_dist, indices = torch.sort(dist, dim=1, descending=False)
            batch_ground_min_dist_square = sorted_dist[:, 1]  # the 0-th column is the distance to oneself
            batch_topk_neighbors = indices[:, 1:1 + top_k]
            print(batch_ground_min_dist_square.shape, batch_topk_neighbors.shape)
            topk_neighbors_list.append(batch_topk_neighbors.cpu())
            ground_min_dist_square_list.append(batch_ground_min_dist_square.cpu())
        ground_min_dist_square = torch.cat(ground_min_dist_square_list, dim=0)
        topk_neighbors = torch.cat(topk_neighbors_list, dim=0)
        print(ground_min_dist_square.shape, topk_neighbors.shape)
    return ground_min_dist_square, topk_neighbors


def kl_div_add_mse_loss(p, q, lam):
    """
    calculate the sum of kl divergence and mse loss
    :param p: p in the formula (P20-2) output similarities
    :param q: q in the formula (P20-2) input similarities
    :param lam: the constant that balances the influence of two losses
    :return: torch.tensor of the shape (,)
    """
    return torch.sum(p * torch.log(p / q)) + lam * torch.sum((p - q) ** 2)


def input_inverse_similarity(x, anchor_idx, min_dist_square, approximate_min_dist=False):
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
    print(y.shape, x.shape)
    din = (x.unsqueeze(dim=1) - y).square().sum(dim=2)   # (n, m)
    dmin_x = min_dist_square.unsqueeze(dim=1)  # (n, 1)
    dmin_y = min_dist_square[anchor_idx]  # (n, m)
    if approximate_min_dist:
        raise NotImplementedError("min_dist_square should give ground-true values")
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
    anchors = y[anchor_idx, :]  # (n, m, d2)
    y = y.unsqueeze(dim=1)  # (n, 1, d2)
    return 1 / (1 + torch.sum((y - anchors).square(), dim=2))
