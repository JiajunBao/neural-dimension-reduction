import torch
from scipy.spatial.distance import cdist
STABLE_FACTOR = 1e-7


def nearest_neighbors(x):
    """
    calculate the nearest neighbors of x, return the
    :param x:
    :return:
            dist: torch.tensor (n, n) nearest neighbor rankings,
                the distance between i and other points is sorted from low to high, the index is recorded.
            In the below return values, the original first column contains the points themselves, we remove them.
            sorted_dist: torch.tensor (n, n - 1) whole distance matrix;
            indices: torch.tensor (n, n - 1);
    """
    y = x.numpy()
    dist = torch.tensor(cdist(y, y, 'euclidean'))
    # dist = torch.cdist(x1=x, x2=x, p=2)  # (n, n)
    sorted_dist, indices = torch.sort(dist, dim=1, descending=False)
    return dist, sorted_dist[:, 1:], indices[:, 1:]


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
