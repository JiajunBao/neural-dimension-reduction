import torch
from torch.nn import functional as F
# from tqdm.auto import tqdm


def pairwise_distance(x):
    """
    calculate the nearest neighbors of x, return the
    :param x: for matrix to calculate nearest neighbor
    :return:
        dist: pairwise distances
    """
    batch_size = 2000
    if x.shape[0] * x.shape[1] < batch_size * 200:  # direct computes the whole matrix
        dist = torch.cdist(x1=x, x2=x, p=2)  # (n, n)
    else:  # calculate the nearest neighbors in batches
        num_iter = x.shape[0] // batch_size + 1
        dist = list()
        for i in torch.arange(num_iter):
            batch_x = x[i * batch_size: (i + 1) * batch_size, :]
            dist.append(torch.cdist(x1=batch_x, x2=x, p=2))  # (n, n)
        dist = torch.cat(dist, dim=0)
    return dist


def softmin_probability(xdist_batch, sigma=30):
    n = xdist_batch.shape[0]
    idx = torch.arange(n)
    d = -xdist_batch.square() / 2 / (sigma ** 2)
    d = d - torch.mean(d)
    p = d.exp()
    p[idx, idx] = 0
    p = p / p.sum(dim=1, keepdim=True)
    p[idx, idx] = 0
    return p


def calculate_perplexity(prob):
    n = prob.shape[0]
    idx = torch.arange(n)
    entropy = torch.log2(prob) * prob
    entropy[idx, idx] = 0
    hp = -torch.sum(entropy, dim=1)
    return 2 ** hp


def get_output_probability(y, alpha):
    pass