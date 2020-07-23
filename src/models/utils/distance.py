import torch
from tqdm.auto import tqdm
from torch.nn import functional as F


def nearest_neighbors(x, top_k, device):
    """
    calculate the nearest neighbors of x, return the
    :param x: for matrix to calculate nearest neighbor
    :param top_k: number of the nearest neighbor to be returned
    :param device: device used during computation
    :return:
            ground_min_dist_square: torch.tensor (n, ) distance to the nearest neighbor
            topk_neighbors: torch.tensor (n, top_k) the index of the top-k nearest neighbors;
    """
    batch_size = 2000
    x_to_device = x.to(device)
    if x_to_device.shape[0] * x_to_device.shape[1] < batch_size * 200:  # direct computes the whole matrix
        dist = torch.cdist(x1=x_to_device, x2=x_to_device, p=2)  # (n, n)
        sorted_dist, indices = torch.sort(dist, dim=1, descending=False)
        ground_min_dist_square = sorted_dist[:, 1]  # the 0-th column is the distance to oneself
        topk_neighbors = indices[:, 1:1 + top_k]
        topk_dists = sorted_dist[:, 1:1 + top_k]
    else:  # calculate the nearest neighbors in batches
        num_iter = x_to_device.shape[0] // batch_size + 1
        topk_neighbors_list = list()
        ground_min_dist_square_list = list()
        sorted_dist_list = list()
        for i in tqdm(torch.arange(num_iter), desc='computing nearest neighbors'):
            batch_x = x_to_device[i * batch_size: (i + 1) * batch_size, :]
            dist = torch.cdist(x1=batch_x, x2=x_to_device, p=2)  # (n, n)
            sorted_dist, indices = torch.sort(dist, dim=1, descending=False)
            batch_ground_min_dist_square = sorted_dist[:, 1]  # the 0-th column is the distance to oneself
            batch_topk_neighbors = indices[:, 1:1 + top_k]
            topk_neighbors_list.append(batch_topk_neighbors.cpu())
            ground_min_dist_square_list.append(batch_ground_min_dist_square.cpu())
            sorted_dist_list.append(sorted_dist[:, 1:1 + top_k].cpu())
        ground_min_dist_square = torch.cat(ground_min_dist_square_list, dim=0)
        topk_neighbors = torch.cat(topk_neighbors_list, dim=0)
        topk_dists = torch.cat(sorted_dist_list, dim=0)
    return ground_min_dist_square.cpu(), topk_neighbors.cpu(), topk_dists.cpu()


def euclidean_softmax_similarity(vec_i, vec_j, ground_min_dist_square_i, two_eps_square=1):
    """
    calculate inverse similarity for inputs:
        1 / ((d_{in}(x_i, x_j))^2 / d_i^2 + eps)
    :param vec_i:  torch.tensor of the shape (n, m)
                xi
    :param vec_j: torch.tensor of the shape (n, m)
                xj
    :param two_eps_square: 2 * (epsilon)^2
    :param ground_min_dist_square_i:
    :return: q: qij in formula (P20-3) torch.tensor of the shape (n, m)
    """

    din = (vec_i.unsqueeze(dim=1) - vec_j).square().sum(dim=2)  # (n, m)
    din = din / ground_min_dist_square_i.view(-1, 1)
    sim_j_given_i = F.softmin(din ** 2 / two_eps_square, dim=1)  # (n, m)
    print('din', din)
    print('sim_j_given_i', sim_j_given_i)
    return sim_j_given_i


def kl_div_loss(input_similarity, output_similarity):
    """
    calculate the sum of kl divergence and mse loss
    :param input_similarity: input similarity
    :param output_similarity: output similarity
    :return: torch.tensor of the shape (,)
    """
    return torch.sum(input_similarity * torch.log(input_similarity / output_similarity))


def precomputing(x, top_k, device):
    """
    compute ground true nearest neighbors
    :param x:
    :param top_k: top-k neighbors that are considered
    :param device: device used during computation
    :return: anchor_idx: each point has m points as anchors (in the case, we pick m near neighbors of x as anchors)
             input_similarity: input_similarity
    """
    ground_min_dist_square, anchor_idx, topk_dists = nearest_neighbors(x, top_k, device)

    xi = x
    xj = x[anchor_idx, :]
    input_similarity = euclidean_softmax_similarity(xi, xj, ground_min_dist_square)
    return anchor_idx, input_similarity, ground_min_dist_square, topk_dists
