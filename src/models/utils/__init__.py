from src.models.utils.loss import nearest_neighbors, euclidean_softmax_similarity


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
    input_similarity = euclidean_softmax_similarity(xi, xj)
    return anchor_idx, input_similarity, ground_min_dist_square, topk_dists
