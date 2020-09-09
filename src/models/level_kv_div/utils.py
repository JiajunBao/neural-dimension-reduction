import torch
from tqdm.auto import tqdm


def far_func(sorted_dist: torch.tensor, indices: torch.tensor):
    mid = indices.shape[1] // 2
    return sorted_dist[:, mid + 1:], indices[:, mid + 1:]


def close_func(sorted_dist: torch.tensor, indices: torch.tensor):
    mid = indices.shape[1] // 2
    return sorted_dist[:, :mid + 1], indices[:, :mid + 1]


def calculate_distance(x, close_fn=close_func, far_fn=far_func):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 512
    x_device = x.to(device)
    if x.shape[0] * x.shape[1] < batch_size * 200:  # direct computes the whole matrix
        # TODO: we first assume memory fits in memory. Later, we can process the data in batches.
        dist = torch.cdist(x1=x_device, x2=x_device, p=2)  # (n, n)
        sorted_dist, indices = torch.sort(dist, dim=1, descending=False)
        sorted_dist, indices = sorted_dist.cpu(), indices.cpu()
        anchor_idx = torch.arange(x.shape[0])  # (n,)
        # the 0-th column is the distance to oneself
        close_distance, close_idx = close_fn(sorted_dist, indices)  # (n, r)
        far_distance, far_idx = far_fn(sorted_dist, indices)  # (n, r)
    else:
        num_iter = x.shape[0] // batch_size + 1
        anchor_idx_list, close_idx_list, far_idx_list = list(), list(), list()
        close_distance_list, far_distance_list = list(), list()
        for i in tqdm(torch.arange(num_iter), desc='create triplets'):
            batch_x = x[i * batch_size: (i + 1) * batch_size, :].to(device)

            dist = torch.cdist(x1=batch_x, x2=x_device, p=2)  # (n, n)
            sorted_dist, indices = torch.sort(dist, dim=1, descending=False)
            sorted_dist, indices = sorted_dist, indices
            anchor_idx = torch.arange(i * batch_size, i * batch_size + batch_x.shape[0])  # (n,)
            # assert torch.equal(anchor_idx, indices[:, 0].cpu())
            # the 0-th column is the distance to oneself
            close_distance, close_idx = close_fn(sorted_dist, indices)  # (n,)
            far_distance, far_idx = far_fn(sorted_dist, indices)  # (n, r)
            anchor_idx_list.append(anchor_idx.cpu())
            close_idx_list.append(close_idx.cpu())
            far_idx_list.append(far_idx.cpu())
            close_distance_list.append(close_distance.cpu())
            far_distance_list.append(far_distance.cpu())
        anchor_idx = torch.cat(anchor_idx_list, dim=0)
        close_idx = torch.cat(close_idx_list, dim=0)
        far_idx = torch.cat(far_idx_list, dim=0)
        close_distance = torch.cat(close_distance_list, dim=0)
        far_distance = torch.cat(far_distance_list, dim=0)
    return anchor_idx, close_idx, far_idx, close_distance, far_distance


def level_grading(sorted_indexes: torch.tensor, k: int):
    res = torch.ones_like(sorted_indexes)
    n = sorted_indexes.shape[0]
    res[:, 0] = -1
    for i in tqdm(range(n)):
        single_idxs = list()
        mutual_idxs = list()
        for j in range(1, k + 1):
            if i in sorted_indexes[sorted_indexes[i][j]][1:k + 1]:
                mutual_idxs.append(j)
            else:
                single_idxs.append(j)
        res[i][mutual_idxs] = -1
        res[i][single_idxs] = 0
    original_ordered_res = torch.ones_like(res)
    for i, tmp in enumerate(sorted_indexes):
        original_ordered_res[i, tmp] = res[i]
    return original_ordered_res


