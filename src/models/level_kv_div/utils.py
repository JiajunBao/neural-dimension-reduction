import torch
from tqdm.auto import tqdm


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


