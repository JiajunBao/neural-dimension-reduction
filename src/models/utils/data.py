import torch
from torch.utils.data import Dataset
import pandas as pd
from src.models.utils.distance import precomputing


class InsaneDataSet(Dataset):
    def __init__(self, x, top_k):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # device = 'cpu'
        self.anchor_idx, self.q, self.ground_min_dist_square, self.topk_dists = \
            precomputing(x, top_k=top_k, device=device)
        self.top_k = top_k
        self.x = x.cpu()

    @classmethod
    def from_df(cls, path_to_dataframe, top_k):
        print(f'Generate dataset top_k = {top_k}')
        x = torch.from_numpy(pd.read_csv(path_to_dataframe).to_numpy()).to(torch.float32)
        return cls(x, top_k)

    @classmethod
    def from_dataset(cls, path_to_tensor):
        return torch.load(path_to_tensor)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]
