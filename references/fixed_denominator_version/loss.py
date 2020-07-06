import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from settings import Global

## CHECK FOR GPU'S ##
CUDA = torch.cuda.is_available()

if CUDA:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class Module(nn.Module, Global):
    def __init__(self, param):
        super().__init__()
        Global.__init__(self)
        self.param = param

class InvLoss(Module):
    def __init__(self, param):
        super().__init__(param)

    def forward(self, input1, input2, similarity):
        one = torch.as_tensor(np.ones(1) + Global.eps).to(device)
        eucl = nn.PairwiseDistance(p=2, eps=1e-6)
        input1 = input1.view(1, -1).to(device)
        input2 = input2.view(1, -1).to(device)
        dist = eucl(input1, input2).double()
        sim = torch.div(one, torch.addcmul(one, 1, dist, dist)).float()
        similarity = similarity.to(device)
        return torch.add(input=F.kl_div(sim, similarity), alpha=self.param, other=F.mse_loss(sim, similarity)).to(device)