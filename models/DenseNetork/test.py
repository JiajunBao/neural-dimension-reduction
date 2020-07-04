from models.DenseNetork.loss import nearest_neighbors, kl_div_add_mse_loss
import torch


x = torch.tensor([[1.3653, -0.5120, -0.3876,  1.0540],
                  [-0.3208, -0.2595, -0.7641,  2.5738],
                  [1.0413,  0.9428,  0.4569,  0.2637]])
dist, sorted_dist, indices = nearest_neighbors(x)
assert indices == torch.tensor([[2, 1],
                                [0, 2],
                                [0, 1]])
assert dist == torch.tensor([[0.0000, 2.3148, 1.8866],
                             [2.3148, 0.0000, 3.1825],
                            [1.8866, 3.1825, 0.0000]])

logit = torch.tensor([0.10, 0.40, 0.50])
target = torch.tensor([0.80, 0.15, 0.05])
assert kl_div_add_mse_loss(logit,target, 0).item() - 2.0907 < 1e-5
