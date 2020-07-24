import torch
from src.models.DenseNetwork.loss import nearest_neighbors, kl_div_add_mse_loss
from src.models.utils.distance import euclidean_softmax_similarity
import torch


def test_kl_mse_loss():
    x = torch.tensor([[1.3653, -0.5120, -0.3876,  1.0540],
                      [-0.3208, -0.2595, -0.7641,  2.5738],
                      [1.0413,  0.9428,  0.4569,  0.2637]])
    ground_min_dist_square, indices, _ = nearest_neighbors(x, top_k=2, device='cuda')
    assert indices == torch.tensor([[2, 1],
                                    [0, 2],
                                    [0, 1]])
    assert ground_min_dist_square == torch.tensor([[1.8866],
                                 [2.3148],
                                [1.8866]])

    logit = torch.tensor([0.10, 0.40, 0.50])
    target = torch.tensor([0.80, 0.15, 0.05])
    assert kl_div_add_mse_loss(logit,target, 0).item() - 2.0907 < 1e-5


def test_nn_softmax_loss():
    x = torch.tensor([[1., 2, 3,],
                      [5., 1, 7,],
                      [4., 2, 1,]])
    anchor_idx = torch.tensor([[2, 1], [0, 2], [0, 1]])
    y = x[anchor_idx]
    assert torch.all(torch.eq(y, torch.tensor([[[4., 2, 1], [5., 1, 7,]],
                                               [[1., 2, 3], [4., 2, 1,]],
                                               [[1., 2, 3], [5., 1, 7,]]])))
    assert (euclidean_softmax_similarity(x, y, None, 13)[0, 0] - 0.82324) < 1e-4


test_nn_softmax_loss()
