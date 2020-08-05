from matplotlib import pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import torch


def plot_model(embedding, labels):
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], marker='o', s=1, edgecolor='', c=labels)
    fig.tight_layout()


def plot_differences(embedding, actual, lim=1000):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    for a, b in zip(embedding, actual)[:lim]:
        ax.add_line(Line2D((a[0], b[0]), (a[1], b[1]), linewidth=1))
    ax.autoscale_view()
    plt.show()


# P is the joint probabilities for this batch (Keras loss functions call this y_true)
# activations is the low-dimensional output (Keras loss functions call this y_pred)
def tsne_2(P, activations):
#     d = K.shape(activations)[1]
    d = 2 # TODO: should set this automatically, but the above is very slow for some reason
    n = batch_size # TODO: should set this automatically
    v = d - 1.
    eps = K.variable(10e-15) # needs to be at least 10e-8 to get anything after Q /= K.sum(Q)
    sum_act = K.sum(K.square(activations), axis=1)
    Q = K.reshape(sum_act, [-1, 1]) + -2 * K.dot(activations, K.transpose(activations))
    Q = (sum_act + Q) / v
    Q = K.pow(1 + Q, -(v + 1) / 2)
    Q *= K.variable(1 - np.eye(n))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)
    C = K.log((P + eps) / (Q + eps))
    C = K.sum(P * C)
    return C


def tsne(P, activations):
    n, d = activations.shape
    alpha = d - 1
    eps = 10e-8
    act_pdist = torch.cdist(x1=activations, x2=activations, p=2.)
    terms = (1 + act_pdist ** 2 / alpha) ** (-(alpha + 1) / 2)
    idx = torch.arange(n)
    terms[idx, idx] = 0
    Q = terms / terms.sum(dim=1, keepdims=True)
    Q = torch.clamp(Q, min=eps)
    C = P * torch.log((P + eps) / (Q + eps))
    C[idx, idx] = 0  # remove i == j terms
    return C.sum()



