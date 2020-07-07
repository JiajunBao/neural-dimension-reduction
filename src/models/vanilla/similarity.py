import os, math, torch
import numpy as np
from scipy.spatial import distance_matrix

from models.vanilla.settings import Global

## CHECK FOR GPU'S ##
CUDA = torch.cuda.is_available()

def compute_distance(data1, data2):
    diff = np.array(data1) - np.array(data2)
    distance_square = np.dot(diff, diff)
    return np.sqrt(distance_square)

def compute_similarity(data1, data2, min_dist_square=None):
    eps = Global.eps
    diff = np.array(data1) - np.array(data2)
    distance_square = np.dot(diff, diff)
    if min_dist_square is None:
        sim = 1 / (1 + eps)
    else:
        if min_dist_square < distance_square:
            sim = 1 / (1 + eps + (distance_square - min_dist_square) / min_dist_square)
        else:
            sim = 1 / (1 + eps)
    # Compute tensors #
    if CUDA:
        sim_tf = torch.tensor(sim).float().cuda()
    else:
        sim_tf = torch.tensor(sim).float() # double
    return sim_tf, distance_square