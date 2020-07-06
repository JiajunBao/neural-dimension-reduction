import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE
import random
import numpy as np
import pandas as pd
from settings import Global
from similarity import compute_similarity, compute_distance
from loss import InvLoss
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from utils import timer
from tqdm.auto import tqdm

## CHECK FOR GPU'S ##
CUDA = torch.cuda.is_available()

if CUDA:
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


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
    x = x.to(device)
    if x.shape[0] * x.shape[1] < batch_size * 200:  # direct computes the whole matrix
        dist = torch.cdist(x1=x, x2=x, p=2)  # (n, n)
        sorted_dist, indices = torch.sort(dist, dim=1, descending=False)
        ground_min_dist_square = sorted_dist[:, 1]  # the 0-th column is the distance to oneself
        topk_neighbors = indices[:, 1:1 + top_k]
        topk_dists = sorted_dist[:, 1:1 + top_k]
    else:  # calculate the nearest neighbors in batches
        num_iter = x.shape[0] // batch_size + 1
        topk_neighbors_list = list()
        ground_min_dist_square_list = list()
        sorted_dist_list = list()
        for i in tqdm(torch.arange(num_iter), desc='computing nearest neighbors'):
            batch_x = x[i * batch_size: (i + 1) * batch_size, :]
            dist = torch.cdist(x1=batch_x, x2=x, p=2)  # (n, n)
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


class SortDB:
    MAX_LEN = 20

    def __init__(self):
        self.array = []
        self.n = len(self.array)

    def add(self, elem): # elem = (dist, index)
        if self.n == 0 or self.n < SortDB.MAX_LEN or elem[0] < self.array[-1][0]:
            arr = self.array + [elem]
            self.array = sorted(arr, key=lambda x: x[0])
            if len(self.array) > SortDB.MAX_LEN:
                self.array = self.array[:-1]
            self.n = len(self.array)

class Module(nn.Module, Global):
    def __init__(self):
        super().__init__()
        Global.__init__(self)

class DeepNet(Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = []
        size_1 = input_size
        for hidden_size in hidden_sizes:
            size_2 = hidden_size
            self.hidden_layers.append(nn.Linear(size_1, size_2).to(device))
            size_1 = hidden_size
        self.output = nn.Linear(size_1, output_size).to(device)

    def forward(self, out):
        for layer in self.hidden_layers:
            out = F.relu(layer(out))
        out = self.output(out)
        return out

class Reducer(Global):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.correct = {1:[],5:[],10:[],20:[]}
        self.cross_correct = {1:[],5:[],10:[],20:[]}

    def fit(self, data_df, args):
        self.out('Needs to be overriden')

    def transform(self, data_df):
        self.out('Needs to be overriden')

    def create_epoch_plot(self, losses, filename=None):
        epochs = np.array(list(range(1,len(losses)+1)))
        self.increase_plots()
        plt.figure(self.num_of_plots)
        self.save_plot_name(self.num_of_plots, filename) #######
        plt.plot(epochs, losses)
        plt.grid(True)
        plt.title('Train Loss Per Epoch')

    def create_plot(self, data_df, name=None, filename=None):
        self.increase_plots()
        plt.figure(self.num_of_plots)
        self.save_plot_name(self.num_of_plots, filename) #######
        projected_data = self.transform(data_df)
        plt.scatter(projected_data[:, 0], projected_data[:, 1])
        plt.grid(True)
        if name is None:
            plt.title(self.name)
        else:
            plt.title(name)

    def create_r_plot(self, R, name='Distribution of R values', filename=None):
        self.increase_plots()
        plt.figure(self.num_of_plots)
        self.save_plot_name(self.num_of_plots, filename) #######
        plt.hist(R,bins=80)
        plt.grid(True)
        if name is None:
            plt.title(self.name)
        else:
            plt.title(name)

    @timer
    def find_nearest_neighbors(self, data_df):
        self.out('\nComputing neighbors.')
        data_np = data_df.to_numpy()
        print('data_np shape', data_np.shape)
        dist = distance_matrix(data_np, data_np)
        nearest_neighbor_matrix = np.argpartition(dist, 21, axis=1)[:,:21] # d(x,x)=0, so this needs to be ommitted
        nearest_neighbors = {i:SortDB() for i in range(data_df.shape[0])}
        for i in range(data_df.shape[0]):
            for j in range(20):
                neighbor = nearest_neighbor_matrix[i,j]
                if not neighbor == i:
                    nearest_neighbors[i].add((dist[i,neighbor]**2, neighbor))
        return nearest_neighbors

    @timer
    def find_nearest_cross_neighbors(self, train_data_df, test_data_df):
        self.out('\nComputing cross neighbors.')
        dist = distance_matrix(test_data_df.to_numpy(), train_data_df.to_numpy())
        nearest_neighbor_matrix = np.argpartition(dist, 20, axis=1)[:,:20]
        nearest_neighbors = {i:SortDB() for i in range(test_data_df.shape[0])}
        for i in range(test_data_df.shape[0]):
            for j in range(20):
                neighbor = nearest_neighbor_matrix[i,j]
                nearest_neighbors[i].add((dist[i,neighbor]**2, neighbor))
        return nearest_neighbors

    def count_neighbors(self, data_df, test=False):
        if test:
            nearest_neighbors = self.actual_test_nearest_neighbors
        else:
            nearest_neighbors = self.actual_train_nearest_neighbors
        N = data_df.shape[0]
        # project data #
        projected_data_df = pd.DataFrame(self.transform(data_df))
        # find nearest neighbors of projected data #
        projected_nearest_neighbors = self.find_nearest_neighbors(projected_data_df)
        # count corrects #
        correct = [i for i in range(N) if nearest_neighbors[i].array[0][1] == projected_nearest_neighbors[i].array[0][1]]
        correct_5 = [i for i in range(N) if nearest_neighbors[i].array[0][1] in [x[1] for x in projected_nearest_neighbors[i].array[:5]]]
        correct_10 = [i for i in range(N) if nearest_neighbors[i].array[0][1] in [x[1] for x in projected_nearest_neighbors[i].array[:10]]]
        correct_20 = [i for i in range(N) if nearest_neighbors[i].array[0][1] in [x[1] for x in projected_nearest_neighbors[i].array]]
        if not test:
            msg = 'Train'
        else:
            msg = 'Test'
        msg1 = '{} correct neighbors: {} / {} ({:.2f}%)'.format(msg, len(correct), N, 100 * len(correct) / N)
        msg2 = '{} correct neighbors in top 5: {} / {} ({:.2f}%)'.format(msg, len(correct_5), N, 100 * len(correct_5) / N)
        msg3 = '{} correct neighbors in top 10: {} / {} ({:.2f}%)'.format(msg, len(correct_10), N, 100 * len(correct_10) / N)
        msg4 = '{} correct neighbors in top 20: {} / {} ({:.2f}%)'.format(msg, len(correct_20), N, 100 * len(correct_20) / N)
        for msg in [msg1, msg2, msg3, msg4]:
            self.out(msg)
        if test:
            R = [compute_distance(data_df.iloc[i], data_df.iloc[projected_nearest_neighbors[i].array[0][1]]) / compute_distance(data_df.iloc[i], data_df.iloc[nearest_neighbors[i].array[0][1]]) - 1 for i in range(N)]
            self.R_data = R
            self.correct[1].append((msg, len(correct), N, 100 * len(correct) / N))
            self.correct[5].append((msg, len(correct_5), N, 100 * len(correct_5) / N))
            self.correct[10].append((msg, len(correct_10), N, 100 * len(correct_10) / N))
            self.correct[20].append((msg, len(correct_20), N, 100 * len(correct_20) / N))
        return '\n'.join([msg1, msg2, msg3, msg4])

    def count_cross_neighbors(self, train_data_df, test_data_df, test=False):
        if test:
            nearest_neighbors = self.actual_test_cross_nearest_neighbors
        else:
            nearest_neighbors = self.actual_train_cross_nearest_neighbors
        N = test_data_df.shape[0]
        # project data #
        projected_train_data_df = pd.DataFrame(self.transform(train_data_df))
        projected_test_data_df = pd.DataFrame(self.transform(test_data_df))
        # find nearest neighbors of projected data #
        projected_nearest_neighbors = self.find_nearest_cross_neighbors(projected_train_data_df, projected_test_data_df)
        # count corrects #
        correct = [i for i in range(N) if nearest_neighbors[i].array[0][1] == projected_nearest_neighbors[i].array[0][1]]
        correct_5 = [i for i in range(N) if nearest_neighbors[i].array[0][1] in [x[1] for x in projected_nearest_neighbors[i].array[:5]]]
        correct_10 = [i for i in range(N) if nearest_neighbors[i].array[0][1] in [x[1] for x in projected_nearest_neighbors[i].array[:10]]]
        correct_20 = [i for i in range(N) if nearest_neighbors[i].array[0][1] in [x[1] for x in projected_nearest_neighbors[i].array]]
        if not test:
            msg = 'Train'
        else:
            msg = 'Test'
        R = [compute_distance(test_data_df.iloc[i], train_data_df.iloc[projected_nearest_neighbors[i].array[0][1]]) / compute_distance(test_data_df.iloc[i], train_data_df.iloc[nearest_neighbors[i].array[0][1]]) - 1 for i in range(N)]
        self.cross_R_data = R

class Net(Module, Reducer):
    def __init__(self, hidden_model, input_size, output_size, hidden_sizes):
        super().__init__()
        Reducer.__init__(self, 'NET')
        self.input_size = input_size
        self.output_size = output_size
        self.model = hidden_model(input_size, output_size, hidden_sizes).to(device)

    def forward(self, x):
        x = x.to(device)
        out = self.model(x)
        return out

    def transform(self, data_df):
        X = torch.tensor(np.array(data_df)).float()
        output = self.forward(X)
        return output.cpu().data.numpy()

    def fit(self, data_df, args):
        if Global.IN_SAMPLE_TESTING:
            self.actual_train_nearest_neighbors  = self.find_nearest_neighbors(data_df)
        # Prepare inputs to fit and params #
        X = torch.tensor(np.array(data_df)).float()
        N = X.shape[0]
        self.train()
        epochs = args['epochs']
        criterion = InvLoss(args['lambda'])
        optimizer = optim.Adam(self.parameters(), lr=args['learning_rate'])
        # store the minimum square distances #
        min_dist_dict = {i:None for i in range(N)}
        self.out('\nFitting the model...')
        losses = []

        ground_min_dist_square, _, _ = nearest_neighbors(X, 1, 'cpu')

        for epoch in range(epochs):
            running_loss = 0
            self.out('EPOCH: {}'.format(epoch+1))
            for i in self.tqdm(np.random.permutation(N)):
                input1 = X[i]
                # get random elem, diff from i #
                j = i
                while j == i:
                    j = random.randint(0,N-1)
                input2 = X[j]
                # # get minimum distance squares so far #
                # if min_dist_dict[i] is None:
                #     min_dist_square_i = None
                # else:
                #     min_dist_square_i = min_dist_dict[i][0]
                # if min_dist_dict[j] is None:
                #     min_dist_square_j = None
                # else:
                #     min_dist_square_j = min_dist_dict[j][0]
                min_dist_square_i = ground_min_dist_square[i]
                min_dist_square_j = ground_min_dist_square[j]

                # compute similarities #
                sim_i, dist_square = compute_similarity(data_df.iloc[i], data_df.iloc[j], min_dist_square_i)
                sim_j, _ = compute_similarity(data_df.iloc[j], data_df.iloc[i], min_dist_square_j)
                sim = (sim_i + sim_j) / 2
                sim = sim.reshape((1))
                # pass inputs from model #
                output1 = self.forward(input1)
                output2 = self.forward(input2)
                # update storage #
                if min_dist_dict[i] is None or dist_square < min_dist_dict[i][0]:
                    min_dist_dict[i] = (dist_square, j)
                if min_dist_dict[j] is None or dist_square < min_dist_dict[j][0]:
                    min_dist_dict[j] = (dist_square, i)
                # compute loss and backpropagate #
                loss = criterion(output1, output2, sim)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            self.out('Train loss: {:.2f}'.format(running_loss)) 
            losses.append(running_loss)
            # test after every epoch #
            if Global.IN_SAMPLE_TESTING:
                self.count_neighbors(data_df, test=False)
        # plot loss per epoch #
        if args['to_plot']:
            losses = np.array(losses)
            self.create_epoch_plot(losses, filename='loss')
        return self

    def test(self, train_data_df, test_data_df):
        self.actual_test_nearest_neighbors  = self.find_nearest_neighbors(test_data_df)
        self.actual_test_cross_nearest_neighbors = self.find_nearest_cross_neighbors(train_data_df, test_data_df)
        self.count_neighbors(test_data_df, test=True)
        self.count_cross_neighbors(train_data_df, test_data_df, test=True)
        self.create_r_plot(self.R_data, filename='r_hist')
        self.create_r_plot(self.cross_R_data, filename='r_cross_hist')
        return self.R_data, self.cross_R_data, self.correct