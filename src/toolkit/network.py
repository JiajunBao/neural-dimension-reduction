from collections import OrderedDict
import torch.nn as nn
import torch
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.fc = nn.Sequential(
            OrderedDict([
                ('bn0', nn.BatchNorm1d(128)),
                ('relu0', nn.ReLU(inplace=True)),
                ('fc0', nn.Linear(128, 128)),
                ('bn1', nn.BatchNorm1d(128)),
                ('relu1', nn.ReLU(inplace=True)),
                ('fc1', nn.Linear(128, 64)),
                ('bn2', nn.BatchNorm1d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('fc2', nn.Linear(64, 64)),
                ('bn3', nn.BatchNorm1d(64)),
                ('relu3', nn.ReLU(inplace=True)),
                ('fc3', nn.Linear(64, 64)),
                ('bn4', nn.BatchNorm1d(64)),
                ('relu4', nn.ReLU(inplace=True)),
                ('fc4', nn.Linear(64, 32)),
                ('bn5', nn.BatchNorm1d(32)),
                ('relu5', nn.ReLU(inplace=True)),
                ('fc5', nn.Linear(32, 32)),
                ('bn6', nn.BatchNorm1d(32)),
                ('relu6', nn.ReLU(inplace=True)),
                ('fc6', nn.Linear(32, 16)),
                ('bn7', nn.BatchNorm1d(16)),
                ('relu7', nn.ReLU(inplace=True)),
                ('fc7', nn.Linear(16, 16)),
                ('bn8', nn.BatchNorm1d(16)),
            ])
        )

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.norm_layer = nn.BatchNorm1d(128)
        self.encoder = nn.Sequential(
            OrderedDict([
                ('fc0', nn.Linear(128, 128)),
                ('relu0', nn.ReLU(inplace=True)),
                ('bn0', nn.BatchNorm1d(128)),

                ('fc1', nn.Linear(128, 128)),
                ('relu1', nn.ReLU(inplace=True)),
                ('bn1', nn.BatchNorm1d(128)),

                ('fc2', nn.Linear(128, 64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('bn2', nn.BatchNorm1d(64)),

                ('fc3', nn.Linear(64, 32)),
                ('relu3', nn.ReLU(inplace=True)),
                ('bn3', nn.BatchNorm1d(32)),

                ('fc4', nn.Linear(32, 32)),
                ('relu4', nn.ReLU(inplace=True)),
                ('bn4', nn.BatchNorm1d(32)),
            ])
        )
        self.decoder = nn.Sequential(
            OrderedDict([
                ('fc5', nn.Linear(32, 64)),
                ('relu5', nn.ReLU(inplace=True)),
                ('bn5', nn.BatchNorm1d(64)),

                ('fc6', nn.Linear(64, 128)),
                ('relu6', nn.ReLU(inplace=True)),
                ('bn6', nn.BatchNorm1d(128)),
            ])
        )

    def forward(self, x):
        normed_x = self.norm_layer(x)
        low_embed = self.encoder(normed_x)
        reconstructed_embed = self.decoder(low_embed)
        reconstructed_loss = (normed_x - reconstructed_embed).sum()
        return low_embed, reconstructed_loss

    def get_embedding(self, x):
        out = self.norm_layer(x)
        return self.encoder(out)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class ReconstructSiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(ReconstructSiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        embedded_x1, reconstruct_loss1 = self.embedding_net(x1)
        embedded_x2, reconstruct_loss2 = self.embedding_net(x2)
        assert len(x1.shape) == 2 and len(embedded_x1.shape) == 2
        dist1 = torch.sum(((x1 - x2) / x1.shape[1]) ** 2)
        dist2 = torch.sum(((x1 - x2) / embedded_x1.shape[1]) ** 2)
        return reconstruct_loss1 + reconstruct_loss2 + (dist1 - dist2) ** 2

    def get_embedding(self, x):
        return self.embedding_net(x)

