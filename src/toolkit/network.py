from collections import OrderedDict
import torch.nn as nn
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


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


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


