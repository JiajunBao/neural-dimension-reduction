from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.fc = nn.Sequential(
            OrderedDict([
                ('bn0', nn.BatchNorm1d(200)),
                ('relu0', nn.ReLU(inplace=True)),
                ('fc0', nn.Linear(200, 500)),
                ('bn1', nn.BatchNorm1d(500)),
                ('relu1', nn.ReLU(inplace=True)),
                ('fc1', nn.Linear(500, 100)),
                ('bn2', nn.BatchNorm1d(100)),
                ('relu2', nn.ReLU(inplace=True)),
                ('fc2', nn.Linear(100, 20)),
                ('bn3', nn.BatchNorm1d(20)),
                ('relu3', nn.ReLU(inplace=True)),
                ('fc3', nn.Linear(20, 20)),
                ('bn4', nn.BatchNorm1d(20)),
                ('relu4', nn.ReLU(inplace=True)),
                ('fc4', nn.Linear(20, 20)),
                ('bn5', nn.BatchNorm1d(20)),
                ('relu5', nn.ReLU(inplace=True)),
                ('fc5', nn.Linear(20, 20)),
                ('bn6', nn.BatchNorm1d(20)),
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


