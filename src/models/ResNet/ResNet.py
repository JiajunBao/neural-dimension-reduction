from torch import nn
from collections import OrderedDict


class PlainBlock(nn.Module):
    def __init__(self, cin, cout, downsample=False):
        super().__init__()
        if downsample:
            self.net = nn.Sequential(
                OrderedDict([
                    ('bn1', nn.BatchNorm1d(cin)),
                    ('relu1', nn.ReLU()),
                    ('conv1', nn.Conv1d(cin, cout, 3, 2, padding=1)),
                    ('bn2', nn.BatchNorm1d(cout)),
                    ('relu2', nn.ReLU()),
                    ('conv2', nn.Conv1d(cout, cout, 3, 1, padding=1))
                ]))
        else:
            self.net = nn.Sequential(
                OrderedDict([
                    ('bn1', nn.BatchNorm1d(cin)),
                    ('relu1', nn.ReLU()),
                    ('conv1', nn.Conv1d(cin, cout, 3, 1, padding=1)),
                    ('bn2', nn.BatchNorm1d(cout)),
                    ('relu2', nn.ReLU()),
                    ('conv2', nn.Conv1d(cout, cout, 3, 1, padding=1))
                ]))

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, downsample=False):
        super().__init__()
        self.block = PlainBlock(cin=cin, cout=cout, downsample=downsample)
        if downsample:
            self.shortcut = nn.Conv1d(in_channels=cin, out_channels=cout, kernel_size=1, stride=2, padding_mode='zeros')
        else:
            if cin == cout:
                self.shortcut = nn.Identity()
            else:
                nn.Conv1d(in_channels=cin, out_channels=cout, kernel_size=1, stride=1, padding_mode='zeros')

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, cin, cout, downsample=False):
        super().__init__()
        if downsample:
            self.block = nn.Sequential(OrderedDict([
                ('bn1', nn.BatchNorm1d(cin)),
                ('relu1', nn.ReLU()),
                ('conv1', nn.Conv1d(cin, cout // 4, kernel_size=1, stride=2)),
                ('bn2', nn.BatchNorm1d(cout // 4)),
                ('relu2', nn.ReLU()),
                ('conv2', nn.Conv1d(cout // 4, cout // 4, kernel_size=3, stride=1, padding=1)),
                ('bn3', nn.BatchNorm1d(cout // 4)),
                ('relu3', nn.ReLU()),
                ('conv3', nn.Conv1d(cout // 4, cout, kernel_size=1))
            ]))
        else:
            self.block = nn.Sequential(OrderedDict([
                ('bn1', nn.BatchNorm1d(cin)),
                ('relu1', nn.ReLU()),
                ('conv1', nn.Conv1d(cin, cout // 4, kernel_size=1, stride=1)),
                ('bn2', nn.BatchNorm1d(cout // 4)),
                ('relu2', nn.ReLU()),
                ('conv2', nn.Conv1d(cout // 4, cout // 4, kernel_size=3, stride=1, padding=1)),
                ('bn3', nn.BatchNorm1d(cout // 4)),
                ('relu3', nn.ReLU()),
                ('conv3', nn.Conv1d(cout // 4, cout, kernel_size=1))
            ]))

        if downsample:
            self.shortcut = nn.Conv1d(in_channels=cin, out_channels=cout, kernel_size=1, stride=2, padding_mode='zeros')
        else:
            self.shortcut = nn.Identity() if cin == cout else nn.Conv1d(in_channels=cin, out_channels=cout,
                                                                        kernel_size=1, stride=1, padding_mode='zeros')

    def forward(self, x):
        return self.block(x) + self.shortcut(x)
    

class ResNetStage(nn.Module):
    def __init__(self, cin, cout, num_blocks, downsample=True,
                 block=ResidualBlock):
        super().__init__()
        blocks = [block(cin, cout, downsample)]
        for _ in range(num_blocks - 1):
            blocks.append(block(cout, cout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class ResNetStem(nn.Module):
    def __init__(self, cin=1, cout=8):
        super().__init__()
        layers = [
            nn.Conv1d(cin, cout, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    def __init__(self, stage_args, cin=1, block_name="ResidualBlock", ouput_dimension=20):
        super().__init__()
        if block_name == 'PlainBlock':
            block = PlainBlock
        elif block_name == 'ResidualBlock':
            block = ResidualBlock
        elif block_name == 'ResidualBottleneckBlock':
            block = ResidualBottleneckBlock
        else:
            raise NotImplementedError(f'No block module named {block_name}!')
        self.model_construct_dict = stage_args
        self.cnn = None
        blocks = [ResNetStem(cin=cin, cout=stage_args[0][0])]
        self.last = stage_args[-1][1]
        for cin, cout, num_blocks, downsample in stage_args:
            blocks.append(ResNetStage(cin=cin, cout=cout, num_blocks=num_blocks, downsample=downsample, block=block))
        self.cnn = nn.Sequential(*blocks)
        self.fc = nn.Linear(stage_args[-1][1], ouput_dimension)

    def forward(self, x):
        out = x.unsqueeze(dim=1)
        out = self.cnn(out)
        N, C, L = out.shape
        out = nn.AvgPool1d(kernel_size=L).forward(out)
        out = out.view(-1, self.last)
        out = self.fc(out)
        return out


def get_resnet(config):
    return ResNet(**config)
