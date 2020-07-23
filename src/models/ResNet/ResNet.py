from torch import nn
from collections import OrderedDict


class PlainBlock(nn.Module):
    def __init__(self, cin, cout, downsample=False):
        super().__init__()
        if downsample:
            self.net = nn.Sequential(
                OrderedDict([
                    ('bn1', nn.BatchNorm2d(cin)),
                    ('relu1', nn.ReLU()),
                    ('conv1', nn.Conv2d(cin, cout, 3, 2, padding=1)),
                    ('bn2', nn.BatchNorm2d(cout)),
                    ('relu2', nn.ReLU()),
                    ('conv2', nn.Conv2d(cout, cout, 3, 1, padding=1))
                ]))
        else:
            self.net = nn.Sequential(
                OrderedDict([
                    ('bn1', nn.BatchNorm2d(cin)),
                    ('relu1', nn.ReLU()),
                    ('conv1', nn.Conv2d(cin, cout, 3, 1, padding=1)),
                    ('bn2', nn.BatchNorm2d(cout)),
                    ('relu2', nn.ReLU()),
                    ('conv2', nn.Conv2d(cout, cout, 3, 1, padding=1))
                ]))

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, downsample=False):
        super().__init__()
        self.block = PlainBlock(cin=cin, cout=cout, downsample=downsample)
        if downsample:
            self.shortcut = nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=1, stride=2, padding_mode='same')
        else:
            if cin == cout:
                self.shortcut = nn.Identity()
            else:
                nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=1, stride=1, padding_mode='same')

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
    def __init__(self, cin=3, cout=8):
        super().__init__()
        layers = [
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    def __init__(self, stage_args, cin=3, block=ResidualBlock, num_classes=10):
        super().__init__()

        self.cnn = None
        blocks = [ResNetStem(cin=cin, cout=stage_args[0][0])]
        self.last = stage_args[-1][1]
        for cin, cout, num_blocks, downsample in stage_args:
            blocks.append(ResNetStage(cin=cin, cout=cout, num_blocks=num_blocks, downsample=downsample, block=block))
        self.cnn = nn.Sequential(*blocks)
        self.fc = nn.Linear(stage_args[-1][1], num_classes)

    def forward(self, x):
        out = self.cnn(x)
        N, C, H, W = out.shape
        out = nn.AvgPool2d(kernel_size=(H, W)).forward(out)
        out = out.view(-1, self.last)
        scores = self.fc(out)
        return scores


def get_resnet(config):
    return ResNet(**config)
