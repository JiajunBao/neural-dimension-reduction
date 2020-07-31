from torch import nn
from torch.nn import Module


def get_encoder(encoder_config):
    encoder = nn.Sequential(
        nn.Linear(500, 500),
        nn.ReLU(),
        nn.Linear(500, 2000),
        nn.ReLU(),
        nn.Linear(2000, 2),
    )
    return encoder


class TSNEMapper(Module):
    def __init__(self, encoder, config):
        super(TSNEMapper, self).__init__()
        self.encoder = encoder
        self.config = config

    @classmethod
    def from_scratch(cls, encoder_config):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        encoder = get_encoder(encoder_config)
        config = {"encoder_config": encoder_config}
        return cls(encoder, config)

    @classmethod
    def from_pretrained(cls, path_to_checkpoints):
        raise NotImplementedError

    def forward(self, x):
        return self.encoder(x)


class RBM(Module):
    def __init__(self, ):
        super(RBM, self).__init__()

    def forward(self):
        pass
