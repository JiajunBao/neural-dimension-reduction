"""Inductive Stochastic Neighbor Embedding"""
from torch.nn import Module

from src.models.utils.loss import StochasticNeighborLoss
from src.models.ResNet.ResNet import get_resnet


class InsaneModel(Module):
    def __init__(self, encoder, decoder, config):
        super(InsaneModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

    @classmethod
    def from_scratch(cls, encoder_config, input_embedding, top_k, device_for_precomputing):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        encoder = get_resnet(encoder_config)
        decoder = StochasticNeighborLoss.from_scratch(input_embedding, top_k, device_for_precomputing)
        config = {"encoder_config": encoder_config, "input_embedding": input_embedding, "top_k": top_k}
        return cls(encoder, decoder, config)

    @classmethod
    def from_pretrained(cls, path_to_checkpoints):
        raise NotImplementedError

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        output_embedding = self.encoder(x)
        loss, output_similarity = self.decoder(output_embedding)
        return loss, output_embedding, output_similarity
