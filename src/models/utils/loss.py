from src.models.utils.distance import precomputing, euclidean_softmax_similarity, kl_div_loss


class StochasticNeighborLoss:
    def __init__(self, anchor_idx, input_similarity):
        super(StochasticNeighborLoss, self).__init__()
        self.anchor_idx = anchor_idx
        self.input_similarity = input_similarity

    @classmethod
    def from_scratch(cls, x, top_k, device):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        anchor_idx, input_similarity, _, _ = precomputing(x, top_k, device)
        return cls(anchor_idx, input_similarity)

    def forward(self, output_embedding):
        yj = output_embedding[self.anchor_idx, :]  # (n, m, d)
        yi = output_embedding
        output_similarity = euclidean_softmax_similarity(yi, yj)
        loss = kl_div_loss(self.input_similarity.to(output_similarity), output_similarity)
        return loss, output_similarity

