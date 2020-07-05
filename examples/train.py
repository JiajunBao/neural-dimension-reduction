from models.DenseNetork.models import Net, Solver
from pathlib import Path


def train():
    dim_in = 200
    hidden_dims_list = [200, 100, 50, 25, 20]
    dim_out = 20
    model = Net.from_scratch(dim_in, hidden_dims_list, dim_out)
    solver = Solver.from_scratch(model,
                                 input_dir=Path('data'),
                                 output_dir=Path('checkpoints/switch-pq-1e-3'),
                                 learning_rate=1e-3,
                                 n_epoch=600,
                                 per_gpu_batch_size=90000,
                                 weight_decay=1e-5,
                                 seed=42)
    solver.fit(num_eval_per_epoch=2)


if __name__ == '__main__':
    train()
