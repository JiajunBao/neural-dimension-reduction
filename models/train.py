from src.models.DenseNetwork.models import Net, Solver
from pathlib import Path


def train():
    args = Solver.get_solver_arguments()
    dim_in = 200
    dim_out = 20
    hidden_dims_list = [int(x) for x in args.hidden_dims_list.split('-')]
    model = Net.from_scratch(dim_in, hidden_dims_list, dim_out, args.shortcut)
    solver = Solver.from_scratch(model,
                                 input_dir=args.input_dir,
                                 output_dir=args.output_dir,
                                 learning_rate=args.learning_rate,
                                 n_epoch=args.n_epoch,
                                 per_gpu_batch_size=args.per_gpu_batch_size,
                                 weight_decay=args.weight_decay,
                                 seed=args.seed,
                                 top_k=args.top_k)
    solver.fit(num_eval_per_epoch=args.num_eval_per_epoch)


if __name__ == '__main__':
    train()
