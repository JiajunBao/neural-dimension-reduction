from sklearn.manifold import TSNE
from argparse import ArgumentParser
from pathlib import Path
import joblib
import torch
from src.data.utils import import_raw_data
from src.models.DenseNetwork.loss import nearest_neighbors, input_inverse_similarity, kl_div_add_mse_loss
from src.models.DenseNetwork.models import Solver


def main():
    parser = ArgumentParser(description='Arguments for dataset processing')
    parser.add_argument('--input_path', type=Path, required=True, default=None,
                        help='the input path to the input data')
    parser.add_argument('--output_dir', type=Path, required=True, default=None,
                        help='the output directory to save sampled data')
    parser.add_argument('--n_iter', type=int, required=True, default=2000,
                        help='the number of rows to sample')
    parser.add_argument('--dim_out', type=int, default=20,
                        help='n components to remain')
    parser.add_argument('--perplexity', type=int, default=40,
                        help='perplexity')
    parser.add_argument('--seed', type=int, default=42,
                        help='the random seed of the whole process')
    args = parser.parse_args()

    input_embeddings = import_raw_data(args.input_path)

    model = TSNE(n_components=args.dim_out, n_iter=args.n_iter, perplexity=40, verbose=2, random_state=args.seed)
    output_embeddings = model.fit_transform(input_embeddings)

    # evaluation
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ground_min_dist_square, anchor_idx, topk_dists = nearest_neighbors(x=input_embeddings, top_k=20, device=device)
    q = input_inverse_similarity(input_embeddings, anchor_idx, ground_min_dist_square)
    scores, p = Solver.static_get_scores(q, output_embeddings, anchor_idx,
                                         device, kl_div_add_mse_loss, 20, ground_min_dist_square)
    for k, v in scores.items():
        print(f'{k} = {v}')
    args.output_dir.mkdir(exist_ok=True, parents=True)
    torch.save({'output_embeddings': output_embeddings,
                'input_embeddings': input_embeddings,
                'p': p, 'q': q, 'scores': scores,
                'tsne-args': args
                }, args.output_dir / 'embeddings.pth.tar')
    joblib.dump(model, 'tsne-model.pth.tar')


if __name__ == '__main__':
    main()
