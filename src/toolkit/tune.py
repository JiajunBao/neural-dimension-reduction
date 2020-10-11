import torch
from src.toolkit import network
from torch.utils.data import DataLoader
import torch.utils.data
from src.toolkit import learn
from src.datasets import SIFT
import argparse
import pathlib
import optuna

parser = argparse.ArgumentParser(description='training args')
parser.add_argument('--data_path', default=pathlib.Path('/home/jiajunb/neural-dimension-reduction/data/sift/siftsmall'))
parser.add_argument('--model_type', default='ReconstructSiameseNet')
args = parser.parse_args()
print('args: \n', args)
train_set, base_set, eval_set = SIFT.get_datasets(args.data_path, args.model_type)
print('dataset completed')


def objective(trial):
    if args.model_type == 'SiameseNet':
        model = network.SiameseNet(network.EmbeddingNet())
    elif args.model_type == 'ReconstructSiameseNet':
        model = network.ReconstructSiameseNet(network.EmbeddingNet())
    else:
        raise NotImplementedError
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = 32768
    num_epoches = 30

    verbose = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
    log_epoch = 1
    # optimizer
    no_decay = ['bias', 'LayerNorm.weight', 'BatchNorm1d.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=learning_rate)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, pin_memory=True)
    base_loader = DataLoader(base_set, shuffle=False, batch_size=batch_size, pin_memory=True)
    eval_loader = DataLoader(eval_set, shuffle=False, batch_size=batch_size, pin_memory=True)

    model = model.to(device)

    criterion = None
    if isinstance(model, network.SiameseNet):
        margin = trial.suggest_float("margin", 1e-1, 1e2, log=True)
        criterion = learn.PowerMarginLoss(margin, 'mean')

    best_recall_query_set, its_recall_on_base_set, best_model, model = learn.train_with_eval(train_loader, base_loader,
                                                                                             eval_loader, criterion,
                                                                                             model, optimizer,
                                                                                             num_epoches, log_epoch,
                                                                                             verbose, device, trial)
    return best_recall_query_set


def main():
    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=20, timeout=None)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    main()
