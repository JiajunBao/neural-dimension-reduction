import torch
from src.toolkit import network
from torch.utils.data import DataLoader
import torch.utils.data
from src.toolkit import learn
from src.datasets import SIFT

import optuna


def get_datasets():
    train_set, base_set, eval_set = SIFT.get_datasets()
    return train_set, base_set, eval_set


def objective(trial):
    model = network.SiameseNet(network.EmbeddingNet())
    batch_size = 32768
    num_epoches = 5

    verbose = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_decay = 1e-6
    log_epoch = 1

    train_set, base_set, eval_set = get_datasets()

    # optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=lr)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, pin_memory=True)
    base_loader = DataLoader(base_set, shuffle=True, batch_size=batch_size, pin_memory=True)
    eval_loader = DataLoader(eval_set, shuffle=False, batch_size=batch_size, pin_memory=True)

    margin = trial.suggest_int("margin", 1, 5, log=True)
    criterion = learn.PowerMarginLoss(margin, reduction='mean')
    model = model.to(device)
    best_recall_query_set, its_recall_on_base_set, best_model, model = learn.train_with_eval(train_loader, base_loader,
                                                                                             eval_loader, criterion,
                                                                                             model, optimizer,
                                                                                             num_epoches, log_epoch,
                                                                                             verbose, device, trial)
    return best_recall_query_set, its_recall_on_base_set, best_model, model


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=None)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
