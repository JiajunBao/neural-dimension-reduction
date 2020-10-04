import torch
from src.toolkit import network
from torch.utils.data import DataLoader
import torch.utils.data
from src.toolkit import learn
from src.datasets import SIFT


train_set, base_set, eval_set = SIFT.get_datasets()


def objective():
    model = network.SiameseNet(network.EmbeddingNet())
    learning_rate = 0.009359
    batch_size = 32768
    num_epoches = 30

    verbose = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    weight_decay = 5.969e-7
    log_epoch = 1
    # optimizer
    no_decay = ['bias', 'LayerNorm.weight']
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

    margin = 10.8816
    criterion = learn.PowerMarginLoss(margin, 'mean')

    best_recall_query_set, its_recall_on_base_set, best_model, model = learn.train_with_eval(train_loader, base_loader,
                                                                                             eval_loader, criterion,
                                                                                             model, optimizer,
                                                                                             num_epoches, log_epoch,
                                                                                             verbose, device, None)
    return best_recall_query_set, model


def main():
    best_recall_query_set, model = objective()
    print(f'best recall on query set {best_recall_query_set}')
    torch.save(model, f'final-model.pt.{best_recall_query_set: .2f}')


if __name__ == '__main__':
    main()
