"""models and solvers"""
import argparse
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from runx.logx import logx
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.DenseNetwork.loss import kl_div_add_mse_loss, input_inverse_similarity, output_inverse_similarity, \
    nearest_neighbors

# TOP_K = 20


class VecDataSet(Dataset):
    def __init__(self, x, top_k):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # device = 'cpu'
        self.anchor_idx, self.q, self.ground_min_dist_square, self.topk_dists = \
            self.precomputing(x, top_k=top_k, device=device)
        self.top_k = top_k
        self.x = x.cpu()

    @classmethod
    def from_df(cls, path_to_dataframe, top_k):
        x = torch.from_numpy(pd.read_csv(path_to_dataframe).to_numpy()).to(torch.float32)
        return cls(x, top_k)

    @classmethod
    def from_dataset(cls, path_to_tensor):
        return torch.load(path_to_tensor)

    @staticmethod
    def precomputing(x, top_k, device):
        """
        compute ground true nearest neighbors
        :param x:
        :param top_k: top-k neighbors that are considered
        :param device: device used during computation
        :return: anchor_idx: each point has m points as anchors (in the case, we pick m near neighbors of x as anchors)
                 q: input_similarity
        """
        ground_min_dist_square, anchor_idx, topk_dists = nearest_neighbors(x, top_k, device)

        q = input_inverse_similarity(x.to(device),
                                     anchor_idx=anchor_idx,  # (n, n - 1)
                                     min_dist_square=ground_min_dist_square.to(device)).cpu()
        return anchor_idx, q, ground_min_dist_square, topk_dists

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


class Net(nn.Module):
    def __init__(self, hidden_layers, model_construct_dict):
        super(Net, self).__init__()
        self.hidden_layers = hidden_layers
        self.model_construct_dict = model_construct_dict

    @classmethod
    def from_scratch(cls, dim_in, hidden_dims_list, dim_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        in_dims = [dim_in] + hidden_dims_list
        out_dims = hidden_dims_list + [dim_out]
        hidden_layers = nn.ModuleList(
            [nn.Linear(in_features=i, out_features=o) for i, o in zip(in_dims, out_dims)])
        model_construct_dict = {
            'dim_in': dim_in,
            'hidden_dims_list': hidden_dims_list,
            'dim_out': dim_out,
        }
        return cls(hidden_layers, model_construct_dict)

    @classmethod
    def from_pretrained(cls, path_to_checkpoints):
        checkpoints = torch.load(path_to_checkpoints)
        model = cls(**checkpoints['model_construct_dict'])
        model.load_state_dict(checkpoints['model_state_dict'])
        model.eval()
        return model

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        for layer in self.hidden_layers:
            x = F.relu(layer.forward(x))
        return x


class Solver(object):

    def __init__(self, input_dir, output_dir, model, device, per_gpu_batch_size, n_gpu, batch_size, learning_rate,
                 weight_decay, n_epoch, seed, top_k, **kwargs):
        # construct param dict
        self.construct_param_dict = OrderedDict({
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "learning_rate": learning_rate,
            "n_epoch": n_epoch,
            "per_gpu_batch_size": per_gpu_batch_size,
            "weight_decay": weight_decay,
            "seed": seed,
            "top_k": top_k,
        })

        # build log
        logx.initialize(logdir=output_dir,
                        coolname=True,
                        tensorboard=True,
                        no_timestamp=False,
                        hparams={"solver_construct_dict": self.construct_param_dict,
                                 "model_construct_dict": model.model_construct_dict},
                        eager_flush=True)
        # arguments
        self.record_training_loss_per_epoch = False
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.top_k = top_k
        # training utilities
        self.model = model

        # data utilities
        self.train_dataloader = kwargs.pop("train_dataloader", None)
        self.dev_dataloader = kwargs.pop("dev_dataloader", None)
        self.batch_size = batch_size

        self.n_epoch = n_epoch
        self.seed = seed
        # device
        self.device = device
        self.n_gpu = n_gpu
        logx.msg(f'Number of GPU: {self.n_gpu}.')

        self.criterion = kl_div_add_mse_loss

        # optimizer and scheduler
        if self.train_dataloader:
            self.optimizer, self.scheduler = self.get_optimizer(named_parameters=self.model.named_parameters(),
                                                                learning_rate=learning_rate,
                                                                weight_decay=weight_decay,
                                                                train_dataloader=self.train_dataloader,
                                                                n_epoch=n_epoch)
        # set up random seeds and model location
        self.setup()

    @classmethod
    def from_scratch(cls, model, input_dir, output_dir, learning_rate, n_epoch,
                     per_gpu_batch_size, weight_decay, seed, top_k):
        # check the validity of the directory
        if os.path.exists(output_dir) and os.listdir(output_dir):
            raise ValueError(f"Output directory ({output_dir}) already exists "
                             "and is not empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        # data utilities
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        n_gpu = torch.cuda.device_count()
        batch_size = per_gpu_batch_size * max(1, n_gpu)

        # dataloader
        train_dataloader = cls.get_train_dataloader(input_dir, batch_size, top_k)
        dev_dataloader = cls.get_dev_dataloader(input_dir, batch_size, top_k)

        return cls(input_dir, output_dir, model, device, per_gpu_batch_size, n_gpu, batch_size, learning_rate,
                   weight_decay, n_epoch, seed, top_k, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader)

    @classmethod
    def from_pretrained(cls, model_constructor, pretrained_system_name_or_path, resume_training=False,
                        input_dir=None, output_dir=None, top_k=None, **kwargs):
        # load checkpoints
        checkpoint = torch.load(pretrained_system_name_or_path)
        meta = {k: v for k, v in checkpoint.items() if k != 'state_dict'}

        # load model
        model = model_constructor.from_pretrained(pretrained_system_name_or_path)  #

        # load arguments
        solver_args = meta["solver_construct_params_dict"]
        solver_args["model"] = model
        # update some parameters
        solver_args["device"] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        solver_args["n_gpu"] = torch.cuda.device_count()
        old_batch_size = solver_args["per_gpu_batch_size"] * max(1, solver_args["n_gpu"])
        solver_args["batch_size"] = kwargs.pop("batch_size", old_batch_size)
        # load dataset
        if resume_training:
            if input_dir is None or output_dir is None or top_k is None:
                raise AssertionError("Either input_dir and output_dir (for resuming) is None!")
            solver_args["input_dir"] = input_dir
            solver_args["output_dir"] = output_dir
            solver_args["train_dataloader"] = cls.get_train_dataloader(input_dir, solver_args["batch_size"], top_k)
            solver_args["dev_dataloader"] = cls.get_dev_dataloader(input_dir, solver_args["batch_size"], top_k)
            solver_args["top_k"] = top_k
        return cls(**solver_args)

    def fit(self, num_eval_per_epoch=5):
        steps_per_eval = len(self.train_dataloader) // num_eval_per_epoch
        steps_per_eval = steps_per_eval if steps_per_eval > 0 else 1
        self.train(steps_per_eval)
        # test_dataloader = self.get_test_dataloader(self.input_dir, self.batch_size)
        # mean_loss, metrics_scores = self.validate(test_dataloader)
        # logx.msg("Scores on test set: ")
        # logx.msg(str(metrics_scores))

    def setup(self):
        # put onto cuda
        self.model = self.model.to(self.device)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        # fix random seed
        self.fix_random_seed()

    def fix_random_seed(self):
        # Set seed
        random.seed(self.seed)
        numpy.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

    def train(self, steps_per_eval):
        # TensorBoard
        for epoch_idx in tqdm(range(self.n_epoch)):
            self.__train_per_epoch(epoch_idx, steps_per_eval)

    def validate(self, dataloader):
        outputs = self.__forward_batch_plus(dataloader)
        metrics_scores, p = self.get_scores(q=dataloader.dataset.q,
                                            output_embeddings=outputs,
                                            anchor_idx=dataloader.dataset.anchor_idx)
        return outputs, metrics_scores, p

    def __train_per_epoch(self, epoch_idx, steps_per_eval):
        # with tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch_idx}") as pbar:
        for batch_idx, batch in enumerate(self.train_dataloader):
            # assume that the whole input matrix fits the GPU memory
            global_step = epoch_idx * len(self.train_dataloader) + batch_idx
            training_set_loss, training_set_outputs, training_set_p = self.__training_step(batch)
            if batch_idx + 1 == len(self.train_dataloader):
                # validate and save checkpoints
                developing_set_outputs, developing_set_metrics_scores, developing_set_p = \
                    self.validate(self.dev_dataloader)
                # TODO: this part can be optimized to batchwise computing
                if self.record_training_loss_per_epoch:
                    training_set_metrics_scores, training_set_p = \
                        self.get_scores(q=self.train_dataloader.dataset.q,
                                        output_embeddings=training_set_outputs,
                                        anchor_idx=self.train_dataloader.dataset.anchor_idx)
                    training_set_metrics_scores['train_p'] = training_set_p.cpu(),
                else:
                    training_set_metrics_scores = dict()
                training_set_metrics_scores['train_loss'] = training_set_loss.item()
                if self.scheduler:
                    training_set_metrics_scores['learning_rate'] = self.scheduler.get_last_lr()[0]
                logx.metric('train', training_set_metrics_scores, global_step)
                logx.metric('val', developing_set_metrics_scores, global_step)
                if self.n_gpu > 1:
                    save_dict = {"model_construct_dict": self.model.model_construct_dict,
                                 "model_state_dict": self.model.module.state_dict(),
                                 "solver_construct_params_dict": self.construct_param_dict,
                                 "optimizer": self.optimizer.state_dict(),
                                 "train_metrics_scores": training_set_metrics_scores,
                                 "train_output_embeddings": training_set_outputs.cpu(),
                                 "train_q": self.train_dataloader.dataset.q.cpu(),
                                 "dev_metrics_scores": developing_set_metrics_scores,
                                 "dev_output_embeddings": developing_set_outputs.cpu(),
                                 "dev_q": self.dev_dataloader.dataset.q.cpu(),
                                 "dev_p": developing_set_p.cpu()}
                else:
                    save_dict = {"model_construct_dict": self.model.model_construct_dict,
                                 "model_state_dict": self.model.state_dict(),
                                 "solver_construct_params_dict": self.construct_param_dict,
                                 "optimizer": self.optimizer.state_dict(),
                                 "train_metrics_scores": training_set_metrics_scores,
                                 "train_output_embeddings": training_set_outputs.cpu(),
                                 "train_q": self.train_dataloader.dataset.q.cpu(),
                                 "dev_metrics_scores": developing_set_metrics_scores,
                                 "dev_output_embeddings": developing_set_outputs.cpu(),
                                 "dev_q": self.dev_dataloader.dataset.q.cpu(),
                                 "dev_p": developing_set_p.cpu()}
                logx.save_model(save_dict,
                                metric=developing_set_metrics_scores['Recall@1'],
                                epoch=global_step,
                                higher_better=True)
                # pbar.update(1)

    def batch_to_device(self, batch):
        return batch.to(self.device)

    def __training_step(self, batch):
        """
        a single forwarding step for training
        :param self: a solver
        :param batch: a batch of input for model
        :return: training loss for this batch
        """
        self.model.zero_grad()  # reset gradient
        self.model.train()
        outputs = self.__forwarding_step(batch)
        p = output_inverse_similarity(y=outputs.to(self.device),
                                      anchor_idx=self.train_dataloader.dataset.anchor_idx.to(self.device)).cpu()
        loss = self.criterion(p.to(self.device),
                              self.train_dataloader.dataset.q.to(self.device), lam=1)

        if self.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        loss.backward()
        # pbar.set_postfix_str(f"tr_loss: {loss.item():.5f}")
        # update weights
        self.optimizer.step()
        # self.scheduler.step()  # Update learning rate schedule

        return loss.cpu().detach(), outputs.cpu().detach(), p.cpu().detach()

    def __forwarding_step(self, batch):
        """
        a single forwarding pass
        e.g.
        meta_features, input_ids, input_mask, segment_ids, labels = batch
        batch_input = {'meta_features': meta_features.to(self.device),
                       'input_ids': input_ids.to(self.device),
                       'attention_mask': input_mask.to(self.device),
                       'token_type_ids': segment_ids.to(self.device),
                       'labels': labels}
        logits = self.model(**batch_input)
        return logits, labels
        :param self: a solver
        :param batch: a batch of input for model
        :return: logits and ground true label for this batch
        """
        batch_inputs = self.batch_to_device(batch)
        outputs = self.model(batch_inputs)
        return outputs.cpu()

    def get_scores(self, q, output_embeddings, anchor_idx):
        """

        :param q: torch.tensor (n, ) input similarity
        :param output_embeddings: torch.tensor (n, d2) output embeddings from the network
        :param anchor_idx: (n, m) each point has m points as anchors
        :return:
        """
        scores = dict()
        # calculate loss
        p = output_inverse_similarity(y=output_embeddings.to(self.device),
                                      anchor_idx=anchor_idx).cpu()
        scores['loss'] = self.criterion(p.to(self.device), q.to(self.device), lam=1).cpu().detach().item()
        # recalls
        _, topk_neighbors, _ = nearest_neighbors(x=output_embeddings, top_k=self.top_k, device=self.device)
        ground_nn = anchor_idx[:, 0].unsqueeze(dim=1)
        for r in [1, 5, 10, 20]:
            top_predictions = topk_neighbors[:, :r]  # (n, r)
            print(top_predictions.device, ground_nn.device)
            scores[f'Recall@{r}'] = \
                torch.sum(top_predictions == ground_nn, dtype=torch.float).item() / ground_nn.shape[0]
        return scores, p

    def __forward_batch_plus(self, dataloader, verbose=False):
        preds_list = list()
        if verbose:
            with tqdm(total=len(dataloader), desc=f"Evaluating: ") as pbar:
                with torch.no_grad():
                    for batch_idx, batch in enumerate(dataloader):
                        outputs = self.__forwarding_step(batch)
                        preds_list.append(outputs)
                        pbar.update(1)
        else:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    outputs = self.__forwarding_step(batch)
                    preds_list.append(outputs)
        # collect the whole chunk
        reduced_embeddings = torch.cat(preds_list, dim=0)
        return reduced_embeddings

    @classmethod
    def get_train_dataloader(cls, input_dir, batch_size, top_k):
        return cls.__set_dataset(input_dir, 'train', batch_size, top_k)

    @classmethod
    def get_dev_dataloader(cls, input_dir, batch_size, top_k):
        return cls.__set_dataset(input_dir, 'dev', batch_size, top_k)

    @classmethod
    def get_test_dataloader(cls, input_dir, batch_size, top_k):
        return cls.__set_dataset(input_dir, 'test', batch_size, top_k)

    @classmethod
    def __set_dataset(cls, input_dir, split_name, batch_size, top_k):
        encoded_data_path = input_dir / f'{split_name}.pth.tar'
        if encoded_data_path.is_file():
            dataset = torch.load(encoded_data_path)
            print(f'load dataset from {encoded_data_path}')
            if dataset.top_k == top_k:
                return DataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True)
            else:
                print(f'inconsistent top_k: {dataset.top_k} vs {top_k}')
        dataset = VecDataSet.from_df(input_dir / f'{split_name}.csv', top_k)
        torch.save(dataset, encoded_data_path)
        print(f'construct dataset from dataframe and save dataset at ({encoded_data_path})')
        return DataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True)

    # def infer(self, data_path):
    #     data_path = Path(data_path)
    #     dataset = cls.__set_dataset(data_, 'test', batch_size)
    #     dataloader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size)
    #     preds, golds = self.__forward_batch_plus(dataloader, verbose=True)
    #     return preds, golds

    @staticmethod
    def get_optimizer(named_parameters, learning_rate, weight_decay, train_dataloader, n_epoch):
        """
        get the optimizer and the learning rate scheduler
        :param named_parameters:
        :param learning_rate:
        :param weight_decay:
        :param train_dataloader:
        :param n_epoch:
        :return:
        """
        # Prepare optimizer and schedule (linear warm-up and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in named_parameters if not any(
                nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': weight_decay},
            {'params': [p for n, p in named_parameters if any(
                nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=learning_rate, weight_decay=weight_decay)
        '''
        # get a linear scheduler
        num_steps_epoch = len(train_dataloader)
        ReduceLROnPlateau(self.optimizer, 'min')
        num_train_optimization_steps = int(num_steps_epoch * n_epoch) + 1
        warmup_steps = 100
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        '''
        return optimizer, None

    @staticmethod
    def get_solver_arguments():
        parser = argparse.ArgumentParser(description='Arguments for Eigenmetric Regression')
        # model parameters

        # solver parameters
        parser.add_argument('--input_dir', type=Path, default=None,
                            help='the input directory to the input data')
        parser.add_argument('--output_dir', type=Path, default=None,
                            help='the output directory for saving the regressor')
        parser.add_argument('--learning_rate', type=float, default=1e-5,
                            help='learning rate for training')
        parser.add_argument('--n_epoch', type=int, default=3,
                            help='the number of epochs for training')
        parser.add_argument('--num_eval_per_epoch', type=int, default=5,
                            help='number of evaluation per epoch')
        parser.add_argument('--per_gpu_batch_size', type=int, default=32,
                            help='the batch size per gpu')
        parser.add_argument('--weight_decay', type=float, default=1e-6,
                            help='weight_decay for the optimizer (l2 regularization)')
        parser.add_argument('--seed', type=int, default=42,
                            help='the random seed of the whole process')
        parser.add_argument('--top_k', type=int, default=20,
                            help='the top-k nearest neighbors that are considered.')
        parser.add_argument('--hidden_dims_list', type=str,
                            help='list of hidden dimensions')
        args = parser.parse_args()

        return args
