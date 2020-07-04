"""models and solvers"""
import os
import random
from pathlib import Path
from collections import OrderedDict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import numpy
from runx.logx import logx
from sklearn.metrics import f1_score

from models.DenseNetork.loss import kl_div_add_mse_loss, input_inverse_similarity, output_inverse_similarity, nearest_neighbors


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
                 weight_decay, n_epoch, seed, **kwargs):
        # construct param dict
        self.construct_param_dict = OrderedDict({
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "learning_rate": learning_rate,
            "n_epoch": n_epoch,
            "per_gpu_batch_size": per_gpu_batch_size,
            "weight_decay": weight_decay,
            "seed": seed,
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
        self.input_dir = input_dir
        self.output_dir = output_dir

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
        self.q = None
        self.anchor_idx = None
        # self.ground_min_dist_square = None  # if we use the ground values, we do not need to keep it here

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
                     per_gpu_batch_size, weight_decay, seed):
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
        train_dataloader = cls.get_train_dataloader(input_dir, batch_size)
        dev_dataloader = cls.get_dev_dataloader(input_dir, batch_size)

        return cls(input_dir, output_dir, model, device, per_gpu_batch_size, n_gpu, batch_size, learning_rate,
                   weight_decay, n_epoch, seed, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader)

    @classmethod
    def from_pretrained(cls, model_constructor, pretrained_system_name_or_path, resume_training=False,
                        input_dir=None, output_dir=None, **kwargs):
        # load checkpoints
        checkpoint = torch.load(pretrained_system_name_or_path)
        state_dict = checkpoint['state_dict']
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
            if input_dir is None or output_dir is None:
                raise AssertionError("Either input_dir and output_dir (for resuming) is None!")
            solver_args["input_dir"] = input_dir
            solver_args["output_dir"] = output_dir
            solver_args["train_dataloader"] = cls.get_train_dataloader(input_dir, solver_args["batch_size"])
            solver_args["dev_dataloader"] = cls.get_dev_dataloader(input_dir, solver_args["batch_size"])

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

    @staticmethod
    def precomputing(x):
        dist, sorted_dist, indices = nearest_neighbors(x)
        ground_min_dist_square = sorted_dist[:, 1]  # the 0-th column is the distance to oneself
        anchor_idx = indices[:, 1:]
        q = input_inverse_similarity(x,
                                     anchor_idx=anchor_idx,  # (n, n - 1)
                                     min_dist_square=ground_min_dist_square)
        return anchor_idx, q

    def train(self, steps_per_eval):
        # pre-compute q and anchor indexes
        self.anchor_idx, self.q = self.precomputing(self.train_dataloader.dataset.x)
        # TensorBoard
        for epoch_idx in range(self.n_epoch):
            self.__train_per_epoch(epoch_idx, steps_per_eval)

    # def validate(self, dataloader):
    #     preds, golds = self.__forward_batch_plus(dataloader)
    #     preds = preds.detach().cpu()
    #     golds = golds.detach().cpu()
    #     mean_loss = self.criterion(preds, golds)  # num_of_label should be 1 for it to work
    #     if self.n_gpu > 1:
    #         mean_loss = mean_loss.mean()  # mean() to average on multi-gpu.
    #     metrics_scores = self.get_scores(preds, golds)
    #     return mean_loss, metrics_scores

    def __train_per_epoch(self, epoch_idx, steps_per_eval):
        with tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch_idx}") as pbar:
            for batch_idx, batch in enumerate(self.train_dataloader):
                # assume that the whole input matrix fits the GPU memory
                global_step = epoch_idx * len(self.train_dataloader) + batch_idx
                loss = self.__training_step(batch)
                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                loss.backward()
                if self.scheduler:
                    logx.metric('train', {"tr_loss": loss.item(),
                                          "learning_rate": self.scheduler.get_last_lr()[0]}, global_step)
                else:
                    logx.metric('train', {"tr_loss": loss.item()}, global_step)
                pbar.set_postfix_str(f"tr_loss: {loss.item():.5f}")
                # update weights
                self.optimizer.step()
                # self.scheduler.step()  # Update learning rate schedule
                if (batch_idx + 1) % steps_per_eval == 0:
                    # validate and save checkpoints
                    # mean_loss, metrics_scores = self.validate(self.dev_dataloader)
                    # logx.metric('val', metrics_scores, global_step)
                    if self.n_gpu > 1:
                        save_dict = {"model_construct_dict": self.model.model_construct_dict,
                                     "model_state_dict": self.model.module.state_dict(),
                                     "solver_construct_params_dict": self.construct_param_dict,
                                     "optimizer": self.optimizer.state_dict()}
                    else:
                        save_dict = {"model_construct_dict": self.model.model_construct_dict,
                                     "model_state_dict": self.model.state_dict(),
                                     "solver_construct_params_dict": self.construct_param_dict,
                                     "optimizer": self.optimizer.state_dict()}
                    #  TODO: here we use training loss as metrics; switch to dev loss in the future
                    logx.save_model(save_dict,
                                    metric=loss.item(),
                                    epoch=global_step,
                                    higher_better=False)
                pbar.update(1)

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
        p = output_inverse_similarity(y=outputs, anchor_idx=self.anchor_idx)
        loss = self.criterion(p, self.q, lam=1)
        return loss

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
        return outputs

    @staticmethod
    def get_scores(soft_preds, golds):
        """
        It is going to be registered.
        :param soft_preds:
        :param golds:
        :return: a dictionary of all the measure
        """
        loss_measure = nn.CrossEntropyLoss()
        hard_preds = torch.argmax(F.softmax(soft_preds, dim=1), dim=1)
        accuracy = float((hard_preds == golds).sum().item()) / golds.shape[0]
        return {'CrossEntropy': loss_measure.forward(input=soft_preds, target=golds).item(),
                'Accuracy': accuracy,
                'F1_score': f1_score(y_true=golds, y_pred=hard_preds)}

    def __forward_batch_plus(self, dataloader, verbose=False):
        preds_list = list()
        golds_list = list()
        if verbose:
            with tqdm(total=len(dataloader), desc=f"Evaluating: ") as pbar:
                with torch.no_grad():
                    for batch_idx, batch in enumerate(dataloader):
                        logits, labels = self.__forwarding_step(batch)
                        preds_list.append(logits)
                        golds_list.append(labels)
                        pbar.update(1)
        else:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    logits, labels = self.__forwarding_step(batch)
                    preds_list.append(logits)
                    golds_list.append(labels)
        # collect the whole chunk
        preds = torch.cat(preds_list, dim=0).cpu()
        golds = torch.cat(golds_list, dim=0).cpu()
        return preds, golds

    @classmethod
    def get_train_dataloader(cls, input_dir, batch_size):
        encoded_data_path = input_dir / 'train.pth.tar'
        dataset = cls.__set_dataset(encoded_data_path)
        train_dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
        return train_dataloader

    @classmethod
    def get_dev_dataloader(cls, input_dir, batch_size):
        encoded_data_path = input_dir / 'dev.pth.tar'
        dataset = cls.__set_dataset(encoded_data_path)
        dev_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True)
        return dev_dataloader

    @classmethod
    def get_test_dataloader(cls, input_dir, batch_size):
        encoded_data_path = input_dir / 'test.pth.tar'
        dataset = cls.__set_dataset(encoded_data_path)
        test_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True)
        return test_dataloader

    @classmethod
    def __set_dataset(cls, encoded_data_path):
        return torch.load(encoded_data_path)

    def infer(self, data_path):
        data_path = Path(data_path)
        dataset = self.__set_dataset(data_path)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size)
        preds, golds = self.__forward_batch_plus(dataloader, verbose=True)
        return preds, golds

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
        # Prepare optimizer and schedule (linear warmup and decay)
        # no_decay = ['bias', 'LayerNorm.weight']
        optimizer = torch.optim.Adam(params=named_parameters, lr=learning_rate, weight_decay=weight_decay)

        ## get a linear scheduler
        # num_steps_epoch = len(train_dataloader)
        # ReduceLROnPlateau(self.optimizer, 'min')
        # num_train_optimization_steps = int(num_steps_epoch * n_epoch) + 1
        # warmup_steps = 100
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
        #                                             num_training_steps=num_train_optimization_steps)
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
        args = parser.parse_args()

        return args