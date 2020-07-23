import argparse
import os
import random
from collections import OrderedDict
from pathlib import Path
import numpy
import torch
from runx.logx import logx
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.utils.loss import StochasticNeighborLoss, nearest_neighbors
from src.models.utils.data import InsaneDataSet


class InsaneTrainer(object):

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
                        hparams={"solver_construct_dict": self.construct_param_dict},
                        eager_flush=True)
        # arguments
        self.record_training_loss_per_epoch = kwargs.pop("record_training_loss_per_epoch", False)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.top_k = top_k
        # training utilities
        self.model = model
        self.train_decoder = None
        self.dev_decoder = None

        # data utilities
        self.train_dataloader = kwargs.pop("train_dataloader", None)
        self.dev_dataloader = kwargs.pop("dev_dataloader", None)
        if self.train_dataloader is not None:
            self.train_decoder = StochasticNeighborLoss(self.train_dataloader.dataset.anchor_idx,
                                                        self.train_dataloader.dataset.input_similarity)
        if self.dev_decoder is not None:
            self.dev_decoder = StochasticNeighborLoss(self.dev_dataloader.anchor_idx,
                                                      self.dev_dataloader.input_similarity)

        self.batch_size = batch_size

        self.n_epoch = n_epoch
        self.seed = seed
        # device
        self.device = device
        self.n_gpu = n_gpu
        logx.msg(f'Number of GPU: {self.n_gpu}.')

        # optimizer and scheduler
        if self.train_dataloader:
            self.optimizer, self.scheduler = self.get_optimizer(named_parameters=self.model.named_parameters(),
                                                                learning_rate=learning_rate,
                                                                weight_decay=weight_decay)
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
        raise NotImplementedError

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
        metrics_scores, output_similarity = self.get_scores(self.dev_decoder, outputs, self.dev_dataloader.anchor_idx)
        return outputs, metrics_scores, output_similarity

    def __train_per_epoch(self, epoch_idx, steps_per_eval):
        # with tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch_idx}") as pbar:
        for batch_idx, batch in enumerate(self.train_dataloader):
            # assume that the whole input matrix fits the GPU memory
            global_step = epoch_idx * len(self.train_dataloader) + batch_idx
            training_set_loss, training_set_outputs, training_set_output_similarity = self.__training_step(batch)
            if batch_idx + 1 == len(self.train_dataloader):
                # validate and save checkpoints
                developing_set_outputs, developing_set_metrics_scores, developing_set_output_similarity = \
                    self.validate(self.dev_dataloader)
                # TODO: this part can be optimized to batchwise computing
                if self.record_training_loss_per_epoch:
                    training_set_metrics_scores, _ = \
                        self.get_scores(self.train_decoder,
                                        training_set_outputs,
                                        self.train_dataloader.dataset.anchor_idx)
                else:
                    training_set_metrics_scores = dict()
                training_set_metrics_scores['loss'] = training_set_loss.item()
                if self.scheduler:
                    training_set_metrics_scores['learning_rate'] = self.scheduler.get_last_lr()[0]
                logx.metric('train', training_set_metrics_scores, global_step)
                logx.metric('val', developing_set_metrics_scores, global_step)
                if self.n_gpu > 1:
                    save_dict = {"model_construct_dict": self.model.module.config,
                                 "model_state_dict": self.model.module.state_dict(),
                                 "solver_construct_params_dict": self.construct_param_dict,
                                 "optimizer": self.optimizer.state_dict(),
                                 "train_scores": training_set_metrics_scores,
                                 "train_input_embedding": self.train_dataloader.dataset.x,
                                 "train_input_similarity": self.train_dataloader.dataset.input_similarity,
                                 "train_output_embedding": training_set_outputs,
                                 "train_output_similarity": training_set_output_similarity,
                                 "dev_scores": developing_set_metrics_scores,
                                 "dev_input_embeddings": self.dev_dataloader.dataset.x,
                                 "dev_input_similarity": self.dev_dataloader.dataset.input_similarity,
                                 "dev_output_embedding": developing_set_outputs,
                                 "dev_output_similarity": developing_set_output_similarity,
                                 }
                else:
                    save_dict = {"model_construct_dict": self.model.config,
                                 "model_state_dict": self.model.state_dict(),
                                 "solver_construct_params_dict": self.construct_param_dict,
                                 "optimizer": self.optimizer.state_dict(),
                                 "train_scores": training_set_metrics_scores,
                                 "train_input_embedding": self.train_dataloader.dataset.x,
                                 "train_input_similarity": self.train_dataloader.dataset.input_similarity,
                                 "train_output_embedding": training_set_outputs,
                                 "train_output_similarity": training_set_output_similarity,
                                 "dev_scores": developing_set_metrics_scores,
                                 "dev_input_embeddings": self.dev_dataloader.dataset.x,
                                 "dev_input_similarity": self.dev_dataloader.dataset.input_similarity,
                                 "dev_output_embedding": developing_set_outputs,
                                 "dev_output_similarity": developing_set_output_similarity,
                                 }
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

        loss, output_similarity = self.train_decoder.forward(outputs)
        if self.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        loss.backward()
        # pbar.set_postfix_str(f"tr_loss: {loss.item():.5f}")
        # update weights
        self.optimizer.step()
        # self.scheduler.step()  # Update learning rate schedule
        return loss.cpu().detach(), outputs.cpu().detach(), output_similarity.cpu().detach()

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

    def get_scores(self, decoder, output_embedding, anchor_idx):
        scores = dict()
        loss, output_similarity = decoder.forward(output_embedding.to(self.device))
        loss, output_similarity = loss.cpu(), output_similarity.cpu()
        scores['loss'] = loss.item()
        _, topk_neighbors, _ = nearest_neighbors(x=output_embedding, top_k=anchor_idx.shape[1], device=self.device)
        ground_nn = anchor_idx[:, 0].unsqueeze(dim=1)
        for r in [1, 5, 10, 20]:
            top_predictions = topk_neighbors[:, :r]  # (n, r)
            scores[f'Recall@{r}'] = \
                torch.sum(top_predictions == ground_nn, dtype=torch.float).item() / ground_nn.shape[0]
        return scores, output_similarity

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
            if dataset.top_k >= top_k and dataset.top_k >= 20:
                return DataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True)
            else:
                print(f'inconsistent top_k: {dataset.top_k} vs {top_k}')
        dataset = InsaneDataSet.from_df(input_dir / f'{split_name}.csv', max(top_k, 20))
        torch.save(dataset, encoded_data_path)
        print(f'construct dataset from dataframe and save dataset at ({encoded_data_path})')
        return DataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True)

    @staticmethod
    def get_optimizer(named_parameters, learning_rate, weight_decay):
        """
        get the optimizer and the learning rate scheduler
        :param named_parameters:
        :param learning_rate:
        :param weight_decay:
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

        optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                      lr=learning_rate, weight_decay=weight_decay)
        return optimizer, None

    @staticmethod
    def get_solver_arguments():
        parser = argparse.ArgumentParser(description='Arguments for Eigenmetric Regression')
        # parameters
        parser.add_argument('--input_dir', type=Path, default=None,
                            help='the input directory to the input data')
        parser.add_argument('--output_dir', type=Path, default=None,
                            help='the output directory for saving the regressor')
        parser.add_argument('--config_name', type=str,
                            help='name of the model configuration file')
        parser.add_argument('--top_k', type=int, default=20,
                            help='the top-k nearest neighbors that are considered.')
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
