import torch
import wandb
from loguru import logger
from tqdm import tqdm

from torch import nn, tensor, Tensor, FloatTensor
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.optim import Optimizer, lr_scheduler
import sys, os, gc, argparse
from typing import Any, Tuple, List, Callable
from argparse import ArgumentParser, Namespace

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
sys.path.append(ROOT_DIR)
from models.full_model import GraphTransModel

from deprecated_trainers import register_trainer
from deprecated_trainers.base_trainer import BaseTrainer


@register_trainer("flag")
class FlagTrainer(BaseTrainer):
    """
    Flag trainer, an adversarial trainer on graphs (not to be instantiated).

    We perturb the embeddings in the forward pass (random, uniform) of the model.
    A forward and backward pass are run for M steps per epoch, where M is a hyperparameter.
    In each step, the perturbation is updated by step_size in the direction of its gradients after loss
    computation (magnitude of gradient is ignored), while model gradients accumulate.
    After M steps, the model parameters are updated by the optimizer, then gradients reset to 0.
    """
    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        """
        Add arguments to the argument parser: the step size and the hyperparameter "m".

        Args:
            parser: The argument parser to enhance.
        """
        # fmt: off
        parser.add_argument('--step-size', type=float, default=8e-3)
        parser.add_argument('-m', type=int, default=3)
        # fmt: on

    @staticmethod
    def train(
        model: GraphTransModel,
        device: torch.device,
        loader: DataLoader,
        optimizer: Optimizer,
        args: Namespace,
        calc_loss: Callable
    ):
        """
        The training loop. Each epoch involves "m" forward passes, with gradients for the
        model accumulating, while the perturbation is updated each step by step_size, only in sign of gradients.

        Args:
            model: The model to train.
            device: The device to run the training on.
            loader: The data loader to load training data from.
            optimizer: The optimizer to use for training.
            args: The command line arguments, which should include "step_size" and "m".
            calc_loss: function that takes in the model predictions, the batch, and "m,
                and calculates the loss to optimize.
        """
        model.train()

        loss_accum = 0
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)

            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                pass
            else:
                optimizer.zero_grad()

                perturb = FloatTensor(batch.x.shape[0], args.gnn_emb_dim).uniform_(-args.step_size, args.step_size).to(device)
                perturb.requires_grad_()

                pred_list = model(batch, perturb)

                loss: Tensor = calc_loss(pred_list, batch, args.m)

                for _ in range(args.m - 1):
                    loss.backward()
                    perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.grad.detach()) # type: ignore
                    perturb.data = perturb_data.data
                    perturb.grad[:] = 0 # type: ignore

                    pred_list = model(batch, perturb)

                    loss = calc_loss(pred_list, batch, args.m)

                loss.backward()
                optimizer.step()

                detached_loss = loss.item()
                loss_accum += detached_loss
                wandb.log({"train/iter-loss": detached_loss})

        return loss_accum / (step + 1)

    @staticmethod
    def name(args: Namespace) -> str:
        """
        Return the name of the trainer, used for logging and checkpoint naming.
        """
        return "flag"
