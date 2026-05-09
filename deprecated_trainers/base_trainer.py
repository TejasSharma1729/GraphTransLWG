import torch
import wandb
from loguru import logger
from tqdm import tqdm

from torch import nn, tensor, Tensor
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.optim import Optimizer, lr_scheduler
import sys, os, gc, argparse
from typing import Any, Tuple, List, Callable
from argparse import ArgumentParser, Namespace

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
sys.path.append(ROOT_DIR)
from models.full_model import GraphTransModel


class BaseTrainer:
    """
    Static class for training models (not to be instantiated).

    Call BaseTrainer.train(...) to train a model, and 
    override the train method in subclasses to implement different training algorithms.
    """
    @staticmethod
    def transform(args: Namespace) -> Any:
        """
        Transform args into any object or tuple thereof, as per necessity.

        This method is to be overridden by subclasses (the base class does nothing and returns None).
        """
        return None

    @staticmethod
    def add_args(parser: ArgumentParser) -> Any:
        """
        Enhance parsers; no need to define all arguments in the main file.
        
        This method is to be overridden by subclasses (the base class does nothing).
        """
        pass

    @staticmethod
    def train(
        model: GraphTransModel, 
        device: torch.device, 
        loader: DataLoader, 
        optimizer: Optimizer, 
        args: Namespace, 
        calc_loss: Callable, 
        scheduler: lr_scheduler.LRScheduler | None = None
    ):
        """
        The main train method. Just runs batch-training, zeros the model parameters' gradients
        before each epoch, does forward pass, computes loss, does the backward pass, and
        steps the optimizer (and scheduler, if given) to update model parameters, each epoch.

        Args:
            model: The model to train
            device: The device to train on
            loader: The DataLoader for the training data
            optimizer: The optimizer to use for training
            args: The argparse.Namespace object containing any additional arguments
            calc_loss: A function that takes in the model's predictions and the batch, and calculates the loss
            scheduler: An optional learning rate scheduler to step after each optimizer step
        """
        model.train()

        loss_accum = 0
        t = tqdm(loader, desc="Train")
        for step, batch in enumerate(t):
            batch = batch.to(device)

            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                pass
            else:
                optimizer.zero_grad()
                pred_list = model(batch)

                loss: Tensor = calc_loss(pred_list, batch)

                loss.backward()
                if args.grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                if scheduler:
                    scheduler.step()

                detached_loss = loss.item()
                loss_accum += detached_loss
                t.set_description(f"Train (loss = {detached_loss:.4f}, smoothed = {loss_accum / (step + 1):.4f})")
                wandb.log({"train/iter-loss": detached_loss, "train/iter-loss-smoothed": loss_accum / (step + 1)})

        logger.info("Average training loss: {:.4f}".format(loss_accum / (step + 1)))
        return loss_accum / (step + 1)

    @staticmethod
    def name(args: Namespace) -> str:
        """
        Method to return the name of the trainer, used for logging and checkpoint naming.

        To be overridden by subclasses (the base class does not have a name and raises NotImplementedError).
        """
        raise NotImplemented
