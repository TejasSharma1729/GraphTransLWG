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

from deprecated_trainers import register_trainer
from deprecated_trainers.base_trainer import BaseTrainer


@register_trainer("baseline")
class BaselineTrainer(BaseTrainer):
    """
    A BaseTrainer class, just has the name defined. Is a concrete static class.
    """
    @staticmethod
    def name(args: Namespace) -> str:
        return "baseline"
