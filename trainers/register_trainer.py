import torch
import wandb
from loguru import logger
from tqdm import tqdm
from importlib import import_module

from torch import nn, tensor, Tensor
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.optim import Optimizer, lr_scheduler
import sys, os, gc, argparse
from typing import Any, Tuple, List, Callable, Type
from argparse import ArgumentParser, Namespace

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
sys.path.append(ROOT_DIR)
from models.full_model import GraphTransModel

from trainers.base_trainer import BaseTrainer

TRAINER_REGISTRY: dict[str, Type[BaseTrainer]] = {}
TRAINER_CLASS_NAMES: set[str] = set()

def get_trainer_and_parser(args: Namespace, parser: ArgumentParser) -> Type[BaseTrainer]:
    trainer = TRAINER_REGISTRY[args.aug]
    trainer.add_args(parser)
    return trainer


def register_trainer(name: str, dataclass=None) -> Callable[[Type[BaseTrainer]], Type[BaseTrainer]]:
    """
    New tasks can be added to fairseq with the
    :func:`~fairseq.tasks.register_task` function decorator.
    For example::
        @register_task('classification')
        class ClassificationTask(FairseqTask):
            (...)
    .. note::
        All Tasks must implement the :class:`~fairseq.tasks.FairseqTask`
        interface.
    Args:
        name (str): the name of the task
    """

    def register_trainer_cls(cls: Type[BaseTrainer]) -> Type[BaseTrainer]:
        if name in TRAINER_REGISTRY:
            raise ValueError("Cannot register duplicate task ({})".format(name))
        if not issubclass(cls, BaseTrainer):
            raise ValueError("Trainer ({}: {}) must extend BaseTrainer".format(name, cls.__name__))
        if cls.__name__ in TRAINER_CLASS_NAMES:
            raise ValueError("Cannot register task with duplicate class name ({})".format(cls.__name__))
        TRAINER_REGISTRY[name] = cls
        TRAINER_CLASS_NAMES.add(cls.__name__)

        return cls

    return register_trainer_cls
