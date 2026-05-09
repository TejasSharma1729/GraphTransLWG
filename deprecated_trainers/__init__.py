import importlib
import os

from .base_trainer import BaseTrainer
from .baseline_trainer import BaselineTrainer
from .flag_trainer import FlagTrainer
from .register_trainer import register_trainer, get_trainer_and_parser
from .register_trainer import TRAINER_REGISTRY, TRAINER_CLASS_NAMES


__all__ = [
    "BaseTrainer",
    "BaselineTrainer",
    "FlagTrainer",
    "register_trainer",
    "get_trainer_and_parser",
    "TRAINER_REGISTRY",
    "TRAINER_CLASS_NAMES"
]
