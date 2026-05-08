"""ogbg-code2 dataset utilities and training config factory."""
from .dataset import build_code2_dataset
from .train_config import get_code2_train_config

__all__ = ["build_code2_dataset", "get_code2_train_config"]
