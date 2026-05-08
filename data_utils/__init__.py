#!/usr/bin/env python3
"""
Module to load datasets for graph learning, and to run experiments on the datasets using the GraphTransModel.
"""

from .extract_datasets import get_graph_dataset
from .tu_to_pyg import PyGAsTorchDataset, load_tudataset_as_torch_dataset, save_dataset_as_pt
from .config_objects import DATASET_CONFIGS, TRAIN_CONFIGS

__all__ = [
    "get_graph_dataset",
    "PyGAsTorchDataset",
    "load_tudataset_as_torch_dataset",
    "save_dataset_as_pt",
    "DATASET_CONFIGS",
    "TRAIN_CONFIGS",
]