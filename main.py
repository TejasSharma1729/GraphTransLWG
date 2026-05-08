#!/usr/bin/env python3

import sys, os, gc, argparse
from dataclasses import replace
import torch, torch_geometric

from torch import nn, tensor, Tensor, optim, cuda, mps, cpu, distributions, autograd
from torch.nn import Module, Parameter, ModuleList, ModuleDict, functional as F
from torch.optim import Optimizer, Adam, AdamW, SGD, RMSprop, lr_scheduler
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv, GINConv, GIN
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, to_dense_batch, coalesce
from torch_geometric.datasets import TUDataset

from data_utils.config_objects import DATASET_CONFIGS, TRAIN_CONFIGS
from models.full_model import ModelTrainConfig
from models.train import train_graph_transformer

REPO_ROOT: str = os.path.dirname(os.path.abspath(__file__))
sys.path.append(REPO_ROOT)

from models import GNNLayer, GNN, AttentionLayer, MLP, TransformerLayer, Transformer, GraphTransConfig, GraphTransModel
from data_utils import get_graph_dataset, load_tudataset_as_torch_dataset, save_dataset_as_pt, PyGAsTorchDataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Graph Transformer model on a given dataset.")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="The name of the dataset to train on (e.g., NCI1, NCI109, ogbg-code2).")
    parser.add_argument("--batch_size", "-b", type=int, default=None, help="Batch size for training (default: 8 for most datasets, see config).")
    parser.add_argument("--num_epochs", "-e", type=int, default=None, help="Number of epochs to train for (default: 1).")
    parser.add_argument("--learning_rate", "-l", type=float, default=None, help="Learning rate for the optimizer (default: 0.001).")
    args = parser.parse_args()

    dataset_name: str = args.dataset
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not found in DATASET_CONFIGS. Available datasets: {list(DATASET_CONFIGS.keys())}")
    
    config: GraphTransConfig = DATASET_CONFIGS[dataset_name]
    train_config: ModelTrainConfig = TRAIN_CONFIGS[dataset_name]
    
    # Override training hyperparameters if provided via command line
    overrides = {}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.num_epochs is not None:
        overrides["num_epochs"] = args.num_epochs
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    
    if overrides:
        train_config = replace(train_config, **overrides)

    train_graph_transformer(train_config)