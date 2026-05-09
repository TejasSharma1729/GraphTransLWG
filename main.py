#!/usr/bin/env python3

import sys, os, gc, argparse
from dataclasses import asdict, replace
import torch, torch_geometric

from torch import nn, tensor, Tensor, optim, cuda, mps, cpu, distributions, autograd
from torch.nn import Module, Parameter, ModuleList, ModuleDict, functional as F
from torch.optim import Optimizer, Adam, AdamW, SGD, RMSprop, lr_scheduler
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv, GINConv, GIN
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, to_dense_batch, coalesce
from torch_geometric.datasets import TUDataset

REPO_ROOT: str = os.path.dirname(os.path.abspath(__file__))
sys.path.append(REPO_ROOT)

from models.train import train_graph_transformer, main as train_main
from data_utils.config_objects import DATASET_CONFIGS, TRAIN_CONFIGS
from models.full_model import ModelTrainConfig

from models import GNNLayer, GNN, AttentionLayer, MLP, TransformerLayer, Transformer, GraphTransConfig, GraphTransModel
from data_utils import get_graph_dataset, load_tudataset_as_torch_dataset, save_dataset_as_pt, PyGAsTorchDataset

DATASETS: list[str] = ["ogbg-molpcba", "ogbg-molhiv", "NCI1", "NCI109"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Graph Transformer model on a given dataset.")
    parser.add_argument("--dataset", "-d", type=str, nargs="+", default="all", help="The name of the dataset to train on (e.g., NCI1, NCI109, ogbg-code2).")
    parser.add_argument("--train", "-t", action="store_true", help="Whether to train the model (default: False, i.e., only load and save a checkpoint).")
    parser.add_argument("--batch_size", "-b", type=int, default=None, help="Batch size for training (default: 32 for most datasets, see config).")
    parser.add_argument("--num_epochs", "-e", type=int, default=None, help="Number of epochs to train for (default: 1 for most datasets, see config).")
    parser.add_argument("--learning_rate", "-l", type=float, default=None, help="Learning rate for training (default: 0.0001 for most datasets, see config).")
    parser.add_argument("--load_path", "-p", type=str, default=os.path.join(REPO_ROOT, "checkpoints"), help="Path to a saved model checkpoint to load after training (default: None).")
    parser.add_argument("--num_points", "-n", type=int, default=100, help="Number of data points to use from the dataset for testing (default: 100).")
    parser.add_argument("--outfile", "-o", type=str, default=os.path.join(REPO_ROOT, "results.txt"), help="File to save test results to (default: results.txt).")
    args = parser.parse_args()

    dataset_names: list[str] = DATASETS if args.dataset == "all" else args.dataset
    for dataset_name in dataset_names:
        if dataset_name not in DATASET_CONFIGS:
            print(f"Dataset {dataset_name} not found in configuration. Skipping.")
            continue

        if args.train:
            train_main(dataset_name, args.batch_size, args.num_epochs, args.learning_rate, args.load_path)
            continue
        
        print(f"Testing loading and saving checkpoint for dataset {dataset_name}...")
        dataloader: DataLoader = DataLoader(get_graph_dataset(dataset_name), shuffle=True)
        model = GraphTransModel(DATASET_CONFIGS[dataset_name])
        for i in range(args.num_points):
            batch: Data = next(iter(dataloader))
            out = model(batch.to(model.device))
            print(f"Output for data point {i}: {out}; correct: {batch.y}")