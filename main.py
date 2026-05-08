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

from data_utils.config_objects import DATASET_CONFIGS, TRAIN_CONFIGS
from models.full_model import ModelTrainConfig
from models.train import TrainingResult, train_graph_transformer

REPO_ROOT: str = os.path.dirname(os.path.abspath(__file__))
sys.path.append(REPO_ROOT)

from models import GNNLayer, GNN, AttentionLayer, MLP, TransformerLayer, Transformer, GraphTransConfig, GraphTransModel
from data_utils import get_graph_dataset, load_tudataset_as_torch_dataset, save_dataset_as_pt, PyGAsTorchDataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Graph Transformer model on a given dataset.")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="The name of the dataset to train on (e.g., NCI1, NCI109, ogbg-code2).")
    parser.add_argument("--batch_size", "-b", type=int, default=None, help="Batch size for training (default: 32 for most datasets, see config).")
    parser.add_argument("--num_epochs", "-e", type=int, default=None, help="Number of epochs to train for (default: 1).")
    parser.add_argument("--learning_rate", "-l", type=float, default=None, help="Learning rate for the optimizer (default: 0.001).")
    parser.add_argument("--save_path", type=str, default="checkpoints", help="Directory to save the trained model checkpoint (default: checkpoints).")
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

    result: TrainingResult = train_graph_transformer(train_config)
    model = result.model

    os.makedirs(args.save_path, exist_ok=True)
    save_name = f"{dataset_name}_model.pt"
    save_file = os.path.join(args.save_path, save_name)
    torch.save(
        {
            "dataset": dataset_name,
            "model_state_dict": model.state_dict(),
            "model_config": asdict(train_config.model_config),
            "train_config": {
                "num_epochs": train_config.num_epochs,
                "batch_size": train_config.batch_size,
                "learning_rate": train_config.learning_rate,
                "train_ratio": train_config.train_ratio,
                "val_ratio": train_config.val_ratio,
                "random_seed": train_config.random_seed,
            },
            "metrics": {
                "best_val_metric": result.best_val_metric,
                "test_metric_at_best_val": result.test_metric_at_best_val,
                "final_test_metric": result.final_test_metric,
            },
        },
        save_file,
    )
    print(f"Saved model checkpoint to {save_file}")
    print(f"Best validation metric: {result.best_val_metric:.4f}")
    print(f"Test metric at best validation: {result.test_metric_at_best_val:.4f}")
