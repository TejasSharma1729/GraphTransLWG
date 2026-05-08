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
from models.train import ExperimentResult, run_graph_transformer_experiments

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
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay for Adam (paper default: 0.0001).")
    parser.add_argument("--scheduler", type=str, default=None, choices=["none", "cosine"], help="Learning rate scheduler.")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed for split/training.")
    parser.add_argument("--runs", type=int, default=None, help="Number of repeated runs with consecutive seeds.")
    parser.add_argument("--train_ratio", type=float, default=None, help="Fraction of data used for training.")
    parser.add_argument("--val_ratio", type=float, default=None, help="Fraction of data used for validation.")
    parser.add_argument("--save_path", type=str, default="checkpoints", help="Directory to save the trained model checkpoint (default: checkpoints).")
    parser.add_argument("--max_graphs", type=int, default=None, help="Subsample ogbg-code2 to this many total graphs (e.g. 4000). Ignored for other datasets.")
    args = parser.parse_args()

    dataset_name: str = args.dataset

    if dataset_name == "ogbg-code2":
        from code2.train_config import get_code2_train_config
        config, train_config = get_code2_train_config(max_graphs=args.max_graphs)
    else:
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Dataset {dataset_name} not found in DATASET_CONFIGS. Available: {list(DATASET_CONFIGS.keys())}")
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
    if args.weight_decay is not None:
        overrides["weight_decay"] = args.weight_decay
    if args.scheduler is not None:
        overrides["scheduler"] = None if args.scheduler == "none" else args.scheduler
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    if args.runs is not None:
        overrides["runs"] = args.runs
    if args.train_ratio is not None:
        overrides["train_ratio"] = args.train_ratio
    if args.val_ratio is not None:
        overrides["val_ratio"] = args.val_ratio
    
    if overrides:
        train_config = replace(train_config, **overrides)

    experiment: ExperimentResult = run_graph_transformer_experiments(train_config)

    os.makedirs(args.save_path, exist_ok=True)
    for run_id, result in enumerate(experiment.results):
        save_name = f"{dataset_name}_run{run_id}_model.pt"
        save_file = os.path.join(args.save_path, save_name)
        torch.save(
            {
                "dataset": dataset_name,
                "run_id": run_id,
                "model_state_dict": result.model.state_dict(),
                "model_config": asdict(train_config.model_config),
                "train_config": {
                    "num_epochs": train_config.num_epochs,
                    "batch_size": train_config.batch_size,
                    "learning_rate": train_config.learning_rate,
                    "weight_decay": train_config.weight_decay,
                    "scheduler": train_config.scheduler,
                    "runs": train_config.runs,
                    "train_ratio": train_config.train_ratio,
                    "val_ratio": train_config.val_ratio,
                    "random_seed": result.seed,
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

    print(f"Mean test metric: {experiment.mean_test_metric:.4f}")
    print(f"Std test metric: {experiment.std_test_metric:.4f}")
