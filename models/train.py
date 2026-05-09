#!/usr/bin/env python3

from typing import List, Tuple, Dict, Set, Iterable, Callable, Literal, Optional, Any, Union
from tqdm import tqdm, trange
from dataclasses import dataclass, field, asdict, replace

import sys, os, gc, argparse
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
sys.path.append(ROOT_DIR)

import torch
from torch import nn, tensor, Tensor, autograd, optim, cuda, mps, cpu, distributions
from torch.nn import Module, Parameter, ModuleList, ModuleDict, functional as F
from torch.optim import Optimizer, Adam, AdamW, SGD, RMSprop, lr_scheduler
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv, GINConv, GIN
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, to_dense_batch, coalesce

from models.gnn import GNNLayer, GNN
from models.attention import AttentionLayer
from models.mlp import MLP
from models.transformer import TransformerLayer, Transformer
from models.full_model import GraphTransConfig, GraphTransModel, ModelTrainConfig

from data_utils.config_objects import DATASET_CONFIGS, TRAIN_CONFIGS

    
def train_graph_transformer(
        config: ModelTrainConfig
) -> GraphTransModel:
    """
    Train the Graph Transformer model on the given dataset.

    Args:
        model: The GraphTransModel to train
        dataset: The dataset to train on, as a PyTorch Dataset object
        out_mapping_fn: A function that maps the output of the model to the ground truth output features for the loss function. 
            This is necessary since the model outputs a tensor of shape [num_graphs, y_dim] but ground truth may be inhomogeneous lists.
        loss_fn: The loss function to use for training, as a function that takes in the model output and ground truth and returns a scalar loss tensor.
        num_epochs: The number of epochs to train for (default: 1)
        batch_size: The batch size to use for training (default: 8)
        learning_rate: The learning rate to use for training (default: 0.0001)
    """
    model = config.model_loader()
    dataset = config.dataset_loader()
    device: torch.device = model.device
    dtype: torch.dtype = model.dtype

    num_batches = dataset.__len__() // config.batch_size # type: ignore
    pbar = tqdm(range(config.num_epochs * num_batches), desc="Training Graph Transformer")
    optimizer: Optimizer = Adam(model.parameters(), lr=config.learning_rate)

    for epoch in pbar:
        optimizer.zero_grad() # zero the gradients
        start = (epoch % num_batches) * config.batch_size
        end = start + config.batch_size
        # Get the data for the batch of graphs, and move it to the device and data type of the model
        batch: List[Data] = [pt.to(device=device) for pt in dataset[start:end]] # type: ignore
        ground_truth: Tensor = dataset.y[start:end] # type: ignore

        # The main computation: run the model
        output: Tensor = model(batch)
        loss: Tensor = config.loss_fn(output, ground_truth) # Compute the loss
        loss.backward() # backpropagate the loss
        
        # Update the model parameters using an optimizer (e.g., Adam)
        optimizer.step() # update the model parameters
        pbar.set_postfix({"loss": loss.item()}) # update the progress bar with the current loss

    return model



def main(
    dataset_name: str,
    batch_size: Optional[int] = None,
    num_epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Main function to train the Graph Transformer model on a specified dataset -- CLI-compatible.

    Args:
        dataset_name: The name of the dataset to train on (e.g., NCI1, NCI109, ogbg-code2) (required).
        batch_size: Batch size for training (default: 32 for most datasets, see config).
        num_epochs: Number of epochs to train for (default: 1 for most datasets, see config).
        learning_rate: Learning rate for training (default: 0.0001 for most datasets, see config).
        save_path: Directory to save the trained model checkpoint (default: checkpoints).
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not found in DATASET_CONFIGS. Available datasets: {list(DATASET_CONFIGS.keys())}")
    
    config: GraphTransConfig = DATASET_CONFIGS[dataset_name]
    train_config: ModelTrainConfig = TRAIN_CONFIGS[dataset_name]
    
    # Override training hyperparameters if provided via command line
    overrides = {}
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if num_epochs is not None:
        overrides["num_epochs"] = num_epochs
    if learning_rate is not None:
        overrides["learning_rate"] = learning_rate
    
    if overrides:
        train_config = replace(train_config, **overrides)

    model = train_graph_transformer(train_config)

    os.makedirs(save_path, exist_ok=True)
    save_name = f"{dataset_name}_model.pt"
    save_file = os.path.join(save_path, save_name)
    torch.save(
        {
            "dataset": dataset_name,
            "model_state_dict": model.state_dict(),
            "model_config": asdict(train_config.model_config),
            "train_config": {
                "num_epochs": train_config.num_epochs,
                "batch_size": train_config.batch_size,
                "learning_rate": train_config.learning_rate,
            },
        },
        save_file,
    )
    print(f"Saved model checkpoint to {save_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Graph Transformer model on a given dataset.")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="The name of the dataset to train on (e.g., NCI1, NCI109, ogbg-code2).")
    parser.add_argument("--batch_size", "-b", type=int, default=None, help="Batch size for training (default: 32 for most datasets, see config).")
    parser.add_argument("--num_epochs", "-e", type=int, default=None, help="Number of epochs to train for (default: 1).")
    parser.add_argument("--learning_rate", "-l", type=float, default=None, help="Learning rate for the optimizer (default: 0.001).")
    parser.add_argument("--save_path", type=str, default="checkpoints", help="Directory to save the trained model checkpoint (default: checkpoints).")
    args = parser.parse_args()
    main(args.dataset, args.batch_size, args.num_epochs, args.learning_rate, args.save_path)