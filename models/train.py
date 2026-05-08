#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Iterable, Callable, Literal, Optional, Any, Union
from tqdm import tqdm, trange
import copy

import sys, os, gc
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


@dataclass
class TrainingResult:
    model: GraphTransModel
    best_val_metric: float
    test_metric_at_best_val: float
    final_test_metric: float


def _split_indices(
    dataset_size: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    return train_indices, val_indices, test_indices


def _default_metric(output: Tensor, target: Tensor) -> float:
    if output.ndim == 2 and output.shape[1] > 1 and target.dtype in (torch.int64, torch.long):
        pred = output.argmax(dim=1)
        return float((pred == target.view(-1)).float().mean().item())

    if output.shape == target.shape or output.numel() == target.numel():
        pred = (output.reshape_as(target).sigmoid() >= 0.5)
        truth = target >= 0.5
        return float((pred == truth).float().mean().item())

    return 0.0


def _make_batches(indices: List[int], batch_size: int) -> Iterable[List[int]]:
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


@torch.no_grad()
def _evaluate(
    model: GraphTransModel,
    dataset: TorchDataset[Data],
    indices: List[int],
    config: ModelTrainConfig,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    total_graphs = 0
    device: torch.device = model.device

    for batch_indices in _make_batches(indices, config.batch_size):
        samples: List[Data] = [dataset[i] for i in batch_indices] # type: ignore
        batch: List[Data] = [pt.to(device=device) for pt in samples]
        ground_truth: Tensor = config.out_mapping_fn(samples).to(device=device)
        output: Tensor = model(batch)
        loss: Tensor = config.loss_fn(output, ground_truth)
        metric_fn = config.metric_fn or _default_metric
        metric = metric_fn(output, ground_truth)
        batch_size = len(samples)
        total_loss += float(loss.item()) * batch_size
        total_metric += metric * batch_size
        total_graphs += batch_size

    return total_loss / max(total_graphs, 1), total_metric / max(total_graphs, 1)


def train_graph_transformer(
        config: ModelTrainConfig
) -> TrainingResult:
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

    train_indices, val_indices, test_indices = _split_indices(
        dataset.__len__(), # type: ignore
        config.train_ratio,
        config.val_ratio,
        config.random_seed,
    )
    optimizer: Optimizer = Adam(model.parameters(), lr=config.learning_rate)
    best_val_metric = float("-inf")
    test_metric_at_best_val = 0.0
    best_state_dict = copy.deepcopy(model.state_dict())
    shuffle_generator = torch.Generator().manual_seed(config.random_seed)

    pbar = tqdm(range(config.num_epochs), desc="Training Graph Transformer")
    for epoch in pbar:
        model.train()
        shuffled_order = torch.randperm(len(train_indices), generator=shuffle_generator).tolist()
        shuffled_train_indices = [train_indices[i] for i in shuffled_order]
        total_train_loss = 0.0
        total_train_graphs = 0

        for batch_indices in _make_batches(shuffled_train_indices, config.batch_size):
            optimizer.zero_grad()
            samples: List[Data] = [dataset[i] for i in batch_indices] # type: ignore
            batch: List[Data] = [pt.to(device=device) for pt in samples]
            ground_truth: Tensor = config.out_mapping_fn(samples).to(device=device)
            output: Tensor = model(batch)
            loss: Tensor = config.loss_fn(output, ground_truth)
            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item()) * len(samples)
            total_train_graphs += len(samples)

        train_loss = total_train_loss / max(total_train_graphs, 1)
        val_loss, val_metric = _evaluate(model, dataset, val_indices, config)
        test_loss, test_metric = _evaluate(model, dataset, test_indices, config)

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            test_metric_at_best_val = test_metric
            best_state_dict = copy.deepcopy(model.state_dict())

        pbar.set_postfix({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_metric": val_metric,
            "test_metric": test_metric,
        })

    model.load_state_dict(best_state_dict)
    _, final_test_metric = _evaluate(model, dataset, test_indices, config)
    return TrainingResult(
        model=model,
        best_val_metric=best_val_metric,
        test_metric_at_best_val=test_metric_at_best_val,
        final_test_metric=final_test_metric,
    )
