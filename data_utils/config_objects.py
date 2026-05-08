#!/usr/bin/env python3
from typing import List, Tuple, Dict, Set, Iterable, Callable, Literal, Optional, Any, Union
from dataclasses import dataclass, field
import sys, os, gc
import torch, torch_geometric

from torch import nn, tensor, Tensor, optim, cuda, mps, cpu, distributions, autograd
from torch.nn import Module, Parameter, ModuleList, ModuleDict, functional as F
from torch.optim import Optimizer, Adam, AdamW, SGD, RMSprop, lr_scheduler
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv, GINConv, GIN
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, to_dense_batch, coalesce
from torch_geometric.datasets import TUDataset

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
sys.path.append(ROOT_DIR)

from models.full_model import GraphTransConfig, ModelTrainConfig, GraphTransModel
from data_utils.extract_datasets import get_graph_dataset
from data_utils.tu_to_pyg import PyGAsTorchDataset, load_tudataset_as_torch_dataset, save_dataset_as_pt


TORCH_DEVICE_STR: str = "cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu" # type: ignore
TORCH_DEVICE = torch.device(TORCH_DEVICE_STR) # type: ignore
TORCH_DTYPE = torch.bfloat16

DATASET_CONFIGS: Dict[str, GraphTransConfig] = {
    "NCI1": GraphTransConfig(
        x_dim=37,
        num_transformer_layers=6,
        embed_dim=256,
        num_heads=8,
        head_dim=32,
        y_dim=2,
        num_gnn_layers=2,
        attn_distance_factors=None,
        num_mlp_layers=2,
        device=TORCH_DEVICE,
        dtype=TORCH_DTYPE,
    ),
    "NCI109": GraphTransConfig(
        x_dim=38,
        num_transformer_layers=6,
        embed_dim=256,
        num_heads=8,
        head_dim=32,
        y_dim=2,
        num_gnn_layers=2,
        attn_distance_factors=None,
        num_mlp_layers=2,
        device=TORCH_DEVICE,
        dtype=TORCH_DTYPE,
    ),
    "ogbg-code2": GraphTransConfig(
        x_dim=2,
        num_transformer_layers=6,
        embed_dim=256,
        num_heads=8,
        head_dim=32,
        y_dim=19, # TODO: check how to tokenizer
        num_gnn_layers=2,
        attn_distance_factors=None,
        num_mlp_layers=2,
        device=TORCH_DEVICE,
        dtype=TORCH_DTYPE,
    ),
    "ogbg-molhiv": GraphTransConfig(
        x_dim=3,
        num_transformer_layers=6,
        embed_dim=256,
        num_heads=8,
        head_dim=32,
        y_dim=1,
        num_gnn_layers=2,
        attn_distance_factors=None,
        num_mlp_layers=2,
        device=TORCH_DEVICE,
        dtype=TORCH_DTYPE,
    ),
    "ogbg-molpcba": GraphTransConfig(
        x_dim=9,
        num_transformer_layers=6,
        embed_dim=256,
        num_heads=8,
        head_dim=32,
        y_dim=128,
        num_gnn_layers=2,
        attn_distance_factors=None,
        num_mlp_layers=2,
        device=TORCH_DEVICE,
        dtype=TORCH_DTYPE,
    ),
}

def bad():
    raise NotImplementedError()


def masked_bce_with_logits(pred_logits: Tensor, target: Tensor) -> Tensor:
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype)
    return F.binary_cross_entropy_with_logits(pred_logits[mask], target[mask])


TRAIN_CONFIGS: Dict[str, ModelTrainConfig] = {
    "NCI1": ModelTrainConfig(
        model_config=DATASET_CONFIGS["NCI1"],
        model_loader=lambda: GraphTransModel(DATASET_CONFIGS["NCI1"]),
        dataset_loader=lambda: load_tudataset_as_torch_dataset("NCI1")[1].to(TORCH_DEVICE_STR),
        out_mapping_fn=lambda x: x,
        loss_fn=lambda x, y: F.cross_entropy(x, y),
    ),
    "NCI109": ModelTrainConfig(
        model_config=DATASET_CONFIGS["NCI109"],
        model_loader=lambda: GraphTransModel(DATASET_CONFIGS["NCI109"]),
        dataset_loader=lambda: load_tudataset_as_torch_dataset("NCI109")[1].to(TORCH_DEVICE_STR),
        out_mapping_fn=lambda x: x,
        loss_fn=lambda x, y: F.cross_entropy(x, y),
    ),
    "ogbg-code2": ModelTrainConfig(
        model_config=DATASET_CONFIGS["ogbg-code2"],
        model_loader=lambda: bad(), # not sure of output format
        dataset_loader=lambda: get_graph_dataset("ogbg-code2").to(TORCH_DEVICE_STR), # type: ignore
        out_mapping_fn=lambda x: bad(),
        loss_fn=lambda x, y: bad(),
    ),
    "ogbg-molhiv": ModelTrainConfig(
        model_config=DATASET_CONFIGS["ogbg-molhiv"],
        model_loader=lambda: GraphTransModel(DATASET_CONFIGS["ogbg-molhiv"]),
        dataset_loader=lambda: get_graph_dataset("ogbg-molhiv").to(TORCH_DEVICE_STR), # type: ignore
        out_mapping_fn=lambda x: x,
        loss_fn=lambda x, y: F.binary_cross_entropy(x, y),
    ),
    "ogbg-molpcba": ModelTrainConfig(
        model_config=DATASET_CONFIGS["ogbg-molpcba"],
        model_loader=lambda: GraphTransModel(DATASET_CONFIGS["ogbg-molpcba"]),
        dataset_loader=lambda: get_graph_dataset("ogbg-molpcba").to(TORCH_DEVICE_STR), # type: ignore
        out_mapping_fn=lambda x: x,
        loss_fn=lambda x, y: masked_bce_with_logits(x, y),
    ),
}