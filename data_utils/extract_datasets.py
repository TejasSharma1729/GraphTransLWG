#!/usr/bin/env python3

import sys, os, gc
CUR_DIR: str = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR: str = os.path.dirname(CUR_DIR)
DATASET_DIR: str = os.path.join(ROOT_DIR, "dataset")

from typing import List, Tuple, Dict, Set, Iterable, Callable, Literal, Optional, Any, Union

import torch
from torch import nn, tensor, Tensor, autograd, optim, cuda, mps, cpu, distributions
from torch.nn import Module, Parameter, ModuleList, ModuleDict, functional as F
from torch.optim import Optimizer, Adam, AdamW, SGD, RMSprop, lr_scheduler
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv, GINConv, GIN
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, to_dense_batch, coalesce

import ogb
from ogb.linkproppred import LinkPropPredDataset, PygLinkPropPredDataset, DglLinkPropPredDataset
from ogb.nodeproppred import NodePropPredDataset, PygNodePropPredDataset, DglNodePropPredDataset
from ogb.graphproppred import GraphPropPredDataset, PygGraphPropPredDataset, DglGraphPropPredDataset

def get_graph_dataset(name: str) -> TorchDataset[Data]:
    """
    Utility to standardize dataset loading PygGraphPropPredDataset as a 
    TorchDataset of graph Data objects, for use in graph transformers.

    It auto-saves the dataset into directory "dataset/" and in subsequent runs,
    loads from the saved dataset instead of re-downloading, for efficiency.

    Args:
        name: The name of the dataset to load, such as "ogbd-code2" (required)

    Returns:
        TorchDataset[Data], a dataset of graphs.
    """
    return PygGraphPropPredDataset(name=name, root=DATASET_DIR)

# MOLHIV = get_graph_dataset("ogbg-molhiv")
# CODE2 = get_graph_dataset("ogbd-code2")
# MOLPCBA = get_graph_dataset("ogbg-molpcba")
# MOLESOL = get_graph_dataset("ogbg-molesol")
# MOLLIPO = get_graph_dataset("ogbg-mollipo")