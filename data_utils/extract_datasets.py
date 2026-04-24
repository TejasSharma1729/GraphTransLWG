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
from ogb.graphproppred import GraphPropPredDataset, PygGraphPropPredDataset

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

if __name__ == "__main__":
    print("==== LOADING OPEN GRAPH BENCHMARK DATASETS ====")
    print("Loading ogbg-molhiv dataset...")
    MOLHIV = get_graph_dataset("ogbg-molhiv")
    print("Loading ogbg-code2 dataset...")
    CODE2 = get_graph_dataset("ogbg-code2")
    print("Loading ogbg-molpcba dataset...")
    MOLPCBA = get_graph_dataset("ogbg-molpcba")
    print("Loading ogbg-molesol dataset...")
    MOLESOL = get_graph_dataset("ogbg-molesol")
    print("Loading ogbg-mollipo dataset...")
    MOLLIPO = get_graph_dataset("ogbg-mollipo")
    print("NOTE: ogbg-ppa is not loaded since it is too large, and kills the process")
    print()

    print("==== EXAMPLE GRAPHS FROM THE DATASETS ====")
    print(f"ogbg-code dataset has {CODE2.__len__()} graphs")
    datapoint: Data = CODE2[0]
    print(f"An example graph from ogbg-code dataset: {datapoint}")
    print(f"This has {datapoint.num_nodes} nodes and {datapoint.edge_index.shape[1]} edges")
    print()