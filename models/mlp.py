#!/usr/bin/env python3

from typing import List, Tuple, Dict, Set, Iterable, Callable, Literal, Optional, Any, Union
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


class MLP(Module):
    """
    An MLP module, consisting of multiple linear layers with GELU activation functions in between.

    This is for each node in the graph, and does not take into account the graph structure. 
    It is used in the transformer layer after the attention layer.
    """
    def __init__(
            self,
            embed_dim: int,
            num_layers: int,
            device: torch.device,
            dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """
        Initialize the MLP.

        Args:
            embed_dim: The embedding dimension
            num_layers: The number of layers in the MLP
            device: The device to run the MLP on
            dtype: The data type to use for the layer (default: torch.bfloat16)
        """
        super().__init__()
        self.embed_dim: int = embed_dim # dimension of the input and output embeddings
        self.num_layers: int = num_layers # number of layers in the MLP
        module_list: List[Module] = []
        module_list_cls: List[Module] = []
        for layer in range(num_layers - 1):
            module_list.append(nn.Linear(embed_dim, embed_dim)) # linear transformation for the layer
            module_list.append(nn.GELU()) # GELU activation function for the layer
            module_list_cls.append(nn.Linear(embed_dim, embed_dim)) # linear transformation for the CLS token
            module_list_cls.append(nn.GELU()) # GELU activation function for the CLS token
        module_list.append(nn.Linear(embed_dim, embed_dim)) # linear transformation for the final layer
        module_list_cls.append(nn.Linear(embed_dim, embed_dim)) # linear transformation for the final layer of the CLS token
        self.layers = ModuleList(module_list) # list of layers in the MLP
        self.layers_cls = ModuleList(module_list_cls) # list of layers for the CLS token in the MLP
        self.device: torch.device = device # device to run the MLP on
        self.dtype: torch.dtype = dtype # data type to use for the layer
        self.to(device) # move the MLP to the device
        self.to(dtype) # move the MLP to the data type

    def forward(
            self,
            input_graphs: Data | List[Data],
            input_embeddings: Tensor,
    ) -> Tensor:
        """
        Forward pass for a batch of graphs or a single graph.

        This function computes the output embeddings for all vertices of all graphs in the batch, in order.
        It applies each layer of the MLP sequentially to the input embeddings, and returns the output embeddings.

        The graph is not provided here, since it is not needed for the MLP (per node computation).

        Args:
            input_graphs: The input graphs (or a single graph) as Data objects or list thereof.
            input_embeddings: The input embeddings for all vertices of all graphs in the batch, in order.
        
        Returns:
            The output embeddings.
        """
        cls_mask: Tensor = torch.zeros((input_embeddings.shape[0],)).bool() # mask for the CLS tokens
        net_num_vertices: int = 0
        for graph in input_graphs if isinstance(input_graphs, list) else [input_graphs]:
            assert graph.num_nodes is not None
            net_num_vertices += graph.num_nodes
            cls_mask[net_num_vertices] = True # Mark the CLS token for this graph.
            net_num_vertices += 1 # Add 1 for the CLS token.
        cls_mask = cls_mask.to(self.device)
        
        out_embeddings = input_embeddings
        for layer, cls_layer in zip(self.layers, self.layers_cls):
            out_embeddings[~cls_mask] = layer(out_embeddings[~cls_mask]) # Apply the layer to non-CLS tokens only.
            out_embeddings[cls_mask] = cls_layer(out_embeddings[cls_mask]) # Apply the corresponding layer for the CLS token.
        return out_embeddings
    