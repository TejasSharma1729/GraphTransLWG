#!/usr/bin/env python3

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
            device: torch.device
    ) -> None:
        """
        Initialize the MLP.

        Args:
            embed_dim: The embedding dimension
            num_layers: The number of layers in the MLP
            device: The device to run the MLP on
        """
        super().__init__()
        self.embed_dim: int = embed_dim # dimension of the input and output embeddings
        self.num_layers: int = num_layers # number of layers in the MLP
        module_list: List[Module] = []
        for layer in range(num_layers - 1):
            module_list.append(nn.Linear(embed_dim, embed_dim)) # linear transformation for the layer
            module_list.append(nn.GELU()) # GELU activation function for the layer
        module_list.append(nn.Linear(embed_dim, embed_dim)) # linear transformation for the final layer
        self.layers = ModuleList(module_list) # list of layers in the MLP
        self.device: torch.device = device # device to run the MLP on
        self.to(device) # move the MLP to the device

    def forward(
            self,
            input_embeddings: Tensor,
    ) -> Tensor:
        """
        Forward pass for a batch of graphs or a single graph.

        This function computes the output embeddings for all vertices of all graphs in the batch, in order.
        It applies each layer of the MLP sequentially to the input embeddings, and returns the output embeddings.

        The graph is not provided here, since it is not needed for the MLP (per node computation).

        Args:
            input_embeddings: The input embeddings for all vertices of all graphs in the batch, in order.
        
        Returns:
            The output embeddings.
        """
        out_embeddings = input_embeddings
        for layer in self.layers:
            out_embeddings = layer(out_embeddings) # apply the layer to the embeddings
        return out_embeddings
    