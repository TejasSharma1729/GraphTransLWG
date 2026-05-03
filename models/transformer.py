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

from models.gnn import GNNLayer, GNN
from models.attention import AttentionLayer
from models.mlp import MLP


class TransformerLayer(Module):
    """
    Transformer layer. This consists of a single GNN layer, followed by a single attention layer, then an MLP.
    The GNN itself may contain multiple layers, and so may the MLP. But there is only one attention layer.

    This allows chosing not just the size but also the number of GNN layers and MLP layers and
    the attention distance factors for the attention layer, which are important hyperparameters.
    """
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            head_dim: int,
            num_gnn_layers: int,
            attn_distance_factors: List[float] | None,
            num_mlp_layers: int,
            device: torch.device,
            dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """
        Initialize the transformer layer.

        Args:
            embed_dim: The embedding dimension
            num_heads: The number of attention heads
            head_dim: The dimension of each attention head
            num_gnn_layers: The number of gnn layers in the gnn layer
            attn_distance_factors: The attention distance factors (hyperparameters for weighing attention)
            num_mlp_layers: The number of MLP layers in the MLP layer
            device: The device to run the layer on
            dtype: The data type to use for the layer (default: torch.bfloat16)
        """
        super().__init__()
        self.embed_dim: int = embed_dim # dimension of the input and output embeddings
        self.num_heads: int = num_heads # number of attention heads
        self.head_dim: int = head_dim # dimension of each attention head
        self.num_gnn_layers: int = num_gnn_layers # number of gnn layers
        self.attn_distance_factors: List[float] | None = attn_distance_factors # for weighted attention, preferential to neighbors
        self.num_mlp_layers: int = num_mlp_layers # number of mlp layers
        self.attention_layer = AttentionLayer(embed_dim, num_heads, head_dim, attn_distance_factors, device, dtype) # attention layer
        self.gnn_layer = GNN(embed_dim, num_gnn_layers, device, dtype) # gnn layer
        self.mlp_layer = MLP(embed_dim, num_mlp_layers, device, dtype) # mlp layer
        self.device = device # device to run the layer on
        self.dtype = dtype # data type to use for the layer
        self.to(device) # move the layer to the device
        self.to(dtype) # move the layer to the data type

    def forward(
            self,
            input_graphs: Data | List[Data],
            input_embeddings: Tensor,
    ) -> Tensor:
        """
        Forward pass for a batch of graphs or a single graph.

        This computes the output of the attention layer and feeds it into the gnn layer, and returns the output of the gnn layer.

        Args:
            input_graphs: The input graphs (or a single graph)
            input_embeddings: [net_num_vertices, embed_dim] The input embeddings for all vertices of all graphs, in order.
        Returns:
            The output embeddings.
        """
        gnn_output = self.gnn_layer(input_graphs, input_embeddings) # output of the gnn layer
        attention_output = self.attention_layer(input_graphs, gnn_output) # output of the attention layer
        mlp_output = self.mlp_layer(input_graphs, attention_output) # output of the mlp layer
        return mlp_output


class Transformer(Module):
    """
    Graph transformer. This consists of multiple transformer layers, each containing a GNN, an Attention layer, and an MLP.
    The output of each layer is fed into the next layer, and the final output is the output.

    Note that this does not include the initial node embeddings or any unembeddings / algorithms after the output.

    This allows fine tuning not just the size but also the number of GNN layers, MLP layers and the attention distance factors 
    for each layer, which are important hyperparameters (they can be different for different layers).
    """
    def __init__(
            self,
            num_layers: int,
            embed_dim: int,
            num_heads: int,
            head_dim: int,
            num_gnn_layers: int | List[int],
            attn_distance_factors: List[List[float] | None] | None,
            num_mlp_layers: int | List[int],
            device: torch.device,
            dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """
        Initialize the transformer.

        Args:
            num_layers: The number of transformer layers
            embed_dim: The embedding dimension
            num_heads: The number of attention heads
            head_dim: The dimension of each attention head
            num_gnn_layers: The number of gnn layers in the gnn layer per layer
            attn_distance_factors: The attention distance factors (hyperparameters for weighing attention) per layer
            num_mlp_layers: The number of MLP layers in the MLP layer per layer
            device: The device to run the layer on
        """
        super().__init__()
        self.num_layers: int = num_layers # number of transformer layers
        self.embed_dim: int = embed_dim # dimension of the input and output embeddings
        self.num_heads: int = num_heads # number of attention heads
        self.head_dim: int = head_dim # dimension of each attention head
        
        self.num_gnn_layers: List[int] # list of number of gnn layers for each transformer layer
        if isinstance(num_gnn_layers, int):
            self.num_gnn_layers = [num_gnn_layers] * num_layers
        else:
            assert len(num_gnn_layers) == num_layers
            self.num_gnn_layers = num_gnn_layers

        self.attn_distance_factors: List[List[float] | None] # list of attention distance factors for each layer
        if attn_distance_factors is None:
            self.attn_distance_factors = [None] * num_layers
        else:
            assert len(attn_distance_factors) == num_layers
            self.attn_distance_factors = attn_distance_factors
        
        self.num_mlp_layers: List[int] # list of number of mlp layers for each transformer layer
        if isinstance(num_mlp_layers, int):
            self.num_mlp_layers = [num_mlp_layers] * num_layers
        else:
            assert len(num_mlp_layers) == num_layers
            self.num_mlp_layers = num_mlp_layers

        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim,
                num_heads,
                head_dim,
                self.num_gnn_layers[i],
                self.attn_distance_factors[i],
                self.num_mlp_layers[i],
                device,
                dtype
            ) for i in range(num_layers)
        ]) # list of transformer layers

        self.device: torch.device = device # device to run the layer on
        self.dtype: torch.dtype = dtype # data type to use for the layer
        self.to(device) # move the layer to the device
        self.to(dtype)

    def forward(
            self,
            input_graphs: Data | List[Data],
            input_embeddings: Tensor,
    ) -> Tensor:
        """
        Forward pass for a batch of graphs or a single graph.

        This feeds the input through all transformer layers sequentially and returns the output of the last layer.

        Args:
            input_graphs: The input graphs (or a single graph)
            input_embeddings: [net_num_vertices, embed_dim] The input embeddings for all vertices of all graphs, in order.
        Returns:
            The output embeddings.
        """
        embeddings = input_embeddings
        for layer in self.layers:
            embeddings = layer(input_graphs, embeddings)
        return embeddings