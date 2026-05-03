#!/usr/bin/env python3

from typing import List, Tuple, Dict, Set, Iterable, Callable, Literal, Optional, Any, Union
from dataclasses import dataclass, field
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

TORCH_DEVICE: str = "cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu"

@dataclass
class GraphTransConfig:
    """
    Class that stores the configuration for the GraphTransModel. 
    This is used to initialize the model, and also to store the hyperparameters for the model.

    Note that instances of this must be created for each dataset (and pertain to a GNN for the dataset).

    Args:
        x_dim: The dimension of the input vertex features
        num_transformer_layers: The number of transformer layers in the transformer module
        embed_dim: The embedding dimension
        num_heads: The number of attention heads in the attention layers
        head_dim: The dimension of each attention head in the attention layers
        y_dim: The dimension of the output features
        num_gnn_layers: The number of gnn layers in each transformer layer of the transformer module
        attn_distance_factors: The attention distance factors for each transformer layer of the transformer module (hyperparameters for weighing attention)
        num_mlp_layers: The number of mlp layers in each transformer layer of the transformer module
        device: The device to run the model on
        dtype: The data type to use for the model (default: torch.bfloat16)
    """
    x_dim: int = 2  
    num_transformer_layers: int = 6
    embed_dim: int = 256
    num_heads: int = 8
    head_dim: int = 32
    y_dim: int = 10
    num_gnn_layers: int | List[int] = 2
    attn_distance_factors: List[List[float] | None] | None = None
    num_mlp_layers: int | List[int] = 2
    device: torch.device = torch.device(TORCH_DEVICE)
    dtype: torch.dtype = torch.bfloat16


class GraphTransModel(Module):
    """
    The full Graph Transformer model, consisting of the following components:
    - An input embedding layer, which is a linear transformation from the input vertex features to the input embeddings.
    - A CLS token embedding, which is a learnable parameter that serves as the embedding for the CLS token for each graph.
    - A transformer module, consisting of multiple transformer layers. Each transformer layer consists of a GNN layer, followed by an attention layer, then an MLP layer.
    - The output layer, which is a linear transformation from the output embedding of the CLS token to the output features.

    Note that the GNNs do nothing to the CLS embeddings, and MLPs (like first input embeddings) are seperate for CLS and non-CLS tokens.
    But the attention is common for CLS and non-CLS tokens.
    """
    def __init__(
            self,
            config: GraphTransConfig
    ) -> None:
        """
        Initialize the GraphTransModel given the configuration.

        Args:
            config: The configuration for the model, as a GraphTransConfig object (required).
        """
        super().__init__()
        self.config: GraphTransConfig = config # full configuration for the model

        self.input_embedding = nn.Linear(config.x_dim, config.embed_dim).to(device=config.device, dtype=config.dtype)
        self.cls_embedding = nn.Parameter(torch.zeros((1, config.embed_dim), device=config.device, dtype=config.dtype))
        # linear transformation for input vertex features to input embeddings and CLS nodes.
        
        self.transformer = Transformer(
            config.num_transformer_layers,
            config.embed_dim,
            config.num_heads,
            config.head_dim,
            config.num_gnn_layers,
            config.attn_distance_factors,
            config.num_mlp_layers,
            config.device,
            config.dtype
        ) # transformer module consisting of multiple transformer layers

        self.output_layer = nn.Linear(config.embed_dim, config.y_dim).to(device=config.device, dtype=config.dtype)
        # linear transformation for output CLS embedding to output features

        self.device = config.device # device to run the model on
        self.dtype = config.dtype # data type to use for the model
        self.to(config.device) # move the model to the device
        self.to(config.dtype) # move the model to the data type
    
    def forward(
            self,
            input_graphs: Data | List[Data],
    ) -> Tensor:
        """
        Forward pass for a batch of graphs or a single graph.
        This computes the output features for all vertices of all graphs in the batch, in order.

        It first computes the input embeddings from the input vertex features,
        then applies the transformer to get the output embeddings, 
        and finally applies the output layer to get the output features.

        Args:
            input_graphs: The input graphs (or a single graph)
        
        Returns:
            The output features (Tensor).
        """
        if isinstance(input_graphs, Data):
            input_graphs = [input_graphs]
        assert isinstance(input_graphs, list)
        
        net_num_vertices: int = 0
        cls_mask: Tensor = torch.zeros((net_num_vertices,)).bool() # mask for the CLS tokens
        for graph in input_graphs:
            assert graph.num_nodes is not None
            net_num_vertices += graph.num_nodes
            cls_mask[net_num_vertices] = True # Mark the CLS token for this graph.
            net_num_vertices += 1 # Add 1 for the CLS token.
        
        cls_mask = cls_mask.to(self.device)
        input_embeddings: Tensor = torch.zeros((net_num_vertices, self.config.embed_dim), device=self.device, dtype=self.dtype)
        x_tensor: Tensor = torch.cat([graph.x for graph in input_graphs], dim=0) # [net_num_vertices, x_dim]
        cls_tensor: Tensor = torch.ones((input_graphs.__len__(),), device=self.device, dtype=self.dtype) # [num_graphs,] all ones for CLS tokens
        
        # Compute input embeddings from input vertex features, and set CLS token embeddings to the CLS embedding.
        input_embeddings[~cls_mask] = self.input_embedding(x_tensor)
        input_embeddings[cls_mask] = self.cls_embedding(cls_tensor)
        
        # Compute transformer output embeddings from input embeddings
        transformer_output: Tensor = self.transformer(input_graphs, input_embeddings)
        
        # Compute final output features from unembedding or output layer, only for CLS tokens
        out_embeddings: Tensor = self.output_layer(transformer_output[cls_mask])
        
        assert out_embeddings.shape == torch.Size([len(input_graphs), self.config.y_dim]) 
        # output features for all graphs in the batch, in order
        return out_embeddings