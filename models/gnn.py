#!/usr/bin/env python3

from typing import List, Tuple, Dict, Set, Iterable, Callable, Literal, Optional, Any, Union

import torch_geometric
import torch
from torch import nn, tensor, Tensor, autograd, optim, cuda, mps, cpu, distributions
from torch.nn import Module, Parameter, ModuleList, ModuleDict, functional as F
from torch.optim import Optimizer, Adam, AdamW, SGD, RMSprop, lr_scheduler

import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv, GINConv, GIN
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, to_dense_batch, coalesce

class GNNLayer(Module):
    def __init__(
        self,
        embed_dim: int,
        device: torch.device
    ) -> None:
        """
        Initialize the GNN layer.
        """
        self.embed_dim: int = embed_dim
        self.node_weights = nn.Linear(embed_dim, embed_dim)
        self.edge_weights = nn.Linear(2 * embed_dim, embed_dim)
        self.embed_gelu = nn.GELU()
        self.embed_update = nn.Linear(3 * embed_dim, embed_dim)
        self.device = device
        self.to(device)
    
    def forward(
        self,
        input_graph: Data,
        input_embeddings: Tensor,
    ) -> Tensor:
        """
        Forward pass for a single graph.
        """
        num_vertices: int = input_graph.num_nodes
        assert input_embeddings.shape == torch.Size(num_vertices, self.embed_dim)

        node_embeddings = self.node_weights(input_embeddings)
        edge_embeddings = self.edge_weights(torch.cat(
            [input_embeddings[input_graph.edge_index[0]],
             input_embeddings[input_graph.edge_index[1]]
        ], dim=1))
        
        pre_gelu_embeddings = torch.zeros((num_vertices, 3 * self.embed_dim))
        for i in range(num_vertices):
            pre_gelu_embeddings[i, :self.embed_dim] = node_embeddings[i]
        for (start, end) in zip(input_graph.edge_index[0], input_graph.edge_index[1]):
            pre_gelu_embeddings[start, self.embed_dim:2*self.embed_dim] += node_embeddings[end]
            pre_gelu_embeddings[start, 2*self.embed_dim:3*self.embed_dim] += edge_embeddings[start, end]
        
        gelu_embeddings = self.embed_gelu(pre_gelu_embeddings)
        out_embeddings = self.embed_update(gelu_embeddings)
        assert out_embeddings.shape == torch.Size(num_vertices, self.embed_dim)
        return out_embeddings
    
    def forward(
        self,
        input_graphs: List[Data],
        input_embeddings: Tensor,
    ) -> Tensor:
        """
        Forward pass for a batch of graphs.
        """
        num_vertices: List[int] = [graph.num_nodes for graph in input_graphs]
        assert input_embeddings.shape == torch.Size(sum(num_vertices).__int__(), self.embed_dim)

        node_embeddings = self.node_weights(input_embeddings)
        vertices_cumsum: List[int] = [0]
        for num_vertex in num_vertices:
            vertices_cumsum.append(vertices_cumsum[-1] + num_vertex)
        edge_indices = torch.cat([graph.edge_index + vertices_cumsum[i] for i, graph in enumerate(input_graphs)])
        edge_embeddings = self.edge_weights(torch.cat(
            [node_embeddings[edge_indices[0]],
             node_embeddings[edge_indices[1]]
        ], dim=1))

        pre_gelu_embeddings = torch.zeros((vertices_cumsum[-1], 3 * self.embed_dim))
        for i in range(num_vertices):
            pre_gelu_embeddings[i, :self.embed_dim] = node_embeddings[i]
        for (start, end) in zip(edge_indices[0], edge_indices[1]):
            pre_gelu_embeddings[start, self.embed_dim:2*self.embed_dim] += node_embeddings[end]
            pre_gelu_embeddings[start, 2*self.embed_dim:3*self.embed_dim] += edge_embeddings[start, end]
        
        gelu_embeddings = self.embed_gelu(pre_gelu_embeddings)
        out_embeddings = self.embed_update(gelu_embeddings)
        assert out_embeddings.shape == torch.Size(vertices_cumsum[-1], self.embed_dim)
        return out_embeddings
        
        
class GNN(Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        device: torch.device
    ) -> None:
        """
        Initialize the Graph Neural Network
        """
        self.embed_dim: int = embed_dim
        self.num_layers: int = num_layers
        self.layers = ModuleList([GNNLayer for layer in range(num_layers)])
        self.to(device)
    
    def forward(
        self,
        input_graph: Data,
        input_embeddings: Tensor,
    ) -> Tensor:
        """
        Forward pass for a single graph.
        """
        out_embeddings = input_embeddings
        for layer in self.layers:
            out_embeddings = layer(input_graph, out_embeddings)
        return out_embeddings
    
    def forward(
        self,
        input_graphs: List[Data],
        input_embeddings: Tensor,
    ) -> Tensor:
        """
        Forward pass for a batch of graphs.
        """
        out_embeddings = input_embeddings
        for layer in self.layers:
            out_embeddings = layer(input_graph, out_embeddings)
        return out_embeddings