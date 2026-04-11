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
    """
    A single layer in the GNN consists of the following steps:
    1. For each vertex, compute a node embedding by applying a linear transformation to the input embedding
    2. For each edge, compute an edge embedding by applying a linear transformation to the concatenation of the 
            node embeddings of the two vertices connected by the edge
    3. For each vertex, compute a pre-GELU embedding by concatenating the node embedding of the vertex, 
            the sum of the node embeddings of the neighboring vertices, 
            and the sum of the edge embeddings of the edges connected to the vertex
    4. For each vertex, compute a post-GELU embedding by applying the GELU activation function
    5. For each vertex, compute the output embedding by applying a linear transformation to the post-GELU embedding
    """
    def __init__(
        self,
        embed_dim: int,
        device: torch.device
    ) -> None:
        """
        Initialize the GNN layer.

        Args:
            embed_dim: The embedding dimension.
            device: The device to run the layer on.
        """
        super().__init__()
        self.embed_dim: int = embed_dim # dimension of the input and output embeddings
        self.node_weights = nn.Linear(embed_dim, embed_dim) # linear transformation for node embeddings
        self.edge_weights = nn.Linear(2 * embed_dim, embed_dim) # linear transformation for edge embeddings 
        self.embed_gelu = nn.GELU() # GELU activation function
        self.embed_update = nn.Linear(3 * embed_dim, embed_dim) # linear transformation for output embeddings
        self.device: torch.device = device # device to run the layer on
        self.to(device) # move the layer to the device
    
    def forward(
        self,
        input_graphs: Data | List[Data],
        input_embeddings: Tensor,
    ) -> Tensor:
        """
        Forward pass for a batch of graphs or a single graph.

        This function computes the output embeddings for all vertices of all graphs in the batch, in order.
        It first computes the node embeddings and edge embeddings, and then computes the pre-GELU embeddings 
        per vertex by concatenating its node embedding, edge embedding and the sum of the neighbors' node embeddings.
        A GELU transformation followed by a final linear transformation gives the output embeddings.

        Args:
            input_graphs: The input graphs (or a single graph)
            input_embeddings; [net_num_vertices, embed_dim] The input embeddings for all vertices of all graphs, in order.
        
        Returns:
            The output embeddings.
        """
        single_graph: bool = isinstance(input_graphs, Data) or not isinstance(input_graphs, list)
        if single_graph:
            assert isinstance(input_graphs, Data)
            input_graphs = [input_graphs]
        
        assert isinstance(input_graphs, list)
        # One per graph: number of vertices (in order)
        num_vertices: List[int] = []
        for graph in input_graphs:
            assert graph.num_nodes is not None
            num_vertices.append(graph.num_nodes)
        assert input_embeddings.shape == torch.Size([sum(num_vertices).__int__(), self.embed_dim])

        # Compute edge embeddings
        net_num_vertices: int = 0
        edge_indices_list: List[Tensor] = []
        for i, graph in enumerate(input_graphs):
            assert graph.edge_index is not None
            assert graph.edge_index.shape == torch.Size([2, graph.edge_index.shape[1]])
            net_num_vertices += num_vertices[i]
            edge_indices_list.append(graph.edge_index + net_num_vertices)
        edge_indices = torch.cat(edge_indices_list)

        # GPU work: compute node and edge embeddings
        node_embeddings: Tensor = self.node_weights(input_embeddings)
        edge_embeddings: Tensor = self.edge_weights(torch.cat(
            [node_embeddings[edge_indices[0]],
             node_embeddings[edge_indices[1]]
        ], dim=1))

        # Move embeddings to pre-GELU embeddings 
        # (concatenation of node embedding, sum of neighbors' node embeddings and sum of edge embeddings)
        pre_gelu_embeddings: Tensor = torch.zeros((net_num_vertices, 3 * self.embed_dim))
        for i in range(num_vertices.__len__()):
            pre_gelu_embeddings[i, :self.embed_dim] = node_embeddings[i]
        for (start, end) in zip(edge_indices[0], edge_indices[1]):
            pre_gelu_embeddings[start, self.embed_dim:2*self.embed_dim] += node_embeddings[end]
            pre_gelu_embeddings[start, 2*self.embed_dim:3*self.embed_dim] += edge_embeddings[start, end]
        
        # GPU work: GELU --> output embeddings
        gelu_embeddings: Tensor = self.embed_gelu(pre_gelu_embeddings)
        out_embeddings: Tensor = self.embed_update(gelu_embeddings)
        assert out_embeddings.shape == torch.Size([net_num_vertices, self.embed_dim])
        return out_embeddings
        
        
class GNN(Module):
    """
    The full GNN consists of multiple layers of type GNNLayer.

    Each GNN layer compute the node embeddings, edge embeddings and concatenates them with the
    sum of neighbors' node embeddings (per vertex) before applying the GELU activation function.
    A final linear transformation is applied to the post-GELU embeddings to get the output embeddings.

    This is repeated for each layer in the GNN.
    """
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        device: torch.device
    ) -> None:
        """
        Initialize the Graph Neural Network
        """
        self.embed_dim: int = embed_dim # dimension of the input and output embeddings
        self.num_layers: int = num_layers # number of layers in the GNN
        self.layers = ModuleList([
            GNNLayer(embed_dim, device) for layer in range(num_layers)
        ]) # list of GNN layers
        self.device: torch.device = device # device to run the GNN on
        self.to(device) # move the GNN to the device
    
    def forward(
        self,
        input_graphs: Data | List[Data],
        input_embeddings: Tensor,
    ) -> Tensor:
        """
        Forward pass for a batch of graphs.
        """
        out_embeddings = input_embeddings
        for layer in self.layers:
            out_embeddings = layer(input_graphs, out_embeddings)
        return out_embeddings