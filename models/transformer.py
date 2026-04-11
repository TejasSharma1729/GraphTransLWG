# This transformer implementation uses the gnn and attention layers defined in gnn.py and attention.py, respectively.
# The transformer consists of multiple layers, each of which consists of an attention layer and a gnn layer. The output of each layer is fed into the next layer, and the final output is the output of the last layer.

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Linear
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, to_dense_batch, coalesce
from .gnn import GNNLayer
from .attention import AttentionLayer
from typing import List, Optional

"""
Transformer layer.
This consists of an attention layer and a gnn layer. The output of the attention layer is fed into the gnn layer, and the output of the gnn layer is fed into the next transformer layer.
"""
class TransformerLayer(Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            head_dim: int,
            attn_distance_factors: List[float] | None,
            gnn_hidden_dim: int,
            device: torch.device
    ) -> None:
        """
        Initialize the transformer layer.

        Args:
            embed_dim: The embedding dimension
            num_heads: The number of attention heads
            head_dim: The dimension of each attention head
            attn_distance_factors: The attention distance factors (hyperparameters for weighing attention)
            gnn_hidden_dim: The hidden dimension of the gnn layer
            device: The device to run the layer on
        """
        super().__init__()
        self.attention_layer = AttentionLayer(embed_dim, num_heads, head_dim, attn_distance_factors, device) # attention layer
        self.gnn_layer = GNNLayer(embed_dim, gnn_hidden_dim, device) # gnn layer
        self.device = device # device to run the layer on
        self.to(device) # move the layer to the device

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
            input_embeddings [net_num_vertices, embed_dim]: The input embeddings for all vertices of all graphs, in order.
        Returns:
            The output embeddings.
        """
        attention_output = self.attention_layer(input_graphs, input_embeddings) # output of the attention layer
        gnn_output = self.gnn_layer(input_graphs, attention_output) # output of the gnn layer
        return gnn_output
    
class Transformer(Module):
    def __init__(
            self,
            num_layers: int,
            embed_dim: int,
            num_heads: int,
            head_dim: int,
            attn_distance_factors: List[float] | None,
            gnn_hidden_dim: int,
            device: torch.device
    ) -> None:
        """
        Initialize the transformer.

        Args:
            num_layers: The number of transformer layers
            embed_dim: The embedding dimension
            num_heads: The number of attention heads
            head_dim: The dimension of each attention head
            attn_distance_factors: The attention distance factors (hyperparameters for weighing attention)
            gnn_hidden_dim: The hidden dimension of the gnn layer
            device: The device to run the layer on
        """
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, num_heads, head_dim, attn_distance_factors, gnn_hidden_dim, device) for _ in range(num_layers)]) # list of transformer layers
        self.device = device # device to run the layer on
        self.to(device) # move the layer to the device

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
            input_embeddings [net_num_vertices, embed_dim]: The input embeddings for all vertices of all graphs, in order.
        Returns:
            The output embeddings.
        """
        x = input_embeddings
        for layer in self.layers:
            x = layer(input_graphs, x)
        return x