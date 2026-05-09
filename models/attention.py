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

from models.batch_utils import PackedGraphBatch, build_cls_mask, pack_edge_index


class AttentionLayer(Module):
    """
    Graph attention layer.

    This computes the attention of matrices similar to attention in transformer networks.
    The twist is that attention is zero for nodes not reachable from each other, and
    there is an option to have weighted attention based on distance (number of hops) between nodes, 
    but those distance attention factors are fixed during initialization.
    """
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            head_dim: int,
            attn_distance_factors: List[float] | None,
            dropout: float,
            device: torch.device,
            dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """
        Initialize the attention layer.

        Args:
            embed_dim: The input and output embedding dimension.
            num_heads: The number of attention heads.
            head_dim: The dimension of each individual attention head.
            attn_distance_factors: Attention distance factors for weighted neighbourhood attention;
                None means binary reachability masking (attend equally to all reachable nodes).
            dropout: Dropout probability applied to attention weights.
            device: The device to run the layer on.
            dtype: The data type to use for the layer parameters (default: torch.bfloat16).
        """
        super().__init__()
        self.embed_dim: int = embed_dim # dimension of the input and output embeddings
        self.num_heads: int = num_heads # number of attention heads
        self.head_dim: int = head_dim # dimension of each attention head
        self.attn_distance_factors: List[float] | None = attn_distance_factors # for weighted attention, preferential to neighbors
        self.dropout: float = dropout
        self.query_weights = nn.Linear(embed_dim, num_heads * head_dim) # linear transformation for query embeddings
        self.key_weights = nn.Linear(embed_dim, num_heads * head_dim)   # linear transformation for key embeddings
        self.value_weights = nn.Linear(embed_dim, num_heads * head_dim) # linear transformation for value embeddings
        self.update_weights = nn.Linear(num_heads * head_dim, embed_dim) # linear transformation for output embeddings
        self.device: torch.device = device # device to run the layer on
        self.dtype: torch.dtype = dtype # data type to use for the layer
        self.to(device) # move the layer to the device
        self.to(dtype)
    
    def forward(
            self,
            input_graphs: Data | List[Data] | PackedGraphBatch,
            input_embeddings: Tensor,
    ) -> Tensor:
        """
        Forward pass for a batch of graphs or a single graph.

        This computes the Q, K and V embeddings of the input embeddings as they are.

        For the attention, it computes a dynamic attention mask, which
        - if there are no distance factors, it is just the reachability matrix (1 or 0)
        - if there are distance factors, the mask is a floating matrix, based on the distance between the nodes
        - between 2 non-reachable nodes or nodes in different graphs, the attention mask is zero

        With the mask, it uses flash attention and also computes the attention projection.
        
        Args:
            input_graphs: The input graphs (or a single graph)
            input_embeddings: [net_num_vertices, embed_dim] The input embeddings for all vertices of all graphs, in order.
        
        Returns:
            The output embeddings.
        """
        if isinstance(input_graphs, PackedGraphBatch):
            batch = input_graphs
            net_num_vertices = int(batch.cls_mask.shape[0])
            base_attn_mask = batch.base_attn_mask
            if self.attn_distance_factors is None:
                attn_factors = batch.attn_factors
            else:
                attn_factors = torch.eye(net_num_vertices, device=self.device, dtype=self.dtype)
                attn_mask: Tensor = base_attn_mask
                for factor in self.attn_distance_factors:
                    attn_mask = torch.clamp(attn_mask @ base_attn_mask, max=1.0)
                    attn_factors = torch.max(attn_factors, factor * attn_mask)
        else:
            single_graph: bool = isinstance(input_graphs, Data) or not isinstance(input_graphs, list)
            if single_graph:
                assert isinstance(input_graphs, Data)
                input_graphs = [input_graphs]

            assert isinstance(input_graphs, list)
            num_vertices: List[int] = []
            for graph in input_graphs:
                assert graph.num_nodes is not None
                num_vertices.append(graph.num_nodes)
            net_num_vertices = sum(num_vertices).__int__() + num_vertices.__len__()
            assert input_embeddings.shape == torch.Size([net_num_vertices, self.embed_dim])

            base_attn_mask = torch.eye(net_num_vertices, device=self.device, dtype=self.dtype)
            prev_num_vertices = 0
            for i, graph in enumerate(input_graphs):
                assert graph.edge_index is not None
                offset = prev_num_vertices
                base_attn_mask[graph.edge_index[0] + offset, graph.edge_index[1] + offset] = 1.0
                for offset_ in range(num_vertices[i]):
                    base_attn_mask[offset + offset_, offset + num_vertices[i]] = 1.0
                    base_attn_mask[offset + num_vertices[i], offset + offset_] = 1.0
                prev_num_vertices += num_vertices[i] + 1

            # The mask that is actually used for attention.
            attn_factors = torch.eye(net_num_vertices, device=self.device, dtype=self.dtype)
            if self.attn_distance_factors is None:
                while True:
                    # Since discrete mask, it will converge in finite steps
                    attn_factors_new: Tensor = torch.clamp(attn_factors @ base_attn_mask, max=1.0)
                    if torch.allclose(attn_factors_new, attn_factors):
                        # Stop when fixed point reachability converges
                        break
                    attn_factors = attn_factors_new
            else:
                attn_mask: Tensor = base_attn_mask
                for factor in self.attn_distance_factors:
                    attn_mask = torch.clamp(attn_mask @ base_attn_mask, max=1.0)
                    attn_factors = torch.max(attn_factors, factor * attn_mask)
        
        # The main (GPU-heavy) computation
        attn_shape = torch.Size([net_num_vertices, self.num_heads, self.head_dim])
        query_embeddings: Tensor = self.query_weights(input_embeddings).view(attn_shape).transpose(0, 1).contiguous()
        key_embeddings: Tensor = self.key_weights(input_embeddings).view(attn_shape).transpose(0, 1).contiguous()
        value_embeddings: Tensor = self.value_weights(input_embeddings).view(attn_shape).transpose(0, 1).contiguous()

        # GPU work: flash attention with floating attention mask (log for proper float masking)
        attn_output: Tensor = F.scaled_dot_product_attention(
            query_embeddings,
            key_embeddings,
            value_embeddings,
            attn_mask=attn_factors.log(), # Log for proper float masking.
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        # Finally, compute the output embeddings with a linear transformation.
        update_in_shape = torch.Size([net_num_vertices, self.num_heads * self.head_dim])
        attn_output = attn_output.transpose(0, 1).contiguous().view(update_in_shape)
        out_embeddings: Tensor = self.update_weights(attn_output)
        assert out_embeddings.shape == torch.Size([net_num_vertices, self.embed_dim])
        return out_embeddings
