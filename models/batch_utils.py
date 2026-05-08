#!/usr/bin/env python3
"""
Batch utilities for efficient graph batch packing and preprocessing.

This module provides TorchScript-compiled functions and data structures for packing multiple
graph instances into a single batch representation. It pre-computes masks and connectivity
information to avoid redundant computation during forward passes through transformer layers.

Key components:
  - build_cls_mask: Marks CLS token positions in the concatenated vertex tensor
  - pack_edge_index: Merges edge indices from multiple graphs with correct vertex offsets
  - build_attn_factors: Computes transitive closure of attention connectivity via fixed-point iteration
  - PackedGraphBatch: Container holding all precomputed batch data
  - pack_graph_batch: Main entry point that orchestrates batch packing
"""

from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor
from torch.jit._script import script
from torch_geometric.data import Data


@script
def build_cls_mask(num_nodes: List[int]) -> Tensor:
    """
    Build a boolean mask marking CLS token positions in the packed batch.
    
    For each graph with n nodes, a CLS token is appended after those n nodes.
    This function returns a boolean tensor of shape [total_vertices + num_graphs]
    where True indicates a CLS token position.
    
    Args:
        num_nodes: List of node counts for each graph in the batch.
                   Example: [3, 2] for two graphs with 3 and 2 nodes respectively.
    
    Returns:
        Boolean tensor of shape [sum(num_nodes) + len(num_nodes)] where True marks CLS positions.
        For num_nodes=[3,2], output has shape [7] with True at indices 3 and 6.
    
    Example:
        >>> mask = build_cls_mask([3, 2])
        >>> mask
        tensor([False, False, False,  True, False,  True])  # Shape: [6]
    """
    total_vertices = 0
    for count in num_nodes:
        total_vertices += count + 1

    cls_mask = torch.zeros((total_vertices,), dtype=torch.bool)
    offset = 0
    for count in num_nodes:
        offset += count
        cls_mask[offset] = True
        offset += 1
    return cls_mask


@script
def pack_edge_index(edge_indices: List[Tensor], num_nodes: List[int]) -> Tensor:
    """
    Pack edge indices from multiple graphs into a single tensor with correct vertex offsets.
    
    When multiple graphs are concatenated, their vertex indices need to be offset to maintain
    correctness. Graph 0 uses vertices [0, n0), graph 1 uses [n0, n0+n1), etc.
    This function applies these offsets to all edge indices and concatenates them.
    
    Args:
        edge_indices: List of edge_index tensors, each of shape [2, num_edges_i].
                      Standard PyG format with source indices in row 0, target in row 1.
        num_nodes: List of node counts for each graph, corresponding to edge_indices.
    
    Returns:
        Packed edge_index tensor of shape [2, total_edges] with adjusted vertex indices.
    
    Example:
        >>> edge_indices = [
        ...     torch.tensor([[0, 1], [1, 0]]).T,  # Graph 0: 3 nodes, 2 edges
        ...     torch.tensor([[0], [1]]).T          # Graph 1: 2 nodes, 1 edge
        ... ]
        >>> num_nodes = [3, 2]
        >>> packed = pack_edge_index(edge_indices, num_nodes)
        >>> packed
        tensor([[0, 1, 3],
                [1, 0, 4]])  # Graph 1 edges offset by 3 (nodes in graph 0)
    """
    packed_edges: List[Tensor] = []
    offset = 0
    for idx in range(len(edge_indices)):
        packed_edges.append(edge_indices[idx] + offset)
        offset += num_nodes[idx]
    return torch.cat(packed_edges, dim=1)


@script
def build_attn_factors(base_attn_mask: Tensor) -> Tensor:
    """
    Compute transitive closure of attention connectivity via fixed-point iteration.
    
    Starting from direct edges (1-hop) and self-loops (0-hop), computes all reachable vertices
    through paths of any length in the graph. Uses matrix multiplication to propagate connectivity:
    each iteration computes attn_factors @ base_attn_mask to extend reachability by one hop.
    Iteration terminates when convergence is reached (attn_factors stops changing).
    
    This enables attention layers to model relationships at any distance by pre-computing
    which vertices can attend to which, avoiding expensive per-layer reachability checks.
    
    Args:
        base_attn_mask: Boolean or float tensor of shape [V, V] representing direct connectivity
                        (1 for edges, 1 for self-loops, 0 elsewhere). Should be on desired device.
    
    Returns:
        Float tensor of shape [V, V] with values in {0, 1} indicating reachability.
        Entry [i, j] = 1 means vertex i can reach vertex j in 0 or more hops.
    
    Complexity:
        O(V^3 * D) where V is num vertices and D is diameter (convergence iterations).
        Typically D is small (2-5), making this efficient for small-medium graphs.
    
    Example:
        >>> base_mask = torch.tensor([
        ...     [1., 1., 0.],  # Vertex 0 connects to 0 and 1
        ...     [1., 1., 1.],  # Vertex 1 connects to 0, 1, 2
        ...     [0., 1., 1.]   # Vertex 2 connects to 1 and 2
        ... ])
        >>> factors = build_attn_factors(base_mask)
        >>> factors  # All vertices eventually reach each other
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]])
    """
    net_num_vertices = int(base_attn_mask.shape[0])
    attn_factors = torch.eye(net_num_vertices, device=base_attn_mask.device, dtype=base_attn_mask.dtype)
    while True:
        attn_factors_new = torch.clamp(attn_factors @ base_attn_mask, max=1.0)
        if torch.allclose(attn_factors_new, attn_factors):
            break
        attn_factors = attn_factors_new
    return attn_factors


@dataclass
class PackedGraphBatch:
    """
    Container holding precomputed batch data for efficient graph transformer inference/training.
    
    Packs multiple graph instances along with their CLS tokens and precomputes connectivity
    masks to avoid redundant computation across transformer layers. All tensors are already
    on the target device and in the correct dtype.
    
    Attributes:
        graphs: List of input Data objects (PyTorch Geometric graph instances).
        num_nodes: List of node counts for each graph in the batch.
                   Example: [3, 2] means first graph has 3 nodes, second has 2.
        cls_mask: Boolean tensor of shape [sum(num_nodes) + len(num_nodes)].
                  True marks CLS token positions. Precomputed to avoid per-layer rebuilds.
        edge_index: Packed edge indices of shape [2, total_edges] with vertex offsets applied.
                    Concatenates and offsets edges from all graphs to match the packed vertex layout.
        base_attn_mask: Base attention connectivity mask of shape [V, V] where V = sum(num_nodes) + len(num_nodes).
                        Represents 1-hop connectivity (direct edges + self-loops + edges to CLS).
                        Float tensor with 1.0 for allowed connections, 0.0 otherwise.
        attn_factors: Transitive closure tensor of shape [V, V].
                      Computed via fixed-point iteration from base_attn_mask.
                      Entry [i, j] = 1.0 means i can attend to j (reachability).
                      Cached here to avoid per-layer recomputation in transformer.
    
    Usage:
        >>> batch = pack_graph_batch([graph1, graph2], device, dtype)
        >>> # batch.cls_mask can be reused across all transformer layers
        >>> # batch.attn_factors pre-computes all reachability, avoiding per-layer closures
    
    Note:
        All tensors are device-aligned (on the same device). dtype of attn_factors
        matches the provided dtype parameter; other tensors follow their natural types.
    """
    graphs: List[Data]
    num_nodes: List[int]
    cls_mask: Tensor
    edge_index: Tensor
    base_attn_mask: Tensor
    attn_factors: Tensor


def pack_graph_batch(graphs: List[Data], device: torch.device, dtype: torch.dtype) -> PackedGraphBatch:
    """
    Main entry point: pack a batch of graphs with precomputed masks and connectivity.
    
    Orchestrates the full batch packing pipeline:
    1. Concatenates all graphs into a single vertex tensor (plus CLS token per graph)
    2. Builds the CLS mask to identify CLS positions
    3. Packs and offsets edge indices from all graphs
    4. Builds base attention connectivity (edges + self-loops + edges-to-CLS)
    5. Computes transitive closure (reachability) via fixed-point iteration
    
    All returned tensors are on the target device and in the correct dtype (where applicable).
    This function is the recommended API; individual helper functions are primarily for internal use.
    
    Args:
        graphs: List of PyTorch Geometric Data objects to pack. Each must have:
                - num_nodes: Integer number of nodes
                - edge_index: Tensor of shape [2, num_edges]
        device: Target device (torch.device('cuda'), torch.device('cpu'), etc.).
        dtype: Target data type for mask/connectivity tensors (usually torch.float32 or torch.bfloat16).
    
    Returns:
        PackedGraphBatch: Container with precomputed masks, edge indices, and attention factors.
                         All tensors are on the specified device and dtype.
    
    Complexity:
        Time: O(E + V^3*D) where E=total edges, V=total vertices, D=reachability diameter
        Space: O(V^2) for the dense base_attn_mask and attn_factors matrices
    
    Example:
        >>> from torch_geometric.data import Data
        >>> g1 = Data(x=torch.randn(3, 16), edge_index=torch.tensor([[0,1],[1,0]]))
        >>> g2 = Data(x=torch.randn(2, 16), edge_index=torch.tensor([[0,1],[1,0]]))
        >>> batch = pack_graph_batch([g1, g2], torch.device('cpu'), torch.float32)
        >>> batch.cls_mask.shape
        torch.Size([7])  # 3+1 + 2+1 = 7 total (nodes + CLS tokens)
        >>> batch.base_attn_mask.shape
        torch.Size([7, 7])
    
    Warning:
        Builds dense attention masks of shape [V, V]. For very large batches (100+ nodes),
        memory usage and computation time may become prohibitive. Consider batching
        strategy or using sparse attention for large graphs.
    """
    num_nodes = []
    edge_indices = []
    for graph in graphs:
        assert graph.num_nodes is not None
        assert graph.edge_index is not None
        num_nodes.append(int(graph.num_nodes))
        edge_indices.append(graph.edge_index)

    cls_mask = build_cls_mask(num_nodes).to(device=device)
    edge_index = pack_edge_index(edge_indices, num_nodes).to(device=device)
    total_vertices = int(cls_mask.shape[0])
    base_attn_mask = torch.eye(total_vertices, device=device, dtype=dtype)
    base_attn_mask[edge_index[0], edge_index[1]] = 1.0

    offset = 0
    for count in num_nodes:
        cls_index = offset + count
        node_indices = torch.arange(offset, cls_index, device=device)
        base_attn_mask[node_indices, cls_index] = 1.0
        base_attn_mask[cls_index, node_indices] = 1.0
        offset += count + 1

    attn_factors = build_attn_factors(base_attn_mask)

    return PackedGraphBatch(
        graphs=graphs,
        num_nodes=num_nodes,
        cls_mask=cls_mask,
        edge_index=edge_index,
        base_attn_mask=base_attn_mask,
        attn_factors=attn_factors,
    )


# Example usage note:
# >>> batch = pack_graph_batch(list_of_graphs, device, dtype)
# >>> # Use batch throughout model forward pass to access precomputed masks
# >>> # GNN, Attention, and MLP layers all check for PackedGraphBatch instance
# >>> # and reuse cached masks instead of recomputing them per layer.