#!/usr/bin/env python3
"""
GNNOnlyModel — GNN stack + global mean pooling, no attention, no MLP.

This is the pre-trained GNN baseline row in Table 4.
For code2: ASTNodeEncoder → GNN → mean pool → max_seq_len output heads.
For other datasets: nn.Linear → GNN → mean pool → output layer.
"""

import os, sys
from typing import List

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # GraphTransLWG/
sys.path.insert(0, _ROOT)

from torch_geometric.data import Data                          # noqa: E402
from models.gnn import GNN                                     # noqa: E402
from models.batch_utils import pack_graph_batch                # noqa: E402
from models.full_model import GraphTransConfig, ASTNodeEncoder # noqa: E402


class GNNOnlyModel(Module):
    """
    GNN-only baseline (no attention, no MLP transformer layers).

    Node embeddings are computed by a GNN stack. Graph-level
    representation is obtained by global mean pooling over all
    non-CLS node embeddings.

    For code2, uses ASTNodeEncoder and max_seq_len prediction heads.
    For other datasets, uses a linear input encoder and a single output layer.
    """

    def __init__(self, config: GraphTransConfig) -> None:
        super().__init__()
        self.config = config
        self._is_code2: bool = (
            config.num_node_types is not None and config.max_seq_len is not None
        )

        # ── input encoder ─────────────────────────────────────────────────────
        if self._is_code2:
            assert config.num_node_types and config.num_node_attrs
            self.input_embedding: Module = ASTNodeEncoder(
                config.embed_dim,
                config.num_node_types,
                config.num_node_attrs,
                config.max_node_depth,
            )
        else:
            self.input_embedding = nn.Linear(config.x_dim, config.embed_dim)

        # ── GNN ───────────────────────────────────────────────────────────────
        num_layers = (
            config.num_gnn_layers
            if isinstance(config.num_gnn_layers, int)
            else config.num_gnn_layers[0]
        )
        self.gnn = GNN(config.embed_dim, num_layers, config.device, config.dtype)

        # ── output head(s) ────────────────────────────────────────────────────
        if self._is_code2:
            assert config.max_seq_len and config.num_vocab
            self.output_layer: Module = nn.ModuleList([
                nn.Linear(config.embed_dim, config.num_vocab)
                for _ in range(config.max_seq_len)
            ])
        else:
            self.output_layer = nn.Linear(config.embed_dim, config.y_dim)

        self.device = config.device
        self.dtype  = config.dtype
        self.to(config.device)
        self.to(config.dtype)

    def forward(self, input_graphs: Data | List[Data]) -> Tensor:
        if isinstance(input_graphs, Data):
            input_graphs = [input_graphs]

        batch           = pack_graph_batch(input_graphs, self.device, self.dtype)
        total_v         = int(batch.cls_mask.shape[0])
        input_emb       = torch.zeros(
            (total_v, self.config.embed_dim), device=self.device, dtype=self.dtype
        )

        # Fill in node embeddings (CLS positions stay 0 — GNN won't touch them)
        if self._is_code2:
            x_int   = torch.cat([g.x for g in input_graphs], dim=0).long().to(self.device)
            x_depth = torch.cat(
                [g.node_depth.view(-1) for g in input_graphs], dim=0
            ).long().to(self.device)
            input_emb[~batch.cls_mask] = self.input_embedding(x_int, x_depth).to(self.dtype)
        else:
            x_t = torch.cat([g.x for g in input_graphs], dim=0).to(
                device=self.device, dtype=self.dtype
            )
            input_emb[~batch.cls_mask] = self.input_embedding(x_t)

        # GNN forward (returns 0 for CLS positions)
        gnn_out     = self.gnn(batch, input_emb)       # [total_v, embed_dim]
        non_cls_out = gnn_out[~batch.cls_mask]          # [total_nodes, embed_dim]

        # Global mean pool per graph
        graph_emb_list: List[Tensor] = []
        offset = 0
        for n in batch.num_nodes:
            graph_emb_list.append(non_cls_out[offset : offset + n].mean(dim=0))
            offset += n
        graph_emb = torch.stack(graph_emb_list, dim=0)  # [B, embed_dim]

        if self._is_code2:
            assert isinstance(self.output_layer, nn.ModuleList)
            return torch.stack([h(graph_emb) for h in self.output_layer], dim=1)
        else:
            return self.output_layer(graph_emb)  # type: ignore[return-value]
