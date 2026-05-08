#!/usr/bin/env python3
"""
GraphTransModelAblation — identical to GraphTransModel but with
a configurable graph-level readout for Table 5.

Four readout variants (non-code2 datasets only):
  cls      : [CLS] token embedding                     (paper default, current impl)
  mean     : global mean of all node embeddings
  last     : last node embedding per graph (index n-1)
  cls_cat  : concat([CLS], mean_nodes)  →  2×embed_dim  →  output layer

No changes to models/ are needed; this file subclasses GraphTransModel
and overrides only __init__ (for cls_cat output layer) and forward
(for the readout step).
"""

import os, sys
from typing import List

import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Data

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from models.full_model import GraphTransModel, GraphTransConfig  # noqa: E402
from models.batch_utils import pack_graph_batch                  # noqa: E402

READOUT_TYPES = ("cls", "mean", "last", "cls_cat")


class GraphTransModelAblation(GraphTransModel):
    """GraphTransModel with a swappable readout head."""

    def __init__(self, config: GraphTransConfig, readout_type: str = "cls") -> None:
        assert readout_type in READOUT_TYPES, f"readout_type must be one of {READOUT_TYPES}"
        super().__init__(config)
        self.readout_type = readout_type

        # cls_cat needs a 2× wider output layer
        if readout_type == "cls_cat":
            self.output_layer = nn.Linear(
                config.embed_dim * 2, config.y_dim
            ).to(device=config.device, dtype=config.dtype)

    # ------------------------------------------------------------------
    def forward(self, input_graphs: Data | List[Data]) -> Tensor:
        if isinstance(input_graphs, Data):
            input_graphs = [input_graphs]

        batch    = pack_graph_batch(input_graphs, self.device, self.dtype)
        total_v  = int(batch.cls_mask.shape[0])
        emb      = torch.zeros((total_v, self.config.embed_dim),
                               device=self.device, dtype=self.dtype)
        cls_ones = torch.ones((len(input_graphs), 1),
                              device=self.device, dtype=self.dtype)

        x_t = torch.cat([g.x for g in input_graphs], dim=0).to(
            device=self.device, dtype=self.dtype
        )
        emb[~batch.cls_mask] = self.input_embedding(x_t)
        emb[batch.cls_mask]  = self.cls_embedding(cls_ones)

        out_emb = self.transformer(batch, emb)   # [total_v, d]

        graph_emb = self._readout(out_emb, batch)  # [B, d] or [B, 2d]
        result = self.output_layer(graph_emb)
        assert result.shape[0] == len(input_graphs)
        return result

    # ------------------------------------------------------------------
    def _readout(self, transformer_output: Tensor, batch) -> Tensor:
        non_cls = transformer_output[~batch.cls_mask]  # [total_nodes, d]

        if self.readout_type == "cls":
            return transformer_output[batch.cls_mask]  # [B, d]

        if self.readout_type == "mean":
            parts, offset = [], 0
            for n in batch.num_nodes:
                parts.append(non_cls[offset : offset + n].mean(dim=0))
                offset += n
            return torch.stack(parts, dim=0)

        if self.readout_type == "last":
            # last non-CLS node in each graph's block
            parts, offset = [], 0
            for n in batch.num_nodes:
                parts.append(non_cls[offset + n - 1])
                offset += n
            return torch.stack(parts, dim=0)

        # cls_cat: concat([CLS], mean_nodes)  →  [B, 2d]
        cls_emb = transformer_output[batch.cls_mask]
        parts, offset = [], 0
        for n in batch.num_nodes:
            parts.append(non_cls[offset : offset + n].mean(dim=0))
            offset += n
        mean_emb = torch.stack(parts, dim=0)
        return torch.cat([cls_emb, mean_emb], dim=1)
