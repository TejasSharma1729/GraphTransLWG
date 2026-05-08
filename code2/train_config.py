#!/usr/bin/env python3
"""
Factory that builds GraphTransConfig + ModelTrainConfig for ogbg-code2.

Usage
-----
    from code2.train_config import get_code2_train_config
    config, train_config = get_code2_train_config()
"""

import os, sys
from typing import List

import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Data

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from code2.dataset import (          # noqa: E402
    build_code2_dataset, decode_arr_to_seq, MAX_SEQ_LEN
)
from models.full_model import GraphTransConfig, ModelTrainConfig, GraphTransModel  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Loss / metric / mapping helpers
# ─────────────────────────────────────────────────────────────────────────────

def code2_out_mapping_fn(samples: List[Data]) -> Tensor:
    """Stack y_arr tensors into [B, MAX_SEQ_LEN]."""
    return torch.cat([s.y_arr for s in samples], dim=0)


def code2_ref_seqs(samples: List[Data]) -> List[List[str]]:
    """Return original token-sequence labels (data.y) — used for F1, not loss."""
    return [list(s.y) for s in samples]


def code2_loss_fn(output: Tensor, y_arr: Tensor) -> Tensor:
    """
    Cross-entropy averaged over all sequence positions.
    output : [B, max_seq_len, num_vocab]
    y_arr  : [B, max_seq_len]  int64
    """
    B, S, V = output.shape
    return F.cross_entropy(
        output.reshape(B * S, V),
        y_arr.to(output.device).reshape(B * S).long(),
    )


def _make_metric_fn(idx2vocab: List[str]):
    """Return metric_fn(output, y_arr) -> float  computing F1 score."""
    try:
        from ogb.graphproppred import Evaluator
        _ev = Evaluator("ogbg-code2")

        def _f1(preds, refs):
            return float(_ev.eval({"seq_ref": refs, "seq_pred": preds})["F1"])

    except Exception:
        # Fallback: token-overlap F1 (no OGB dependency)
        def _f1(preds, refs):
            total = 0.0
            for p, r in zip(preds, refs):
                ps, rs = set(p), set(r)
                if not ps and not rs:
                    total += 1.0
                elif ps and rs:
                    inter = len(ps & rs)
                    pr = inter / len(ps)
                    rc = inter / len(rs)
                    total += 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            return total / max(len(preds), 1)

    _iv = idx2vocab

    def metric_fn(output: Tensor, y_arr: Tensor) -> float:
        # Decode model predictions using training vocab
        preds_idx = output.detach().cpu().argmax(dim=-1)   # [B, S]
        seq_pred = [decode_arr_to_seq(preds_idx[i], _iv) for i in range(len(preds_idx))]
        # References: decode from y_arr using vocab (same as predictions space)
        # Note: OOV tokens in val/test are encoded as __UNK__ in y_arr.
        # Using y_arr-decoded refs keeps pred/ref in the same vocab space,
        # which gives meaningful relative comparisons across ablation rows.
        refs_idx = y_arr.detach().cpu()
        seq_ref  = [decode_arr_to_seq(refs_idx[i], _iv) for i in range(len(refs_idx))]
        return _f1(seq_pred, seq_ref)

    return metric_fn


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────

def get_code2_train_config(
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    # model
    embed_dim: int = 300,
    num_transformer_layers: int = 4,
    num_gnn_layers: int = 4,
    num_heads: int = 6,
    head_dim: int = 50,
    num_mlp_layers: int = 2,
    mlp_hidden_dim: int = 600,
    dropout: float = 0.3,
    # training
    num_epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    runs: int = 5,
    max_graphs: int | None = None,
):
    """
    Build GraphTransConfig + ModelTrainConfig for ogbg-code2.
    Downloads and preprocesses the dataset on the first call.

    Hyperparameters follow the paper:
      4 transformer layers, embed_dim=300, 4 GNN layers,
      dropout=0.3, 30 epochs, batch_size=16, lr=1e-4, 5 runs.
    """
    if device is None:
        import torch.cuda as _cuda, torch.mps as _mps
        device = torch.device(
            "cuda" if _cuda.is_available() else
            "mps"  if _mps.is_available()  else "cpu"
        )

    (
        dataset, train_idx, val_idx, test_idx,
        vocab2idx, idx2vocab,

        num_nodetypes, num_nodeattributes,
    ) = build_code2_dataset(max_graphs=max_graphs)

    num_vocab = len(vocab2idx)  # NUM_VOCAB + 2

    config = GraphTransConfig(
        x_dim=2,                          # not used – ASTNodeEncoder takes int x
        num_transformer_layers=num_transformer_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        y_dim=num_vocab,                  # not used – seq heads take over
        num_gnn_layers=num_gnn_layers,
        attn_distance_factors=None,
        num_mlp_layers=num_mlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout=dropout,
        num_node_types=num_nodetypes,
        num_node_attrs=num_nodeattributes,
        max_node_depth=20,
        max_seq_len=MAX_SEQ_LEN,
        num_vocab=num_vocab,
        device=device,
        dtype=dtype,
    )

    metric_fn = _make_metric_fn(idx2vocab)
    _ds = dataset   # keep reference alive in lambda closures

    train_config = ModelTrainConfig(
        model_config=config,
        model_loader=lambda: GraphTransModel(config),
        dataset_loader=lambda: _ds,
        out_mapping_fn=code2_out_mapping_fn,
        loss_fn=code2_loss_fn,
        metric_fn=metric_fn,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler="cosine",
        runs=runs,
        fixed_split_indices=(train_idx, val_idx, test_idx),
    )

    return config, train_config
