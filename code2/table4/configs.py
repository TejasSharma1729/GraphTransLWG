#!/usr/bin/env python3
"""
Config factories for Table 4's three rows.

Row 1 – GNN-only (no transformer):
    config, kw = get_gnn_only_config(dataset_info)

Row 2 – GraphTrans with frozen GNN (pre-trained GNN weights, GNN layers frozen):
    config, kw = get_frozen_gnn_config(dataset_info, pretrained_ckpt)

Row 3 – GraphTrans with fine-tuned GNN (pre-trained GNN weights, all trained):
    config, kw = get_finetune_gnn_config(dataset_info, pretrained_ckpt)

`kw` is a dict that can be unpacked directly into run_experiment().
"""

import os, sys
from typing import Any, Dict, List

import torch
from torch.nn import Module

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # GraphTransLWG/
sys.path.insert(0, _ROOT)

from models.full_model import GraphTransConfig, GraphTransModel   # noqa: E402
from code2.table4.gnn_only_model import GNNOnlyModel             # noqa: E402
from code2.train_config import (                                   # noqa: E402
    code2_out_mapping_fn, code2_loss_fn, _make_metric_fn,
)


# ─────────────────────────────────────────────────────────────────────────────
# GNN weight transfer helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_gnn_weights_into_graphtrans(
    graphtrans_model: GraphTransModel,
    gnn_only_ckpt: str,
) -> None:
    """
    Copy GNNOnlyModel's .gnn weights into every transformer layer's .gnn_layer.

    GNNOnlyModel saves state as:  gnn.layers.{k}.node_weights.weight  ...
    GraphTransModel expects:       transformer.layers.{i}.gnn_layer.layers.{k}...

    All transformer layers are initialised with the same pre-trained GNN weights.
    """
    ckpt  = torch.load(gnn_only_ckpt, map_location="cpu")
    state = ckpt["model_state_dict"]

    # Extract gnn.* keys, strip 'gnn.' prefix
    gnn_state = {k[4:]: v for k, v in state.items() if k.startswith("gnn.")}

    for i, layer in enumerate(graphtrans_model.transformer.layers):
        missing, unexpected = layer.gnn_layer.load_state_dict(gnn_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Layer {i} GNN weight mismatch: missing={missing}, unexpected={unexpected}"
            )

    print(
        f"Loaded GNN weights from '{gnn_only_ckpt}' "
        f"into {len(graphtrans_model.transformer.layers)} transformer layers.",
        flush=True,
    )


def freeze_gnn_layers(model: GraphTransModel) -> None:
    """Freeze all GNN sub-layer parameters in a GraphTransModel."""
    frozen = 0
    for layer in model.transformer.layers:
        for param in layer.gnn_layer.parameters():
            param.requires_grad = False
            frozen += param.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen {frozen:,} GNN params. Trainable: {trainable:,}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Config builders
# ─────────────────────────────────────────────────────────────────────────────

def _base_config(dataset_info: Dict[str, Any]) -> GraphTransConfig:
    """Build the shared GraphTransConfig for all three rows."""
    return GraphTransConfig(
        x_dim=2,
        num_transformer_layers=4,
        embed_dim=300,
        num_heads=6,
        head_dim=50,
        y_dim=dataset_info["num_vocab"],
        num_gnn_layers=4,
        attn_distance_factors=None,
        num_mlp_layers=2,
        mlp_hidden_dim=600,
        dropout=0.3,
        num_node_types=dataset_info["num_nodetypes"],
        num_node_attrs=dataset_info["num_nodeattributes"],
        max_node_depth=20,
        max_seq_len=5,
        num_vocab=dataset_info["num_vocab"],
        device=dataset_info["device"],
        dtype=dataset_info["dtype"],
    )


def _common_kw(
    config: GraphTransConfig,
    model_factory,
    dataset_info: Dict[str, Any],
    num_epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-4,
    wd: float = 0.0,
    runs: int = 5,
) -> Dict[str, Any]:
    """Build the keyword dict for run_experiment()."""
    idx2vocab = dataset_info["idx2vocab"]
    return dict(
        model_factory   = model_factory,
        device          = config.device,
        dataset_factory = lambda: dataset_info["dataset"],
        train_idx       = dataset_info["train_idx"],
        val_idx         = dataset_info["val_idx"],
        test_idx        = dataset_info["test_idx"],
        out_mapping_fn  = code2_out_mapping_fn,
        loss_fn         = code2_loss_fn,
        metric_fn       = _make_metric_fn(idx2vocab),
        num_epochs      = num_epochs,
        batch_size      = batch_size,
        learning_rate   = lr,
        weight_decay    = wd,
        scheduler_name  = "cosine",
        base_seed       = 12344,
        runs            = runs,
    )


def get_gnn_only_config(
    dataset_info: Dict[str, Any],
    num_epochs: int = 30,
    batch_size: int = 16,
    runs: int = 5,
) -> Dict[str, Any]:
    """
    Row 1 — GNN-only baseline.
    No transformer layers; uses global mean pooling.
    """
    config = _base_config(dataset_info)
    return _common_kw(
        config, lambda: GNNOnlyModel(config),
        dataset_info, num_epochs, batch_size, runs=runs,
    )


def get_frozen_gnn_config(
    dataset_info: Dict[str, Any],
    pretrained_ckpt: str,
    num_epochs: int = 30,
    batch_size: int = 16,
    runs: int = 5,
) -> Dict[str, Any]:
    """
    Row 2 — GraphTrans with pre-trained + frozen GNN.
    GNN weights loaded from gnn_only checkpoint; GNN params frozen.
    Only attention, MLP, and output heads are trained.
    """
    config = _base_config(dataset_info)

    def model_factory() -> Module:
        model = GraphTransModel(config)
        load_gnn_weights_into_graphtrans(model, pretrained_ckpt)
        freeze_gnn_layers(model)
        return model

    return _common_kw(config, model_factory, dataset_info, num_epochs, batch_size, runs=runs)


def get_finetune_gnn_config(
    dataset_info: Dict[str, Any],
    pretrained_ckpt: str,
    num_epochs: int = 30,
    batch_size: int = 16,
    runs: int = 5,
) -> Dict[str, Any]:
    """
    Row 3 — GraphTrans with pre-trained + fine-tuned GNN.
    GNN weights loaded from gnn_only checkpoint; everything is trained.
    """
    config = _base_config(dataset_info)

    def model_factory() -> Module:
        model = GraphTransModel(config)
        load_gnn_weights_into_graphtrans(model, pretrained_ckpt)
        return model

    return _common_kw(config, model_factory, dataset_info, num_epochs, batch_size, runs=runs)
