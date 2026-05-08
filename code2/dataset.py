#!/usr/bin/env python3
"""
ogbg-code2 dataset preprocessing.

Provides:
  - get_vocab_mapping   : build vocab from training sequences
  - encode_y_to_arr     : encode target string list → fixed-length int tensor
  - decode_arr_to_seq   : decode int tensor → string list
  - augment_edge        : add inverse + next-token edges to AST graphs
  - build_code2_dataset : full pipeline (download → vocab → transform)

All utilities are standalone (no loguru / torchvision dependency).
"""

import os, sys
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data

# reach root so absolute imports work when this file is run directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from data_utils.extract_datasets import get_graph_dataset  # noqa: E402

NUM_VOCAB   = 5000
MAX_SEQ_LEN = 5


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_vocab_mapping(seq_list: List[List[str]], num_vocab: int) -> Tuple[Dict[str, int], List[str]]:
    """
    Build vocab2idx / idx2vocab from a list of token sequences.
    Top-num_vocab most frequent tokens are kept; '__UNK__' and '__EOS__' are appended.
    """
    cnt: Dict[str, int] = {}
    order: List[str] = []
    for seq in seq_list:
        for w in seq:
            if w not in cnt:
                cnt[w] = 0
                order.append(w)
            cnt[w] += 1

    top = sorted(order, key=lambda w: -cnt[w])[:num_vocab]
    vocab2idx: Dict[str, int] = {w: i for i, w in enumerate(top)}
    idx2vocab: List[str] = top[:]
    vocab2idx["__UNK__"] = num_vocab
    idx2vocab.append("__UNK__")
    vocab2idx["__EOS__"] = num_vocab + 1
    idx2vocab.append("__EOS__")
    return vocab2idx, idx2vocab


def encode_y_to_arr(data: Data, vocab2idx: Dict[str, int], max_seq_len: int) -> Data:
    """Add data.y_arr: [1, max_seq_len] int64 tensor encoding data.y (list of strings)."""
    seq = list(data.y)
    padded = seq[:max_seq_len] + ["__EOS__"] * max(0, max_seq_len - len(seq))
    data.y_arr = torch.tensor(
        [[vocab2idx.get(w, vocab2idx["__UNK__"]) for w in padded]],
        dtype=torch.long,
    )
    return data


def decode_arr_to_seq(arr: Tensor, idx2vocab: List[str]) -> List[str]:
    """Decode int tensor to token list, stopping at __EOS__."""
    eos = len(idx2vocab) - 1
    pos = (arr == eos).nonzero()
    if len(pos) > 0:
        arr = arr[: int(pos[0].item())]
    return [idx2vocab[int(i)] for i in arr.cpu()]


# ─────────────────────────────────────────────────────────────────────────────
# Edge augmentation  (matches reference GraphTrans implementation)
# ─────────────────────────────────────────────────────────────────────────────

def augment_edge(data: Data) -> Data:
    """
    Augment AST edge_index with:
      - inverse AST edges
      - next-token edges between attributed nodes (DFS order)
      - inverse next-token edges
    Also adds data.edge_attr  [E, 2]  (edge_type, direction).
    """
    ei = data.edge_index  # [2, E_ast]

    ea_ast = torch.zeros((ei.size(1), 2), dtype=torch.long)
    ei_inv = torch.stack([ei[1], ei[0]], dim=0)
    ea_inv = torch.zeros((ei_inv.size(1), 2), dtype=torch.long)
    ea_inv[:, 1] = 1  # inverse direction flag

    attr_nodes = torch.where(data.node_is_attributed.view(-1) == 1)[0]
    if attr_nodes.numel() > 1:
        ei_next     = torch.stack([attr_nodes[:-1], attr_nodes[1:]], dim=0)
        ea_next     = torch.zeros((ei_next.size(1), 2), dtype=torch.long)
        ea_next[:, 0] = 1  # next-token edge type flag
        ei_next_inv = torch.stack([ei_next[1], ei_next[0]], dim=0)
        ea_next_inv = torch.ones((ei_next.size(1), 2), dtype=torch.long)

        data.edge_index = torch.cat([ei, ei_inv, ei_next, ei_next_inv], dim=1)
        data.edge_attr  = torch.cat([ea_ast, ea_inv, ea_next, ea_next_inv], dim=0)
    else:
        data.edge_index = torch.cat([ei, ei_inv], dim=1)
        data.edge_attr  = torch.cat([ea_ast, ea_inv], dim=0)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Full dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_code2_dataset(max_graphs: int | None = None):
    """
    Download (or load cached) ogbg-code2, build vocabulary from training set,
    and register the on-the-fly transform.

    Parameters
    ----------
    max_graphs : int | None
        If set, subsample the dataset to at most this many graphs total,
        preserving the train/val/test ratio of the OGB split.
        Useful for smoke-testing or memory-constrained runs (e.g. max_graphs=4000).

    Returns
    -------
    dataset              PygGraphPropPredDataset with .transform set
    train_idx            list[int]
    val_idx              list[int]
    test_idx             list[int]
    vocab2idx            dict[str, int]
    idx2vocab            list[str]
    num_nodetypes        int
    num_nodeattributes   int
    """
    dataset   = get_graph_dataset("ogbg-code2")
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"].tolist()
    val_idx   = split_idx["valid"].tolist()
    test_idx  = split_idx["test"].tolist()

    # ── optional subsampling ──────────────────────────────────────────────────
    if max_graphs is not None:
        total = len(train_idx) + len(val_idx) + len(test_idx)
        frac  = max_graphs / total
        train_idx = train_idx[: max(1, int(len(train_idx) * frac))]
        val_idx   = val_idx  [: max(1, int(len(val_idx)   * frac))]
        test_idx  = test_idx [: max(1, int(len(test_idx)  * frac))]
        print(f"[code2] Subsampled to {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    else:
        print(f"[code2] {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    print("[code2] Building vocabulary from training sequences...")
    train_seqs = [dataset[i].y for i in train_idx]
    vocab2idx, idx2vocab = get_vocab_mapping(train_seqs, NUM_VOCAB)
    print(f"[code2] Vocab size: {len(vocab2idx)} (incl. __UNK__, __EOS__)")

    ds_root = dataset.root
    nodetypes_df      = pd.read_csv(os.path.join(ds_root, "mapping", "typeidx2type.csv.gz"))
    nodeattributes_df = pd.read_csv(os.path.join(ds_root, "mapping", "attridx2attr.csv.gz"))
    num_nodetypes      = len(nodetypes_df["type"])
    num_nodeattributes = len(nodeattributes_df["attr"])
    print(f"[code2] Node types: {num_nodetypes}, Node attrs: {num_nodeattributes}")

    _v2i = vocab2idx   # capture in closure
    def _transform(data: Data) -> Data:
        data = augment_edge(data)
        data = encode_y_to_arr(data, _v2i, MAX_SEQ_LEN)
        return data

    dataset.transform = _transform

    return (
        dataset, train_idx, val_idx, test_idx,
        vocab2idx, idx2vocab,
        num_nodetypes, num_nodeattributes,
    )
