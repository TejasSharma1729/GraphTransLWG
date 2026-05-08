#!/usr/bin/env python3

from typing import List, Tuple, Dict, Set, Iterable, Callable, Literal, Optional, Any, Union
from dataclasses import dataclass, field
import sys, os, gc
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
sys.path.append(ROOT_DIR)

import torch
from torch import nn, tensor, Tensor, autograd, optim, cuda, mps, cpu, distributions
from torch.nn import Module, Parameter, ModuleList, ModuleDict, functional as F
from torch.optim import Optimizer, Adam, AdamW, SGD, RMSprop, lr_scheduler
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv, GINConv, GIN
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, to_dense_batch, coalesce

from models.gnn import GNNLayer, GNN
from models.attention import AttentionLayer
from models.mlp import MLP
from models.transformer import TransformerLayer, Transformer
from models.batch_utils import PackedGraphBatch, pack_graph_batch

TORCH_DEVICE: str = "cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu" # type: ignore


class ASTNodeEncoder(Module):
    """
    Encodes AST node features (type index, attribute index, depth) into a dense vector
    using three separate embedding tables, then sums them.

    Used for ogbg-code2 where x[:,0]=node_type, x[:,1]=node_attr, and node_depth is
    a separate integer tensor.
    """
    def __init__(self, embed_dim: int, num_nodetypes: int, num_nodeattributes: int, max_depth: int = 20) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.type_encoder      = nn.Embedding(num_nodetypes,      embed_dim)
        self.attribute_encoder = nn.Embedding(num_nodeattributes, embed_dim)
        self.depth_encoder     = nn.Embedding(max_depth + 1,      embed_dim)

    def forward(self, x: Tensor, depth: Tensor) -> Tensor:
        depth_clipped = depth.clamp(max=self.max_depth)
        return (self.type_encoder(x[:, 0])
                + self.attribute_encoder(x[:, 1])
                + self.depth_encoder(depth_clipped))


@dataclass
class GraphTransConfig:
    """
    Class that stores the configuration for the GraphTransModel. 
    This is used to initialize the model, and also to store the hyperparameters for the model.

    Note that instances of this must be created for each dataset (and pertain to a GNN for the dataset).

    Args:
        x_dim: The dimension of the input vertex features
        num_transformer_layers: The number of transformer layers in the transformer module
        embed_dim: The embedding dimension
        num_heads: The number of attention heads in the attention layers
        head_dim: The dimension of each attention head in the attention layers
        y_dim: The dimension of the output features
        num_gnn_layers: The number of gnn layers in each transformer layer of the transformer module
        attn_distance_factors: The attention distance factors for each transformer layer of the transformer module (hyperparameters for weighing attention)
        num_mlp_layers: The number of mlp layers in each transformer layer of the transformer module
        mlp_hidden_dim: Hidden dimension inside the transformer feedforward/MLP subnetwork.
        dropout: Dropout used in transformer attention and residual branches.
        device: The device to run the model on
        dtype: The data type to use for the model (default: torch.float32)
    """
    x_dim: int = 2  
    num_transformer_layers: int = 6
    embed_dim: int = 256
    num_heads: int = 8
    head_dim: int = 32
    y_dim: int = 10
    num_gnn_layers: int | List[int] = 2
    attn_distance_factors: List[List[float] | None] | None = None
    num_mlp_layers: int | List[int] = 2
    mlp_hidden_dim: int | None = None
    dropout: float = 0.0
    # ── code2 / sequence-prediction mode ──────────────────────────────────────
    num_node_types: int | None = None   # set for code2; enables ASTNodeEncoder
    num_node_attrs: int | None = None
    max_node_depth: int = 20
    max_seq_len: int | None = None      # set for code2; enables multi-head output
    num_vocab: int | None = None        # vocabulary size (num_vocab output heads)
    # ──────────────────────────────────────────────────────────────────────────
    device: torch.device = torch.device(TORCH_DEVICE)
    dtype: torch.dtype = torch.float32



class GraphTransModel(Module):
    """
    The full Graph Transformer model, consisting of the following components:
    - An input embedding layer, which is a linear transformation from the input vertex features to the input embeddings.
    - A CLS token embedding, which is a learnable parameter that serves as the embedding for the CLS token for each graph.
    - A transformer module, consisting of multiple transformer layers. Each transformer layer consists of a GNN layer, followed by an attention layer, then an MLP layer.
    - The output layer, which is a linear transformation from the output embedding of the CLS token to the output features.

    Note that the GNNs do nothing to the CLS embeddings, and MLPs (like first input embeddings) are seperate for CLS and non-CLS tokens.
    But the attention is common for CLS and non-CLS tokens.
    """
    def __init__(
            self,
            config: GraphTransConfig
    ) -> None:
        """
        Initialize the GraphTransModel given the configuration.

        Args:
            config: The configuration for the model, as a GraphTransConfig object (required).
        """
        super().__init__()
        self.config: GraphTransConfig = config
        self._is_code2: bool = (config.num_node_types is not None and config.max_seq_len is not None)

        if self._is_code2:
            assert config.num_node_types is not None and config.num_node_attrs is not None
            assert config.max_seq_len    is not None and config.num_vocab      is not None
            self.input_embedding: Module = ASTNodeEncoder(
                config.embed_dim, config.num_node_types, config.num_node_attrs, config.max_node_depth
            )
            self.output_layer: Module = nn.ModuleList([
                nn.Linear(config.embed_dim, config.num_vocab)
                for _ in range(config.max_seq_len)
            ])
        else:
            self.input_embedding = nn.Linear(config.x_dim, config.embed_dim)
            self.output_layer    = nn.Linear(config.embed_dim, config.y_dim)

        self.cls_embedding = nn.Linear(1, config.embed_dim).to(device=config.device, dtype=config.dtype)
        
        self.transformer = Transformer(
            config.num_transformer_layers,
            config.embed_dim,
            config.num_heads,
            config.head_dim,
            config.num_gnn_layers,
            config.attn_distance_factors,
            config.num_mlp_layers,
            config.mlp_hidden_dim,
            config.dropout,
            config.device,
            config.dtype
        ) # transformer module consisting of multiple transformer layers

        self.output_layer = nn.Linear(config.embed_dim, config.y_dim).to(device=config.device, dtype=config.dtype)
        # linear transformation for output CLS embedding to output features

        self.device = config.device # device to run the model on
        self.dtype = config.dtype # data type to use for the model
        self.to(config.device) # move the model to the device
        self.to(config.dtype) # move the model to the data type
    
    def forward(
            self,
            input_graphs: Data | List[Data],
    ) -> Tensor:
        """
        Forward pass for a batch of graphs or a single graph.
        This computes the output features for all vertices of all graphs in the batch, in order.

        It first computes the input embeddings from the input vertex features,
        then applies the transformer to get the output embeddings, 
        and finally applies the output layer to get the output features.

        Args:
            input_graphs: The input graphs (or a single graph)
        
        Returns:
            The output features (Tensor).
        """
        if isinstance(input_graphs, Data):
            input_graphs = [input_graphs]
        assert isinstance(input_graphs, list)

        batch: PackedGraphBatch = pack_graph_batch(input_graphs, self.device, self.dtype)
        net_num_vertices = int(batch.cls_mask.shape[0])
        input_embeddings: Tensor = torch.zeros((net_num_vertices, self.config.embed_dim), device=self.device, dtype=self.dtype)
        cls_tensor: Tensor = torch.ones((len(input_graphs), 1), device=self.device, dtype=self.dtype)

        # Node embeddings: use ASTNodeEncoder for code2, else nn.Linear
        if self._is_code2:
            x_int   = torch.cat([g.x          for g in input_graphs], dim=0).long().to(self.device)
            x_depth = torch.cat([g.node_depth.view(-1) for g in input_graphs], dim=0).long().to(self.device)
            input_embeddings[~batch.cls_mask] = self.input_embedding(x_int, x_depth).to(self.dtype)
        else:
            x_tensor = torch.cat([g.x for g in input_graphs], dim=0).to(device=self.device, dtype=self.dtype)
            input_embeddings[~batch.cls_mask] = self.input_embedding(x_tensor)

        input_embeddings[batch.cls_mask] = self.cls_embedding(cls_tensor)

        transformer_output: Tensor = self.transformer(batch, input_embeddings)
        cls_emb: Tensor = transformer_output[batch.cls_mask]  # [num_graphs, embed_dim]

        if self._is_code2:
            # Stack max_seq_len prediction heads → [num_graphs, max_seq_len, num_vocab]
            assert isinstance(self.output_layer, nn.ModuleList)
            return torch.stack([head(cls_emb) for head in self.output_layer], dim=1)
        else:
            out_embeddings = self.output_layer(cls_emb)
            assert out_embeddings.shape == torch.Size([len(input_graphs), self.config.y_dim])
            return out_embeddings
    

@dataclass
class ModelTrainConfig:
    """
    Class that stores the configuration for training the GraphTransModel on a dataset. 
    This includes the model loader, dataset loader, output mapping function, loss function, and training hyperparameters.

    Args:
        model_config: The configuration for the GraphTransModel, as a GraphTransConfig object (required).
        model_loader: A function that returns an instance of the GraphTransModel to train (required).
        dataset_loader: A function that returns a dataset of graphs to train on (required).
        out_mapping_fn: A function that maps the output of the model to the ground truth output features for the loss function (required).
        loss_fn: The loss function to use for training (required).
        metric_fn: The metric function to use for validation/testing. Higher is better.
        num_epochs: The number of epochs to train for (default: 1)
        batch_size: The batch size to use for training (default: 32)
        learning_rate: The learning rate to use for training (default: 0.001)
        weight_decay: Adam weight decay.
        scheduler: Optional learning rate scheduler name.
        runs: Number of repeated random-seed runs.
        train_ratio: Fraction of the dataset used for training.
        val_ratio: Fraction of the dataset used for validation.
        random_seed: Seed for reproducible train/validation/test splits.
    """
    model_config: GraphTransConfig
    model_loader: Callable[[], GraphTransModel]
    dataset_loader: Callable[[], TorchDataset[Data]]
    out_mapping_fn: Callable[[Any], Tensor]
    loss_fn: Callable[[Tensor, Tensor], Tensor]
    metric_fn: Callable[[Tensor, Tensor], float] | None = None
    num_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    scheduler: str | None = None
    runs: int = 1
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    random_seed: int = 12344
    fixed_split_indices: tuple[List[int], List[int], List[int]] | None = None
