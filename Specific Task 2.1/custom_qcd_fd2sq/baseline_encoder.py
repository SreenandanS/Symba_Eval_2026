"""
Simple encoder baselines for factorized custom-encoder ablations.

These baselines preserve the same output contract as `CustomQCDFd2SqEncoder`
while removing the symmetry-aware message-passing structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch

from .contracts import NUM_NODE_ROLES
from .encoder import CrossStreamAttention, GraphReadout


class BaselineFeatureBlock(nn.Module):
    """Edge-blind per-node MLP used as a lightweight encoder branch."""

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        role_emb_dim: int = 8,
    ):
        super().__init__()
        self.role_emb = nn.Embedding(NUM_NODE_ROLES, role_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(node_in_dim + role_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, node_features: Tensor, node_role: Tensor) -> Tensor:
        role_features = self.role_emb(node_role.clamp(min=0))
        return self.net(torch.cat([node_features, role_features], dim=-1))


class BaselineEncoder(nn.Module):
    """Graph-contract baseline with no explicit relation-aware message passing."""

    def __init__(
        self,
        node_in_dim: int = 19,
        edge_in_dim: int = 15,
        hidden_dim: int = 128,
        stream_dim: int = 64,
        graph_dim: int = 128,
        num_mp_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        del edge_in_dim, num_mp_layers

        self.kinematic_block = BaselineFeatureBlock(
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            out_dim=stream_dim,
            dropout=dropout,
        )
        self.color_block = BaselineFeatureBlock(
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            out_dim=stream_dim,
            dropout=dropout,
        )
        self.spinor_block = BaselineFeatureBlock(
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            out_dim=stream_dim,
            dropout=dropout,
        )
        self.fusion = CrossStreamAttention(
            stream_dim=stream_dim,
            num_streams=3,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.readout = GraphReadout(
            node_dim=stream_dim,
            out_dim=graph_dim,
            dropout=dropout,
        )

    @staticmethod
    def _batch_vector(data: Data | Batch) -> Tensor:
        if hasattr(data, "batch") and data.batch is not None:
            return data.batch
        return torch.zeros(
            data.x.size(0),
            dtype=torch.long,
            device=data.x.device,
        )

    def _encode_streams(
        self,
        data: Data | Batch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = self._batch_vector(data)
        node_role = data.node_role
        node_features = data.x

        kinematic_h = self.kinematic_block(node_features, node_role)
        color_h = self.color_block(node_features, node_role)
        spinor_h = self.spinor_block(node_features, node_role)
        fused = self.fusion(kinematic_h, color_h, spinor_h)
        return fused, batch, kinematic_h, color_h, spinor_h

    def forward(self, data: Data | Batch) -> Tensor:
        fused, batch, *_ = self._encode_streams(data)
        node_type_mask = data.x[:, -3:]
        return self.readout(fused, batch, node_type_mask)

    def encode_nodes(self, data: Data | Batch) -> tuple[Tensor, Tensor]:
        fused, batch, *_ = self._encode_streams(data)
        memory, mask = to_dense_batch(fused, batch)
        return memory, mask

    def encode_context(self, data: Data | Batch) -> dict[str, Tensor]:
        fused, batch, kinematic_h, color_h, spinor_h = self._encode_streams(data)
        memory, mask = to_dense_batch(fused, batch)
        return {
            "memory": memory,
            "memory_mask": mask,
            "batch": batch,
            "kinematic": kinematic_h,
            "color": color_h,
            "spinor": spinor_h,
            "fused": fused,
        }

    def get_stream_embeddings(self, data: Data | Batch) -> dict[str, Tensor]:
        fused, _batch, kinematic_h, color_h, spinor_h = self._encode_streams(data)
        return {
            "kinematic": kinematic_h,
            "lorentz": kinematic_h,
            "color": color_h,
            "spinor": spinor_h,
            "fused": fused,
        }
