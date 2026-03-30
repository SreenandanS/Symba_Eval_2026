"""
Static kinematic message passing for the Lorentz stream.

The design uses fixed symbolic and Minkowski descriptors derived
from graph topology and momentum signatures, so the stream respects
Lorentz structure and momentum conservation without evolving coordinates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .contracts import (
    NUM_KIN_RELATIONS,
    NUM_NODE_ROLES,
)


def signed_log1p(x: Tensor) -> Tensor:
    """Stable odd normalization for signed symbolic descriptors."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def unsorted_segment_sum(data: Tensor, segment_ids: Tensor, num_segments: int) -> Tensor:
    """Sum data elements grouped by segment ids."""
    out = data.new_zeros((num_segments, data.size(-1)))
    if data.numel() == 0:
        return out
    out.index_add_(0, segment_ids, data)
    return out


class StaticKinematicLayer(nn.Module):
    """One round of static invariant message passing."""

    def __init__(
        self,
        hidden_dim: int,
        num_relations: int = NUM_KIN_RELATIONS,
        num_roles: int = NUM_NODE_ROLES,
        num_channels: int = 4,
        rel_emb_dim: int = 16,
        role_emb_dim: int = 8,
        channel_emb_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, rel_emb_dim)
        self.role_emb = nn.Embedding(num_roles, role_emb_dim)
        self.channel_emb = nn.Embedding(num_channels, channel_emb_dim)

        edge_in = (
            hidden_dim * 2
            + 4 * 4  # src / dst / sum / |diff| signatures
            + 4      # overlap / diff norm / endpoint norms
            + 2 * 2  # src / dst mass features
            + rel_emb_dim
            + 2 * role_emb_dim
            + channel_emb_dim
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4 + 2 + role_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_relation: Tensor,
        node_momentum_signature: Tensor,
        node_role: Tensor,
        node_mass_features: Tensor,
        edge_channel: Tensor,
    ) -> Tensor:
        src, dst = edge_index
        src_sig = node_momentum_signature[src]
        dst_sig = node_momentum_signature[dst]
        sum_sig = src_sig + dst_sig
        diff_sig = dst_sig - src_sig
        abs_diff_sig = diff_sig.abs()

        overlap = (src_sig * dst_sig).sum(dim=-1, keepdim=True)
        diff_norm = (diff_sig.pow(2)).sum(dim=-1, keepdim=True)
        src_norm = (src_sig.pow(2)).sum(dim=-1, keepdim=True)
        dst_norm = (dst_sig.pow(2)).sum(dim=-1, keepdim=True)
        static_scalars = signed_log1p(
            torch.cat([overlap, diff_norm, src_norm, dst_norm], dim=-1)
        )

        edge_input = torch.cat(
            [
                h[src],
                h[dst],
                signed_log1p(src_sig),
                signed_log1p(dst_sig),
                signed_log1p(sum_sig),
                signed_log1p(abs_diff_sig),
                static_scalars,
                node_mass_features[src],
                node_mass_features[dst],
                self.rel_emb(edge_relation),
                self.role_emb(node_role[src]),
                self.role_emb(node_role[dst]),
                self.channel_emb(edge_channel),
            ],
            dim=-1,
        )
        m = self.edge_mlp(edge_input)
        agg = unsorted_segment_sum(m, dst, num_segments=h.size(0))

        node_input = torch.cat(
            [
                h,
                agg,
                signed_log1p(node_momentum_signature),
                node_mass_features,
                self.role_emb(node_role),
            ],
            dim=-1,
        )
        update = self.node_mlp(node_input)
        return self.norm(h + update)


class StaticKinematicBlock(nn.Module):
    """Static symmetry-respecting kinematic stream."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.role_emb = nn.Embedding(NUM_NODE_ROLES, 8)
        self.input_proj = nn.Linear(in_dim + 4 + 2 + 8, hidden_dim)
        self.layers = nn.ModuleList(
            StaticKinematicLayer(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        )
        self.output_proj = (
            nn.Linear(hidden_dim, out_dim) if out_dim != hidden_dim else nn.Identity()
        )

    def encode_input(
        self,
        scalars: Tensor,
        node_momentum_signature: Tensor,
        node_role: Tensor,
        node_mass_features: Tensor,
    ) -> Tensor:
        node_input = torch.cat(
            [
                scalars,
                signed_log1p(node_momentum_signature),
                node_mass_features,
                self.role_emb(node_role),
            ],
            dim=-1,
        )
        return self.input_proj(node_input)

    def step_layer(
        self,
        layer_idx: int,
        h: Tensor,
        edge_index: Tensor,
        edge_relation: Tensor,
        node_momentum_signature: Tensor,
        node_role: Tensor,
        node_mass_features: Tensor,
        edge_channel: Tensor,
    ) -> Tensor:
        return self.layers[layer_idx](
            h=h,
            edge_index=edge_index,
            edge_relation=edge_relation,
            node_momentum_signature=node_momentum_signature,
            node_role=node_role,
            node_mass_features=node_mass_features,
            edge_channel=edge_channel,
        )

    def project_output(self, h: Tensor) -> Tensor:
        return self.output_proj(h)

    def forward(
        self,
        scalars: Tensor,
        edge_index: Tensor,
        edge_relation: Tensor,
        node_momentum_signature: Tensor,
        node_role: Tensor,
        node_mass_features: Tensor,
        edge_channel: Tensor,
    ) -> Tensor:
        h = self.encode_input(
            scalars=scalars,
            node_momentum_signature=node_momentum_signature,
            node_role=node_role,
            node_mass_features=node_mass_features,
        )
        for layer_idx in range(self.num_layers):
            h = self.step_layer(
                layer_idx=layer_idx,
                h=h,
                edge_index=edge_index,
                edge_relation=edge_relation,
                node_momentum_signature=node_momentum_signature,
                node_role=node_role,
                node_mass_features=node_mass_features,
                edge_channel=edge_channel,
            )
        return self.project_output(h)


# Alias for the kinematic stream block.
LorentzBlock = StaticKinematicBlock
