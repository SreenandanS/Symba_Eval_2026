"""
Relation-aware SU(3) color-flow message passing.

The color stream treats color as an internal routing symmetry, not a
geometric coordinate system. Each edge is annotated with an explicit
color-flow relation, and gluon adjoint structure is materialised in the
graph as fundamental / anti-fundamental routed edges.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .contracts import NUM_COLOR_RELATIONS, NUM_NODE_ROLES
from .lorentz_mp import unsorted_segment_sum


class ColorFlowLayer(nn.Module):
    """One round of relation-aware color transport."""

    def __init__(
        self,
        hidden_dim: int,
        num_relations: int = NUM_COLOR_RELATIONS,
        num_roles: int = NUM_NODE_ROLES,
        num_color_reps: int = 3,
        rel_emb_dim: int = 16,
        role_emb_dim: int = 8,
        color_emb_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, rel_emb_dim)
        self.role_emb = nn.Embedding(num_roles, role_emb_dim)
        self.color_emb = nn.Embedding(num_color_reps, color_emb_dim)

        edge_in = (
            hidden_dim * 2
            + rel_emb_dim
            + 2 * role_emb_dim
            + 2 * color_emb_dim
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + role_emb_dim + color_emb_dim, hidden_dim),
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
        node_color_rep: Tensor,
        node_role: Tensor,
    ) -> Tensor:
        src, dst = edge_index
        edge_input = torch.cat(
            [
                h[src],
                h[dst],
                self.rel_emb(edge_relation),
                self.role_emb(node_role[src]),
                self.role_emb(node_role[dst]),
                self.color_emb(node_color_rep[src]),
                self.color_emb(node_color_rep[dst]),
            ],
            dim=-1,
        )
        m = self.edge_mlp(edge_input)
        agg = unsorted_segment_sum(m, dst, num_segments=h.size(0))

        node_input = torch.cat(
            [
                h,
                agg,
                self.role_emb(node_role),
                self.color_emb(node_color_rep),
            ],
            dim=-1,
        )
        update = self.node_mlp(node_input)
        return self.norm(h + update)


class ColorFlowBlock(nn.Module):
    """Stack of symmetry-aware color-flow layers."""

    def __init__(
        self,
        in_dim: int | None = None,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        n_scalar: int | None = None,
        n_hidden: int | None = None,
        n_output: int | None = None,
        n_layers: int | None = None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected ColorFlowBlock kwargs: {unknown}")

        if in_dim is None:
            in_dim = n_scalar
        if in_dim is None:
            raise TypeError("ColorFlowBlock requires `in_dim` or alias `n_scalar`.")
        if n_hidden is not None:
            hidden_dim = n_hidden
        if n_output is not None:
            out_dim = n_output
        if n_layers is not None:
            num_layers = n_layers

        self.num_layers = num_layers
        self.role_emb = nn.Embedding(NUM_NODE_ROLES, 8)
        self.color_emb = nn.Embedding(3, 8)
        self.input_proj = nn.Linear(in_dim + 8 + 8, hidden_dim)
        self.layers = nn.ModuleList(
            ColorFlowLayer(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        )
        self.output_proj = (
            nn.Linear(hidden_dim, out_dim) if out_dim != hidden_dim else nn.Identity()
        )

    def encode_input(
        self,
        scalars: Tensor,
        node_color_rep: Tensor,
        node_role: Tensor,
    ) -> Tensor:
        node_input = torch.cat(
            [
                scalars,
                self.color_emb(node_color_rep),
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
        node_color_rep: Tensor,
        node_role: Tensor,
    ) -> Tensor:
        return self.layers[layer_idx](
            h=h,
            edge_index=edge_index,
            edge_relation=edge_relation,
            node_color_rep=node_color_rep,
            node_role=node_role,
        )

    def project_output(self, h: Tensor) -> Tensor:
        return self.output_proj(h)

    def forward(
        self,
        scalars: Tensor,
        edge_index: Tensor,
        edge_relation: Tensor,
        node_color_rep: Tensor,
        node_role: Tensor,
    ) -> Tensor:
        h = self.encode_input(
            scalars=scalars,
            node_color_rep=node_color_rep,
            node_role=node_role,
        )
        for layer_idx in range(self.num_layers):
            h = self.step_layer(
                layer_idx=layer_idx,
                h=h,
                edge_index=edge_index,
                edge_relation=edge_relation,
                node_color_rep=node_color_rep,
                node_role=node_role,
            )
        return self.project_output(h)
