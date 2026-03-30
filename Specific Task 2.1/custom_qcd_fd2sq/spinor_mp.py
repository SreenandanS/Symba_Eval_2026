"""
Directed fermion-line transport with typed interaction mixing.

The spinor stream is built around explicit fermion lines. Messages move
along annotated fermion-flow edges, while a separate interaction phase
mixes local qqg / ggg context at nodes without inventing a fake bosonic
spinor transport channel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .contracts import (
    NUM_NODE_ROLES,
    NUM_SPINOR_RELATIONS,
    NUM_VERTEX_INTERACTIONS,
    SPINOR_REL_BOSON,
    SPINOR_REL_FERMION_AGAINST,
    SPINOR_REL_FERMION_ALONG,
)
from .lorentz_mp import unsorted_segment_sum


class SpinorFlowLayer(nn.Module):
    """One round of fermion transport followed by vertex interaction mixing."""

    def __init__(
        self,
        hidden_dim: int,
        num_relations: int = NUM_SPINOR_RELATIONS,
        num_roles: int = NUM_NODE_ROLES,
        num_vertex_types: int = NUM_VERTEX_INTERACTIONS,
        num_fermion_numbers: int = 3,
        rel_emb_dim: int = 16,
        role_emb_dim: int = 8,
        vertex_emb_dim: int = 8,
        fermion_emb_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, rel_emb_dim)
        self.role_emb = nn.Embedding(num_roles, role_emb_dim)
        self.vertex_emb = nn.Embedding(num_vertex_types, vertex_emb_dim)
        self.fermion_emb = nn.Embedding(num_fermion_numbers, fermion_emb_dim)

        transport_in = (
            hidden_dim * 2
            + rel_emb_dim
            + 2 * role_emb_dim
            + 2 * fermion_emb_dim
            + 1
        )
        interaction_in = (
            hidden_dim * 2
            + rel_emb_dim
            + 2 * role_emb_dim
            + 2 * vertex_emb_dim
            + 2 * fermion_emb_dim
        )
        self.transport_mlp = nn.Sequential(
            nn.Linear(transport_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.interaction_mlp = nn.Sequential(
            nn.Linear(interaction_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.transport_norm = nn.LayerNorm(hidden_dim)
        self.interaction_norm = nn.LayerNorm(hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(
                hidden_dim * 4 + role_emb_dim + vertex_emb_dim + fermion_emb_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _fermion_index(node_fermion_num: Tensor) -> Tensor:
        return (node_fermion_num + 1).clamp_(0, 2)

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_relation: Tensor,
        fermion_line_id: Tensor,
        node_fermion_num: Tensor,
        node_role: Tensor,
        vertex_interaction_type: Tensor,
    ) -> Tensor:
        src, dst = edge_index
        fermion_idx = self._fermion_index(node_fermion_num)

        transport_mask = (fermion_line_id >= 0) & (edge_relation != SPINOR_REL_BOSON)
        transport_input = torch.cat(
            [
                h[src],
                h[dst],
                self.rel_emb(edge_relation),
                self.role_emb(node_role[src]),
                self.role_emb(node_role[dst]),
                self.fermion_emb(fermion_idx[src]),
                self.fermion_emb(fermion_idx[dst]),
                transport_mask.unsqueeze(-1).to(h.dtype),
            ],
            dim=-1,
        )
        transport_messages = self.transport_mlp(transport_input)
        transport_messages = transport_messages * transport_mask.unsqueeze(-1).to(h.dtype)
        along_mask = transport_mask & (edge_relation == SPINOR_REL_FERMION_ALONG)
        against_mask = transport_mask & (edge_relation == SPINOR_REL_FERMION_AGAINST)
        agg_transport_along = unsorted_segment_sum(
            transport_messages * along_mask.unsqueeze(-1).to(h.dtype),
            dst,
            num_segments=h.size(0),
        )
        agg_transport_against = unsorted_segment_sum(
            transport_messages * against_mask.unsqueeze(-1).to(h.dtype),
            dst,
            num_segments=h.size(0),
        )
        agg_transport_along = self.transport_norm(agg_transport_along)
        agg_transport_against = self.transport_norm(agg_transport_against)

        interaction_input = torch.cat(
            [
                h[src],
                h[dst],
                self.rel_emb(edge_relation),
                self.role_emb(node_role[src]),
                self.role_emb(node_role[dst]),
                self.vertex_emb(vertex_interaction_type[src]),
                self.vertex_emb(vertex_interaction_type[dst]),
                self.fermion_emb(fermion_idx[src]),
                self.fermion_emb(fermion_idx[dst]),
            ],
            dim=-1,
        )
        interaction_messages = self.interaction_mlp(interaction_input)
        agg_interaction = unsorted_segment_sum(
            interaction_messages,
            dst,
            num_segments=h.size(0),
        )
        agg_interaction = self.interaction_norm(agg_interaction)

        node_input = torch.cat(
            [
                h,
                agg_transport_along,
                agg_transport_against,
                agg_interaction,
                self.role_emb(node_role),
                self.vertex_emb(vertex_interaction_type),
                self.fermion_emb(fermion_idx),
            ],
            dim=-1,
        )
        update = self.node_mlp(node_input)
        return self.norm(h + update)


class SpinorFlowBlock(nn.Module):
    """Stack of line-aware spinor-flow layers."""

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
            raise TypeError(f"Unexpected SpinorFlowBlock kwargs: {unknown}")

        if in_dim is None:
            in_dim = n_scalar
        if in_dim is None:
            raise TypeError("SpinorFlowBlock requires `in_dim` or alias `n_scalar`.")
        if n_hidden is not None:
            hidden_dim = n_hidden
        if n_output is not None:
            out_dim = n_output
        if n_layers is not None:
            num_layers = n_layers

        self.num_layers = num_layers
        self.role_emb = nn.Embedding(NUM_NODE_ROLES, 8)
        self.vertex_emb = nn.Embedding(NUM_VERTEX_INTERACTIONS, 8)
        self.fermion_emb = nn.Embedding(3, 8)
        self.input_proj = nn.Linear(in_dim + 8 + 8 + 8, hidden_dim)
        self.layers = nn.ModuleList(
            SpinorFlowLayer(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        )
        self.output_proj = (
            nn.Linear(hidden_dim, out_dim) if out_dim != hidden_dim else nn.Identity()
        )

    @staticmethod
    def _fermion_index(node_fermion_num: Tensor) -> Tensor:
        return (node_fermion_num + 1).clamp_(0, 2)

    def encode_input(
        self,
        scalars: Tensor,
        node_fermion_num: Tensor,
        node_role: Tensor,
        vertex_interaction_type: Tensor,
    ) -> Tensor:
        node_input = torch.cat(
            [
                scalars,
                self.role_emb(node_role),
                self.vertex_emb(vertex_interaction_type),
                self.fermion_emb(self._fermion_index(node_fermion_num)),
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
        fermion_line_id: Tensor,
        node_fermion_num: Tensor,
        node_role: Tensor,
        vertex_interaction_type: Tensor,
    ) -> Tensor:
        return self.layers[layer_idx](
            h=h,
            edge_index=edge_index,
            edge_relation=edge_relation,
            fermion_line_id=fermion_line_id,
            node_fermion_num=node_fermion_num,
            node_role=node_role,
            vertex_interaction_type=vertex_interaction_type,
        )

    def project_output(self, h: Tensor) -> Tensor:
        return self.output_proj(h)

    def forward(
        self,
        scalars: Tensor,
        edge_index: Tensor,
        edge_relation: Tensor,
        fermion_line_id: Tensor,
        node_fermion_num: Tensor,
        node_role: Tensor,
        vertex_interaction_type: Tensor,
    ) -> Tensor:
        h = self.encode_input(
            scalars=scalars,
            node_fermion_num=node_fermion_num,
            node_role=node_role,
            vertex_interaction_type=vertex_interaction_type,
        )
        for layer_idx in range(self.num_layers):
            h = self.step_layer(
                layer_idx=layer_idx,
                h=h,
                edge_index=edge_index,
                edge_relation=edge_relation,
                fermion_line_id=fermion_line_id,
                node_fermion_num=node_fermion_num,
                node_role=node_role,
                vertex_interaction_type=vertex_interaction_type,
            )
        return self.project_output(h)
