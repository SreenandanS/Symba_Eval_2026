"""Relation builders for the fixed-slot QED contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import (
    CHANNEL_PROP_EDGE_COUNT,
    CHANNEL_PROP_REL_LEG,
    CHANNEL_PROP_REL_PROP_S,
    CHANNEL_PROP_REL_PROP_T,
    CHANNEL_PROP_REL_PROP_U,
    CHANNEL_TO_IDX,
    EXTERNAL_SLOT_INDICES,
    FERMION_EDGE_COUNT,
    FERMION_REL_AGAINST,
    FERMION_REL_ALONG,
    FERMION_REL_BOSON,
    PROPAGATOR_SLOT_INDEX,
    VERTEX_SLOT_INDICES,
)
from .interaction import QEDInteraction, external_flow_sign


@dataclass(frozen=True)
class InteractionRelations:
    channel_prop_edge_index: Tensor
    channel_prop_edge_type: Tensor
    fermion_line_edge_index: Tensor
    fermion_line_edge_type: Tensor
    external_mask: Tensor
    vertex_mask: Tensor
    propagator_mask: Tensor


def _prop_edge_relation(channel: str) -> int:
    if channel == "s":
        return CHANNEL_PROP_REL_PROP_S
    if channel == "t":
        return CHANNEL_PROP_REL_PROP_T
    if channel == "u":
        return CHANNEL_PROP_REL_PROP_U
    raise ValueError(f"Unsupported channel '{channel}' in audited QED scope.")


def _append_edge(
    edge_pairs: list[list[int]],
    edge_types: list[int],
    src: int,
    dst: int,
    relation: int,
) -> None:
    edge_pairs.append([src, dst])
    edge_types.append(relation)


def _external_fermion_relations(interaction: QEDInteraction) -> tuple[list[list[int]], list[int]]:
    edge_pairs: list[list[int]] = []
    edge_types: list[int] = []
    for external in interaction.externals:
        vertex_slot = 4 + interaction.external_to_vertex[external.slot_index]
        flow_sign = external_flow_sign(external)
        if flow_sign == 0:
            forward_rel = FERMION_REL_BOSON
            backward_rel = FERMION_REL_BOSON
        elif flow_sign > 0:
            forward_rel = FERMION_REL_ALONG
            backward_rel = FERMION_REL_AGAINST
        else:
            forward_rel = FERMION_REL_AGAINST
            backward_rel = FERMION_REL_ALONG

        _append_edge(edge_pairs, edge_types, external.slot_index, vertex_slot, forward_rel)
        _append_edge(edge_pairs, edge_types, vertex_slot, external.slot_index, backward_rel)
    return edge_pairs, edge_types


def _propagator_fermion_relations(
    interaction: QEDInteraction,
    edge_pairs: list[list[int]],
    edge_types: list[int],
) -> None:
    propagator = interaction.propagator
    for vertex_slot in VERTEX_SLOT_INDICES:
        if propagator is None or propagator.is_photon:
            forward_rel = FERMION_REL_BOSON
            backward_rel = FERMION_REL_BOSON
        else:
            endpoint_idx = propagator.endpoint_vertices.index(vertex_slot - 4)
            sign = propagator.endpoint_signs[endpoint_idx]
            if sign > 0:
                forward_rel = FERMION_REL_ALONG
                backward_rel = FERMION_REL_AGAINST
            elif sign < 0:
                forward_rel = FERMION_REL_AGAINST
                backward_rel = FERMION_REL_ALONG
            else:
                forward_rel = FERMION_REL_BOSON
                backward_rel = FERMION_REL_BOSON

        _append_edge(edge_pairs, edge_types, vertex_slot, PROPAGATOR_SLOT_INDEX, forward_rel)
        _append_edge(edge_pairs, edge_types, PROPAGATOR_SLOT_INDEX, vertex_slot, backward_rel)


def build_interaction_relations(interaction: QEDInteraction) -> InteractionRelations:
    channel_prop_pairs: list[list[int]] = []
    channel_prop_types: list[int] = []
    for external_slot in EXTERNAL_SLOT_INDICES:
        vertex_slot = 4 + interaction.external_to_vertex[external_slot]
        _append_edge(
            channel_prop_pairs,
            channel_prop_types,
            external_slot,
            vertex_slot,
            CHANNEL_PROP_REL_LEG,
        )
        _append_edge(
            channel_prop_pairs,
            channel_prop_types,
            vertex_slot,
            external_slot,
            CHANNEL_PROP_REL_LEG,
        )

    prop_relation = _prop_edge_relation(interaction.channel)
    for vertex_slot in VERTEX_SLOT_INDICES:
        _append_edge(
            channel_prop_pairs,
            channel_prop_types,
            vertex_slot,
            PROPAGATOR_SLOT_INDEX,
            prop_relation,
        )
        _append_edge(
            channel_prop_pairs,
            channel_prop_types,
            PROPAGATOR_SLOT_INDEX,
            vertex_slot,
            prop_relation,
        )

    fermion_pairs, fermion_types = _external_fermion_relations(interaction)
    _propagator_fermion_relations(interaction, fermion_pairs, fermion_types)

    if len(channel_prop_pairs) != CHANNEL_PROP_EDGE_COUNT:
        raise ValueError(
            f"Expected {CHANNEL_PROP_EDGE_COUNT} channel/prop edges, got {len(channel_prop_pairs)}."
        )
    if len(fermion_pairs) != FERMION_EDGE_COUNT:
        raise ValueError(
            f"Expected {FERMION_EDGE_COUNT} fermion-line edges, got {len(fermion_pairs)}."
        )

    return InteractionRelations(
        channel_prop_edge_index=torch.tensor(channel_prop_pairs, dtype=torch.long),
        channel_prop_edge_type=torch.tensor(channel_prop_types, dtype=torch.long),
        fermion_line_edge_index=torch.tensor(fermion_pairs, dtype=torch.long),
        fermion_line_edge_type=torch.tensor(fermion_types, dtype=torch.long),
        external_mask=torch.tensor([1, 1, 1, 1, 0, 0, 0], dtype=torch.bool),
        vertex_mask=torch.tensor([0, 0, 0, 0, 1, 1, 0], dtype=torch.bool),
        propagator_mask=torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.bool),
    )


__all__ = ["InteractionRelations", "build_interaction_relations"]
