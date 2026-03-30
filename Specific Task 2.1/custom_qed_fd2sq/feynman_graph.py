"""Fixed-slot tensor packaging for the QED tree-level 2->2 custom encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import (
    CHANNEL_PROP_REL_LEG,
    CHANNEL_TO_IDX,
    NUM_CHANNEL_PROP_RELATIONS,
    NUM_FERMION_RELATIONS,
    PATTERN_TO_IDX,
)
from .features import MASS_VALUES, build_interaction_features
from .interaction import signed_charge as _signed_charge
from .parser import FeynmanDiagram
from .relations import build_interaction_relations


@dataclass(frozen=True)
class FixedSlotQEDGraph:
    slot_features: Tensor
    slot_type_ids: Tensor
    slot_position_ids: Tensor
    fermion_line_ids: Tensor
    channel_prop_edge_index: Tensor
    channel_prop_edge_attr: Tensor
    channel_prop_edge_type: Tensor
    fermion_line_edge_index: Tensor
    fermion_line_edge_attr: Tensor
    fermion_line_edge_type: Tensor
    static_charge_features: Tensor
    topology_features: Tensor
    external_mass_summary: Tensor
    channel_id: Tensor
    process_family_id: Tensor
    pattern_id: Tensor
    external_mask: Tensor
    vertex_mask: Tensor
    propagator_mask: Tensor

    @property
    def x(self) -> Tensor:
        return self.slot_features


def _one_hot_edge_attr(edge_types: Tensor, num_classes: int) -> Tensor:
    return torch.nn.functional.one_hot(
        edge_types,
        num_classes=num_classes,
    ).to(torch.float32)


def diagram_to_fixed_slot_graph(diagram: FeynmanDiagram) -> FixedSlotQEDGraph:
    features = build_interaction_features(diagram)
    relations = build_interaction_relations(diagram)

    channel_prop_edge_attr = _one_hot_edge_attr(
        relations.channel_prop_edge_type,
        NUM_CHANNEL_PROP_RELATIONS,
    )
    fermion_line_edge_attr = _one_hot_edge_attr(
        relations.fermion_line_edge_type,
        NUM_FERMION_RELATIONS,
    )

    return FixedSlotQEDGraph(
        slot_features=features.slot_features,
        slot_type_ids=features.slot_type_ids,
        slot_position_ids=features.slot_position_ids,
        fermion_line_ids=features.fermion_line_ids,
        channel_prop_edge_index=relations.channel_prop_edge_index,
        channel_prop_edge_attr=channel_prop_edge_attr,
        channel_prop_edge_type=relations.channel_prop_edge_type,
        fermion_line_edge_index=relations.fermion_line_edge_index,
        fermion_line_edge_attr=fermion_line_edge_attr,
        fermion_line_edge_type=relations.fermion_line_edge_type,
        static_charge_features=features.static_charge_features,
        topology_features=features.topology_features,
        external_mass_summary=features.external_mass_summary,
        channel_id=features.channel_id,
        process_family_id=features.process_family_id,
        pattern_id=features.pattern_id,
        external_mask=relations.external_mask,
        vertex_mask=relations.vertex_mask,
        propagator_mask=relations.propagator_mask,
    )


def diagram_to_homogeneous_graph(diagram: FeynmanDiagram) -> FixedSlotQEDGraph:
    return diagram_to_fixed_slot_graph(diagram)


__all__ = [
    "CHANNEL_PROP_REL_LEG",
    "CHANNEL_TO_IDX",
    "FixedSlotQEDGraph",
    "MASS_VALUES",
    "PATTERN_TO_IDX",
    "_signed_charge",
    "diagram_to_fixed_slot_graph",
    "diagram_to_homogeneous_graph",
]
