"""
Feynman diagram to graph conversion for the symmetry-respecting custom graph encoder.

The graph remains a fixed 7-node tree for 2→2 tree-level QCD diagrams:
4 external legs, 2 interaction vertices, and 1 internal propagator. The
builder emits explicit per-stream relation tensors so the encoder can
respect static kinematic, color-flow, and fermion-line structure directly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from .contracts import (
    COLOR_REL_ADJ_ANTIFUND,
    COLOR_REL_ADJ_FUND,
    COLOR_REL_ANTIFUND,
    COLOR_REL_FUND,
    KIN_REL_LEG,
    KIN_REL_PROP_S,
    KIN_REL_PROP_T,
    KIN_REL_PROP_U,
    KIN_REL_PROP_UNKNOWN,
    NODE_ROLE_EXTERNAL,
    NODE_ROLE_PROPAGATOR,
    NODE_ROLE_VERTEX,
    SPINOR_REL_BOSON,
    SPINOR_REL_FERMION_AGAINST,
    SPINOR_REL_FERMION_ALONG,
    VERTEX_INT_GGG,
    VERTEX_INT_NONE,
    VERTEX_INT_QQG,
)
from .parser import (
    ALL_PARTICLES,
    PARTICLE_ANTIQUARK,
    PARTICLE_GLUON,
    PARTICLE_QUARK,
    ExternalParticle,
    FeynmanDiagram,
    Propagator,
    Vertex,
)

# Particle flavor one-hot (u, d, s, t, c, b, G)
FLAVOR_TO_IDX = {f: i for i, f in enumerate(ALL_PARTICLES)}
NUM_FLAVORS = len(ALL_PARTICLES)

# Particle type one-hot (quark, antiquark, gluon)
PTYPE_TO_IDX = {
    PARTICLE_QUARK: 0,
    PARTICLE_ANTIQUARK: 1,
    PARTICLE_GLUON: 2,
}
NUM_PTYPES = 3

# Channel encoding
CHANNEL_TO_IDX = {"s": 0, "t": 1, "u": 2, "unknown": 3}
NUM_CHANNELS = 4

# Mandelstam labels
MANDELSTAM_LABELS = ["s_12", "s_13", "s_14", "s_23", "s_24", "s_34"]
NUM_MANDELSTAM = len(MANDELSTAM_LABELS)

# Relative mass features
MASS_VALUES = {
    "u": 0.0022,
    "d": 0.0047,
    "s": 0.095,
    "c": 1.275,
    "b": 4.18,
    "t": 173.0,
    "G": 0.0,
}

COLOR_DIM = {
    PARTICLE_QUARK: 3,
    PARTICLE_ANTIQUARK: 3,
    PARTICLE_GLUON: 8,
}
SPIN_DIM = {
    PARTICLE_QUARK: 0.5,
    PARTICLE_ANTIQUARK: 0.5,
    PARTICLE_GLUON: 1.0,
}


def _encode_flavor_onehot(flavor: str) -> torch.Tensor:
    vec = torch.zeros(NUM_FLAVORS, dtype=torch.float32)
    idx = FLAVOR_TO_IDX.get(flavor)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _encode_ptype_onehot(ptype: str) -> torch.Tensor:
    vec = torch.zeros(NUM_PTYPES, dtype=torch.float32)
    idx = PTYPE_TO_IDX.get(ptype)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _log_mass(flavor: str) -> float:
    return float(np.log1p(MASS_VALUES.get(flavor, 0.0)))


def _external_node_features(particle: ExternalParticle) -> torch.Tensor:
    return torch.cat(
        [
            _encode_flavor_onehot(particle.flavor),
            _encode_ptype_onehot(particle.particle_type),
            torch.tensor(
                [
                    1.0 if particle.is_incoming else 0.0,
                    1.0 if particle.is_antiparticle else 0.0,
                    _log_mass(particle.flavor),
                    COLOR_DIM.get(particle.particle_type, 3) / 8.0,
                    SPIN_DIM.get(particle.particle_type, 0.5),
                ],
                dtype=torch.float32,
            ),
        ],
        dim=0,
    )


def _vertex_node_features(vertex: Vertex, diagram: FeynmanDiagram) -> torch.Tensor:
    propagator = vertex.propagator
    if propagator is None:
        flavor_oh = torch.zeros(NUM_FLAVORS, dtype=torch.float32)
        ptype_oh = torch.zeros(NUM_PTYPES, dtype=torch.float32)
    else:
        flavor_oh = _encode_flavor_onehot(propagator.flavor)
        ptype_oh = _encode_ptype_onehot(propagator.particle_type)
    return torch.cat(
        [
            flavor_oh,
            ptype_oh,
            torch.tensor(
                [
                    len(vertex.external_legs) / 3.0,
                    float(vertex.vertex_id),
                ],
                dtype=torch.float32,
            ),
        ],
        dim=0,
    )


def _propagator_node_features(diagram: FeynmanDiagram) -> torch.Tensor:
    propagator = next((v.propagator for v in diagram.vertices if v.propagator), None)
    if propagator is None:
        flavor_oh = torch.zeros(NUM_FLAVORS, dtype=torch.float32)
        ptype_oh = torch.zeros(NUM_PTYPES, dtype=torch.float32)
        mass_val = 0.0
    else:
        flavor_oh = _encode_flavor_onehot(propagator.flavor)
        ptype_oh = _encode_ptype_onehot(propagator.particle_type)
        mass_val = MASS_VALUES.get(propagator.flavor, 0.0)

    channel_oh = torch.zeros(NUM_CHANNELS, dtype=torch.float32)
    channel_oh[CHANNEL_TO_IDX.get(diagram.channel, CHANNEL_TO_IDX["unknown"])] = 1.0
    return torch.cat(
        [
            flavor_oh,
            ptype_oh,
            torch.tensor(
                [
                    float(np.log1p(mass_val)),
                    1.0 if mass_val < 1e-6 else 0.0,
                ],
                dtype=torch.float32,
            ),
            channel_oh,
        ],
        dim=0,
    )


def _leg_edge_features(particle: ExternalParticle) -> torch.Tensor:
    return torch.cat(
        [
            _encode_flavor_onehot(particle.flavor),
            _encode_ptype_onehot(particle.particle_type),
            torch.tensor(
                [
                    1.0 if particle.is_incoming else -1.0,
                    1.0 if particle.is_conjugate else 0.0,
                ],
                dtype=torch.float32,
            ),
        ],
        dim=0,
    )


def _internal_edge_features(diagram: FeynmanDiagram) -> torch.Tensor:
    propagator = next((v.propagator for v in diagram.vertices if v.propagator), None)
    if propagator is None:
        flavor_oh = torch.zeros(NUM_FLAVORS, dtype=torch.float32)
        ptype_oh = torch.zeros(NUM_PTYPES, dtype=torch.float32)
        mass_val = 0.0
    else:
        flavor_oh = _encode_flavor_onehot(propagator.flavor)
        ptype_oh = _encode_ptype_onehot(propagator.particle_type)
        mass_val = MASS_VALUES.get(propagator.flavor, 0.0)

    channel_oh = torch.zeros(NUM_CHANNELS, dtype=torch.float32)
    channel_oh[CHANNEL_TO_IDX.get(diagram.channel, CHANNEL_TO_IDX["unknown"])] = 1.0
    return torch.cat(
        [
            flavor_oh,
            ptype_oh,
            channel_oh,
            torch.tensor([float(np.log1p(mass_val))], dtype=torch.float32),
        ],
        dim=0,
    )


def _build_mandelstam_features(
    diagram: FeynmanDiagram,
    mandelstam_values: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    if mandelstam_values is None:
        return torch.ones(NUM_MANDELSTAM, dtype=torch.float32)
    return torch.tensor(
        [mandelstam_values.get(label, 0.0) for label in MANDELSTAM_LABELS],
        dtype=torch.float32,
    )


def _build_color_flow_features(diagram: FeynmanDiagram) -> torch.Tensor:
    leg_colors: List[float] = []
    for particle in diagram.externals:
        if particle.particle_type == PARTICLE_QUARK:
            leg_colors.append(3.0 / 8.0)
        elif particle.particle_type == PARTICLE_ANTIQUARK:
            leg_colors.append(-3.0 / 8.0)
        else:
            leg_colors.append(1.0)
    while len(leg_colors) < 4:
        leg_colors.append(0.0)

    propagator = next((v.propagator for v in diagram.vertices if v.propagator), None)
    if propagator is None:
        prop_color = 0.0
        is_singlet = 0.0
        is_octet = 0.0
    elif propagator.is_gluon:
        prop_color = 1.0
        is_singlet = 0.0
        is_octet = 1.0
    else:
        prop_color = 3.0 / 8.0
        is_singlet = 1.0
        is_octet = 0.0

    return torch.tensor(
        leg_colors + [prop_color, 0.5, is_singlet, is_octet],
        dtype=torch.float32,
    )


def _build_fermion_flow_features(diagram: FeynmanDiagram) -> torch.Tensor:
    fermion_nums: List[float] = []
    for particle in diagram.externals:
        if particle.particle_type == PARTICLE_QUARK:
            fermion_nums.append(1.0)
        elif particle.particle_type == PARTICLE_ANTIQUARK:
            fermion_nums.append(-1.0)
        else:
            fermion_nums.append(0.0)
    while len(fermion_nums) < 4:
        fermion_nums.append(0.0)

    total_fermion = sum(fermion_nums)
    n_fermion_lines = (
        sum(p.particle_type != PARTICLE_GLUON for p in diagram.externals) / 2.0
    )
    propagator = next((v.propagator for v in diagram.vertices if v.propagator), None)
    carries_fermion = float(propagator is not None and propagator.is_quark)
    n_gamma = 0.5 if carries_fermion else 0.0

    return torch.tensor(
        fermion_nums + [total_fermion, n_fermion_lines / 2.0, carries_fermion, n_gamma],
        dtype=torch.float32,
    )


def _canonical_vertices(diagram: FeynmanDiagram) -> List[Vertex]:
    return sorted(
        diagram.vertices,
        key=lambda vertex: (tuple(sorted(vertex.external_legs)), vertex.vertex_id),
    )


def _all_outgoing_signature(particle: ExternalParticle) -> torch.Tensor:
    signature = torch.zeros(4, dtype=torch.float32)
    idx = max(min(particle.momentum_label - 1, 3), 0)
    signature[idx] = -1.0 if particle.is_incoming else 1.0
    return signature


def _external_flow_direction(particle: ExternalParticle) -> int:
    if particle.particle_type == PARTICLE_GLUON:
        return 0
    if particle.particle_type == PARTICLE_QUARK:
        return 1 if particle.is_incoming else -1
    return -1 if particle.is_incoming else 1


def _spinor_relation_from_flow(flow_dir: int) -> int:
    if flow_dir > 0:
        return SPINOR_REL_FERMION_ALONG
    if flow_dir < 0:
        return SPINOR_REL_FERMION_AGAINST
    return SPINOR_REL_BOSON


def _kin_relation_from_channel(channel_idx: int) -> int:
    if channel_idx == CHANNEL_TO_IDX["s"]:
        return KIN_REL_PROP_S
    if channel_idx == CHANNEL_TO_IDX["t"]:
        return KIN_REL_PROP_T
    if channel_idx == CHANNEL_TO_IDX["u"]:
        return KIN_REL_PROP_U
    return KIN_REL_PROP_UNKNOWN


def _vertex_interaction_type(
    vertex: Vertex,
    external_by_label: Dict[int, ExternalParticle],
) -> int:
    attached_types = [
        external_by_label[leg].particle_type
        for leg in vertex.external_legs
        if leg in external_by_label
    ]
    if vertex.propagator is not None:
        attached_types.append(vertex.propagator.particle_type)

    n_gluons = sum(ptype == PARTICLE_GLUON for ptype in attached_types)
    n_fermions = len(attached_types) - n_gluons
    if n_gluons == 1 and n_fermions == 2:
        return VERTEX_INT_QQG
    if n_gluons == 3:
        return VERTEX_INT_GGG
    return VERTEX_INT_NONE


def _vertex_line_ids(
    vertices: List[Vertex],
    external_by_label: Dict[int, ExternalParticle],
    propagator: Optional[Propagator],
) -> Dict[int, int]:
    if propagator is not None and propagator.is_quark:
        has_fermion = any(
            external_by_label.get(leg) is not None
            and external_by_label[leg].particle_type != PARTICLE_GLUON
            for vertex in vertices
            for leg in vertex.external_legs
        )
        return {idx: (0 if has_fermion else -1) for idx in range(len(vertices))}

    line_ids: Dict[int, int] = {}
    next_line_id = 0
    for idx, vertex in enumerate(vertices):
        local_fermion_legs = [
            leg
            for leg in vertex.external_legs
            if external_by_label.get(leg) is not None
            and external_by_label[leg].particle_type != PARTICLE_GLUON
        ]
        if len(local_fermion_legs) >= 2:
            line_ids[idx] = next_line_id
            next_line_id += 1
        else:
            line_ids[idx] = -1
    return line_ids


def _color_relations_for_particle_type(particle_type: str) -> List[int]:
    if particle_type == PARTICLE_QUARK:
        return [COLOR_REL_FUND]
    if particle_type == PARTICLE_ANTIQUARK:
        return [COLOR_REL_ANTIFUND]
    return [COLOR_REL_ADJ_FUND, COLOR_REL_ADJ_ANTIFUND]


def _node_mass_features(
    external_by_label: Dict[int, ExternalParticle],
    vertices: List[Vertex],
    propagator: Optional[Propagator],
) -> torch.Tensor:
    masses: List[torch.Tensor] = []
    for leg in range(1, 5):
        particle = external_by_label.get(leg)
        if particle is None:
            masses.append(torch.zeros(2, dtype=torch.float32))
            continue
        mass_val = MASS_VALUES.get(particle.flavor, 0.0)
        masses.append(
            torch.tensor(
                [float(np.log1p(mass_val)), 1.0 if mass_val < 1e-6 else 0.0],
                dtype=torch.float32,
            )
        )

    masses.extend(torch.zeros(2, dtype=torch.float32) for _ in vertices)
    while len(masses) < 6:
        masses.append(torch.zeros(2, dtype=torch.float32))

    if propagator is None:
        masses.append(torch.zeros(2, dtype=torch.float32))
    else:
        mass_val = MASS_VALUES.get(propagator.flavor, 0.0)
        masses.append(
            torch.tensor(
                [float(np.log1p(mass_val)), 1.0 if mass_val < 1e-6 else 0.0],
                dtype=torch.float32,
            )
        )

    return torch.stack(masses)


def _node_color_representations(
    external_by_label: Dict[int, ExternalParticle],
    vertices: List[Vertex],
    propagator: Optional[Propagator],
) -> List[int]:
    representations: List[int] = []
    for leg in range(1, 5):
        particle = external_by_label.get(leg)
        if particle is None or particle.particle_type == PARTICLE_GLUON:
            representations.append(2)
        elif particle.particle_type == PARTICLE_QUARK:
            representations.append(0)
        else:
            representations.append(1)

    representations.extend(2 for _ in vertices)
    while len(representations) < 6:
        representations.append(2)

    if propagator is None or propagator.is_gluon:
        representations.append(2)
    elif propagator.is_antiparticle:
        representations.append(1)
    else:
        representations.append(0)
    return representations


def _node_fermion_numbers(
    external_by_label: Dict[int, ExternalParticle],
    vertices: List[Vertex],
    propagator: Optional[Propagator],
) -> List[int]:
    numbers: List[int] = []
    for leg in range(1, 5):
        particle = external_by_label.get(leg)
        if particle is None or particle.particle_type == PARTICLE_GLUON:
            numbers.append(0)
        elif particle.particle_type == PARTICLE_QUARK:
            numbers.append(1)
        else:
            numbers.append(-1)

    numbers.extend(0 for _ in vertices)
    while len(numbers) < 6:
        numbers.append(0)

    if propagator is None or propagator.is_gluon:
        numbers.append(0)
    elif propagator.is_antiparticle:
        numbers.append(-1)
    else:
        numbers.append(1)
    return numbers


def _node_momentum_signatures(
    external_by_label: Dict[int, ExternalParticle],
    vertices: List[Vertex],
) -> torch.Tensor:
    external_signatures: List[torch.Tensor] = []
    for leg in range(1, 5):
        particle = external_by_label.get(leg)
        if particle is None:
            external_signatures.append(torch.zeros(4, dtype=torch.float32))
        else:
            external_signatures.append(_all_outgoing_signature(particle))

    vertex_signatures: List[torch.Tensor] = []
    for vertex in vertices:
        signature = torch.zeros(4, dtype=torch.float32)
        for leg in vertex.external_legs:
            if 1 <= leg <= 4:
                signature = signature + external_signatures[leg - 1]
        vertex_signatures.append(signature)
    while len(vertex_signatures) < 2:
        vertex_signatures.append(torch.zeros(4, dtype=torch.float32))

    propagator_signature = (
        vertex_signatures[0].clone()
        if vertex_signatures
        else torch.zeros(4, dtype=torch.float32)
    )

    return torch.stack(external_signatures + vertex_signatures[:2] + [propagator_signature])


def _pad_feature(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    if tensor.shape[0] >= target_dim:
        return tensor
    return torch.cat(
        [tensor, torch.zeros(target_dim - tensor.shape[0], dtype=tensor.dtype)],
        dim=0,
    )


def diagram_to_homogeneous_graph(
    diagram: FeynmanDiagram,
    mandelstam_values: Optional[Dict[str, float]] = None,
) -> Data:
    """
    Convert a diagram to a homogeneous `torch_geometric.data.Data` object.

    The returned graph includes the generic graph tensors (`edge_index`,
    `edge_attr`, `x`) together with explicit per-stream relation tensors.
    """
    external_by_label = {particle.momentum_label: particle for particle in diagram.externals}
    canonical_vertices = _canonical_vertices(diagram)
    canonical_vertices = canonical_vertices[:2]
    while len(canonical_vertices) < 2:
        canonical_vertices.append(Vertex(vertex_id=len(canonical_vertices)))

    propagator = next((v.propagator for v in canonical_vertices if v.propagator), None)
    channel_idx = CHANNEL_TO_IDX.get(diagram.channel, CHANNEL_TO_IDX["unknown"])

    external_features: List[torch.Tensor] = []
    for leg in range(1, 5):
        particle = external_by_label.get(leg)
        if particle is None:
            external_features.append(torch.zeros(15, dtype=torch.float32))
        else:
            external_features.append(_external_node_features(particle))

    vertex_features = [_vertex_node_features(vertex, diagram) for vertex in canonical_vertices]
    propagator_features = _propagator_node_features(diagram)

    max_node_dim = max(
        external_features[0].shape[0],
        vertex_features[0].shape[0],
        propagator_features.shape[0],
    )
    node_type_indicators = (
        [torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32) for _ in range(4)]
        + [torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32) for _ in range(2)]
        + [torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)]
    )
    flat_node_features = [
        *[_pad_feature(feat, max_node_dim) for feat in external_features],
        *[_pad_feature(feat, max_node_dim) for feat in vertex_features],
        _pad_feature(propagator_features, max_node_dim),
    ]
    x = torch.cat([torch.stack(flat_node_features), torch.stack(node_type_indicators)], dim=1)

    node_role = torch.tensor(
        [
            NODE_ROLE_EXTERNAL,
            NODE_ROLE_EXTERNAL,
            NODE_ROLE_EXTERNAL,
            NODE_ROLE_EXTERNAL,
            NODE_ROLE_VERTEX,
            NODE_ROLE_VERTEX,
            NODE_ROLE_PROPAGATOR,
        ],
        dtype=torch.long,
    )
    node_color_rep = torch.tensor(
        _node_color_representations(external_by_label, canonical_vertices, propagator),
        dtype=torch.long,
    )
    node_fermion_num = torch.tensor(
        _node_fermion_numbers(external_by_label, canonical_vertices, propagator),
        dtype=torch.long,
    )
    node_mass_features = _node_mass_features(external_by_label, canonical_vertices, propagator)
    node_momentum_signature = _node_momentum_signatures(external_by_label, canonical_vertices)
    node_momentum_label = torch.tensor([1, 2, 3, 4, 0, 0, 0], dtype=torch.long)

    vertex_types = [VERTEX_INT_NONE] * 4
    vertex_types.extend(
        _vertex_interaction_type(vertex, external_by_label) for vertex in canonical_vertices
    )
    vertex_types.append(VERTEX_INT_NONE)
    vertex_interaction_type = torch.tensor(vertex_types, dtype=torch.long)

    vertex_line_ids = _vertex_line_ids(canonical_vertices, external_by_label, propagator)

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_attr_list: List[torch.Tensor] = []

    kin_edge_src: List[int] = []
    kin_edge_dst: List[int] = []
    edge_kin_relation: List[int] = []
    kin_edge_channel: List[int] = []

    color_edge_src: List[int] = []
    color_edge_dst: List[int] = []
    edge_color_relation: List[int] = []

    spinor_edge_src: List[int] = []
    spinor_edge_dst: List[int] = []
    edge_spinor_relation: List[int] = []
    fermion_line_id: List[int] = []

    for canonical_idx, vertex in enumerate(canonical_vertices):
        vertex_node = 4 + canonical_idx
        for leg in sorted(vertex.external_legs):
            external_node = leg - 1
            particle = external_by_label.get(leg)
            if particle is None or not 0 <= external_node < 4:
                continue

            generic_edge = _leg_edge_features(particle)
            edge_src.extend([external_node, vertex_node])
            edge_dst.extend([vertex_node, external_node])
            edge_attr_list.extend([generic_edge, generic_edge])

            kin_edge_src.extend([external_node, vertex_node])
            kin_edge_dst.extend([vertex_node, external_node])
            edge_kin_relation.extend([KIN_REL_LEG, KIN_REL_LEG])
            kin_edge_channel.extend([channel_idx, channel_idx])

            for relation in _color_relations_for_particle_type(particle.particle_type):
                color_edge_src.extend([external_node, vertex_node])
                color_edge_dst.extend([vertex_node, external_node])
                edge_color_relation.extend([relation, relation])

            flow = _external_flow_direction(particle)
            line_id = (
                vertex_line_ids.get(canonical_idx, -1)
                if particle.particle_type != PARTICLE_GLUON
                else -1
            )
            spinor_edge_src.extend([external_node, vertex_node])
            spinor_edge_dst.extend([vertex_node, external_node])
            edge_spinor_relation.extend(
                [
                    _spinor_relation_from_flow(flow),
                    _spinor_relation_from_flow(-flow),
                ]
            )
            fermion_line_id.extend([line_id, line_id])

    propagator_node = 6
    generic_internal_edge = _internal_edge_features(diagram)
    internal_kin_relation = _kin_relation_from_channel(channel_idx)
    propagator_type = propagator.particle_type if propagator is not None else PARTICLE_GLUON
    propagator_color_relations = _color_relations_for_particle_type(propagator_type)
    propagator_line_id = 0 if propagator is not None and propagator.is_quark else -1

    for canonical_idx, vertex in enumerate(canonical_vertices):
        vertex_node = 4 + canonical_idx
        edge_src.extend([vertex_node, propagator_node])
        edge_dst.extend([propagator_node, vertex_node])
        edge_attr_list.extend([generic_internal_edge, generic_internal_edge])

        kin_edge_src.extend([vertex_node, propagator_node])
        kin_edge_dst.extend([propagator_node, vertex_node])
        edge_kin_relation.extend([internal_kin_relation, internal_kin_relation])
        kin_edge_channel.extend([channel_idx, channel_idx])

        for relation in propagator_color_relations:
            color_edge_src.extend([vertex_node, propagator_node])
            color_edge_dst.extend([propagator_node, vertex_node])
            edge_color_relation.extend([relation, relation])

        if propagator is not None and propagator.is_quark:
            local_flows = [
                _external_flow_direction(external_by_label[leg])
                for leg in sorted(vertex.external_legs)
                if leg in external_by_label
                and external_by_label[leg].particle_type != PARTICLE_GLUON
            ]
            flow_v_to_prop = local_flows[0] if local_flows else 0
        else:
            flow_v_to_prop = 0

        spinor_edge_src.extend([vertex_node, propagator_node])
        spinor_edge_dst.extend([propagator_node, vertex_node])
        edge_spinor_relation.extend(
            [
                _spinor_relation_from_flow(flow_v_to_prop),
                _spinor_relation_from_flow(-flow_v_to_prop),
            ]
        )
        fermion_line_id.extend([propagator_line_id, propagator_line_id])

    max_edge_dim = max((edge.shape[0] for edge in edge_attr_list), default=15)
    edge_attr = (
        torch.stack([_pad_feature(edge, max_edge_dim) for edge in edge_attr_list])
        if edge_attr_list
        else torch.zeros((0, max_edge_dim), dtype=torch.float32)
    )

    return Data(
        x=x,
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_attr=edge_attr,
        # Compatibility: expose the static momentum signature in `pos`.
        pos=node_momentum_signature.clone(),
        kin_edge_index=torch.tensor([kin_edge_src, kin_edge_dst], dtype=torch.long),
        edge_kin_relation=torch.tensor(edge_kin_relation, dtype=torch.long),
        kin_edge_channel=torch.tensor(kin_edge_channel, dtype=torch.long),
        color_edge_index=torch.tensor([color_edge_src, color_edge_dst], dtype=torch.long),
        edge_color_relation=torch.tensor(edge_color_relation, dtype=torch.long),
        spinor_edge_index=torch.tensor([spinor_edge_src, spinor_edge_dst], dtype=torch.long),
        edge_spinor_relation=torch.tensor(edge_spinor_relation, dtype=torch.long),
        fermion_line_id=torch.tensor(fermion_line_id, dtype=torch.long),
        node_role=node_role,
        node_color_rep=node_color_rep,
        node_fermion_num=node_fermion_num,
        node_mass_features=node_mass_features,
        node_momentum_signature=node_momentum_signature,
        node_momentum_label=node_momentum_label,
        vertex_interaction_type=vertex_interaction_type,
        mandelstam=_build_mandelstam_features(diagram, mandelstam_values).unsqueeze(0),
        color_flow=_build_color_flow_features(diagram).unsqueeze(0),
        fermion_flow=_build_fermion_flow_features(diagram).unsqueeze(0),
        channel=torch.tensor(channel_idx, dtype=torch.long),
        num_nodes=7,
    )


def build_graph_dataset(
    diagrams: List[FeynmanDiagram],
    mandelstam_values: Optional[List[Dict[str, float]]] = None,
) -> List[Data]:
    graphs: List[Data] = []
    for idx, diagram in enumerate(diagrams):
        values = mandelstam_values[idx] if mandelstam_values is not None else None
        graphs.append(diagram_to_homogeneous_graph(diagram, values))
    return graphs
