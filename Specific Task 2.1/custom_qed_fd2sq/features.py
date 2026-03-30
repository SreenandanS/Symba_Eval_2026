"""Feature builders for the canonical QED slot contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import (
    CHANNEL_TO_IDX,
    FLAVOR_TO_IDX,
    NUM_CHANNELS,
    NUM_FIXED_SLOTS,
    NUM_FLAVORS,
    NUM_PATTERNS,
    NUM_PROCESS_FAMILIES,
    NUM_PTYPES,
    PATTERN_TO_IDX,
    PROCESS_FAMILY_TO_IDX,
    PROPAGATOR_SLOT_INDEX,
    SLOT_KIND_EXTERNAL,
    SLOT_KIND_PROPAGATOR,
    SLOT_KIND_VERTEX,
)
from .interaction import (
    BASE_CHARGES,
    LEPTON_FLAVORS,
    PARTICLE_PHOTON,
    QUARK_FLAVORS,
    QEDExternalSlot,
    QEDInteraction,
)


MASS_VALUES = {
    "A": 0.0,
    "e": 0.000511,
    "mu": 0.10566,
    "tt": 1.77686,
    "u": 0.0022,
    "d": 0.0047,
    "s": 0.095,
    "c": 1.275,
    "b": 4.18,
    "t": 173.0,
}

SPIN_VALUES = {
    "fermion": 0.5,
    "antifermion": 0.5,
    "photon": 1.0,
}

SLOT_FEATURE_DIM = 25
STATIC_CHARGE_FEATURE_DIM = 34
TOPOLOGY_FEATURE_DIM = 20
EXTERNAL_MASS_SUMMARY_DIM = 6


@dataclass(frozen=True)
class InteractionFeatures:
    slot_features: Tensor
    slot_type_ids: Tensor
    slot_position_ids: Tensor
    fermion_line_ids: Tensor
    static_charge_features: Tensor
    topology_features: Tensor
    external_mass_summary: Tensor
    channel_id: Tensor
    process_family_id: Tensor
    pattern_id: Tensor


def _one_hot(index: int | None, size: int) -> Tensor:
    vec = torch.zeros(size, dtype=torch.float32)
    if index is not None and 0 <= index < size:
        vec[index] = 1.0
    return vec


def _log_mass(flavor: str) -> float:
    return math.log1p(MASS_VALUES.get(flavor, 0.0))


def _ptype_one_hot(particle_type: str) -> Tensor:
    index = {"fermion": 0, "antifermion": 1, "photon": 2}.get(particle_type)
    return _one_hot(index, NUM_PTYPES)


def _flavor_one_hot(flavor: str) -> Tensor:
    return _one_hot(FLAVOR_TO_IDX.get(flavor), NUM_FLAVORS)


def _charge_stats(external: QEDExternalSlot) -> tuple[float, float, float]:
    charge = external.charge
    return charge, abs(charge), charge * charge


def _external_features(external: QEDExternalSlot) -> Tensor:
    charge, abs_charge, charge_sq = _charge_stats(external)
    return torch.cat(
        [
            _flavor_one_hot(external.flavor),
            _ptype_one_hot(external.particle_type),
            torch.tensor(
                [
                    1.0 if external.is_incoming else 0.0,
                    1.0 if external.is_antiparticle else 0.0,
                    1.0 if external.is_conjugate else 0.0,
                    _log_mass(external.flavor),
                    SPIN_VALUES[external.particle_type],
                    charge,
                    abs_charge,
                    charge_sq,
                    1.0 if external.flavor in QUARK_FLAVORS else 0.0,
                    1.0 if external.flavor in LEPTON_FLAVORS else 0.0,
                    1.0 if external.flavor == "A" else 0.0,
                    1.0,
                ],
                dtype=torch.float32,
            ),
        ],
        dim=0,
    )


def _vertex_features(interaction: QEDInteraction, vertex_slot_index: int) -> Tensor:
    vertex = interaction.vertices[vertex_slot_index]
    attached = [interaction.externals[idx] for idx in vertex.external_slots]
    charges = [external.charge for external in attached if external.flavor != "A"]
    n_incoming = sum(external.is_incoming for external in attached)
    n_outgoing = len(attached) - n_incoming
    n_quarks = sum(external.flavor in QUARK_FLAVORS for external in attached)
    n_leptons = sum(external.flavor in LEPTON_FLAVORS for external in attached)
    n_photons = sum(external.flavor == "A" for external in attached)
    n_fermions = len(attached) - n_photons
    return torch.cat(
        [
            torch.zeros(NUM_FLAVORS, dtype=torch.float32),
            torch.zeros(NUM_PTYPES, dtype=torch.float32),
            torch.tensor(
                [
                    len(attached) / 3.0,
                    float(sum(charges)),
                    abs(float(sum(charges))),
                    float(sum(charge * charge for charge in charges)),
                    n_fermions / 2.0,
                    n_photons / 2.0,
                    n_incoming / 2.0,
                    n_outgoing / 2.0,
                    float(n_quarks > 0),
                    float(n_leptons > 0),
                    float(n_photons > 0),
                    float(vertex.fermion_line_id),
                ],
                dtype=torch.float32,
            ),
        ],
        dim=0,
    )


def _propagator_features(interaction: QEDInteraction) -> Tensor:
    propagator = interaction.propagator
    if propagator is None:
        flavor = "A"
        ptype = "photon"
        log_mass = 0.0
        charge = 0.0
        charge_sq = 0.0
        endpoint_signs = (0.0, 0.0)
        is_photon = 0.0
        is_fermion = 0.0
    else:
        flavor = propagator.flavor
        ptype = propagator.particle_type
        log_mass = _log_mass(flavor)
        charge = propagator.charge
        charge_sq = charge * charge
        endpoint_signs = (float(propagator.endpoint_signs[0]), float(propagator.endpoint_signs[1]))
        is_photon = 1.0 if propagator.is_photon else 0.0
        is_fermion = 1.0 if propagator.is_fermion else 0.0

    channel_oh = _one_hot(CHANNEL_TO_IDX.get(interaction.channel, CHANNEL_TO_IDX["unknown"]), NUM_CHANNELS)
    return torch.cat(
        [
            _flavor_one_hot(flavor),
            _ptype_one_hot(ptype),
            torch.tensor(
                [
                    log_mass,
                    is_photon,
                    is_fermion,
                ],
                dtype=torch.float32,
            ),
            channel_oh,
            torch.tensor(
                [
                    charge,
                    abs(charge),
                    charge_sq,
                    endpoint_signs[0],
                    endpoint_signs[1],
                ],
                dtype=torch.float32,
            ),
        ],
        dim=0,
    )


def _static_charge_features(interaction: QEDInteraction) -> Tensor:
    charges = [external.charge for external in interaction.externals]
    abs_charges = [abs(charge) for charge in charges]
    charge_squares = [charge * charge for charge in charges]
    is_quark = [1.0 if external.flavor in QUARK_FLAVORS else 0.0 for external in interaction.externals]
    is_lepton = [1.0 if external.flavor in LEPTON_FLAVORS else 0.0 for external in interaction.externals]
    is_photon = [1.0 if external.flavor == "A" else 0.0 for external in interaction.externals]
    incoming = interaction.externals[:2]
    outgoing = interaction.externals[2:]
    propagator = interaction.propagator
    incoming_abs_product = abs(incoming[0].charge) * abs(incoming[1].charge)
    outgoing_abs_product = abs(outgoing[0].charge) * abs(outgoing[1].charge)
    extras = [
        sum(abs(external.charge) for external in incoming),
        sum(abs(external.charge) for external in outgoing),
        sum(external.charge for external in incoming),
        sum(external.charge for external in outgoing),
        float(any(external.flavor in QUARK_FLAVORS for external in incoming)),
        float(any(external.flavor in QUARK_FLAVORS for external in outgoing)),
        1.0 if propagator is not None and propagator.is_photon else 0.0,
        1.0 if propagator is not None and propagator.is_fermion else 0.0,
        incoming_abs_product,
        outgoing_abs_product,
    ]
    values = charges + abs_charges + charge_squares + is_quark + is_lepton + is_photon + extras
    return torch.tensor(values, dtype=torch.float32)


def _topology_features(interaction: QEDInteraction) -> Tensor:
    pattern_oh = _one_hot(PATTERN_TO_IDX.get(interaction.external_pattern, PATTERN_TO_IDX["unknown"]), NUM_PATTERNS)
    family_oh = _one_hot(
        PROCESS_FAMILY_TO_IDX.get(interaction.process_family, PROCESS_FAMILY_TO_IDX["unknown"]),
        NUM_PROCESS_FAMILIES,
    )
    channel_oh = _one_hot(
        CHANNEL_TO_IDX.get(interaction.channel, CHANNEL_TO_IDX["unknown"]),
        NUM_CHANNELS,
    )
    propagator = interaction.propagator
    prop_kind = torch.tensor(
        [
            1.0 if propagator is not None and propagator.is_photon else 0.0,
            1.0 if propagator is not None and propagator.is_fermion else 0.0,
        ],
        dtype=torch.float32,
    )
    ext_to_vertex = torch.tensor(
        [float(index) for index in interaction.external_to_vertex],
        dtype=torch.float32,
    )
    return torch.cat([pattern_oh, family_oh, channel_oh, prop_kind, ext_to_vertex], dim=0)


def _external_mass_summary(interaction: QEDInteraction) -> Tensor:
    masses = [_log_mass(external.flavor) for external in interaction.externals]
    return torch.tensor(
        masses
        + [
            masses[0] + masses[1],
            masses[2] + masses[3],
        ],
        dtype=torch.float32,
    )


def _fermion_line_ids(interaction: QEDInteraction) -> Tensor:
    propagator_line_id = (
        interaction.vertices[0].fermion_line_id
        if interaction.propagator is not None and interaction.propagator.is_fermion
        else -1
    )
    values = [
        *interaction.fermion_line_ids,
        interaction.vertices[0].fermion_line_id,
        interaction.vertices[1].fermion_line_id,
        propagator_line_id,
    ]
    return torch.tensor(values, dtype=torch.long)


def build_interaction_features(interaction: QEDInteraction) -> InteractionFeatures:
    slot_features = torch.stack(
        [
            *[_external_features(external) for external in interaction.externals],
            _vertex_features(interaction, 0),
            _vertex_features(interaction, 1),
            _propagator_features(interaction),
        ],
        dim=0,
    )
    slot_type_ids = torch.tensor(
        [
            SLOT_KIND_EXTERNAL,
            SLOT_KIND_EXTERNAL,
            SLOT_KIND_EXTERNAL,
            SLOT_KIND_EXTERNAL,
            SLOT_KIND_VERTEX,
            SLOT_KIND_VERTEX,
            SLOT_KIND_PROPAGATOR,
        ],
        dtype=torch.long,
    )
    slot_position_ids = torch.arange(NUM_FIXED_SLOTS, dtype=torch.long)
    return InteractionFeatures(
        slot_features=slot_features,
        slot_type_ids=slot_type_ids,
        slot_position_ids=slot_position_ids,
        fermion_line_ids=_fermion_line_ids(interaction),
        static_charge_features=_static_charge_features(interaction),
        topology_features=_topology_features(interaction),
        external_mass_summary=_external_mass_summary(interaction),
        channel_id=torch.tensor(
            CHANNEL_TO_IDX.get(interaction.channel, CHANNEL_TO_IDX["unknown"]),
            dtype=torch.long,
        ),
        process_family_id=torch.tensor(
            PROCESS_FAMILY_TO_IDX.get(
                interaction.process_family,
                PROCESS_FAMILY_TO_IDX["unknown"],
            ),
            dtype=torch.long,
        ),
        pattern_id=torch.tensor(
            PATTERN_TO_IDX.get(interaction.external_pattern, PATTERN_TO_IDX["unknown"]),
            dtype=torch.long,
        ),
    )


__all__ = [
    "EXTERNAL_MASS_SUMMARY_DIM",
    "InteractionFeatures",
    "MASS_VALUES",
    "SLOT_FEATURE_DIM",
    "STATIC_CHARGE_FEATURE_DIM",
    "TOPOLOGY_FEATURE_DIM",
    "build_interaction_features",
]
