"""Canonical fixed-slot QED interaction contract."""

from __future__ import annotations

from dataclasses import dataclass


LEPTON_FLAVORS = ("e", "mu", "tt")
QUARK_FLAVORS = ("u", "d", "s", "c", "b", "t")
GAUGE_BOSONS = ("A",)
ALL_FLAVORS = ("A", "e", "mu", "t", "tt", "u", "d", "s", "c", "b")

PARTICLE_FERMION = "fermion"
PARTICLE_ANTIFERMION = "antifermion"
PARTICLE_PHOTON = "photon"

PROCESS_FERMION_FERMION = "fermion_fermion"
PROCESS_FERMION_PHOTON = "fermion_photon"
PROCESS_PHOTON_PHOTON = "photon_photon"
PROCESS_MIXED = "mixed"

MASS_SYMBOLS = {
    "A": "0",
    "e": "m_e",
    "mu": "m_mu",
    "tt": "m_tt",
    "u": "m_u",
    "d": "m_d",
    "s": "m_s",
    "c": "m_c",
    "b": "m_b",
    "t": "m_t",
}

BASE_CHARGES = {
    "A": 0.0,
    "e": -1.0,
    "mu": -1.0,
    "tt": -1.0,
    "u": 2.0 / 3.0,
    "c": 2.0 / 3.0,
    "t": 2.0 / 3.0,
    "d": -1.0 / 3.0,
    "s": -1.0 / 3.0,
    "b": -1.0 / 3.0,
}

NUM_EXTERNAL_SLOTS = 4
NUM_VERTEX_SLOTS = 2
NUM_TOTAL_SLOTS = 7
PROPAGATOR_SLOT_INDEX = 6


def particle_type_from(flavor: str, is_antiparticle: bool) -> str:
    if flavor == "A":
        return PARTICLE_PHOTON
    return PARTICLE_ANTIFERMION if is_antiparticle else PARTICLE_FERMION


def signed_charge(flavor: str, is_antiparticle: bool) -> float:
    base_charge = BASE_CHARGES.get(flavor, 0.0)
    if flavor == "A":
        return 0.0
    return -base_charge if is_antiparticle else base_charge


def external_pattern_from(externals: tuple["QEDExternalSlot", ...]) -> str:
    return "".join("A" if slot.flavor == "A" else "F" for slot in externals)


def process_family_from(pattern: str) -> str:
    n_photons = pattern.count("A")
    if n_photons == 0:
        return PROCESS_FERMION_FERMION
    if n_photons == 2:
        return PROCESS_FERMION_PHOTON
    if n_photons == 4:
        return PROCESS_PHOTON_PHOTON
    return PROCESS_MIXED


def infer_channel(vertex_external_slots: tuple[tuple[int, ...], tuple[int, ...]]) -> str:
    incoming = {0, 1}
    v0 = set(vertex_external_slots[0])
    v1 = set(vertex_external_slots[1])
    if incoming <= v0 or incoming <= v1:
        return "s"
    if {0, 2} <= v0 or {0, 2} <= v1:
        return "t"
    if {0, 3} <= v0 or {0, 3} <= v1:
        return "u"
    return "unknown"


def external_flow_sign(external: "QEDExternalSlot") -> int:
    if external.particle_type == PARTICLE_PHOTON:
        return 0
    if external.particle_type == PARTICLE_FERMION:
        return 1 if external.is_incoming else -1
    return -1 if external.is_incoming else 1


@dataclass(frozen=True)
class QEDExternalSlot:
    slot_index: int
    momentum_label: int
    flavor: str
    is_incoming: bool
    is_antiparticle: bool
    is_conjugate: bool
    label: str = ""

    @property
    def particle_type(self) -> str:
        return particle_type_from(self.flavor, self.is_antiparticle)

    @property
    def mass_symbol(self) -> str:
        return MASS_SYMBOLS.get(self.flavor, "0")

    @property
    def charge(self) -> float:
        return signed_charge(self.flavor, self.is_antiparticle)


@dataclass(frozen=True)
class QEDVertexSlot:
    slot_index: int
    raw_vertex_id: int
    external_slots: tuple[int, ...]
    fermion_line_id: int = -1
    interaction_type: str = "ffA"


@dataclass(frozen=True)
class QEDPropagatorSlot:
    flavor: str
    endpoint_vertices: tuple[int, int]
    endpoint_signs: tuple[int, int]
    raw_is_antiparticle: bool = False

    @property
    def particle_type(self) -> str:
        if self.flavor == "A":
            return PARTICLE_PHOTON
        return PARTICLE_FERMION

    @property
    def is_photon(self) -> bool:
        return self.flavor == "A"

    @property
    def is_fermion(self) -> bool:
        return self.flavor != "A"

    @property
    def charge(self) -> float:
        if self.flavor == "A":
            return 0.0
        return BASE_CHARGES.get(self.flavor, 0.0)

    @property
    def species(self) -> str:
        return "photon" if self.is_photon else "fermion"


@dataclass(frozen=True)
class QEDTarget:
    prefactor: str
    denominator: str
    numerator_infix: str
    full_infix: str


@dataclass(frozen=True)
class QEDInteraction:
    sample_id: str
    source_file: str
    source_line_index: int
    externals: tuple[QEDExternalSlot, ...]
    vertices: tuple[QEDVertexSlot, ...]
    propagator: QEDPropagatorSlot | None
    raw_interaction: str
    raw_topology: str
    raw_amplitude: str
    raw_squared: str
    channel: str
    process_family: str
    external_pattern: str
    external_to_vertex: tuple[int, int, int, int]
    fermion_line_ids: tuple[int, int, int, int]

    @property
    def topology_signature(self) -> tuple[str, str, str]:
        propagator_kind = "unknown"
        if self.propagator is not None:
            propagator_kind = self.propagator.species
        return (self.external_pattern, propagator_kind, self.channel)

    @property
    def vertex_external_slots(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return tuple(vertex.external_slots for vertex in self.vertices)  # type: ignore[return-value]

    @property
    def incoming(self) -> tuple[QEDExternalSlot, ...]:
        return tuple(slot for slot in self.externals if slot.is_incoming)

    @property
    def outgoing(self) -> tuple[QEDExternalSlot, ...]:
        return tuple(slot for slot in self.externals if not slot.is_incoming)


ExternalParticle = QEDExternalSlot
Vertex = QEDVertexSlot
Propagator = QEDPropagatorSlot
FeynmanDiagram = QEDInteraction
