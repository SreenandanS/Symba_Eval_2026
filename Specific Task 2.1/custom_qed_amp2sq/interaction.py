"""QED amplitude-side interaction contracts and metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass


LEPTON_FLAVORS = ("e", "mu", "tt")
QUARK_FLAVORS = ("u", "d", "s", "c", "b", "t")
GAUGE_BOSONS = ("A",)
ALL_FLAVORS = ("A", "e", "mu", "t", "tt", "u", "d", "s", "c", "b")

PROCESS_FERMION_FERMION = "fermion_fermion"
PROCESS_FERMION_PHOTON = "fermion_photon"
PROCESS_PHOTON_PHOTON = "photon_photon"
PROCESS_MIXED = "mixed"


def external_pattern_from(slot_kinds: dict[int, str]) -> str:
    pattern = []
    for pindex in range(1, 5):
        pattern.append(slot_kinds.get(pindex, "F"))
    return "".join(pattern)


def process_family_from(pattern: str) -> str:
    n_photons = pattern.count("A")
    if n_photons == 0:
        return PROCESS_FERMION_FERMION
    if n_photons == 2:
        return PROCESS_FERMION_PHOTON
    if n_photons == 4:
        return PROCESS_PHOTON_PHOTON
    return PROCESS_MIXED


@dataclass(frozen=True)
class QEDAmplitudeRecord:
    sample_id: str
    source_file: str
    source_line_index: int
    raw_interaction: str
    raw_topology: str
    raw_amplitude: str
    raw_squared: str
