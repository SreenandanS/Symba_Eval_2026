"""Parser for canonical fixed-slot QED interactions."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .interaction import (
    ALL_FLAVORS,
    GAUGE_BOSONS,
    LEPTON_FLAVORS,
    PARTICLE_ANTIFERMION,
    PARTICLE_FERMION,
    PARTICLE_PHOTON,
    PROCESS_FERMION_FERMION,
    PROCESS_FERMION_PHOTON,
    PROCESS_MIXED,
    PROCESS_PHOTON_PHOTON,
    QUARK_FLAVORS,
    ExternalParticle,
    FeynmanDiagram,
    Propagator,
    QEDExternalSlot,
    QEDInteraction,
    QEDPropagatorSlot,
    QEDVertexSlot,
    Vertex,
    external_pattern_from,
    infer_channel,
    process_family_from,
)


_INTERACTION_TOKEN_RE = re.compile(
    r"(AntiPart\s+)?([A-Za-z]+)_([^\s()]+)\(X\)(\^\(\*\))?"
)
_VERTEX_RE = re.compile(r"Vertex\s+V_(\d+)\s*:")
_LEG_RE = re.compile(
    r"(AntiPart\s+)?"
    r"(OffShell\s+)?"
    r"([A-Za-z]+)"
    r"\(([^)]+)\)"
)


class _RawVertex:
    def __init__(self, raw_vertex_id: int):
        self.raw_vertex_id = raw_vertex_id
        self.external_slots: list[int] = []


def _split_sections(raw: str) -> tuple[str, str, str, str]:
    raw = raw.strip()
    interaction_end = raw.find(" : Vertex")
    if interaction_end == -1:
        interaction_end = raw.find(":Vertex")
    if interaction_end == -1:
        raise ValueError("Could not find interaction/topology boundary.")

    sec1 = raw[:interaction_end].strip()
    rest = raw[interaction_end + 3 :].strip()

    v1_pos = rest.find("Vertex V_1")
    if v1_pos == -1:
        remaining_parts = rest.split(" : ", 2)
        sec2 = remaining_parts[0] if len(remaining_parts) > 0 else ""
        sec3 = remaining_parts[1] if len(remaining_parts) > 1 else ""
        sec4 = remaining_parts[2] if len(remaining_parts) > 2 else ""
        return sec1, sec2, sec3, sec4

    search_start = v1_pos + len("Vertex V_1")
    topology_end = rest.find(" : ", search_start)
    if topology_end == -1:
        return sec1, rest, "", ""

    sec2 = rest[:topology_end].strip()
    amplitude_and_squared = rest[topology_end + 3 :]
    last_colon = amplitude_and_squared.rfind(" : ")
    if last_colon == -1:
        return sec1, sec2, amplitude_and_squared.strip(), ""
    sec3 = amplitude_and_squared[:last_colon].strip()
    sec4 = amplitude_and_squared[last_colon + 3 :].strip()
    return sec1, sec2, sec3, sec4


def _parse_external_slots(section: str) -> tuple[QEDExternalSlot, ...]:
    body = section.strip()
    if body.startswith("Interaction"):
        body = body.split(":", 1)[1].strip()

    parts = re.split(r"\s+to\s+", body)
    if len(parts) != 2:
        raise ValueError(f"Unexpected interaction format: {section}")

    slots: list[QEDExternalSlot] = []
    momentum = 1
    for group_idx, group in enumerate(parts):
        is_incoming = group_idx == 0
        for anti, flavor, label, conjugate in _INTERACTION_TOKEN_RE.findall(group):
            slots.append(
                QEDExternalSlot(
                    slot_index=momentum - 1,
                    momentum_label=momentum,
                    flavor=flavor,
                    is_incoming=is_incoming,
                    is_antiparticle=bool(anti),
                    is_conjugate=bool(conjugate),
                    label=label,
                )
            )
            momentum += 1

    if len(slots) != 4:
        raise ValueError(
            f"Expected exactly 4 external particles, found {len(slots)}."
        )
    return tuple(slots)


def _parse_topology(
    section: str,
) -> tuple[list[_RawVertex], list[tuple[str, bool, int]]]:
    raw_vertices: list[_RawVertex] = []
    propagator_mentions: list[tuple[str, bool, int]] = []
    vertex_blocks = _VERTEX_RE.split(section)
    idx = 1
    while idx < len(vertex_blocks) - 1:
        vertex_id = int(vertex_blocks[idx])
        content = vertex_blocks[idx + 1]
        idx += 2

        raw_vertex = _RawVertex(raw_vertex_id=vertex_id)
        for anti, offshell, flavor, location in _LEG_RE.findall(content):
            if offshell:
                match = re.match(r"V_(\d+)", location)
                mention_vertex_id = int(match.group(1)) if match else vertex_id
                propagator_mentions.append((flavor, bool(anti), mention_vertex_id))
                continue

            leg_match = re.match(r"X_(\d+)", location)
            if leg_match is None:
                continue
            raw_vertex.external_slots.append(int(leg_match.group(1)) - 1)
        raw_vertices.append(raw_vertex)

    if len(raw_vertices) != 2:
        raise ValueError(
            f"Expected exactly 2 vertices in audited QED scope, found {len(raw_vertices)}."
        )
    return raw_vertices, propagator_mentions


def _canonicalize_vertices(
    raw_vertices: list[_RawVertex],
) -> tuple[tuple[QEDVertexSlot, ...], dict[int, int], tuple[int, int, int, int]]:
    ordered = sorted(
        raw_vertices,
        key=lambda vertex: (tuple(sorted(vertex.external_slots)), vertex.raw_vertex_id),
    )
    raw_to_canonical = {
        vertex.raw_vertex_id: canonical_idx
        for canonical_idx, vertex in enumerate(ordered)
    }
    external_to_vertex = [-1, -1, -1, -1]
    vertex_slots: list[QEDVertexSlot] = []
    for canonical_idx, vertex in enumerate(ordered):
        sorted_slots = tuple(sorted(vertex.external_slots))
        for ext_slot in sorted_slots:
            external_to_vertex[ext_slot] = canonical_idx
        vertex_slots.append(
            QEDVertexSlot(
                slot_index=canonical_idx,
                raw_vertex_id=vertex.raw_vertex_id,
                external_slots=sorted_slots,
            )
        )

    if any(slot < 0 for slot in external_to_vertex):
        raise ValueError(
            f"Expected every external slot to attach to a vertex, got {external_to_vertex}."
        )
    return tuple(vertex_slots), raw_to_canonical, tuple(external_to_vertex)


def _build_propagator(
    propagator_mentions: list[tuple[str, bool, int]],
    raw_to_canonical: dict[int, int],
) -> QEDPropagatorSlot | None:
    if not propagator_mentions:
        return None

    flavor = propagator_mentions[0][0]
    raw_is_antiparticle = propagator_mentions[0][1]
    signs_by_vertex: dict[int, int] = {}
    for mention_flavor, anti_flag, raw_vertex_id in propagator_mentions:
        if mention_flavor != flavor:
            raise ValueError(
                f"Mixed propagator flavors in audited single-propagator scope: "
                f"{flavor} vs {mention_flavor}."
            )
        canonical_vertex = raw_to_canonical[raw_vertex_id]
        if flavor == "A":
            signs_by_vertex[canonical_vertex] = 0
        else:
            signs_by_vertex[canonical_vertex] = -1 if anti_flag else 1

    endpoint_vertices = tuple(sorted(signs_by_vertex.keys()))
    if len(endpoint_vertices) != 2:
        raise ValueError(
            f"Expected propagator to touch exactly 2 canonical vertices, got {endpoint_vertices}."
        )
    endpoint_signs = tuple(signs_by_vertex[vertex] for vertex in endpoint_vertices)
    return QEDPropagatorSlot(
        flavor=flavor,
        endpoint_vertices=(endpoint_vertices[0], endpoint_vertices[1]),
        endpoint_signs=(endpoint_signs[0], endpoint_signs[1]),
        raw_is_antiparticle=raw_is_antiparticle,
    )


def _fermion_line_ids(
    externals: tuple[QEDExternalSlot, ...],
    vertices: tuple[QEDVertexSlot, ...],
    propagator: QEDPropagatorSlot | None,
) -> tuple[tuple[int, int, int, int], tuple[QEDVertexSlot, ...]]:
    line_ids = [-1, -1, -1, -1]
    vertex_line_ids = [-1, -1]

    if propagator is not None and propagator.is_fermion:
        for external in externals:
            if external.flavor != "A":
                line_ids[external.slot_index] = 0
        vertex_line_ids = [0, 0]
    else:
        next_line_id = 0
        for vertex in vertices:
            local_fermions = [
                ext_slot
                for ext_slot in vertex.external_slots
                if externals[ext_slot].flavor != "A"
            ]
            if not local_fermions:
                continue
            vertex_line_ids[vertex.slot_index] = next_line_id
            for ext_slot in local_fermions:
                line_ids[ext_slot] = next_line_id
            next_line_id += 1

    updated_vertices = tuple(
        QEDVertexSlot(
            slot_index=vertex.slot_index,
            raw_vertex_id=vertex.raw_vertex_id,
            external_slots=vertex.external_slots,
            fermion_line_id=vertex_line_ids[vertex.slot_index],
            interaction_type=vertex.interaction_type,
        )
        for vertex in vertices
    )
    return tuple(line_ids), updated_vertices


def parse_diagram(
    line: str,
    source_file: str = "<memory>",
    source_line_index: int = 0,
) -> QEDInteraction:
    sec1, sec2, sec3, sec4 = _split_sections(line)
    externals = _parse_external_slots(sec1)
    raw_vertices, propagator_mentions = _parse_topology(sec2)
    vertices, raw_to_canonical, external_to_vertex = _canonicalize_vertices(raw_vertices)
    propagator = _build_propagator(propagator_mentions, raw_to_canonical)
    external_pattern = external_pattern_from(externals)
    process_family = process_family_from(external_pattern)
    channel = infer_channel(tuple(vertex.external_slots for vertex in vertices))
    fermion_line_ids, vertices = _fermion_line_ids(externals, vertices, propagator)

    return QEDInteraction(
        sample_id=f"{source_file}:{source_line_index}",
        source_file=source_file,
        source_line_index=source_line_index,
        externals=externals,
        vertices=vertices,
        propagator=propagator,
        raw_interaction=sec1,
        raw_topology=sec2,
        raw_amplitude=sec3,
        raw_squared=sec4,
        channel=channel,
        process_family=process_family,
        external_pattern=external_pattern,
        external_to_vertex=external_to_vertex,
        fermion_line_ids=fermion_line_ids,
    )


def parse_file(filepath: str | Path) -> list[QEDInteraction]:
    filepath = Path(filepath)
    diagrams: list[QEDInteraction] = []
    with open(filepath, "r") as handle:
        for line_index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                diagrams.append(
                    parse_diagram(
                        line,
                        source_file=filepath.name,
                        source_line_index=line_index,
                    )
                )
            except Exception as exc:
                print(f"Warning: failed to parse QED line in {filepath.name}: {exc}")
    return diagrams


@lru_cache(maxsize=None)
def _parse_all_qed_cached(data_dir_str: str) -> tuple[QEDInteraction, ...]:
    data_dir = Path(data_dir_str)
    all_diagrams: list[QEDInteraction] = []
    for filepath in sorted(data_dir.glob("QED-2-to-2-diag-TreeLevel-*.txt")):
        diagrams = parse_file(filepath)
        all_diagrams.extend(diagrams)
        print(f"  Parsed {filepath.name}: {len(diagrams)} diagrams")
    print(f"Total QED diagrams: {len(all_diagrams)}")
    return tuple(all_diagrams)


def parse_all_qed(data_dir: str | Path) -> list[QEDInteraction]:
    normalized = str(Path(data_dir).expanduser().resolve())
    return list(_parse_all_qed_cached(normalized))


__all__ = [
    "ALL_FLAVORS",
    "GAUGE_BOSONS",
    "LEPTON_FLAVORS",
    "PARTICLE_FERMION",
    "PARTICLE_ANTIFERMION",
    "PARTICLE_PHOTON",
    "PROCESS_FERMION_FERMION",
    "PROCESS_FERMION_PHOTON",
    "PROCESS_PHOTON_PHOTON",
    "PROCESS_MIXED",
    "QUARK_FLAVORS",
    "ExternalParticle",
    "Vertex",
    "Propagator",
    "FeynmanDiagram",
    "QEDInteraction",
    "parse_diagram",
    "parse_file",
    "parse_all_qed",
]
