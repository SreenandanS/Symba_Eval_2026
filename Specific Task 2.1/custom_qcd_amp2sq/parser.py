"""
QCD Feynman diagram parser.

Parses the SYMBA data format:
  Section 1: External particles   (Interaction: p1 p2 to p3 p4)
  Section 2: Graph topology       (Vertex V_0: ..., Vertex V_1: ...)
  Section 3: Symbolic amplitude M
  Section 4: Squared amplitude |M|²
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


QUARK_FLAVORS = ("u", "d", "s", "t", "c", "b")
GAUGE_BOSONS = ("G",)
ALL_PARTICLES = QUARK_FLAVORS + GAUGE_BOSONS

MASS_MAP = {f: f"m_{f}" for f in QUARK_FLAVORS}
MASS_MAP["G"] = "0"

PARTICLE_QUARK = "quark"
PARTICLE_ANTIQUARK = "antiquark"
PARTICLE_GLUON = "gluon"


@dataclass
class ExternalParticle:
    flavor: str
    is_antiparticle: bool
    momentum_label: int
    is_incoming: bool
    color_label: str = ""
    spinor_label: str = ""
    is_conjugate: bool = False

    @property
    def particle_type(self) -> str:
        if self.flavor == "G":
            return PARTICLE_GLUON
        return PARTICLE_ANTIQUARK if self.is_antiparticle else PARTICLE_QUARK

    @property
    def mass_symbol(self) -> str:
        return MASS_MAP.get(self.flavor, "0")


@dataclass
class Propagator:
    flavor: str
    is_antiparticle: bool
    vertex_id: int

    @property
    def particle_type(self) -> str:
        if self.flavor == "G":
            return PARTICLE_GLUON
        return PARTICLE_ANTIQUARK if self.is_antiparticle else PARTICLE_QUARK

    @property
    def is_gluon(self) -> bool:
        return self.flavor == "G"

    @property
    def is_quark(self) -> bool:
        return self.flavor in QUARK_FLAVORS


@dataclass
class Vertex:
    vertex_id: int
    external_legs: List[int] = field(default_factory=list)
    propagator: Optional[Propagator] = None


@dataclass
class FeynmanDiagram:
    externals: List[ExternalParticle] = field(default_factory=list)
    vertices: List[Vertex] = field(default_factory=list)
    raw_interaction: str = ""
    raw_topology: str = ""
    raw_amplitude: str = ""
    raw_squared: str = ""
    channel: str = ""

    @property
    def incoming(self) -> List[ExternalParticle]:
        return [p for p in self.externals if p.is_incoming]

    @property
    def outgoing(self) -> List[ExternalParticle]:
        return [p for p in self.externals if not p.is_incoming]

    @property
    def propagator_type(self) -> str:
        for vertex in self.vertices:
            if vertex.propagator:
                return vertex.propagator.particle_type
        return "unknown"

    def get_channel(self) -> str:
        if len(self.vertices) < 2:
            return "unknown"
        v0_legs = set(self.vertices[0].external_legs)
        v1_legs = set(self.vertices[1].external_legs)
        incoming_momenta = {p.momentum_label for p in self.incoming}
        if incoming_momenta <= v0_legs or incoming_momenta <= v1_legs:
            return "s"
        if (1 in v0_legs and 3 in v0_legs) or (1 in v1_legs and 3 in v1_legs):
            return "t"
        if (1 in v0_legs and 4 in v0_legs) or (1 in v1_legs and 4 in v1_legs):
            return "u"
        return "unknown"


_EXT_PARTICLE_RE = re.compile(
    r"(AntiPart\s+)?"
    r"(\w+)"
    r"_\{([^,}]+),([^}]+)\}"
    r"\(X\)"
    r"(\^\(\*\))?"
)
_VERTEX_RE = re.compile(r"Vertex\s+V_(\d+)\s*:")
_LEG_RE = re.compile(
    r"(AntiPart\s+)?"
    r"(OffShell\s+)?"
    r"(\w+)"
    r"\(([^)]+)\)"
)


def _parse_external_particles(section: str) -> List[ExternalParticle]:
    body = section.strip()
    if body.startswith("Interaction"):
        body = body.split(":", 1)[1].strip()
    parts = body.split(" to ")
    if len(parts) != 2:
        parts = re.split(r"\s+to\s+", body)

    particles = []
    momentum = 1
    for group_idx, group in enumerate(parts):
        is_incoming = group_idx == 0
        for match in _EXT_PARTICLE_RE.finditer(group):
            particles.append(
                ExternalParticle(
                    flavor=match.group(2),
                    is_antiparticle=match.group(1) is not None,
                    momentum_label=momentum,
                    is_incoming=is_incoming,
                    color_label=match.group(3),
                    spinor_label=match.group(4),
                    is_conjugate=match.group(5) is not None,
                )
            )
            momentum += 1
    return particles


def _parse_topology(section: str) -> List[Vertex]:
    vertices: List[Vertex] = []
    vertex_blocks = _VERTEX_RE.split(section)
    idx = 1
    while idx < len(vertex_blocks) - 1:
        vid = int(vertex_blocks[idx])
        content = vertex_blocks[idx + 1]
        idx += 2
        vertex = Vertex(vertex_id=vid)
        for match in _LEG_RE.finditer(content):
            is_anti = match.group(1) is not None
            is_offshell = match.group(2) is not None
            flavor = match.group(3)
            location = match.group(4)
            if is_offshell:
                v_match = re.match(r"V_(\d+)", location)
                v_id = int(v_match.group(1)) if v_match else vid
                vertex.propagator = Propagator(
                    flavor=flavor,
                    is_antiparticle=is_anti,
                    vertex_id=v_id,
                )
            else:
                x_match = re.match(r"X_(\d+)", location)
                if x_match:
                    vertex.external_legs.append(int(x_match.group(1)))
        vertices.append(vertex)
    return vertices


def parse_diagram(line: str) -> FeynmanDiagram:
    raw = line.strip()
    interaction_end = raw.find(" : Vertex")
    if interaction_end == -1:
        interaction_end = raw.find(":Vertex")

    sec1 = raw[:interaction_end].strip()
    rest = raw[interaction_end + 3 :].strip()
    v1_pos = rest.find("Vertex V_1")
    if v1_pos == -1:
        remaining_parts = rest.split(" : ", 2)
        sec2 = remaining_parts[0] if len(remaining_parts) > 0 else ""
        sec3 = remaining_parts[1] if len(remaining_parts) > 1 else ""
        sec4 = remaining_parts[2] if len(remaining_parts) > 2 else ""
    else:
        search_start = v1_pos + 10
        colon_pos = rest.find(" : ", search_start)
        if colon_pos == -1:
            sec2 = rest
            sec3 = ""
            sec4 = ""
        else:
            sec2 = rest[:colon_pos].strip()
            amplitude_and_squared = rest[colon_pos + 3 :]
            last_colon = amplitude_and_squared.rfind(" : ")
            if last_colon == -1:
                sec3 = amplitude_and_squared
                sec4 = ""
            else:
                sec3 = amplitude_and_squared[:last_colon].strip()
                sec4 = amplitude_and_squared[last_colon + 3 :].strip()

    diagram = FeynmanDiagram(
        raw_interaction=sec1,
        raw_topology=sec2,
        raw_amplitude=sec3,
        raw_squared=sec4,
    )
    diagram.externals = _parse_external_particles(sec1)
    diagram.vertices = _parse_topology(sec2)
    diagram.channel = diagram.get_channel()
    return diagram


def parse_file(filepath: str | Path) -> List[FeynmanDiagram]:
    filepath = Path(filepath)
    diagrams: List[FeynmanDiagram] = []
    with open(filepath, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                diagrams.append(parse_diagram(line))
            except Exception as exc:
                print(f"Warning: failed to parse line: {exc}")
                continue
    return diagrams


def parse_all_qcd(data_dir: str | Path) -> List[FeynmanDiagram]:
    data_dir = Path(data_dir)
    all_diagrams: List[FeynmanDiagram] = []
    for file_path in sorted(data_dir.glob("QCD-2-to-2-diag-TreeLevel-*.txt")):
        diagrams = parse_file(file_path)
        all_diagrams.extend(diagrams)
        print(f"  Parsed {file_path.name}: {len(diagrams)} diagrams")
    print(f"Total QCD diagrams: {len(all_diagrams)}")
    return all_diagrams
