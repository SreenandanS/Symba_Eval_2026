"""
QCD Feynman diagram parser.

Parses the SYMBA data format:
  Section 1: External particles   (Interaction: p1 p2 to p3 p4)
  Section 2: Graph topology       (Vertex V_0: ..., Vertex V_1: ...)
  Section 3: Symbolic amplitude M
  Section 4: Squared amplitude |M|²

Each line in a QCD-2-to-2-diag-TreeLevel-*.txt file encodes one tree-level
2→2 Feynman diagram.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Particle Representation
# ---------------------------------------------------------------------------

# All 6 quark flavors + gluon
QUARK_FLAVORS = ("u", "d", "s", "t", "c", "b")
GAUGE_BOSONS = ("G",)
ALL_PARTICLES = QUARK_FLAVORS + GAUGE_BOSONS

# Mass symbols
MASS_MAP = {f: f"m_{f}" for f in QUARK_FLAVORS}
MASS_MAP["G"] = "0"  # gluons are massless

# Particle type enum-like
PARTICLE_QUARK = "quark"
PARTICLE_ANTIQUARK = "antiquark"
PARTICLE_GLUON = "gluon"


@dataclass
class ExternalParticle:
    """An external leg of the Feynman diagram."""
    flavor: str              # u, d, s, t, c, b, G
    is_antiparticle: bool    # True for antiquarks / anti-gluons
    momentum_label: int      # 1-4  (p_1 through p_4)
    is_incoming: bool        # True for p_1, p_2; False for p_3, p_4
    color_label: str = ""    # e.g. "A_74"
    spinor_label: str = ""   # e.g. "alpha_104"
    is_conjugate: bool = False  # ^(*) suffix on spinors

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
    """The internal (off-shell) propagator connecting two vertices."""
    flavor: str              # quark flavor or G
    is_antiparticle: bool
    vertex_id: int           # which vertex it emanates from

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
    """A QCD interaction vertex."""
    vertex_id: int
    external_legs: List[int] = field(default_factory=list)   # momentum labels
    propagator: Optional[Propagator] = None


@dataclass
class FeynmanDiagram:
    """Complete parsed Feynman diagram."""
    # External particles
    externals: List[ExternalParticle] = field(default_factory=list)
    # Vertices (always 2 for tree-level 2→2)
    vertices: List[Vertex] = field(default_factory=list)
    # Raw string sections
    raw_interaction: str = ""
    raw_topology: str = ""
    raw_amplitude: str = ""
    raw_squared: str = ""
    # Derived
    channel: str = ""  # "s", "t", "u"

    @property
    def incoming(self) -> List[ExternalParticle]:
        return [p for p in self.externals if p.is_incoming]

    @property
    def outgoing(self) -> List[ExternalParticle]:
        return [p for p in self.externals if not p.is_incoming]

    @property
    def propagator_type(self) -> str:
        """Return the type of the internal propagator."""
        for v in self.vertices:
            if v.propagator:
                return v.propagator.particle_type
        return "unknown"

    def get_channel(self) -> str:
        """Determine s/t/u channel from vertex-leg assignment."""
        if len(self.vertices) < 2:
            return "unknown"
        v0_legs = set(self.vertices[0].external_legs)
        v1_legs = set(self.vertices[1].external_legs)
        incoming_momenta = {p.momentum_label for p in self.incoming}
        outgoing_momenta = {p.momentum_label for p in self.outgoing}

        # s-channel: both incoming at one vertex, both outgoing at other
        if incoming_momenta <= v0_legs or incoming_momenta <= v1_legs:
            return "s"
        # t-channel: one incoming + one outgoing at each vertex,
        # with p1 and p3 at same vertex
        if (1 in v0_legs and 3 in v0_legs) or (1 in v1_legs and 3 in v1_legs):
            return "t"
        # u-channel: p1 and p4 at same vertex
        if (1 in v0_legs and 4 in v0_legs) or (1 in v1_legs and 4 in v1_legs):
            return "u"
        return "unknown"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Regex for external particle in section 1
_EXT_PARTICLE_RE = re.compile(
    r"(AntiPart\s+)?"          # optional AntiPart prefix
    r"(\w+)"                   # flavor
    r"_\{([^,}]+),([^}]+)\}"   # _{color_label, spinor_label}
    r"\(X\)"                   # (X)
    r"(\^\(\*\))?"             # optional ^(*)
)

# Regex for vertex block in section 2
_VERTEX_RE = re.compile(r"Vertex\s+V_(\d+)\s*:")

# Regex for leg in a vertex: e.g. "u(X_1)" or "AntiPart u(X_2)" or "OffShell G(V_0)"
_LEG_RE = re.compile(
    r"(AntiPart\s+)?"         # optional AntiPart
    r"(OffShell\s+)?"         # optional OffShell
    r"(\w+)"                  # particle flavor
    r"\(([^)]+)\)"            # (X_i) or (V_n)
)


def _parse_external_particles(section: str) -> List[ExternalParticle]:
    """Parse section 1: 'Interaction: p1 p2 to p3 p4'."""
    # Remove the "Interaction:" prefix
    body = section.strip()
    if body.startswith("Interaction"):
        body = body.split(":", 1)[1].strip()

    # Split on " to " to get incoming / outgoing
    parts = body.split(" to ")
    if len(parts) != 2:
        # Try splitting differently
        parts = re.split(r"\s+to\s+", body)

    particles = []
    momentum = 1

    for group_idx, group in enumerate(parts):
        is_incoming = (group_idx == 0)
        # Find all particle matches in this group
        for m in _EXT_PARTICLE_RE.finditer(group):
            anti = m.group(1) is not None
            flavor = m.group(2)
            color_label = m.group(3)
            spinor_label = m.group(4)
            conjugate = m.group(5) is not None
            particles.append(ExternalParticle(
                flavor=flavor,
                is_antiparticle=anti,
                momentum_label=momentum,
                is_incoming=is_incoming,
                color_label=color_label,
                spinor_label=spinor_label,
                is_conjugate=conjugate,
            ))
            momentum += 1

    return particles


def _parse_topology(section: str) -> List[Vertex]:
    """Parse section 2: 'Vertex V_0: ..., Vertex V_1: ...'."""
    vertices = []

    # Split into vertex blocks
    vertex_blocks = _VERTEX_RE.split(section)
    # vertex_blocks[0] is stuff before first vertex (empty)
    # then alternating: vertex_id_str, content, vertex_id_str, content, ...
    idx = 1
    while idx < len(vertex_blocks) - 1:
        vid = int(vertex_blocks[idx])
        content = vertex_blocks[idx + 1]
        idx += 2

        vertex = Vertex(vertex_id=vid)
        # Parse each leg in this vertex
        for m in _LEG_RE.finditer(content):
            is_anti = m.group(1) is not None
            is_offshell = m.group(2) is not None
            flavor = m.group(3)
            location = m.group(4)  # X_i or V_n

            if is_offshell:
                # Internal propagator
                v_match = re.match(r"V_(\d+)", location)
                v_id = int(v_match.group(1)) if v_match else vid
                vertex.propagator = Propagator(
                    flavor=flavor,
                    is_antiparticle=is_anti,
                    vertex_id=v_id,
                )
            else:
                # External leg
                x_match = re.match(r"X_(\d+)", location)
                if x_match:
                    vertex.external_legs.append(int(x_match.group(1)))

        vertices.append(vertex)

    return vertices


def parse_diagram(line: str) -> FeynmanDiagram:
    """Parse a single line from a QCD data file into a FeynmanDiagram."""
    # Split on " : " (colon with spaces)
    sections = line.strip().split(" : ")
    if len(sections) < 4:
        # Try splitting on just ":"
        sections = line.strip().split(":")
        if len(sections) >= 4:
            # Recombine — first section is "Interaction", rest need care
            # Actually the first token before first ":" is "Interaction"
            pass

    # For the SYMBA format, we expect exactly 4 colon-separated sections
    # But colons also appear inside expressions, so we need smarter splitting
    # The format is:  Interaction: <particles> : <topology> : <amplitude> : <squared>
    # Let's split on " : " which is the section delimiter
    raw = line.strip()

    # Find section boundaries using the pattern markers
    # Section 1 starts with "Interaction"
    # Section 2 starts with "Vertex V_0"
    # Section 3 and 4 are the amplitude and squared amplitude
    
    interaction_end = raw.find(" : Vertex")
    if interaction_end == -1:
        interaction_end = raw.find(":Vertex")
    
    sec1 = raw[:interaction_end].strip()
    rest = raw[interaction_end + 3:].strip()  # skip " : "

    # Find where topology ends and amplitude begins
    # The amplitude section starts after the last vertex description
    # Look for the pattern that indicates start of amplitude (contains gamma, spinor products, etc.)
    # Actually, let's find the second " : " boundary
    # Topology section contains "Vertex V_0: ... , Vertex V_1: ..."
    # After topology, the amplitude section starts

    # Split rest on " : " but we need to be careful
    # The topology ends and amplitude begins at a " : " boundary
    # Let's find "Vertex V_1:" and then the " : " after that vertex's content
    
    v1_pos = rest.find("Vertex V_1")
    if v1_pos == -1:
        # Fallback: split on " : "
        remaining_parts = rest.split(" : ", 2)
        sec2 = remaining_parts[0] if len(remaining_parts) > 0 else ""
        sec3 = remaining_parts[1] if len(remaining_parts) > 1 else ""
        sec4 = remaining_parts[2] if len(remaining_parts) > 2 else ""
    else:
        # Find the " : " after V_1's content
        # V_1 content ends at the next " : " that's not inside the vertex
        # The vertex content has patterns like "G(X_3)" and "OffShell G(V_1)"
        # The amplitude starts with patterns like "(-1)" or "1/2" or "i*g"
        
        # Search for " : " after V_1, skipping the one between V_0 and V_1
        search_start = v1_pos + 10  # skip "Vertex V_1"
        # Find the next " : " delimiter
        colon_pos = rest.find(" : ", search_start)
        if colon_pos == -1:
            sec2 = rest
            sec3 = ""
            sec4 = ""
        else:
            sec2 = rest[:colon_pos].strip()
            amplitude_and_squared = rest[colon_pos + 3:]
            # Now split amplitude from squared amplitude
            # The squared amplitude is the last section
            # Find the last " : " in the remaining text
            last_colon = amplitude_and_squared.rfind(" : ")
            if last_colon == -1:
                sec3 = amplitude_and_squared
                sec4 = ""
            else:
                sec3 = amplitude_and_squared[:last_colon].strip()
                sec4 = amplitude_and_squared[last_colon + 3:].strip()

    diagram = FeynmanDiagram(
        raw_interaction=sec1,
        raw_topology=sec2,
        raw_amplitude=sec3,
        raw_squared=sec4,
    )

    # Parse external particles
    diagram.externals = _parse_external_particles(sec1)

    # Parse topology
    diagram.vertices = _parse_topology(sec2)

    # Determine channel
    diagram.channel = diagram.get_channel()

    return diagram


def parse_file(filepath: str | Path) -> List[FeynmanDiagram]:
    """Parse an entire QCD data file."""
    filepath = Path(filepath)
    diagrams = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                diagram = parse_diagram(line)
                diagrams.append(diagram)
            except Exception as e:
                print(f"Warning: failed to parse line: {e}")
                continue
    return diagrams


def parse_all_qcd(data_dir: str | Path) -> List[FeynmanDiagram]:
    """Parse all QCD-2-to-2-diag-TreeLevel-*.txt files in a directory."""
    data_dir = Path(data_dir)
    all_diagrams = []
    for f in sorted(data_dir.glob("QCD-2-to-2-diag-TreeLevel-*.txt")):
        diagrams = parse_file(f)
        all_diagrams.extend(diagrams)
        print(f"  Parsed {f.name}: {len(diagrams)} diagrams")
    print(f"Total QCD diagrams: {len(all_diagrams)}")
    return all_diagrams
