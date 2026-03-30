"""Fixed-slot metadata for amplitude -> squared-amplitude experiments."""

from __future__ import annotations

from dataclasses import dataclass

from .parser import FeynmanDiagram


@dataclass(frozen=True)
class Amp2SqInteraction:
    sample_id: str
    diagram: FeynmanDiagram
    external_pattern: str
    channel: str
    propagator_type: str


def build_interaction(diagram: FeynmanDiagram, sample_id: str) -> Amp2SqInteraction:
    external_pattern = "".join(
        "G" if particle.flavor == "G" else ("Q" if particle.is_antiparticle else "q")
        for particle in sorted(diagram.externals, key=lambda item: item.momentum_label)
    )
    return Amp2SqInteraction(
        sample_id=sample_id,
        diagram=diagram,
        external_pattern=external_pattern,
        channel=diagram.channel or "unknown",
        propagator_type=diagram.propagator_type,
    )
