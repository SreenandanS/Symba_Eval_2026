"""Canonical QED amplitude compiler for amplitude -> squared-amplitude learning."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import sympy as sp

from .interaction import external_pattern_from, process_family_from


ATOM_TYPE_GAMMA = "gamma"
ATOM_TYPE_MOMENTUM = "momentum"
ATOM_TYPE_SPINOR = "spinor"
ATOM_TYPE_POLARIZATION = "polarization"
ATOM_TYPES: tuple[str, ...] = (
    ATOM_TYPE_GAMMA,
    ATOM_TYPE_MOMENTUM,
    ATOM_TYPE_SPINOR,
    ATOM_TYPE_POLARIZATION,
)

CONNECTOR_KIND_LORENTZ = "lorentz"
CONNECTOR_KIND_DIRAC = "dirac"
CONNECTOR_SLOT_KINDS: dict[str, tuple[str, ...]] = {
    ATOM_TYPE_GAMMA: (
        CONNECTOR_KIND_LORENTZ,
        CONNECTOR_KIND_DIRAC,
        CONNECTOR_KIND_DIRAC,
    ),
    ATOM_TYPE_MOMENTUM: (CONNECTOR_KIND_LORENTZ,),
    ATOM_TYPE_SPINOR: (CONNECTOR_KIND_DIRAC,),
    ATOM_TYPE_POLARIZATION: (CONNECTOR_KIND_LORENTZ,),
}

FLAVOR_NONE = "<NONE>"
SPIN_ROLE_NONE = "<NONE>"

_MASS_SYMBOLS = {
    name: sp.symbols(name)
    for name in ("m_b", "m_c", "m_d", "m_e", "m_mu", "m_s", "m_t", "m_tt", "m_u")
}
_MANDELSTAM_SYMBOLS = {
    name: sp.symbols(name)
    for name in ("s_12", "s_13", "s_14", "s_23", "s_24", "s_34")
}
_E = sp.symbols("e")
_REG_PROP = sp.symbols("reg_prop")
SYMPY_LOCALS = {
    "e": _E,
    "i": sp.I,
    "reg_prop": _REG_PROP,
    **_MASS_SYMBOLS,
    **_MANDELSTAM_SYMBOLS,
}

_ATOM_PATTERNS = (
    re.compile(r"gamma_\{[^{}]*\}"),
    re.compile(r"[A-Za-z]+_\{[^{}]*\}\(p_[1-4]\)_[uv](?:\^\(\*\))?"),
    re.compile(r"A_\{[^{}]*\}\(p_[1-4]\)(?:\^\(\*\))?"),
    re.compile(r"p_[1-4]_[+%\\A-Za-z0-9_]+"),
)
_TENSOR_ATOM_RE = re.compile(
    r"([A-Za-z]+)_\{([^{}]+)\}\(p_(\d+)\)(?:_([uv]))?(\^\(\*\))?"
)
_GAMMA_RE = re.compile(r"gamma_\{([^{}]+)\}")
_MOM_RE = re.compile(r"p_(\d+)_(.+)")


def _fraction_from_sympy(value: sp.Expr) -> Fraction:
    value = sp.nsimplify(value)
    if isinstance(value, sp.Integer):
        return Fraction(int(value), 1)
    if isinstance(value, sp.Rational):
        return Fraction(int(value.p), int(value.q))
    raise ValueError(f"Expected rational value, got {value!r}")


def _rational_gcd(values: Sequence[Fraction]) -> Fraction:
    if not values:
        return Fraction(1, 1)
    numerators = [abs(value.numerator) for value in values if value.numerator != 0]
    if not numerators:
        return Fraction(1, 1)
    denominator_lcm = 1
    for value in values:
        denominator_lcm = math.lcm(denominator_lcm, value.denominator)
    scaled = [
        abs(value.numerator) * (denominator_lcm // value.denominator)
        for value in values
        if value.numerator != 0
    ]
    gcd_value = scaled[0]
    for current in scaled[1:]:
        gcd_value = math.gcd(gcd_value, current)
    return Fraction(gcd_value, denominator_lcm)


def _basis_symbol(expr: sp.Expr) -> str:
    return sp.sstr(expr, order="lex").replace("**", "^")


def _term_factors(expr: sp.Expr) -> list[sp.Expr]:
    if expr == 1:
        return []
    if isinstance(expr, sp.Mul):
        return list(expr.args)
    return [expr]


def _namespace_label(raw_label: str, kind: str) -> str:
    normalized = raw_label.strip().lstrip("+-")
    normalized = normalized.replace("%\\", "").replace("%", "")
    return f"{kind}:{normalized}"


def _split_brace_fields(body: str) -> list[str]:
    return [field.strip() for field in body.split(",") if field.strip()]


def _channel_from_denominator_tokens(tokens: Sequence[str]) -> str:
    if "s_12" in tokens:
        return "s"
    if "s_13" in tokens:
        return "t"
    if "s_23" in tokens:
        return "u"
    return "unknown"


def _propagator_species_from_denominator(tokens: Sequence[str]) -> str:
    return "fermion" if any(token.startswith("m_") for token in tokens) else "photon"


def _family_scalar_symbol(symbol: str) -> str:
    if symbol.startswith("m_"):
        return "MASS"
    if symbol == "reg_prop":
        return "REG_PROP"
    if symbol.startswith("s_") or symbol.startswith("p_"):
        return "MOMENTUM"
    return symbol


def _denominator_tokens(expr: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[-()+*/^]", expr)


@dataclass(frozen=True)
class CanonicalQEDTensorAtom:
    atom_type: str
    flavor: str = FLAVOR_NONE
    pindex: int = 0
    spin_role: str = SPIN_ROLE_NONE
    conjugated: bool = False
    momentum_index: int = -1
    index_ids: tuple[int, int, int] = (-1, -1, -1)

    def connector_kinds(self) -> tuple[str, ...]:
        return CONNECTOR_SLOT_KINDS[self.atom_type]

    def to_typed_tokens(self) -> list[str]:
        tokens = [f"[{self.atom_type.upper()}]"]
        if self.flavor != FLAVOR_NONE:
            tokens.append(f"FLAVOR_{self.flavor.upper()}")
        if self.pindex > 0:
            tokens.append(f"PINDEX_{self.pindex}")
        if self.spin_role != SPIN_ROLE_NONE:
            tokens.append(f"ROLE_{self.spin_role.upper()}")
        tokens.append(f"CONJ_{1 if self.conjugated else 0}")
        if self.momentum_index >= 0:
            tokens.append(f"MOMENTUM_{self.momentum_index}")
        for index_id in self.index_ids:
            if index_id >= 0:
                tokens.append(f"INDEX_{index_id}")
        return tokens

    def skeleton_tokens(self) -> list[str]:
        tokens = [f"[{self.atom_type.upper()}]"]
        if self.flavor != FLAVOR_NONE:
            tokens.append(f"FLAVOR_{self.flavor.upper()}")
        if self.pindex > 0:
            tokens.append("PINDEX")
        if self.spin_role != SPIN_ROLE_NONE:
            tokens.append(f"ROLE_{self.spin_role.upper()}")
        tokens.append(f"CONJ_{1 if self.conjugated else 0}")
        if self.momentum_index >= 0:
            tokens.append("MOMENTUM")
        for index_id in self.index_ids:
            tokens.append("INDEX" if index_id >= 0 else "INDEX_NONE")
        return tokens

    def to_dict(self) -> dict:
        return {
            "atom_type": self.atom_type,
            "flavor": self.flavor,
            "pindex": self.pindex,
            "spin_role": self.spin_role,
            "conjugated": self.conjugated,
            "momentum_index": self.momentum_index,
            "index_ids": list(self.index_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CanonicalQEDTensorAtom":
        return cls(
            atom_type=payload["atom_type"],
            flavor=payload["flavor"],
            pindex=int(payload["pindex"]),
            spin_role=payload["spin_role"],
            conjugated=bool(payload["conjugated"]),
            momentum_index=int(payload.get("momentum_index", -1)),
            index_ids=tuple(int(value) for value in payload["index_ids"]),
        )


@dataclass(frozen=True)
class CanonicalQEDAmplitudeTerm:
    coefficient: Fraction
    scalar_factors: tuple[str, ...]
    atoms: tuple[CanonicalQEDTensorAtom, ...]
    dirac_chains: tuple[tuple[int, ...], ...]
    lorentz_chains: tuple[tuple[int, ...], ...]
    momentum_insertions: tuple[int, ...]
    external_wavefunctions: tuple[int, ...]

    def coefficient_tokens(self) -> list[str]:
        sign = "NEG" if self.coefficient < 0 else "POS"
        magnitude = abs(self.coefficient.numerator)
        if self.coefficient.denominator == 1:
            return [f"SIGN_{sign}", str(magnitude)]
        return [
            f"SIGN_{sign}",
            str(magnitude),
            "/",
            str(self.coefficient.denominator),
        ]

    def typed_tokens(self) -> list[str]:
        tokens: list[str] = ["[TERM]", "[COEFF]"]
        tokens.extend(self.coefficient_tokens())

        tokens.append("[SCALARS]")
        if self.scalar_factors:
            for symbol in self.scalar_factors:
                tokens.extend(["[SCALAR]", symbol])
        else:
            tokens.append("[NO_SCALARS]")

        tokens.append("[DIRAC]")
        if self.dirac_chains:
            for chain in self.dirac_chains:
                tokens.append("[CHAIN]")
                for atom_index in chain:
                    tokens.extend(self.atoms[atom_index].to_typed_tokens())
                tokens.append("[END_CHAIN]")
        else:
            tokens.append("[NO_DIRAC]")

        tokens.append("[LORENTZ]")
        if self.lorentz_chains:
            for chain in self.lorentz_chains:
                tokens.append("[CHAIN]")
                for atom_index in chain:
                    tokens.extend(self.atoms[atom_index].to_typed_tokens())
                tokens.append("[END_CHAIN]")
        else:
            tokens.append("[NO_LORENTZ]")

        tokens.append("[MOMENTA]")
        if self.momentum_insertions:
            for atom_index in self.momentum_insertions:
                tokens.extend(self.atoms[atom_index].to_typed_tokens())
        else:
            tokens.append("[NO_MOMENTA]")

        tokens.append("[EXT]")
        if self.external_wavefunctions:
            for atom_index in self.external_wavefunctions:
                tokens.extend(self.atoms[atom_index].to_typed_tokens())
        else:
            tokens.append("[NO_EXT]")
        tokens.append("[END_TERM]")
        return tokens

    def skeleton_tokens(self) -> list[str]:
        tokens: list[str] = ["[TERM]", "[COEFF]", "SIGN", "INT", "[SCALARS]"]
        for symbol in self.scalar_factors:
            tokens.extend(["[SCALAR]", _family_scalar_symbol(symbol)])
        tokens.append("[DIRAC]")
        for chain in self.dirac_chains:
            tokens.append("[CHAIN]")
            for atom_index in chain:
                tokens.extend(self.atoms[atom_index].skeleton_tokens())
            tokens.append("[END_CHAIN]")
        tokens.append("[LORENTZ]")
        for chain in self.lorentz_chains:
            tokens.append("[CHAIN]")
            for atom_index in chain:
                tokens.extend(self.atoms[atom_index].skeleton_tokens())
            tokens.append("[END_CHAIN]")
        tokens.append("[END_TERM]")
        return tokens

    def to_dict(self) -> dict:
        return {
            "coefficient": [self.coefficient.numerator, self.coefficient.denominator],
            "scalar_factors": list(self.scalar_factors),
            "atoms": [atom.to_dict() for atom in self.atoms],
            "dirac_chains": [list(chain) for chain in self.dirac_chains],
            "lorentz_chains": [list(chain) for chain in self.lorentz_chains],
            "momentum_insertions": list(self.momentum_insertions),
            "external_wavefunctions": list(self.external_wavefunctions),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CanonicalQEDAmplitudeTerm":
        numerator, denominator = payload["coefficient"]
        return cls(
            coefficient=Fraction(int(numerator), int(denominator)),
            scalar_factors=tuple(payload["scalar_factors"]),
            atoms=tuple(
                CanonicalQEDTensorAtom.from_dict(atom) for atom in payload["atoms"]
            ),
            dirac_chains=tuple(
                tuple(int(value) for value in chain)
                for chain in payload["dirac_chains"]
            ),
            lorentz_chains=tuple(
                tuple(int(value) for value in chain)
                for chain in payload["lorentz_chains"]
            ),
            momentum_insertions=tuple(
                int(value) for value in payload["momentum_insertions"]
            ),
            external_wavefunctions=tuple(
                int(value) for value in payload["external_wavefunctions"]
            ),
        )


@dataclass(frozen=True)
class CanonicalQEDAmplitude:
    sample_id: str
    global_i_power: int
    global_e_power: int
    global_rational: Fraction
    denominator_infix: str
    denominator_tokens: tuple[str, ...]
    external_pattern: str
    process_family: str
    channel: str
    propagator_species: str
    terms: tuple[CanonicalQEDAmplitudeTerm, ...]
    family_signature: str
    raw_amplitude: str = ""

    @property
    def term_count(self) -> int:
        return len(self.terms)

    def typed_term_sequences(self) -> list[list[str]]:
        return [term.typed_tokens() for term in self.terms]

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "global_i_power": self.global_i_power,
            "global_e_power": self.global_e_power,
            "global_rational": [
                self.global_rational.numerator,
                self.global_rational.denominator,
            ],
            "denominator_infix": self.denominator_infix,
            "denominator_tokens": list(self.denominator_tokens),
            "external_pattern": self.external_pattern,
            "process_family": self.process_family,
            "channel": self.channel,
            "propagator_species": self.propagator_species,
            "terms": [term.to_dict() for term in self.terms],
            "family_signature": self.family_signature,
            "raw_amplitude": self.raw_amplitude,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CanonicalQEDAmplitude":
        numerator, denominator = payload["global_rational"]
        return cls(
            sample_id=payload["sample_id"],
            global_i_power=int(payload["global_i_power"]),
            global_e_power=int(payload["global_e_power"]),
            global_rational=Fraction(int(numerator), int(denominator)),
            denominator_infix=payload["denominator_infix"],
            denominator_tokens=tuple(payload["denominator_tokens"]),
            external_pattern=payload["external_pattern"],
            process_family=payload["process_family"],
            channel=payload["channel"],
            propagator_species=payload["propagator_species"],
            terms=tuple(
                CanonicalQEDAmplitudeTerm.from_dict(term)
                for term in payload["terms"]
            ),
            family_signature=payload["family_signature"],
            raw_amplitude=payload.get("raw_amplitude", ""),
        )


@dataclass(frozen=True)
class MaskedAmplitude:
    expr_for_sympy: str
    atoms: tuple[str, ...]


def mask_tensor_atoms(raw_amplitude: str) -> MaskedAmplitude:
    expr = raw_amplitude
    atoms: list[str] = []

    def repl(match: re.Match[str]) -> str:
        index = len(atoms)
        atoms.append(match.group(0))
        return f"A{index}"

    for pattern in _ATOM_PATTERNS:
        expr = pattern.sub(repl, expr)
    expr = expr.replace("^", "**")
    return MaskedAmplitude(expr_for_sympy=expr, atoms=tuple(atoms))


def _parse_momentum(atom_str: str) -> tuple[int, list[tuple[str, str]]]:
    match = _MOM_RE.fullmatch(atom_str)
    if not match:
        raise ValueError(f"Failed to parse momentum insertion: {atom_str}")
    slot = int(match.group(1))
    return slot, [
        (
            CONNECTOR_KIND_LORENTZ,
            _namespace_label(match.group(2), CONNECTOR_KIND_LORENTZ),
        )
    ]


def _parse_gamma(
    atom_str: str,
) -> tuple[int, str, str, bool, list[tuple[str, str]]]:
    match = _GAMMA_RE.fullmatch(atom_str)
    if not match:
        raise ValueError(f"Failed to parse gamma tensor: {atom_str}")
    fields = _split_brace_fields(match.group(1))
    if len(fields) != 3:
        raise ValueError(f"Unexpected gamma arity: {atom_str}")
    return 0, FLAVOR_NONE, SPIN_ROLE_NONE, False, [
        (
            CONNECTOR_KIND_LORENTZ,
            _namespace_label(fields[0], CONNECTOR_KIND_LORENTZ),
        ),
        (CONNECTOR_KIND_DIRAC, _namespace_label(fields[1], CONNECTOR_KIND_DIRAC)),
        (CONNECTOR_KIND_DIRAC, _namespace_label(fields[2], CONNECTOR_KIND_DIRAC)),
    ]


def _parse_tensor_wavefunction(
    atom_str: str,
) -> tuple[str, int, str, bool, list[tuple[str, str]]]:
    match = _TENSOR_ATOM_RE.fullmatch(atom_str)
    if not match:
        raise ValueError(f"Failed to parse tensor atom: {atom_str}")
    flavor = match.group(1)
    fields = _split_brace_fields(match.group(2))
    if not fields:
        raise ValueError(f"Expected indexed tensor fields in: {atom_str}")
    slot = int(match.group(3))
    role = match.group(4)
    conjugated = match.group(5) is not None
    return flavor, slot, role or SPIN_ROLE_NONE, conjugated, fields


def _parse_spinor(
    atom_str: str,
) -> tuple[int, str, str, bool, list[tuple[str, str]]]:
    flavor, slot, role, conjugated, fields = _parse_tensor_wavefunction(atom_str)
    return slot, flavor, role, conjugated, [
        (CONNECTOR_KIND_DIRAC, _namespace_label(fields[-1], CONNECTOR_KIND_DIRAC))
    ]


def _parse_polarization(
    atom_str: str,
) -> tuple[int, str, str, bool, list[tuple[str, str]]]:
    flavor, slot, role, conjugated, fields = _parse_tensor_wavefunction(atom_str)
    if role != SPIN_ROLE_NONE:
        raise ValueError(f"Polarization vector unexpectedly had spin role: {atom_str}")
    return slot, flavor, SPIN_ROLE_NONE, conjugated, [
        (CONNECTOR_KIND_LORENTZ, _namespace_label(fields[-1], CONNECTOR_KIND_LORENTZ))
    ]


def _normalized_momentum_index(slot: int) -> int:
    return max(slot - 1, 0)


def _canonical_atom(
    atom_str: str,
    connector_map: dict[str, int],
) -> CanonicalQEDTensorAtom:
    if atom_str.startswith("gamma_{"):
        slot, flavor, spin_role, conjugated, connectors = _parse_gamma(atom_str)
        atom_type = ATOM_TYPE_GAMMA
        momentum_index = -1
    elif atom_str.startswith("p_"):
        slot, connectors = _parse_momentum(atom_str)
        flavor, spin_role, conjugated = FLAVOR_NONE, SPIN_ROLE_NONE, False
        atom_type = ATOM_TYPE_MOMENTUM
        momentum_index = _normalized_momentum_index(slot)
    else:
        flavor = atom_str.split("_", 1)[0]
        if flavor == "A":
            slot, flavor, spin_role, conjugated, connectors = _parse_polarization(
                atom_str
            )
            atom_type = ATOM_TYPE_POLARIZATION
        else:
            slot, flavor, spin_role, conjugated, connectors = _parse_spinor(atom_str)
            atom_type = ATOM_TYPE_SPINOR
        momentum_index = _normalized_momentum_index(slot)

    index_ids = [-1, -1, -1]
    for position, (_kind, label) in enumerate(connectors):
        if label not in connector_map:
            connector_map[label] = len(connector_map)
        index_ids[position] = connector_map[label]
    return CanonicalQEDTensorAtom(
        atom_type=atom_type,
        flavor=flavor,
        pindex=slot,
        spin_role=spin_role,
        conjugated=conjugated,
        momentum_index=momentum_index,
        index_ids=tuple(index_ids),
    )


def _is_scalar_symbol_factor(expr: sp.Expr) -> bool:
    if not expr.is_commutative:
        return False
    if expr in (sp.Integer(1), sp.Integer(-1), sp.I, _E, _E**2):
        return False
    if expr.is_number or expr.is_rational:
        return False
    free_symbols = expr.free_symbols
    if not free_symbols:
        return False
    return any(symbol != _E for symbol in free_symbols)


def _stream_connector_kinds(
    atom: CanonicalQEDTensorAtom,
    allowed: set[str],
) -> set[int]:
    indices: set[int] = set()
    for index_id, kind in zip(atom.index_ids, atom.connector_kinds(), strict=False):
        if index_id >= 0 and kind in allowed:
            indices.add(index_id)
    return indices


def _connected_chains(
    atoms: Sequence[CanonicalQEDTensorAtom],
    allowed_atom_types: set[str],
    allowed_connector_kinds: set[str],
) -> tuple[tuple[int, ...], ...]:
    candidates = [
        index
        for index, atom in enumerate(atoms)
        if atom.atom_type in allowed_atom_types
    ]
    if not candidates:
        return tuple()

    remaining = set(candidates)
    chains: list[tuple[int, ...]] = []
    while remaining:
        seed = min(remaining)
        queue = [seed]
        component: set[int] = set()
        while queue:
            current = queue.pop()
            if current in component:
                continue
            component.add(current)
            current_connectors = _stream_connector_kinds(
                atoms[current],
                allowed_connector_kinds,
            )
            for other in list(remaining):
                if other == current:
                    continue
                other_connectors = _stream_connector_kinds(
                    atoms[other],
                    allowed_connector_kinds,
                )
                if current_connectors & other_connectors:
                    queue.append(other)
        remaining -= component
        chains.append(tuple(sorted(component)))
    chains.sort(key=lambda chain: chain[0])
    return tuple(chains)


def canonical_family_signature(amplitude: CanonicalQEDAmplitude) -> str:
    pieces: list[str] = [
        f"PATTERN:{amplitude.external_pattern}",
        f"PROCESS:{amplitude.process_family}",
        f"CHANNEL:{amplitude.channel}",
        f"PROP:{amplitude.propagator_species}",
        f"DEN:{' '.join(_family_scalar_symbol(token) for token in amplitude.denominator_tokens)}",
        f"I:{amplitude.global_i_power}",
        f"E:{amplitude.global_e_power}",
        f"T:{amplitude.term_count}",
    ]
    for term in amplitude.terms:
        pieces.append(" ".join(term.skeleton_tokens()))
    return "|".join(pieces)


def compile_qed_amplitude(
    raw_amplitude: str,
    sample_id: str = "",
) -> CanonicalQEDAmplitude:
    masked = mask_tensor_atoms(raw_amplitude)
    locals_map = dict(SYMPY_LOCALS)
    for index in range(len(masked.atoms)):
        locals_map[f"A{index}"] = sp.Symbol(f"A{index}", commutative=False)

    expr = sp.together(sp.sympify(masked.expr_for_sympy, locals=locals_map))
    numerator, denominator = sp.fraction(expr)
    denominator_factor = sp.factor(denominator)
    den_content, denominator_primitive = denominator_factor.as_content_primitive()
    denominator_infix = sp.sstr(
        sp.expand(denominator_primitive),
        order="lex",
    ).replace("**", "^")
    denominator_tokens = tuple(_denominator_tokens(denominator_infix))

    numerator_expanded = sp.expand(numerator)
    raw_terms = (
        list(numerator_expanded.args)
        if isinstance(numerator_expanded, sp.Add)
        else [numerator_expanded]
    )

    raw_coeffs: list[sp.Expr] = []
    raw_scalar_symbols: list[tuple[str, ...]] = []
    raw_residuals: list[sp.Expr] = []
    for term in raw_terms:
        commutative = [arg for arg in sp.Mul.make_args(term) if arg.is_commutative]
        noncommutative = [
            arg for arg in sp.Mul.make_args(term) if not arg.is_commutative
        ]
        symbolic_scalars = [arg for arg in commutative if _is_scalar_symbol_factor(arg)]
        numeric_scalars = [arg for arg in commutative if not _is_scalar_symbol_factor(arg)]
        raw_coeffs.append(sp.Mul(*numeric_scalars) if numeric_scalars else sp.Integer(1))
        raw_scalar_symbols.append(
            tuple(sorted(_basis_symbol(current) for current in symbolic_scalars))
        )
        raw_residuals.append(
            sp.Mul(*noncommutative) if noncommutative else sp.Integer(1)
        )

    e_powers: list[int] = []
    for coeff in raw_coeffs:
        degree = sp.degree(coeff, _E)
        e_powers.append(0 if degree is None else int(degree))
    global_e_power = min(e_powers) if e_powers else 0
    coeffs_no_e = [sp.simplify(coeff / (_E**global_e_power)) for coeff in raw_coeffs]
    global_i_power = (
        1
        if coeffs_no_e and all(sp.simplify(coeff / sp.I).is_rational for coeff in coeffs_no_e)
        else 0
    )
    coeffs_scalar = [
        sp.simplify(coeff / ((_E**global_e_power) * (sp.I**global_i_power)))
        for coeff in raw_coeffs
    ]
    scalar_fractions = [_fraction_from_sympy(value) for value in coeffs_scalar]
    coeff_gcd = _rational_gcd(scalar_fractions)

    canonical_terms: list[CanonicalQEDAmplitudeTerm] = []
    slot_kinds: dict[int, str] = {}
    for residual_expr, scalar_fraction, scalar_symbols in zip(
        raw_residuals,
        scalar_fractions,
        raw_scalar_symbols,
    ):
        term_coeff = scalar_fraction / coeff_gcd
        connector_map: dict[str, int] = {}
        term_atoms: list[CanonicalQEDTensorAtom] = []
        for factor in _term_factors(residual_expr):
            factor_text = sp.sstr(factor)
            if not factor_text.startswith("A"):
                continue
            atom_index = int(factor_text[1:])
            atom = _canonical_atom(masked.atoms[atom_index], connector_map)
            term_atoms.append(atom)
            if 1 <= atom.pindex <= 4 and atom.atom_type == ATOM_TYPE_POLARIZATION:
                slot_kinds[atom.pindex] = "A"
            elif (
                1 <= atom.pindex <= 4
                and atom.atom_type == ATOM_TYPE_SPINOR
                and slot_kinds.get(atom.pindex) != "A"
            ):
                slot_kinds[atom.pindex] = "F"

        atoms_tuple = tuple(term_atoms)
        dirac_chains = _connected_chains(
            atoms_tuple,
            allowed_atom_types={ATOM_TYPE_GAMMA, ATOM_TYPE_SPINOR},
            allowed_connector_kinds={CONNECTOR_KIND_DIRAC},
        )
        lorentz_chains = _connected_chains(
            atoms_tuple,
            allowed_atom_types={
                ATOM_TYPE_GAMMA,
                ATOM_TYPE_MOMENTUM,
                ATOM_TYPE_POLARIZATION,
            },
            allowed_connector_kinds={CONNECTOR_KIND_LORENTZ},
        )
        momentum_insertions = tuple(
            index
            for index, atom in enumerate(atoms_tuple)
            if atom.atom_type == ATOM_TYPE_MOMENTUM
        )
        external_wavefunctions = tuple(
            index
            for index, atom in enumerate(atoms_tuple)
            if atom.atom_type in {ATOM_TYPE_SPINOR, ATOM_TYPE_POLARIZATION}
            and atom.pindex > 0
        )
        canonical_terms.append(
            CanonicalQEDAmplitudeTerm(
                coefficient=term_coeff,
                scalar_factors=scalar_symbols,
                atoms=atoms_tuple,
                dirac_chains=dirac_chains,
                lorentz_chains=lorentz_chains,
                momentum_insertions=momentum_insertions,
                external_wavefunctions=external_wavefunctions,
            )
        )

    canonical_terms.sort(key=lambda term: tuple(term.skeleton_tokens()))
    external_pattern = external_pattern_from(slot_kinds)
    process_family = process_family_from(external_pattern)
    channel = _channel_from_denominator_tokens(denominator_tokens)
    propagator_species = _propagator_species_from_denominator(denominator_tokens)
    amplitude = CanonicalQEDAmplitude(
        sample_id=sample_id,
        global_i_power=global_i_power,
        global_e_power=global_e_power,
        global_rational=coeff_gcd / Fraction(int(den_content), 1),
        denominator_infix=denominator_infix,
        denominator_tokens=denominator_tokens,
        external_pattern=external_pattern,
        process_family=process_family,
        channel=channel,
        propagator_species=propagator_species,
        terms=tuple(canonical_terms),
        family_signature="",
        raw_amplitude=raw_amplitude,
    )
    return CanonicalQEDAmplitude(
        sample_id=amplitude.sample_id,
        global_i_power=amplitude.global_i_power,
        global_e_power=amplitude.global_e_power,
        global_rational=amplitude.global_rational,
        denominator_infix=amplitude.denominator_infix,
        denominator_tokens=amplitude.denominator_tokens,
        external_pattern=amplitude.external_pattern,
        process_family=amplitude.process_family,
        channel=amplitude.channel,
        propagator_species=amplitude.propagator_species,
        terms=amplitude.terms,
        family_signature=canonical_family_signature(amplitude),
        raw_amplitude=amplitude.raw_amplitude,
    )


def save_canonical_corpus(
    path: str | Path,
    amplitudes: Sequence[CanonicalQEDAmplitude],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump([amplitude.to_dict() for amplitude in amplitudes], handle, indent=2)


def load_canonical_corpus(path: str | Path) -> list[CanonicalQEDAmplitude]:
    with open(path) as handle:
        payload = json.load(handle)
    return [CanonicalQEDAmplitude.from_dict(item) for item in payload]


def create_canonical_corpus(
    raw_amplitudes: Iterable[tuple[str, str]],
    cache_dir: str | Path | None = None,
) -> list[CanonicalQEDAmplitude]:
    amplitudes = [
        compile_qed_amplitude(raw_amplitude, sample_id=sample_id)
        for sample_id, raw_amplitude in raw_amplitudes
    ]
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "corpus.json"
        save_canonical_corpus(cache_path, amplitudes)
    return amplitudes
