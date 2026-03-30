"""
Canonical factorization of SYMBA QED squared amplitudes.

Each tree-level 2-to-2 single-diagram target is rewritten as

    charge_factor * numerator / denominator

within the audited QED symbolic scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


TARGET_VARIANT_FACTORIZED = "factorized"
TARGET_VARIANT_RAW_STRING = "raw_string"

SUPPORTED_TARGET_VARIANTS: tuple[str, ...] = (
    TARGET_VARIANT_FACTORIZED,
    TARGET_VARIANT_RAW_STRING,
)

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

_NUMERATOR_SYMBOL_ORDER = tuple(
    _MASS_SYMBOLS[name]
    for name in ("m_b", "m_c", "m_d", "m_e", "m_mu", "m_s", "m_t", "m_tt", "m_u")
) + tuple(
    _MANDELSTAM_SYMBOLS[name]
    for name in ("s_12", "s_13", "s_14", "s_23", "s_24", "s_34")
) + (_REG_PROP,)


def normalize_target_variant(target_variant: str) -> str:
    if target_variant not in SUPPORTED_TARGET_VARIANTS:
        raise ValueError(
            f"Unknown target_variant '{target_variant}'. Expected one of "
            f"{SUPPORTED_TARGET_VARIANTS}."
        )
    return target_variant


def default_decoder_representation(target_variant: str) -> str:
    normalize_target_variant(target_variant)
    return "postfix"


def default_use_grammar(target_variant: str) -> bool:
    normalize_target_variant(target_variant)
    return True


@dataclass(frozen=True)
class FactorizedTarget:
    charge_factor: str
    denominator: str
    numerator_infix: str
    full_infix: str
    raw_string: str
    target_variant: str = TARGET_VARIANT_FACTORIZED

    @property
    def prefactor(self) -> str:
        return self.charge_factor

    def sequence_target_text(self) -> str:
        variant = normalize_target_variant(self.target_variant)
        if variant == TARGET_VARIANT_FACTORIZED:
            return self.full_infix
        return self.raw_string

    def sequence_target_infix(self) -> str:
        return self.sequence_target_text()


def _canonical_expr_string(expr: sp.Expr) -> str:
    return sp.sstr(expr, order="lex")


def _canonicalize_numerator(expr: sp.Expr) -> sp.Expr:
    expanded = sp.expand(expr)
    try:
        poly = sp.Poly(expanded, *_NUMERATOR_SYMBOL_ORDER, domain="ZZ")
    except Exception as exc:  # pragma: no cover - defensive scope guard
        raise ValueError(
            "Canonical numerator left the audited QED polynomial scope."
        ) from exc
    return sp.expand(poly.as_expr())


def reconstruct_full_infix(
    charge_factor: str,
    denominator: str,
    numerator_infix: str,
) -> str:
    denominator_infix = denominator.replace("**", "^")
    charge_factor_infix = charge_factor.replace("**", "^")
    return f"({charge_factor_infix})*(({numerator_infix})/({denominator_infix}))"


def factorize_squared_amplitude(
    raw_squared: str,
    target_variant: str = TARGET_VARIANT_FACTORIZED,
) -> FactorizedTarget:
    normalized_variant = normalize_target_variant(target_variant)
    raw_string = raw_squared.replace("**", "^").strip()

    expr = sp.sympify(raw_string.replace("^", "**"), locals=SYMPY_LOCALS)
    expr = sp.cancel(expr)

    numerator, denominator = sp.fraction(expr)
    denominator = sp.factor(denominator)
    den_const, denominator = denominator.as_coeff_Mul()

    numerator = sp.simplify(sp.collect(numerator, _E**4) / (_E**4))
    num_const, numerator = numerator.as_content_primitive()
    charge_factor = sp.simplify(num_const / den_const)

    if charge_factor.could_extract_minus_sign():
        charge_factor = -charge_factor
        numerator = -numerator

    charge_factor = sp.nsimplify(charge_factor)
    denominator = sp.factor(denominator)
    simplified_numerator = sp.simplify(numerator)
    factorized_numerator = _canonicalize_numerator(simplified_numerator)

    charge_factor_str = _canonical_expr_string(charge_factor)
    denominator_str = _canonical_expr_string(denominator)
    numerator_infix = _canonical_expr_string(factorized_numerator).replace("**", "^")
    return FactorizedTarget(
        charge_factor=charge_factor_str,
        denominator=denominator_str,
        numerator_infix=numerator_infix,
        full_infix=reconstruct_full_infix(
            charge_factor=charge_factor_str,
            denominator=denominator_str,
            numerator_infix=numerator_infix,
        ),
        raw_string=raw_string,
        target_variant=normalized_variant,
    )
