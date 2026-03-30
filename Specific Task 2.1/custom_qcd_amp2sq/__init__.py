"""Canonical amplitude -> squared-amplitude package."""

from importlib import import_module

__all__ = [
    "Amp2SqConfig",
    "Amp2SqTrainer",
    "CanonicalAmplitude",
    "CanonicalAmplitudeTerm",
    "CanonicalTensorAtom",
    "compile_amplitude",
    "create_canonical_corpus",
]

_EXPORTS = {
    "Amp2SqConfig": (".config", "Amp2SqConfig"),
    "Amp2SqTrainer": (".train", "Amp2SqTrainer"),
    "CanonicalAmplitude": (".compiler", "CanonicalAmplitude"),
    "CanonicalAmplitudeTerm": (".compiler", "CanonicalAmplitudeTerm"),
    "CanonicalTensorAtom": (".compiler", "CanonicalTensorAtom"),
    "compile_amplitude": (".compiler", "compile_amplitude"),
    "create_canonical_corpus": (".compiler", "create_canonical_corpus"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
