"""QED amplitude -> simplified squared-amplitude package."""

from importlib import import_module

__all__ = [
    "QEDAmp2SqConfig",
    "QEDAmp2SqTrainer",
    "CanonicalQEDAmplitude",
    "CanonicalQEDAmplitudeTerm",
    "CanonicalQEDTensorAtom",
    "compile_qed_amplitude",
    "create_canonical_corpus",
    "load_qed_amp2sq_corpus",
]

_EXPORTS = {
    "QEDAmp2SqConfig": (".config", "QEDAmp2SqConfig"),
    "QEDAmp2SqTrainer": (".train", "QEDAmp2SqTrainer"),
    "CanonicalQEDAmplitude": (".compiler", "CanonicalQEDAmplitude"),
    "CanonicalQEDAmplitudeTerm": (".compiler", "CanonicalQEDAmplitudeTerm"),
    "CanonicalQEDTensorAtom": (".compiler", "CanonicalQEDTensorAtom"),
    "compile_qed_amplitude": (".compiler", "compile_qed_amplitude"),
    "create_canonical_corpus": (".compiler", "create_canonical_corpus"),
    "load_qed_amp2sq_corpus": (".dataset", "load_qed_amp2sq_corpus"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
