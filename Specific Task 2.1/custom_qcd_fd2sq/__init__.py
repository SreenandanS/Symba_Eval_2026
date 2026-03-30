"""
Custom QCD fd-to-squared-amplitude package.

A multi-stream PyG-based custom graph encoder over kinematic, color, and
spinor relations with symbolic sequence decoding for factorized and raw-string targets.
"""

from importlib import import_module

__version__ = "0.2.0"

__all__ = [
    "CustomQCDFd2SqConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "CustomQCDFd2SqModel",
    "CustomQCDFd2SqEncoder",
    "BaselineEncoder",
    "SequenceDiagramEncoder",
    "AmplitudeTokenizer",
    "DiagramSequenceTokenizer",
    "RPNGrammar",
    "CustomQCDFd2SqTrainer",
    "QCDFeynmanDataset",
    "create_dataloaders",
    "create_sequence_dataloaders",
]

_EXPORTS = {
    "CustomQCDFd2SqConfig": (".config", "CustomQCDFd2SqConfig"),
    "ModelConfig": (".config", "ModelConfig"),
    "TrainingConfig": (".config", "TrainingConfig"),
    "DataConfig": (".config", "DataConfig"),
    "CustomQCDFd2SqModel": (".model", "CustomQCDFd2SqModel"),
    "CustomQCDFd2SqEncoder": (".encoder", "CustomQCDFd2SqEncoder"),
    "BaselineEncoder": (".baseline_encoder", "BaselineEncoder"),
    "SequenceDiagramEncoder": (".sequence_encoder", "SequenceDiagramEncoder"),
    "AmplitudeTokenizer": (".tokenizer", "AmplitudeTokenizer"),
    "DiagramSequenceTokenizer": (".sequence_dataset", "DiagramSequenceTokenizer"),
    "RPNGrammar": (".grammar", "RPNGrammar"),
    "CustomQCDFd2SqTrainer": (".train", "CustomQCDFd2SqTrainer"),
    "QCDFeynmanDataset": (".dataset", "QCDFeynmanDataset"),
    "create_dataloaders": (".dataset", "create_dataloaders"),
    "create_sequence_dataloaders": (
        ".sequence_dataset",
        "create_sequence_dataloaders",
    ),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
