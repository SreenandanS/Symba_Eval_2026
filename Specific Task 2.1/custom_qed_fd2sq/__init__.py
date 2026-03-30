"""Fixed-slot custom package for SYMBA QED tree-level 2->2 interactions."""

from importlib import import_module

__version__ = "0.2.0"

__all__ = [
    "CustomQEDFd2SqConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "CustomQEDFd2SqModel",
    "CustomQEDFd2SqEncoder",
    "SequenceDiagramEncoder",
    "AmplitudeTokenizer",
    "DiagramSequenceTokenizer",
    "RPNGrammar",
    "CustomQEDFd2SqTrainer",
    "QEDFeynmanDataset",
    "create_dataloaders",
    "create_sequence_dataloaders",
    "load_qed_corpus",
]

_EXPORTS = {
    "CustomQEDFd2SqConfig": (".config", "CustomQEDFd2SqConfig"),
    "ModelConfig": (".config", "ModelConfig"),
    "TrainingConfig": (".config", "TrainingConfig"),
    "DataConfig": (".config", "DataConfig"),
    "CustomQEDFd2SqModel": (".model", "CustomQEDFd2SqModel"),
    "CustomQEDFd2SqEncoder": (".encoder", "CustomQEDFd2SqEncoder"),
    "SequenceDiagramEncoder": (".sequence_encoder", "SequenceDiagramEncoder"),
    "AmplitudeTokenizer": (".tokenizer", "AmplitudeTokenizer"),
    "DiagramSequenceTokenizer": (".sequence_dataset", "DiagramSequenceTokenizer"),
    "RPNGrammar": (".grammar", "RPNGrammar"),
    "CustomQEDFd2SqTrainer": (".train", "CustomQEDFd2SqTrainer"),
    "QEDFeynmanDataset": (".dataset", "QEDFeynmanDataset"),
    "create_dataloaders": (".dataset", "create_dataloaders"),
    "create_sequence_dataloaders": (".sequence_dataset", "create_sequence_dataloaders"),
    "load_qed_corpus": (".dataset", "load_qed_corpus"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
