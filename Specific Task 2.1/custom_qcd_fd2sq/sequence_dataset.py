"""
Sequence-based dataset utilities for custom-encoder ablations.

This module linearizes parsed diagrams into token sequences so we can
run graph-free seq2seq baselines against the same factorized targets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .factorization import FactorizedTarget, factorize_squared_amplitude
from .parser import FeynmanDiagram, parse_all_qcd
from .splits import random_split_indices
from .tokenizer import AmplitudeTokenizer


PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"

SPECIAL_TOKENS: tuple[str, ...] = (PAD, SOS, EOS, UNK)

SEGMENT_GLOBAL = 0
SEGMENT_CHANNEL = 1
SEGMENT_EXTERNAL = 2
SEGMENT_VERTEX = 3
SEGMENT_PROPAGATOR = 4
NUM_SEGMENT_TYPES = 5


def serialize_diagram_to_sequence(
    diagram: FeynmanDiagram,
) -> tuple[list[str], list[int]]:
    tokens: list[str] = []
    segments: list[int] = []

    def add(segment: int, *items: str) -> None:
        for item in items:
            tokens.append(item)
            segments.append(segment)

    add(SEGMENT_CHANNEL, "[CHANNEL]", f"CH_{diagram.channel or 'unknown'}")

    for particle in sorted(diagram.externals, key=lambda item: item.momentum_label):
        add(
            SEGMENT_EXTERNAL,
            "[EXT]",
            f"P_{particle.momentum_label}",
            f"F_{particle.flavor}",
            f"T_{particle.particle_type}",
            "IN" if particle.is_incoming else "OUT",
            "ANTI" if particle.is_antiparticle else "PART",
            "CONJ" if particle.is_conjugate else "PLAIN",
        )

    for vertex in sorted(diagram.vertices, key=lambda item: item.vertex_id):
        add(SEGMENT_VERTEX, "[VERTEX]", f"V_{vertex.vertex_id}", "[LEGS]")
        for leg in sorted(vertex.external_legs):
            add(SEGMENT_VERTEX, f"X_{leg}")
        if vertex.propagator is not None:
            prop = vertex.propagator
            add(
                SEGMENT_PROPAGATOR,
                "[PROP]",
                f"F_{prop.flavor}",
                f"T_{prop.particle_type}",
                "ANTI" if prop.is_antiparticle else "PART",
                f"PV_{prop.vertex_id}",
            )

    return tokens, segments


class DiagramSequenceTokenizer:
    def __init__(self, token2id: dict[str, int] | None = None):
        if token2id is not None:
            self.token2id = dict(token2id)
        else:
            self.token2id = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        self._rebuild_reverse()

    def _rebuild_reverse(self) -> None:
        self.id2token = {value: key for key, value in self.token2id.items()}
        self.vocab_size = len(self.token2id)
        self.pad_id = self.token2id[PAD]
        self.sos_id = self.token2id[SOS]
        self.eos_id = self.token2id[EOS]
        self.unk_id = self.token2id[UNK]

    def build_vocab(
        self,
        sequences: Sequence[Sequence[str]],
    ) -> "DiagramSequenceTokenizer":
        self.token2id = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        for sequence in sequences:
            for tok in sequence:
                if tok not in self.token2id:
                    self.token2id[tok] = len(self.token2id)
        self._rebuild_reverse()
        return self

    def encode_tokens(
        self,
        tokens: Sequence[str],
        max_len: int,
        add_sos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        ids: list[int] = []
        if add_sos:
            ids.append(self.sos_id)
        ids.extend(self.token2id.get(tok, self.unk_id) for tok in tokens)
        if add_eos:
            ids.append(self.eos_id)
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.eos_id]
        return ids

    def encode_segments(
        self,
        segments: Sequence[int],
        max_len: int,
        add_sos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        encoded: list[int] = []
        if add_sos:
            encoded.append(SEGMENT_GLOBAL)
        encoded.extend(int(value) for value in segments)
        if add_eos:
            encoded.append(SEGMENT_GLOBAL)
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        return encoded

    def encode_tensor(
        self,
        tokens: Sequence[str],
        segments: Sequence[int],
        max_len: int,
    ) -> tuple[Tensor, Tensor]:
        token_ids = self.encode_tokens(tokens, max_len=max_len)
        segment_ids = self.encode_segments(segments, max_len=max_len)

        token_tensor = torch.full((max_len,), self.pad_id, dtype=torch.long)
        token_tensor[: len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)

        segment_tensor = torch.full(
            (max_len,),
            SEGMENT_GLOBAL,
            dtype=torch.long,
        )
        segment_tensor[: len(segment_ids)] = torch.tensor(
            segment_ids,
            dtype=torch.long,
        )
        return token_tensor, segment_tensor

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump({"token2id": self.token2id}, handle, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DiagramSequenceTokenizer":
        with open(path) as handle:
            payload = json.load(handle)
        return cls(token2id=payload["token2id"])


@dataclass
class SequenceExample:
    src_ids: Tensor
    src_segment_ids: Tensor
    tgt_tokens: Tensor
    diagram_idx: Tensor


@dataclass
class SequenceBatch:
    src_ids: Tensor
    src_segment_ids: Tensor
    tgt_tokens: Tensor
    diagram_idx: Tensor

    def to(self, device: torch.device | str) -> "SequenceBatch":
        return SequenceBatch(
            src_ids=self.src_ids.to(device),
            src_segment_ids=self.src_segment_ids.to(device),
            tgt_tokens=self.tgt_tokens.to(device),
            diagram_idx=self.diagram_idx.to(device),
        )


class SequenceFactorizedDataset(Dataset):
    def __init__(
        self,
        diagrams: Sequence[FeynmanDiagram],
        targets: Sequence[FactorizedTarget],
        indices: Sequence[int],
        source_tokenizer: DiagramSequenceTokenizer,
        target_tokenizer: AmplitudeTokenizer,
        max_src_seq_len: int = 256,
        max_tgt_seq_len: int = 1400,
    ):
        super().__init__()
        self.diagrams = list(diagrams)
        self.targets = list(targets)
        self.indices = list(indices)
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_src_seq_len = max_src_seq_len
        self.max_tgt_seq_len = max_tgt_seq_len

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> SequenceExample:
        diag_idx = self.indices[idx]
        diagram = self.diagrams[diag_idx]
        target = self.targets[diag_idx]
        src_tokens, src_segments = serialize_diagram_to_sequence(diagram)
        src_ids, src_segment_ids = self.source_tokenizer.encode_tensor(
            src_tokens,
            src_segments,
            max_len=self.max_src_seq_len,
        )
        tgt_tokens = self.target_tokenizer.encode_tensor(
            target.sequence_target_text(),
            max_len=self.max_tgt_seq_len,
        )

        return SequenceExample(
            src_ids=src_ids,
            src_segment_ids=src_segment_ids,
            tgt_tokens=tgt_tokens,
            diagram_idx=torch.tensor(diag_idx, dtype=torch.long),
        )


def _collate_sequence_examples(batch: list[SequenceExample]) -> SequenceBatch:
    src_ids = torch.stack([item.src_ids for item in batch], dim=0)
    src_segment_ids = torch.stack([item.src_segment_ids for item in batch], dim=0)
    tgt_tokens = torch.stack([item.tgt_tokens for item in batch], dim=0)
    diagram_idx = torch.stack([item.diagram_idx for item in batch], dim=0)
    return SequenceBatch(
        src_ids=src_ids,
        src_segment_ids=src_segment_ids,
        tgt_tokens=tgt_tokens,
        diagram_idx=diagram_idx,
    )


def create_sequence_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    max_src_seq_len: int = 256,
    max_tgt_seq_len: int = 1400,
    expression_mode: str = "postfix",
    seed: int = 42,
    target_variant: str = "factorized",
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    AmplitudeTokenizer,
    DiagramSequenceTokenizer,
]:
    data_dir = Path(data_dir)
    print(f"Loading QCD data from {data_dir}...")
    diagrams = parse_all_qcd(data_dir)
    print(f"Loaded {len(diagrams)} diagrams")

    print("Factorizing squared amplitudes...")
    factorized_targets = [
        factorize_squared_amplitude(
            diagram.raw_squared,
            target_variant=target_variant,
        )
        for diagram in diagrams
    ]

    serialized = [serialize_diagram_to_sequence(diagram) for diagram in diagrams]

    train_indices, val_indices, test_indices = random_split_indices(
        len(diagrams),
        seed=seed,
    )

    target_tokenizer = AmplitudeTokenizer(
        expression_mode=expression_mode,
    ).build_vocab(
        [factorized_targets[idx].sequence_target_text() for idx in train_indices]
    )
    source_tokenizer = DiagramSequenceTokenizer().build_vocab(
        [serialized[idx][0] for idx in train_indices]
    )
    print(
        f"Tokenizer numerator vocab size: {target_tokenizer.vocab_size} "
        f"({target_tokenizer.expression_mode})"
    )
    print(f"Tokenizer source vocab size: {source_tokenizer.vocab_size}")

    train_ds = SequenceFactorizedDataset(
        diagrams=diagrams,
        targets=factorized_targets,
        indices=train_indices,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_src_seq_len=max_src_seq_len,
        max_tgt_seq_len=max_tgt_seq_len,
    )
    val_ds = SequenceFactorizedDataset(
        diagrams=diagrams,
        targets=factorized_targets,
        indices=val_indices,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_src_seq_len=max_src_seq_len,
        max_tgt_seq_len=max_tgt_seq_len,
    )
    test_ds = SequenceFactorizedDataset(
        diagrams=diagrams,
        targets=factorized_targets,
        indices=test_indices,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_src_seq_len=max_src_seq_len,
        max_tgt_seq_len=max_tgt_seq_len,
    )

    print(f"  Split 'train': {len(train_ds)} total")
    print(f"  Split 'val':   {len(val_ds)} total")
    print(f"  Split 'test':  {len(test_ds)} total")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=_collate_sequence_examples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_sequence_examples,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_sequence_examples,
    )
    return (
        train_loader,
        val_loader,
        test_loader,
        target_tokenizer,
        source_tokenizer,
    )
