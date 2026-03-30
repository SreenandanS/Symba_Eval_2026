"""Sequence-based dataset utilities built from canonical QED interactions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .dataset import LoadedQEDCorpus, _resolve_splits, load_qed_corpus
from .interaction import QEDInteraction
from .tokenizer import AmplitudeTokenizer


PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"

SPECIAL_TOKENS: tuple[str, ...] = (PAD, SOS, EOS, UNK)

TOPO_MARKER = "[TOPO]"
EXT_MARKERS = ("[EXT1]", "[EXT2]", "[EXT3]", "[EXT4]")
VERTEX_MARKERS = ("[V0]", "[V1]")
PROP_MARKER = "[PROP]"
MARKER_ORDER = EXT_MARKERS + VERTEX_MARKERS + (PROP_MARKER, TOPO_MARKER)

SEGMENT_GLOBAL = 0
SEGMENT_TOPO = 1
SEGMENT_EXT1 = 2
SEGMENT_EXT2 = 3
SEGMENT_EXT3 = 4
SEGMENT_EXT4 = 5
SEGMENT_V0 = 6
SEGMENT_V1 = 7
SEGMENT_PROP = 8
NUM_SEGMENT_TYPES = 9

_SEGMENTS_BY_MARKER = {
    TOPO_MARKER: SEGMENT_TOPO,
    "[EXT1]": SEGMENT_EXT1,
    "[EXT2]": SEGMENT_EXT2,
    "[EXT3]": SEGMENT_EXT3,
    "[EXT4]": SEGMENT_EXT4,
    "[V0]": SEGMENT_V0,
    "[V1]": SEGMENT_V1,
    PROP_MARKER: SEGMENT_PROP,
}


def _charge_token(value: float) -> str:
    if abs(value) < 1e-8:
        return "Q_0"
    return f"Q_{value:+.3f}".replace(".", "_")


def serialize_interaction_to_sequence(interaction: QEDInteraction) -> tuple[list[str], list[int], list[int]]:
    tokens: list[str] = []
    segments: list[int] = []
    marker_positions: list[int] = []

    def add(marker: str, *items: str) -> None:
        marker_positions.append(len(tokens) + 1)
        segment_id = _SEGMENTS_BY_MARKER[marker]
        for item in (marker, *items):
            tokens.append(item)
            segments.append(segment_id)

    add(
        TOPO_MARKER,
        f"CH_{interaction.channel}",
        f"PF_{interaction.process_family}",
        f"PAT_{interaction.external_pattern}",
        (
            "PROP_photon"
            if interaction.propagator is not None and interaction.propagator.is_photon
            else "PROP_fermion"
            if interaction.propagator is not None and interaction.propagator.is_fermion
            else "PROP_none"
        ),
    )

    for marker, external in zip(EXT_MARKERS, interaction.externals):
        add(
            marker,
            f"F_{external.flavor}",
            f"T_{external.particle_type}",
            "IN" if external.is_incoming else "OUT",
            "ANTI" if external.is_antiparticle else "PART",
            "CONJ" if external.is_conjugate else "PLAIN",
            _charge_token(external.charge),
        )

    for marker, vertex in zip(VERTEX_MARKERS, interaction.vertices):
        add(
            marker,
            f"LEGS_{'_'.join(str(slot + 1) for slot in vertex.external_slots)}",
            f"LINE_{vertex.fermion_line_id}",
            f"INT_{vertex.interaction_type}",
        )

    propagator = interaction.propagator
    add(
        PROP_MARKER,
        f"F_{propagator.flavor}" if propagator is not None else "F_A",
        (
            "K_PHOTON"
            if propagator is not None and propagator.is_photon
            else "K_FERMION"
            if propagator is not None and propagator.is_fermion
            else "K_NONE"
        ),
        (
            f"ENDS_{propagator.endpoint_vertices[0]}_{propagator.endpoint_vertices[1]}"
            if propagator is not None
            else "ENDS_0_1"
        ),
        (
            f"SIGNS_{propagator.endpoint_signs[0]}_{propagator.endpoint_signs[1]}"
            if propagator is not None
            else "SIGNS_0_0"
        ),
    )
    return tokens, segments, marker_positions


class DiagramSequenceTokenizer:
    def __init__(self, token2id: dict[str, int] | None = None):
        if token2id is not None:
            self.token2id = dict(token2id)
        else:
            self.token2id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self._rebuild_reverse()

    def _rebuild_reverse(self) -> None:
        self.id2token = {value: key for key, value in self.token2id.items()}
        self.vocab_size = len(self.token2id)
        self.pad_id = self.token2id[PAD]
        self.sos_id = self.token2id[SOS]
        self.eos_id = self.token2id[EOS]
        self.unk_id = self.token2id[UNK]

    def build_vocab(self, sequences: Sequence[Sequence[str]]) -> "DiagramSequenceTokenizer":
        self.token2id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        for sequence in sequences:
            for token in sequence:
                if token not in self.token2id:
                    self.token2id[token] = len(self.token2id)
        self._rebuild_reverse()
        return self

    def encode_tokens(self, tokens: Sequence[str], max_len: int) -> list[int]:
        ids = [self.sos_id]
        ids.extend(self.token2id.get(token, self.unk_id) for token in tokens)
        ids.append(self.eos_id)
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.eos_id]
        return ids

    def encode_segments(self, segments: Sequence[int], max_len: int) -> list[int]:
        encoded = [SEGMENT_GLOBAL]
        encoded.extend(int(value) for value in segments)
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
        segment_tensor = torch.full((max_len,), SEGMENT_GLOBAL, dtype=torch.long)
        token_tensor[: len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
        segment_tensor[: len(segment_ids)] = torch.tensor(segment_ids, dtype=torch.long)
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
    slot_marker_positions: Tensor
    static_charge_features: Tensor
    topology_features: Tensor
    external_mass_summary: Tensor
    channel_id: Tensor
    process_family_id: Tensor
    pattern_id: Tensor
    tgt_tokens: Tensor
    diagram_idx: Tensor


@dataclass
class SequenceBatch:
    src_ids: Tensor
    src_segment_ids: Tensor
    slot_marker_positions: Tensor
    static_charge_features: Tensor
    topology_features: Tensor
    external_mass_summary: Tensor
    channel_id: Tensor
    process_family_id: Tensor
    pattern_id: Tensor
    tgt_tokens: Tensor
    diagram_idx: Tensor

    def to(self, device: torch.device | str) -> "SequenceBatch":
        return SequenceBatch(
            src_ids=self.src_ids.to(device),
            src_segment_ids=self.src_segment_ids.to(device),
            slot_marker_positions=self.slot_marker_positions.to(device),
            static_charge_features=self.static_charge_features.to(device),
            topology_features=self.topology_features.to(device),
            external_mass_summary=self.external_mass_summary.to(device),
            channel_id=self.channel_id.to(device),
            process_family_id=self.process_family_id.to(device),
            pattern_id=self.pattern_id.to(device),
            tgt_tokens=self.tgt_tokens.to(device),
            diagram_idx=self.diagram_idx.to(device),
        )


class QEDSequenceDataset(Dataset):
    def __init__(
        self,
        corpus: LoadedQEDCorpus,
        indices: Sequence[int],
        target_tokenizer: AmplitudeTokenizer,
        source_tokenizer: DiagramSequenceTokenizer,
        max_src_seq_len: int = 256,
        max_tgt_seq_len: int = 1400,
    ):
        super().__init__()
        self.corpus = corpus
        self.indices = list(indices)
        self.target_tokenizer = target_tokenizer
        self.source_tokenizer = source_tokenizer
        self.max_src_seq_len = max_src_seq_len
        self.max_tgt_seq_len = max_tgt_seq_len
        self.serialized = [
            serialize_interaction_to_sequence(corpus.interactions[index])
            for index in self.indices
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> SequenceExample:
        corpus_idx = self.indices[index]
        tokens, segments, marker_positions = self.serialized[index]
        src_ids, src_segment_ids = self.source_tokenizer.encode_tensor(
            tokens,
            segments,
            max_len=self.max_src_seq_len,
        )
        target = self.corpus.targets[corpus_idx]
        feature = self.corpus.features[corpus_idx]
        tgt_tokens = self.target_tokenizer.encode_tensor(
            target.sequence_target_text(),
            max_len=self.max_tgt_seq_len,
        )
        marker_tensor = torch.tensor(marker_positions, dtype=torch.long)
        marker_tensor = torch.clamp(marker_tensor, max=self.max_src_seq_len - 1)
        return SequenceExample(
            src_ids=src_ids,
            src_segment_ids=src_segment_ids,
            slot_marker_positions=marker_tensor,
            static_charge_features=feature.static_charge_features,
            topology_features=feature.topology_features,
            external_mass_summary=feature.external_mass_summary,
            channel_id=feature.channel_id.view(1),
            process_family_id=feature.process_family_id.view(1),
            pattern_id=feature.pattern_id.view(1),
            tgt_tokens=tgt_tokens,
            diagram_idx=torch.tensor(corpus_idx, dtype=torch.long),
        )


def collate_sequence_examples(batch: Sequence[SequenceExample]) -> SequenceBatch:
    return SequenceBatch(
        src_ids=torch.stack([item.src_ids for item in batch], dim=0),
        src_segment_ids=torch.stack([item.src_segment_ids for item in batch], dim=0),
        slot_marker_positions=torch.stack([item.slot_marker_positions for item in batch], dim=0),
        static_charge_features=torch.stack([item.static_charge_features for item in batch], dim=0),
        topology_features=torch.stack([item.topology_features for item in batch], dim=0),
        external_mass_summary=torch.stack([item.external_mass_summary for item in batch], dim=0),
        channel_id=torch.cat([item.channel_id for item in batch], dim=0),
        process_family_id=torch.cat([item.process_family_id for item in batch], dim=0),
        pattern_id=torch.cat([item.pattern_id for item in batch], dim=0),
        tgt_tokens=torch.stack([item.tgt_tokens for item in batch], dim=0),
        diagram_idx=torch.stack([item.diagram_idx for item in batch], dim=0),
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
) -> tuple[DataLoader, DataLoader, DataLoader, AmplitudeTokenizer, DiagramSequenceTokenizer]:
    print(f"Loading QED sequence data from {data_dir}...")
    corpus = load_qed_corpus(data_dir, target_variant=target_variant)
    print(f"Loaded {len(corpus.interactions)} QED interactions for sequence baseline")

    train_indices, val_indices, test_indices = _resolve_splits(
        corpus=corpus,
        seed=seed,
    )

    target_tokenizer = AmplitudeTokenizer(expression_mode=expression_mode).build_vocab(
        [corpus.targets[index].sequence_target_text() for index in train_indices]
    )

    source_sequences = [
        serialize_interaction_to_sequence(corpus.interactions[index])[0]
        for index in train_indices
    ]
    source_tokenizer = DiagramSequenceTokenizer().build_vocab(source_sequences)

    train_ds = QEDSequenceDataset(
        corpus=corpus,
        indices=train_indices,
        target_tokenizer=target_tokenizer,
        source_tokenizer=source_tokenizer,
        max_src_seq_len=max_src_seq_len,
        max_tgt_seq_len=max_tgt_seq_len,
    )
    val_ds = QEDSequenceDataset(
        corpus=corpus,
        indices=val_indices,
        target_tokenizer=target_tokenizer,
        source_tokenizer=source_tokenizer,
        max_src_seq_len=max_src_seq_len,
        max_tgt_seq_len=max_tgt_seq_len,
    )
    test_ds = QEDSequenceDataset(
        corpus=corpus,
        indices=test_indices,
        target_tokenizer=target_tokenizer,
        source_tokenizer=source_tokenizer,
        max_src_seq_len=max_src_seq_len,
        max_tgt_seq_len=max_tgt_seq_len,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
        collate_fn=collate_sequence_examples,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_sequence_examples,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_sequence_examples,
    )
    return train_loader, val_loader, test_loader, target_tokenizer, source_tokenizer


__all__ = [
    "DiagramSequenceTokenizer",
    "NUM_SEGMENT_TYPES",
    "QEDSequenceDataset",
    "SEGMENT_PROP",
    "SEGMENT_PROPAGATOR",
    "SequenceBatch",
    "create_sequence_dataloaders",
    "serialize_interaction_to_sequence",
]


SEGMENT_PROPAGATOR = SEGMENT_PROP
