"""Datasets and source serialization for QED amplitude -> squared-amplitude training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .compiler import (
    ATOM_TYPE_GAMMA,
    ATOM_TYPE_MOMENTUM,
    ATOM_TYPE_POLARIZATION,
    ATOM_TYPE_SPINOR,
    CanonicalQEDAmplitude,
    CanonicalQEDAmplitudeTerm,
    create_canonical_corpus,
    load_canonical_corpus,
)
from .factorization import FactorizedTarget, factorize_squared_amplitude
from .interaction import QEDAmplitudeRecord
from .parser import parse_all_qed
from .sequence_utils import (
    CanonicalSourceTokenizer,
    SRC_TYPE_GLOBAL,
    SRC_TYPE_RAW,
    SRC_TYPE_TERM,
    encode_source_type_tensor,
    tokenize_raw_amplitude,
)
from .splits import random_split_indices
from .tokenizer import AmplitudeTokenizer


ATOM_TYPE_ORDER: tuple[str, ...] = (
    ATOM_TYPE_GAMMA,
    ATOM_TYPE_MOMENTUM,
    ATOM_TYPE_SPINOR,
    ATOM_TYPE_POLARIZATION,
)

FLAVOR_ORDER: tuple[str, ...] = (
    "<NONE>",
    "A",
    "e",
    "mu",
    "t",
    "tt",
    "u",
    "d",
    "s",
    "c",
    "b",
)

ROLE_ORDER: tuple[str, ...] = ("<NONE>", "u", "v")


@dataclass(frozen=True)
class LoadedQEDAmp2SqCorpus:
    records: list[QEDAmplitudeRecord]
    amplitudes: list[CanonicalQEDAmplitude]
    targets: list[FactorizedTarget]


@dataclass
class QEDAmp2SqExample:
    global_src_ids: Tensor
    term_src_ids: Tensor
    term_skeleton_ids: Tensor
    family_src_ids: Tensor
    flat_src_ids: Tensor
    flat_src_type_ids: Tensor
    term_count: Tensor
    tgt_tokens: Tensor
    sample_idx: Tensor


@dataclass
class QEDAmp2SqBatch:
    global_src_ids: Tensor
    term_src_ids: Tensor
    term_skeleton_ids: Tensor
    family_src_ids: Tensor
    flat_src_ids: Tensor
    flat_src_type_ids: Tensor
    term_count: Tensor
    tgt_tokens: Tensor
    sample_idx: Tensor

    def to(self, device: torch.device | str) -> "QEDAmp2SqBatch":
        return QEDAmp2SqBatch(
            global_src_ids=self.global_src_ids.to(device),
            term_src_ids=self.term_src_ids.to(device),
            term_skeleton_ids=self.term_skeleton_ids.to(device),
            family_src_ids=self.family_src_ids.to(device),
            flat_src_ids=self.flat_src_ids.to(device),
            flat_src_type_ids=self.flat_src_type_ids.to(device),
            term_count=self.term_count.to(device),
            tgt_tokens=self.tgt_tokens.to(device),
            sample_idx=self.sample_idx.to(device),
        )


def _normalized_scalar_symbol(symbol: str, momentum_map: dict[str, int]) -> str:
    if symbol.startswith("s_") or symbol.startswith("p_"):
        if symbol not in momentum_map:
            momentum_map[symbol] = len(momentum_map)
        return f"MOMENTUM_{momentum_map[symbol]}"
    return symbol


def build_momentum_map(amplitude: CanonicalQEDAmplitude) -> dict[str, int]:
    momentum_map: dict[str, int] = {}
    for token in amplitude.denominator_tokens:
        if token.startswith("s_") or token.startswith("p_") or token.startswith("MOMENTUM_"):
            if token not in momentum_map:
                momentum_map[token] = len(momentum_map)
    for term in amplitude.terms:
        for symbol in term.scalar_factors:
            if symbol.startswith("s_") or symbol.startswith("p_"):
                if symbol not in momentum_map:
                    momentum_map[symbol] = len(momentum_map)
    return momentum_map


def serialize_global_tokens(
    amplitude: CanonicalQEDAmplitude,
    momentum_map: dict[str, int],
) -> list[str]:
    tokens: list[str] = [
        "[GLOBAL]",
        "[I_POW]",
        str(amplitude.global_i_power),
        "[E_POW]",
        str(amplitude.global_e_power),
        "[RAT]",
        str(amplitude.global_rational.numerator),
        "/",
        str(amplitude.global_rational.denominator),
        f"PATTERN_{amplitude.external_pattern}",
        f"PROCESS_{amplitude.process_family.upper()}",
        f"CHANNEL_{amplitude.channel.upper()}",
        f"PROP_{amplitude.propagator_species.upper()}",
        "[DEN]",
    ]
    for token in amplitude.denominator_tokens:
        tokens.append(_normalized_scalar_symbol(token, momentum_map))
    tokens.append("[END_GLOBAL]")
    return tokens


def serialize_term_summary_tokens(
    term: CanonicalQEDAmplitudeTerm,
    momentum_map: dict[str, int],
) -> list[str]:
    atom_counts = Counter(atom.atom_type for atom in term.atoms)
    flavor_counts = Counter(
        atom.flavor for atom in term.atoms if atom.flavor != FLAVOR_ORDER[0]
    )
    role_counts = Counter(
        atom.spin_role for atom in term.atoms if atom.spin_role != ROLE_ORDER[0]
    )
    momentum_slots = sorted(
        {atom.momentum_index for atom in term.atoms if atom.momentum_index >= 0}
    )
    external_pindices = sorted(
        {term.atoms[index].pindex for index in term.external_wavefunctions}
    )

    tokens: list[str] = ["[TERM_SUMMARY]", "[COEFF]"]
    tokens.extend(term.coefficient_tokens())
    tokens.extend(
        [
            "[N_ATOMS]",
            str(len(term.atoms)),
            "[N_DIRAC]",
            str(len(term.dirac_chains)),
            "[N_LORENTZ]",
            str(len(term.lorentz_chains)),
            "[N_MOM]",
            str(len(term.momentum_insertions)),
            "[N_EXT]",
            str(len(term.external_wavefunctions)),
        ]
    )

    if term.scalar_factors:
        tokens.append("[SCALARS]")
        for symbol in term.scalar_factors:
            tokens.extend(["[SCALAR]", _normalized_scalar_symbol(symbol, momentum_map)])
    else:
        tokens.append("[NO_SCALARS]")

    tokens.append("[ATOM_COUNTS]")
    for atom_type in ATOM_TYPE_ORDER:
        count = atom_counts.get(atom_type, 0)
        if count > 0:
            tokens.extend([f"[COUNT_{atom_type.upper()}]", str(count)])

    if flavor_counts:
        tokens.append("[FLAVOR_COUNTS]")
        for flavor in FLAVOR_ORDER[1:]:
            count = flavor_counts.get(flavor, 0)
            if count > 0:
                tokens.extend([f"[COUNT_FLAVOR_{flavor.upper()}]", str(count)])

    if role_counts:
        tokens.append("[ROLE_COUNTS]")
        for role in ROLE_ORDER[1:]:
            count = role_counts.get(role, 0)
            if count > 0:
                tokens.extend([f"[COUNT_ROLE_{role.upper()}]", str(count)])

    if term.dirac_chains:
        tokens.append("[DIRAC_LENS]")
        for chain in term.dirac_chains:
            tokens.append(str(len(chain)))

    if term.lorentz_chains:
        tokens.append("[LORENTZ_LENS]")
        for chain in term.lorentz_chains:
            tokens.append(str(len(chain)))

    if momentum_slots:
        tokens.append("[MOMENTUM_SLOTS]")
        for slot in momentum_slots:
            tokens.append(f"MOMENTUM_{slot}")

    if external_pindices:
        tokens.append("[EXT_PINDEX]")
        for pindex in external_pindices:
            tokens.append(f"PINDEX_{pindex}")

    tokens.append("[END_TERM_SUMMARY]")
    return tokens


def build_physics_augmented_source_tokens(
    record: QEDAmplitudeRecord,
    amplitude: CanonicalQEDAmplitude,
) -> tuple[list[str], list[int]]:
    momentum_map = build_momentum_map(amplitude)
    raw_tokens = tokenize_raw_amplitude(record.raw_amplitude)
    global_tokens = serialize_global_tokens(amplitude, momentum_map)
    term_summary_tokens = [
        token
        for term in amplitude.terms
        for token in serialize_term_summary_tokens(term, momentum_map)
    ]

    tokens: list[str] = []
    type_ids: list[int] = []

    def extend(section_tokens: Sequence[str], type_id: int) -> None:
        tokens.extend(section_tokens)
        type_ids.extend([type_id] * len(section_tokens))

    extend(["[RAW_SRC]"], SRC_TYPE_RAW)
    extend(raw_tokens, SRC_TYPE_RAW)
    extend(["[END_RAW_SRC]"], SRC_TYPE_RAW)
    extend(global_tokens, SRC_TYPE_GLOBAL)
    extend(["[TERM_SUMMARIES]"], SRC_TYPE_TERM)
    extend(term_summary_tokens, SRC_TYPE_TERM)
    extend(["[END_TERM_SUMMARIES]", "[END_SRC]"], SRC_TYPE_TERM)
    return tokens, type_ids


class QEDAmp2SqDataset(Dataset):
    def __init__(
        self,
        corpus: LoadedQEDAmp2SqCorpus,
        indices: Sequence[int],
        source_tokenizer: CanonicalSourceTokenizer,
        target_tokenizer: AmplitudeTokenizer,
        max_src_term_len: int,
        max_src_terms: int,
        max_flat_src_len: int,
        max_tgt_seq_len: int,
        encoder_variant: str,
    ):
        self.corpus = corpus
        self.indices = list(indices)
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_src_term_len = max_src_term_len
        self.max_src_terms = max_src_terms
        self.max_flat_src_len = max_flat_src_len
        self.max_tgt_seq_len = max_tgt_seq_len
        self.encoder_variant = encoder_variant

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> QEDAmp2SqExample:
        corpus_idx = self.indices[index]
        amplitude = self.corpus.amplitudes[corpus_idx]
        record = self.corpus.records[corpus_idx]
        target = self.corpus.targets[corpus_idx]

        raw_tokens = tokenize_raw_amplitude(record.raw_amplitude)
        pad_term_ids = torch.full(
            (self.max_src_terms, self.max_src_term_len),
            self.source_tokenizer.pad_id,
            dtype=torch.long,
        )
        pad_seq_ids = torch.full(
            (self.max_src_term_len,),
            self.source_tokenizer.pad_id,
            dtype=torch.long,
        )
        if self.encoder_variant == "seq2seq":
            source_tokens = raw_tokens
            source_type_ids = [SRC_TYPE_RAW] * len(raw_tokens)
        else:
            source_tokens, source_type_ids = build_physics_augmented_source_tokens(
                record,
                amplitude,
            )

        return QEDAmp2SqExample(
            global_src_ids=pad_seq_ids.clone(),
            term_src_ids=pad_term_ids.clone(),
            term_skeleton_ids=pad_term_ids.clone(),
            family_src_ids=pad_seq_ids.clone(),
            flat_src_ids=self.source_tokenizer.encode_tensor(
                source_tokens,
                max_len=self.max_flat_src_len,
            ),
            flat_src_type_ids=encode_source_type_tensor(
                source_type_ids,
                max_len=self.max_flat_src_len,
            ),
            term_count=torch.tensor(0, dtype=torch.long),
            tgt_tokens=self.target_tokenizer.encode_tensor(
                target.sequence_target_text(),
                max_len=self.max_tgt_seq_len,
            ),
            sample_idx=torch.tensor(corpus_idx, dtype=torch.long),
        )


def collate_examples(batch: Sequence[QEDAmp2SqExample]) -> QEDAmp2SqBatch:
    return QEDAmp2SqBatch(
        global_src_ids=torch.stack([item.global_src_ids for item in batch]),
        term_src_ids=torch.stack([item.term_src_ids for item in batch]),
        term_skeleton_ids=torch.stack([item.term_skeleton_ids for item in batch]),
        family_src_ids=torch.stack([item.family_src_ids for item in batch]),
        flat_src_ids=torch.stack([item.flat_src_ids for item in batch]),
        flat_src_type_ids=torch.stack([item.flat_src_type_ids for item in batch]),
        term_count=torch.stack([item.term_count for item in batch]),
        tgt_tokens=torch.stack([item.tgt_tokens for item in batch]),
        sample_idx=torch.stack([item.sample_idx for item in batch]),
    )


def load_qed_amp2sq_corpus(
    data_dir: str | Path,
    cache_dir: str | Path | None = None,
    target_variant: str = "factorized",
) -> LoadedQEDAmp2SqCorpus:
    records = parse_all_qed(data_dir)

    amplitudes: list[CanonicalQEDAmplitude]
    cache_path = None if cache_dir is None else Path(cache_dir) / "corpus.json"
    if cache_path is not None and cache_path.exists():
        amplitudes = load_canonical_corpus(cache_path)
    else:
        amplitudes = create_canonical_corpus(
            ((record.sample_id, record.raw_amplitude) for record in records),
            cache_dir=cache_dir,
        )

    targets = [
        factorize_squared_amplitude(
            record.raw_squared,
            target_variant=target_variant,
        )
        for record in records
    ]
    return LoadedQEDAmp2SqCorpus(
        records=records,
        amplitudes=amplitudes,
        targets=targets,
    )


def resolve_splits(
    corpus: LoadedQEDAmp2SqCorpus,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    return random_split_indices(len(corpus.targets), seed=seed)


def build_source_tokenizer(
    corpus: LoadedQEDAmp2SqCorpus,
    train_indices: Sequence[int],
    encoder_variant: str,
) -> CanonicalSourceTokenizer:
    sequences: list[list[str]] = []
    if encoder_variant == "seq2seq":
        for index in train_indices:
            sequences.append(tokenize_raw_amplitude(corpus.records[index].raw_amplitude))
    else:
        for index in train_indices:
            source_tokens, _ = build_physics_augmented_source_tokens(
                corpus.records[index],
                corpus.amplitudes[index],
            )
            sequences.append(source_tokens)
    return CanonicalSourceTokenizer().build_vocab(sequences)
