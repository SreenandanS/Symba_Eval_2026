"""Datasets and source serialization for amplitude -> squared-amplitude training."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset
try:
    from torch_geometric.data import Batch, Data
except ModuleNotFoundError:  # pragma: no cover - graph path is inactive here
    class Data:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def to(self, device):
            del device
            return self

    class Batch:
        @classmethod
        def from_data_list(cls, data_list):
            return data_list

from .factorization import FactorizedTarget, factorize_squared_amplitude
from .tokenizer import AmplitudeTokenizer

from .compiler import (
    ATOM_TYPE_COLOR_GEN,
    ATOM_TYPE_GAMMA,
    ATOM_TYPE_GLUON,
    ATOM_TYPE_MOMENTUM,
    ATOM_TYPE_SPINOR,
    ATOM_TYPE_STRUCT,
    CanonicalAmplitude,
    CanonicalAmplitudeTerm,
    CanonicalTensorAtom,
    create_canonical_corpus,
    load_canonical_corpus,
)
from .contracts import (
    COLOR_REL_CHAIN,
    COLOR_REL_INDEX,
    COLOR_REL_TERM_ATOM,
    DIRAC_REL_CHAIN,
    DIRAC_REL_INDEX,
    DIRAC_REL_TERM_ATOM,
    INDEX_KIND_COLOR,
    INDEX_KIND_DIRAC,
    INDEX_KIND_LORENTZ,
    NODE_KIND_ATOM,
    NODE_KIND_GLOBAL,
    NODE_KIND_INDEX,
    NODE_KIND_SCALAR,
    NODE_KIND_TERM,
    NUM_COLOR_RELATIONS,
    NUM_DIRAC_RELATIONS,
    NUM_INDEX_KINDS,
    NUM_NODE_KINDS,
    NUM_SCALAR_KINDS,
    SCALAR_KIND_DENOMINATOR,
    SCALAR_KIND_MASS,
    SCALAR_KIND_MOMENTUM,
    SCALAR_KIND_NONE,
    SCALAR_KIND_NUMERATOR,
    SCALAR_KIND_REG,
    SCALAR_REL_GLOBAL_SCALAR,
    SCALAR_REL_GLOBAL_TERM,
    SCALAR_REL_TERM_SCALAR,
    NUM_SCALAR_RELATIONS,
)
from .interaction import Amp2SqInteraction, build_interaction
from .parser import parse_all_qcd
from .sequence_utils import (
    CanonicalSourceTokenizer,
    SRC_TYPE_FAMILY,
    SRC_TYPE_GLOBAL,
    SRC_TYPE_PAD,
    SRC_TYPE_RAW,
    SRC_TYPE_TERM,
    encode_source_type_tensor,
    tokenize_family_signature,
    tokenize_raw_amplitude,
)
from .splits import random_split_indices


ATOM_TYPE_ORDER: tuple[str, ...] = (
    ATOM_TYPE_GAMMA,
    ATOM_TYPE_MOMENTUM,
    ATOM_TYPE_SPINOR,
    ATOM_TYPE_GLUON,
    ATOM_TYPE_COLOR_GEN,
    ATOM_TYPE_STRUCT,
)
ATOM_TYPE_TO_ID = {value: index for index, value in enumerate(ATOM_TYPE_ORDER)}

FLAVOR_ORDER: tuple[str, ...] = ("<NONE>", "u", "d", "s", "t", "c", "b", "G")
FLAVOR_TO_ID = {value: index for index, value in enumerate(FLAVOR_ORDER)}

ROLE_ORDER: tuple[str, ...] = ("<NONE>", "u", "v")
ROLE_TO_ID = {value: index for index, value in enumerate(ROLE_ORDER)}

TOTAL_EDGE_RELATIONS = NUM_DIRAC_RELATIONS + NUM_COLOR_RELATIONS + NUM_SCALAR_RELATIONS


@dataclass(frozen=True)
class LoadedAmp2SqCorpus:
    interactions: list[Amp2SqInteraction]
    amplitudes: list[CanonicalAmplitude]
    targets: list[FactorizedTarget]


@dataclass
class Amp2SqExample:
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
class Amp2SqBatch:
    global_src_ids: Tensor
    term_src_ids: Tensor
    term_skeleton_ids: Tensor
    family_src_ids: Tensor
    flat_src_ids: Tensor
    flat_src_type_ids: Tensor
    term_count: Tensor
    tgt_tokens: Tensor
    sample_idx: Tensor

    def to(self, device: torch.device | str) -> "Amp2SqBatch":
        return Amp2SqBatch(
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


def _one_hot(index: int, size: int) -> list[float]:
    vector = [0.0] * size
    if 0 <= index < size:
        vector[index] = 1.0
    return vector


def _normalized_scalar_symbol(symbol: str, momentum_map: dict[str, int]) -> str:
    if symbol.startswith("s_") or symbol.startswith("p_"):
        if symbol not in momentum_map:
            momentum_map[symbol] = len(momentum_map)
        return f"MOMENTUM_{momentum_map[symbol]}"
    return symbol


def build_momentum_map(amplitude: CanonicalAmplitude) -> dict[str, int]:
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
    amplitude: CanonicalAmplitude,
    momentum_map: dict[str, int],
) -> list[str]:
    tokens: list[str] = [
        "[GLOBAL]",
        "[I_POW]",
        str(amplitude.global_i_power),
        "[G_POW]",
        str(amplitude.global_g_power),
        "[RAT]",
        str(amplitude.global_rational.numerator),
        "/",
        str(amplitude.global_rational.denominator),
        "[DEN]",
    ]
    for token in amplitude.denominator_tokens:
        tokens.append(_normalized_scalar_symbol(token, momentum_map))
    tokens.append("[END_GLOBAL]")
    return tokens


def serialize_term_tokens(
    term: CanonicalAmplitudeTerm,
    momentum_map: dict[str, int],
) -> list[str]:
    tokens: list[str] = ["[TERM]", "[COEFF]"]
    sign = "NEG" if term.coefficient < 0 else "POS"
    magnitude = abs(term.coefficient.numerator)
    tokens.extend([f"SIGN_{sign}", str(magnitude)])
    if term.coefficient.denominator != 1:
        tokens.extend(["/", str(term.coefficient.denominator)])

    tokens.append("[SCALARS]")
    if term.scalar_factors:
        for symbol in term.scalar_factors:
            tokens.extend(["[SCALAR]", _normalized_scalar_symbol(symbol, momentum_map)])
    else:
        tokens.append("[NO_SCALARS]")

    tokens.append("[DIRAC]")
    if term.dirac_chains:
        for chain in term.dirac_chains:
            tokens.append("[CHAIN]")
            for atom_index in chain:
                tokens.extend(term.atoms[atom_index].to_typed_tokens())
            tokens.append("[END_CHAIN]")
    else:
        tokens.append("[NO_DIRAC]")

    tokens.append("[COLOR]")
    if term.color_chains:
        for chain in term.color_chains:
            tokens.append("[CHAIN]")
            for atom_index in chain:
                tokens.extend(term.atoms[atom_index].to_typed_tokens())
            tokens.append("[END_CHAIN]")
    else:
        tokens.append("[NO_COLOR]")

    tokens.append("[END_TERM]")
    return tokens


def serialize_term_skeleton_tokens(term: CanonicalAmplitudeTerm) -> list[str]:
    return term.skeleton_tokens()


def serialize_family_tokens(amplitude: CanonicalAmplitude) -> list[str]:
    tokens = ["[FAMILY]"]
    tokens.extend(tokenize_family_signature(amplitude.family_signature))
    tokens.append("[END_FAMILY]")
    return tokens


def serialize_term_summary_tokens(
    term: CanonicalAmplitudeTerm,
    momentum_map: dict[str, int],
) -> list[str]:
    atom_counts = Counter(atom.atom_type for atom in term.atoms)
    flavor_counts = Counter(
        atom.flavor for atom in term.atoms if atom.flavor != FLAVOR_ORDER[0]
    )
    momentum_slots = sorted(
        {
            atom.momentum_index
            for atom in term.atoms
            if atom.momentum_index >= 0
        }
    )
    tokens: list[str] = ["[TERM_SUMMARY]", "[COEFF]"]
    tokens.extend(term.coefficient_tokens())
    tokens.extend(
        [
            "[N_ATOMS]",
            str(len(term.atoms)),
            "[N_DIRAC]",
            str(len(term.dirac_chains)),
            "[N_COLOR]",
            str(len(term.color_chains)),
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
            tokens.extend(
                [f"[COUNT_{atom_type.upper().replace('-', '_')}]", str(count)]
            )

    if flavor_counts:
        tokens.append("[FLAVOR_COUNTS]")
        for flavor in FLAVOR_ORDER[1:]:
            count = flavor_counts.get(flavor, 0)
            if count > 0:
                tokens.extend([f"[COUNT_FLAVOR_{flavor.upper()}]", str(count)])

    if term.dirac_chains:
        tokens.append("[DIRAC_LENS]")
        for chain in term.dirac_chains:
            tokens.append(str(len(chain)))

    if term.color_chains:
        tokens.append("[COLOR_LENS]")
        for chain in term.color_chains:
            tokens.append(str(len(chain)))

    if momentum_slots:
        tokens.append("[MOMENTUM_SLOTS]")
        for slot in momentum_slots:
            tokens.append(f"MOMENTUM_{slot}")

    tokens.append("[END_TERM_SUMMARY]")
    return tokens


def build_physics_augmented_source_tokens(
    interaction: Amp2SqInteraction,
    amplitude: CanonicalAmplitude,
) -> tuple[list[str], list[int]]:
    momentum_map = build_momentum_map(amplitude)
    raw_tokens = tokenize_raw_amplitude(interaction.diagram.raw_amplitude)
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


def _index_kind_from_atom(atom: CanonicalTensorAtom, slot: int) -> int:
    connector_kinds = atom.connector_kinds()
    if slot >= len(connector_kinds):
        return 0
    kind = connector_kinds[slot]
    if kind == "lorentz":
        return INDEX_KIND_LORENTZ
    if kind == "dirac":
        return INDEX_KIND_DIRAC
    if kind == "color":
        return INDEX_KIND_COLOR
    return 0


def _scalar_kind(symbol: str, denominator: bool = False) -> int:
    if symbol.startswith("MOMENTUM_") or symbol.startswith("s_") or symbol.startswith("p_"):
        return SCALAR_KIND_MOMENTUM
    if symbol.startswith("m_"):
        return SCALAR_KIND_MASS
    if symbol == "reg_prop":
        return SCALAR_KIND_REG
    return SCALAR_KIND_DENOMINATOR if denominator else SCALAR_KIND_NUMERATOR


def _build_node_feature(
    *,
    node_kind: int,
    atom_type: str | None = None,
    scalar_kind: int = SCALAR_KIND_NONE,
    flavor: str = "<NONE>",
    role: str = "<NONE>",
    index_kind: int = 0,
    conj: bool = False,
    pindex: int = 0,
    momentum_index: int = -1,
    coeff: Fraction | None = None,
    global_i_power: int = 0,
    global_g_power: int = 0,
    size_hint: int = 0,
) -> Tensor:
    numeric = [0.0] * 8
    numeric[0] = 1.0 if conj else 0.0
    numeric[1] = pindex / 4.0 if pindex > 0 else 0.0
    numeric[2] = (
        (momentum_index + 1) / 8.0
        if momentum_index >= 0
        else 0.0
    )
    if coeff is not None:
        numeric[3] = -1.0 if coeff < 0 else (1.0 if coeff > 0 else 0.0)
        numeric[4] = math.log1p(abs(coeff.numerator))
        numeric[5] = math.log1p(coeff.denominator)
    numeric[6] = global_i_power / 4.0
    numeric[7] = max(size_hint, 0) / 16.0

    feature = (
        _one_hot(node_kind, NUM_NODE_KINDS)
        + _one_hot(ATOM_TYPE_TO_ID.get(atom_type, -1), len(ATOM_TYPE_ORDER))
        + _one_hot(scalar_kind, NUM_SCALAR_KINDS)
        + _one_hot(FLAVOR_TO_ID.get(flavor, -1), len(FLAVOR_ORDER))
        + _one_hot(ROLE_TO_ID.get(role, -1), len(ROLE_ORDER))
        + _one_hot(index_kind, NUM_INDEX_KINDS)
        + numeric
    )
    return torch.tensor(feature, dtype=torch.float32)


def _append_bidirectional(
    edges: list[list[int]],
    rels: list[int],
    src: int,
    dst: int,
    relation: int,
) -> None:
    edges.append([src, dst])
    rels.append(relation)
    edges.append([dst, src])
    rels.append(relation)


def canonical_amplitude_to_graph(
    amplitude: CanonicalAmplitude,
    momentum_map: dict[str, int],
) -> Data:
    node_features: list[Tensor] = []
    node_kind_ids: list[int] = []
    term_mask: list[bool] = []
    term_position: list[int] = []

    dirac_edges: list[list[int]] = []
    dirac_rels: list[int] = []
    color_edges: list[list[int]] = []
    color_rels: list[int] = []
    scalar_edges: list[list[int]] = []
    scalar_rels: list[int] = []

    def add_node(
        feature: Tensor,
        node_kind: int,
        is_term: bool = False,
        position: int = -1,
    ) -> int:
        node_id = len(node_features)
        node_features.append(feature)
        node_kind_ids.append(node_kind)
        term_mask.append(is_term)
        term_position.append(position)
        return node_id

    global_node = add_node(
        _build_node_feature(
            node_kind=NODE_KIND_GLOBAL,
            global_i_power=amplitude.global_i_power,
            global_g_power=amplitude.global_g_power,
            coeff=amplitude.global_rational,
            size_hint=amplitude.term_count,
        ),
        NODE_KIND_GLOBAL,
    )

    denominator_scalar_tokens = [
        token
        for token in amplitude.denominator_tokens
        if token not in {"(", ")", "+", "-", "*", "/", "^"}
    ]
    for token in denominator_scalar_tokens:
        normalized = _normalized_scalar_symbol(token, momentum_map)
        scalar_node = add_node(
            _build_node_feature(
                node_kind=NODE_KIND_SCALAR,
                scalar_kind=_scalar_kind(normalized, denominator=True),
                momentum_index=momentum_map.get(token, momentum_map.get(normalized, -1)),
            ),
            NODE_KIND_SCALAR,
        )
        _append_bidirectional(
            scalar_edges,
            scalar_rels,
            global_node,
            scalar_node,
            SCALAR_REL_GLOBAL_SCALAR,
        )

    for term_idx, term in enumerate(amplitude.terms):
        term_node = add_node(
            _build_node_feature(
                node_kind=NODE_KIND_TERM,
                coeff=term.coefficient,
                size_hint=len(term.atoms) + len(term.scalar_factors),
            ),
            NODE_KIND_TERM,
            is_term=True,
            position=term_idx,
        )
        _append_bidirectional(
            scalar_edges,
            scalar_rels,
            global_node,
            term_node,
            SCALAR_REL_GLOBAL_TERM,
        )

        index_nodes: dict[tuple[int, int], int] = {}
        atom_nodes: list[int] = []

        for symbol in term.scalar_factors:
            normalized = _normalized_scalar_symbol(symbol, momentum_map)
            scalar_node = add_node(
                _build_node_feature(
                    node_kind=NODE_KIND_SCALAR,
                    scalar_kind=_scalar_kind(normalized, denominator=False),
                    momentum_index=momentum_map.get(symbol, -1),
                ),
                NODE_KIND_SCALAR,
            )
            _append_bidirectional(
                scalar_edges,
                scalar_rels,
                term_node,
                scalar_node,
                SCALAR_REL_TERM_SCALAR,
            )

        for atom_idx, atom in enumerate(term.atoms):
            atom_node = add_node(
                _build_node_feature(
                    node_kind=NODE_KIND_ATOM,
                    atom_type=atom.atom_type,
                    flavor=atom.flavor,
                    role=atom.spin_role,
                    conj=atom.conjugated,
                    pindex=atom.pindex,
                    momentum_index=atom.momentum_index,
                ),
                NODE_KIND_ATOM,
            )
            atom_nodes.append(atom_node)

            if atom.atom_type in {ATOM_TYPE_GAMMA, ATOM_TYPE_MOMENTUM, ATOM_TYPE_SPINOR, ATOM_TYPE_GLUON}:
                _append_bidirectional(
                    dirac_edges,
                    dirac_rels,
                    term_node,
                    atom_node,
                    DIRAC_REL_TERM_ATOM,
                )
            if atom.atom_type in {ATOM_TYPE_SPINOR, ATOM_TYPE_GLUON, ATOM_TYPE_COLOR_GEN, ATOM_TYPE_STRUCT}:
                _append_bidirectional(
                    color_edges,
                    color_rels,
                    term_node,
                    atom_node,
                    COLOR_REL_TERM_ATOM,
                )

            for slot, index_id in enumerate(atom.index_ids):
                if index_id < 0:
                    continue
                index_kind = _index_kind_from_atom(atom, slot)
                node_key = (index_kind, index_id)
                if node_key not in index_nodes:
                    index_nodes[node_key] = add_node(
                        _build_node_feature(
                            node_kind=NODE_KIND_INDEX,
                            index_kind=index_kind,
                            size_hint=index_id + 1,
                        ),
                        NODE_KIND_INDEX,
                    )
                index_node = index_nodes[node_key]
                if index_kind in {INDEX_KIND_LORENTZ, INDEX_KIND_DIRAC}:
                    _append_bidirectional(
                        dirac_edges,
                        dirac_rels,
                        atom_node,
                        index_node,
                        DIRAC_REL_INDEX,
                    )
                if index_kind == INDEX_KIND_COLOR:
                    _append_bidirectional(
                        color_edges,
                        color_rels,
                        atom_node,
                        index_node,
                        COLOR_REL_INDEX,
                    )

        for chain in term.dirac_chains:
            for left, right in zip(chain, chain[1:]):
                _append_bidirectional(
                    dirac_edges,
                    dirac_rels,
                    atom_nodes[left],
                    atom_nodes[right],
                    DIRAC_REL_CHAIN,
                )
        for chain in term.color_chains:
            for left, right in zip(chain, chain[1:]):
                _append_bidirectional(
                    color_edges,
                    color_rels,
                    atom_nodes[left],
                    atom_nodes[right],
                    COLOR_REL_CHAIN,
                )

    x = torch.stack(node_features)
    node_kind = torch.tensor(node_kind_ids, dtype=torch.long)
    term_mask_tensor = torch.tensor(term_mask, dtype=torch.bool)
    term_position_tensor = torch.tensor(term_position, dtype=torch.long)

    def edge_tensor(edges: list[list[int]]) -> Tensor:
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def rel_one_hot(rels: list[int], size: int) -> Tensor:
        if not rels:
            return torch.zeros((0, size), dtype=torch.float32)
        return torch.nn.functional.one_hot(
            torch.tensor(rels, dtype=torch.long),
            num_classes=size,
        ).to(torch.float32)

    dirac_edge_index = edge_tensor(dirac_edges)
    color_edge_index = edge_tensor(color_edges)
    scalar_edge_index = edge_tensor(scalar_edges)
    dirac_edge_attr = rel_one_hot(dirac_rels, NUM_DIRAC_RELATIONS)
    color_edge_attr = rel_one_hot(color_rels, NUM_COLOR_RELATIONS)
    scalar_edge_attr = rel_one_hot(scalar_rels, NUM_SCALAR_RELATIONS)

    all_edges = dirac_edges + color_edges + scalar_edges
    all_rels: list[int] = (
        dirac_rels
        + [NUM_DIRAC_RELATIONS + value for value in color_rels]
        + [NUM_DIRAC_RELATIONS + NUM_COLOR_RELATIONS + value for value in scalar_rels]
    )
    edge_index = edge_tensor(all_edges)
    edge_attr = rel_one_hot(all_rels, TOTAL_EDGE_RELATIONS)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        dirac_edge_index=dirac_edge_index,
        dirac_edge_attr=dirac_edge_attr,
        color_edge_index=color_edge_index,
        color_edge_attr=color_edge_attr,
        scalar_edge_index=scalar_edge_index,
        scalar_edge_attr=scalar_edge_attr,
        node_kind=node_kind,
        term_mask=term_mask_tensor,
        term_position=term_position_tensor,
    )


class Amp2SqDataset(Dataset):
    def __init__(
        self,
        corpus: LoadedAmp2SqCorpus,
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

    def __getitem__(self, index: int) -> Amp2SqExample:
        corpus_idx = self.indices[index]
        amplitude = self.corpus.amplitudes[corpus_idx]
        interaction = self.corpus.interactions[corpus_idx]
        target = self.corpus.targets[corpus_idx]
        raw_tokens = tokenize_raw_amplitude(interaction.diagram.raw_amplitude)
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
                interaction,
                amplitude,
            )

        return Amp2SqExample(
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


def collate_examples(batch: Sequence[Amp2SqExample]) -> Amp2SqBatch:
    return Amp2SqBatch(
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


def load_amp2sq_corpus(
    data_dir: str | Path,
    cache_dir: str | Path | None = None,
    target_variant: str = "factorized",
) -> LoadedAmp2SqCorpus:
    diagrams = parse_all_qcd(data_dir)
    interactions = [
        build_interaction(diagram, sample_id=f"amp2sq_{index:04d}")
        for index, diagram in enumerate(diagrams)
    ]

    amplitudes: list[CanonicalAmplitude]
    cache_path = None if cache_dir is None else Path(cache_dir) / "corpus.json"
    if cache_path is not None and cache_path.exists():
        amplitudes = load_canonical_corpus(cache_path)
    else:
        amplitudes = create_canonical_corpus(
            (
                (interaction.sample_id, interaction.diagram.raw_amplitude)
                for interaction in interactions
            ),
            cache_dir=cache_dir,
        )

    targets = [
        factorize_squared_amplitude(
            diagram.raw_squared,
            target_variant=target_variant,
        )
        for diagram in diagrams
    ]
    return LoadedAmp2SqCorpus(
        interactions=interactions,
        amplitudes=amplitudes,
        targets=targets,
    )


def resolve_splits(
    corpus: LoadedAmp2SqCorpus,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    return random_split_indices(len(corpus.targets), seed=seed)


def build_source_tokenizer(
    corpus: LoadedAmp2SqCorpus,
    train_indices: Sequence[int],
    encoder_variant: str,
) -> CanonicalSourceTokenizer:
    sequences: list[list[str]] = []
    if encoder_variant == "seq2seq":
        for index in train_indices:
            sequences.append(
                tokenize_raw_amplitude(corpus.interactions[index].diagram.raw_amplitude)
            )
    else:
        for index in train_indices:
            amplitude = corpus.amplitudes[index]
            interaction = corpus.interactions[index]
            source_tokens, _ = build_physics_augmented_source_tokens(
                interaction,
                amplitude,
            )
            sequences.append(source_tokens)
    return CanonicalSourceTokenizer().build_vocab(sequences)
