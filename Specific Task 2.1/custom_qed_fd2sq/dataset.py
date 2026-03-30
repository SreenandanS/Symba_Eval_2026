"""Fixed-slot datasets for the QED tree-level 2->2 custom workflow."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .factorization import FactorizedTarget, factorize_squared_amplitude
from .features import InteractionFeatures, build_interaction_features
from .feynman_graph import FixedSlotQEDGraph, diagram_to_fixed_slot_graph
from .interaction import QEDInteraction
from .parser import parse_all_qed
from .relations import InteractionRelations, build_interaction_relations
from .splits import random_split_indices
from .tokenizer import AmplitudeTokenizer


@dataclass(frozen=True)
class LoadedQEDCorpus:
    interactions: list[QEDInteraction]
    targets: list[FactorizedTarget]
    features: list[InteractionFeatures]
    relations: list[InteractionRelations]
    graphs: list[FixedSlotQEDGraph]


@dataclass(frozen=True)
class QEDFeynmanExample:
    slot_features: Tensor
    slot_type_ids: Tensor
    slot_position_ids: Tensor
    fermion_line_ids: Tensor
    channel_prop_edge_index: Tensor
    channel_prop_edge_attr: Tensor
    channel_prop_edge_type: Tensor
    fermion_line_edge_index: Tensor
    fermion_line_edge_attr: Tensor
    fermion_line_edge_type: Tensor
    static_charge_features: Tensor
    topology_features: Tensor
    external_mass_summary: Tensor
    channel_id: Tensor
    process_family_id: Tensor
    pattern_id: Tensor
    external_mask: Tensor
    vertex_mask: Tensor
    propagator_mask: Tensor
    tgt_tokens: Tensor
    diagram_idx: Tensor

    @property
    def x(self) -> Tensor:
        return self.slot_features


@dataclass
class QEDFeynmanBatch:
    slot_features: Tensor
    slot_type_ids: Tensor
    slot_position_ids: Tensor
    fermion_line_ids: Tensor
    channel_prop_edge_index: Tensor
    channel_prop_edge_attr: Tensor
    channel_prop_edge_type: Tensor
    fermion_line_edge_index: Tensor
    fermion_line_edge_attr: Tensor
    fermion_line_edge_type: Tensor
    static_charge_features: Tensor
    topology_features: Tensor
    external_mass_summary: Tensor
    channel_id: Tensor
    process_family_id: Tensor
    pattern_id: Tensor
    external_mask: Tensor
    vertex_mask: Tensor
    propagator_mask: Tensor
    tgt_tokens: Tensor
    diagram_idx: Tensor

    @property
    def x(self) -> Tensor:
        return self.slot_features

    def to(self, device: torch.device | str) -> "QEDFeynmanBatch":
        return QEDFeynmanBatch(
            slot_features=self.slot_features.to(device),
            slot_type_ids=self.slot_type_ids.to(device),
            slot_position_ids=self.slot_position_ids.to(device),
            fermion_line_ids=self.fermion_line_ids.to(device),
            channel_prop_edge_index=self.channel_prop_edge_index.to(device),
            channel_prop_edge_attr=self.channel_prop_edge_attr.to(device),
            channel_prop_edge_type=self.channel_prop_edge_type.to(device),
            fermion_line_edge_index=self.fermion_line_edge_index.to(device),
            fermion_line_edge_attr=self.fermion_line_edge_attr.to(device),
            fermion_line_edge_type=self.fermion_line_edge_type.to(device),
            static_charge_features=self.static_charge_features.to(device),
            topology_features=self.topology_features.to(device),
            external_mass_summary=self.external_mass_summary.to(device),
            channel_id=self.channel_id.to(device),
            process_family_id=self.process_family_id.to(device),
            pattern_id=self.pattern_id.to(device),
            external_mask=self.external_mask.to(device),
            vertex_mask=self.vertex_mask.to(device),
            propagator_mask=self.propagator_mask.to(device),
            tgt_tokens=self.tgt_tokens.to(device),
            diagram_idx=self.diagram_idx.to(device),
        )


class QEDFeynmanDataset(Dataset):
    """Dataset backed by fixed-slot QED interaction tensors."""

    def __init__(
        self,
        corpus: LoadedQEDCorpus,
        indices: Sequence[int],
        tokenizer: AmplitudeTokenizer,
        max_seq_len: int = 1400,
    ):
        super().__init__()
        self.corpus = corpus
        self.indices = list(indices)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> QEDFeynmanExample:
        corpus_idx = self.indices[index]
        graph = self.corpus.graphs[corpus_idx]
        target = self.corpus.targets[corpus_idx]
        tokens = self.tokenizer.encode_tensor(
            target.sequence_target_text(),
            max_len=self.max_seq_len,
        )
        return QEDFeynmanExample(
            slot_features=graph.slot_features,
            slot_type_ids=graph.slot_type_ids,
            slot_position_ids=graph.slot_position_ids,
            fermion_line_ids=graph.fermion_line_ids,
            channel_prop_edge_index=graph.channel_prop_edge_index,
            channel_prop_edge_attr=graph.channel_prop_edge_attr,
            channel_prop_edge_type=graph.channel_prop_edge_type,
            fermion_line_edge_index=graph.fermion_line_edge_index,
            fermion_line_edge_attr=graph.fermion_line_edge_attr,
            fermion_line_edge_type=graph.fermion_line_edge_type,
            static_charge_features=graph.static_charge_features,
            topology_features=graph.topology_features,
            external_mass_summary=graph.external_mass_summary,
            channel_id=graph.channel_id,
            process_family_id=graph.process_family_id,
            pattern_id=graph.pattern_id,
            external_mask=graph.external_mask,
            vertex_mask=graph.vertex_mask,
            propagator_mask=graph.propagator_mask,
            tgt_tokens=tokens,
            diagram_idx=torch.tensor(corpus_idx, dtype=torch.long),
        )


def collate_graph_examples(batch: Sequence[QEDFeynmanExample]) -> QEDFeynmanBatch:
    return QEDFeynmanBatch(
        slot_features=torch.stack([item.slot_features for item in batch], dim=0),
        slot_type_ids=torch.stack([item.slot_type_ids for item in batch], dim=0),
        slot_position_ids=torch.stack([item.slot_position_ids for item in batch], dim=0),
        fermion_line_ids=torch.stack([item.fermion_line_ids for item in batch], dim=0),
        channel_prop_edge_index=torch.stack(
            [item.channel_prop_edge_index for item in batch],
            dim=0,
        ),
        channel_prop_edge_attr=torch.stack(
            [item.channel_prop_edge_attr for item in batch],
            dim=0,
        ),
        channel_prop_edge_type=torch.stack(
            [item.channel_prop_edge_type for item in batch],
            dim=0,
        ),
        fermion_line_edge_index=torch.stack(
            [item.fermion_line_edge_index for item in batch],
            dim=0,
        ),
        fermion_line_edge_attr=torch.stack(
            [item.fermion_line_edge_attr for item in batch],
            dim=0,
        ),
        fermion_line_edge_type=torch.stack(
            [item.fermion_line_edge_type for item in batch],
            dim=0,
        ),
        static_charge_features=torch.stack(
            [item.static_charge_features for item in batch],
            dim=0,
        ),
        topology_features=torch.stack([item.topology_features for item in batch], dim=0),
        external_mass_summary=torch.stack(
            [item.external_mass_summary for item in batch],
            dim=0,
        ),
        channel_id=torch.stack([item.channel_id for item in batch], dim=0),
        process_family_id=torch.stack(
            [item.process_family_id for item in batch],
            dim=0,
        ),
        pattern_id=torch.stack([item.pattern_id for item in batch], dim=0),
        external_mask=torch.stack([item.external_mask for item in batch], dim=0),
        vertex_mask=torch.stack([item.vertex_mask for item in batch], dim=0),
        propagator_mask=torch.stack([item.propagator_mask for item in batch], dim=0),
        tgt_tokens=torch.stack([item.tgt_tokens for item in batch], dim=0),
        diagram_idx=torch.stack([item.diagram_idx for item in batch], dim=0),
    )


@lru_cache(maxsize=None)
def _load_qed_corpus_cached(
    data_dir_str: str,
    target_variant: str,
) -> LoadedQEDCorpus:
    data_dir = Path(data_dir_str)
    interactions = parse_all_qed(data_dir)
    targets = [
        factorize_squared_amplitude(
            interaction.raw_squared,
            target_variant=target_variant,
        )
        for interaction in interactions
    ]
    features = [build_interaction_features(interaction) for interaction in interactions]
    relations = [build_interaction_relations(interaction) for interaction in interactions]
    graphs = [diagram_to_fixed_slot_graph(interaction) for interaction in interactions]
    return LoadedQEDCorpus(
        interactions=interactions,
        targets=targets,
        features=features,
        relations=relations,
        graphs=graphs,
    )


def load_qed_corpus(
    data_dir: str | Path,
    target_variant: str = "factorized",
) -> LoadedQEDCorpus:
    normalized = str(Path(data_dir).expanduser().resolve())
    return _load_qed_corpus_cached(normalized, target_variant)


def _resolve_splits(
    corpus: LoadedQEDCorpus,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    return random_split_indices(
        len(corpus.targets),
        seed=seed,
    )


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    augment_kinematics: bool = False,
    num_kinematic_samples: int = 1,
    max_seq_len: int = 1400,
    seed: int = 42,
    expression_mode: str = "postfix",
    target_variant: str = "factorized",
) -> tuple[DataLoader, DataLoader, DataLoader, AmplitudeTokenizer]:
    del augment_kinematics, num_kinematic_samples

    print(f"Loading QED data from {data_dir}...")
    corpus = load_qed_corpus(data_dir, target_variant=target_variant)
    print(f"Loaded {len(corpus.interactions)} QED interactions")

    train_indices, val_indices, test_indices = _resolve_splits(
        corpus=corpus,
        seed=seed,
    )

    tokenizer = AmplitudeTokenizer(expression_mode=expression_mode).build_vocab(
        [corpus.targets[index].sequence_target_text() for index in train_indices]
    )
    print(
        f"Target tokenizer vocab size: {tokenizer.vocab_size} "
        f"({tokenizer.expression_mode})"
    )

    train_ds = QEDFeynmanDataset(
        corpus=corpus,
        indices=train_indices,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )
    val_ds = QEDFeynmanDataset(
        corpus=corpus,
        indices=val_indices,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )
    test_ds = QEDFeynmanDataset(
        corpus=corpus,
        indices=test_indices,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    print(f"  Split 'train': {len(train_ds)}")
    print(f"  Split 'val':   {len(val_ds)}")
    print(f"  Split 'test':  {len(test_ds)}")

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
        collate_fn=collate_graph_examples,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graph_examples,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graph_examples,
    )
    return train_loader, val_loader, test_loader, tokenizer


__all__ = [
    "LoadedQEDCorpus",
    "QEDFeynmanBatch",
    "QEDFeynmanDataset",
    "QEDFeynmanExample",
    "collate_graph_examples",
    "create_dataloaders",
    "load_qed_corpus",
]
