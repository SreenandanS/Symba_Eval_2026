"""
PyTorch dataset for factorized QCD tree-level 2→2 targets.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .factorization import FactorizedTarget, factorize_squared_amplitude
from .feynman_graph import diagram_to_homogeneous_graph
from .parser import FeynmanDiagram, parse_all_qcd
from .splits import random_split_indices
from .tokenizer import AmplitudeTokenizer


class QCDFeynmanDataset(Dataset):
    """Dataset backed by shared parsed diagrams and factorized targets."""

    def __init__(
        self,
        diagrams: Sequence[FeynmanDiagram],
        targets: Sequence[FactorizedTarget],
        indices: Sequence[int],
        tokenizer: AmplitudeTokenizer,
        augment_kinematics: bool = False,
        num_kinematic_samples: int = 1,
        max_seq_len: int = 1400,
    ):
        super().__init__()
        self.diagrams = list(diagrams)
        self.targets = list(targets)
        self.indices = list(indices)
        self.tokenizer = tokenizer
        self.augment_kinematics = augment_kinematics
        self.num_kinematic_samples = num_kinematic_samples
        self.max_seq_len = max_seq_len

        if augment_kinematics:
            self._expanded_indices = [
                idx
                for idx in self.indices
                for _ in range(num_kinematic_samples)
            ]
        else:
            self._expanded_indices = list(self.indices)

    def __len__(self) -> int:
        return len(self._expanded_indices)

    def __getitem__(self, idx: int) -> Data:
        diag_idx = self._expanded_indices[idx]
        diagram = self.diagrams[diag_idx]
        target = self.targets[diag_idx]

        mandelstam_values = (
            self._sample_mandelstam(diagram)
            if self.augment_kinematics
            else None
        )
        data = diagram_to_homogeneous_graph(diagram, mandelstam_values)

        data.tgt_tokens = self.tokenizer.encode_tensor(
            target.sequence_target_text(),
            max_len=self.max_seq_len,
        ).unsqueeze(0)
        data.diagram_idx = diag_idx
        return data

    @staticmethod
    def _sample_mandelstam(diagram: FeynmanDiagram) -> Dict[str, float]:
        import math

        masses = []
        for particle in diagram.externals:
            from .feynman_graph import MASS_VALUES

            masses.append(MASS_VALUES.get(particle.flavor, 0.0))
        while len(masses) < 4:
            masses.append(0.0)

        m1, m2, m3, m4 = masses[:4]
        s_threshold = (m1 + m2) ** 2 + 1.0
        s = random.uniform(s_threshold, 1000.0)
        cos_theta = random.uniform(-0.99, 0.99)

        sqrt_s = math.sqrt(s)
        e1 = (s + m1**2 - m2**2) / (2 * sqrt_s)
        p_i = math.sqrt(max(e1**2 - m1**2, 1e-10))
        e3 = (s + m3**2 - m4**2) / (2 * sqrt_s)
        e4 = (s + m4**2 - m3**2) / (2 * sqrt_s)
        p_f = math.sqrt(max(e3**2 - m3**2, 1e-10))

        t = m1**2 + m3**2 - 2 * (e1 * e3 - p_i * p_f * cos_theta)
        u = m1**2 + m4**2 - 2 * (e1 * e4 + p_i * p_f * cos_theta)
        return {
            "s_12": s,
            "s_13": t,
            "s_14": u,
            "s_23": u,
            "s_24": t,
            "s_34": s,
        }


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
) -> Tuple[DataLoader, DataLoader, DataLoader, AmplitudeTokenizer]:
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

    train_indices, val_indices, test_indices = random_split_indices(
        len(diagrams),
        seed=seed,
    )

    tokenizer = AmplitudeTokenizer(expression_mode=expression_mode).build_vocab(
        [factorized_targets[idx].sequence_target_text() for idx in train_indices]
    )
    print(
        f"Tokenizer numerator vocab size: {tokenizer.vocab_size} "
        f"({tokenizer.expression_mode})"
    )

    train_ds = QCDFeynmanDataset(
        diagrams=diagrams,
        targets=factorized_targets,
        indices=train_indices,
        tokenizer=tokenizer,
        augment_kinematics=augment_kinematics,
        num_kinematic_samples=num_kinematic_samples,
        max_seq_len=max_seq_len,
    )
    val_ds = QCDFeynmanDataset(
        diagrams=diagrams,
        targets=factorized_targets,
        indices=val_indices,
        tokenizer=tokenizer,
        augment_kinematics=augment_kinematics,
        num_kinematic_samples=num_kinematic_samples,
        max_seq_len=max_seq_len,
    )
    test_ds = QCDFeynmanDataset(
        diagrams=diagrams,
        targets=factorized_targets,
        indices=test_indices,
        tokenizer=tokenizer,
        augment_kinematics=augment_kinematics,
        num_kinematic_samples=num_kinematic_samples,
        max_seq_len=max_seq_len,
    )

    print(
        f"  Split 'train': {len(train_indices)} diagrams × "
        f"{num_kinematic_samples if augment_kinematics else 1} samples = "
        f"{len(train_ds)} total"
    )
    print(
        f"  Split 'val':   {len(val_indices)} diagrams × "
        f"{num_kinematic_samples if augment_kinematics else 1} samples = "
        f"{len(val_ds)} total"
    )
    print(
        f"  Split 'test':  {len(test_indices)} diagrams × "
        f"{num_kinematic_samples if augment_kinematics else 1} samples = "
        f"{len(test_ds)} total"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, tokenizer
