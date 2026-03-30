"""Fixed-seed random split helpers for QED."""

from __future__ import annotations

import random

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def random_split_indices(
    num_items: int,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)

    n_train = int(num_items * TRAIN_RATIO)
    n_val = int(num_items * VAL_RATIO)

    return (
        sorted(indices[:n_train]),
        sorted(indices[n_train : n_train + n_val]),
        sorted(indices[n_train + n_val :]),
    )
