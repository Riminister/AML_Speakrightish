from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split(
    y: Sequence[int],
    *,
    test_size: float = 0.2,
    seed: int = 42,
):
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=np.asarray(list(y))
    )
    return train_idx, test_idx

