from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LabeledExample:
    path: Path
    label: int


def load_labels(labels_csv: str | Path, audio_dir: str | Path) -> list[LabeledExample]:
    labels_csv = Path(labels_csv)
    audio_dir = Path(audio_dir)

    df = pd.read_csv(labels_csv)
    if "file" not in df.columns or "label" not in df.columns:
        raise ValueError("labels.csv must contain columns: file,label")

    examples: list[LabeledExample] = []
    for _, row in df.iterrows():
        rel = str(row["file"])
        p = (audio_dir / rel).resolve()
        label = int(row["label"])
        examples.append(LabeledExample(path=p, label=label))

    return examples

