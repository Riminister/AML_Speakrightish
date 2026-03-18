from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from aml_speakrightish.features.audio_features import featurize_file
from aml_speakrightish.models.registry import create_model
from aml_speakrightish.models.sklearn_model import SklearnBinaryClassifier
from aml_speakrightish.models.tf_model import TFBinaryClassifier


@dataclass(frozen=True)
class LabeledExample:
    path: Path
    label: int


def load_examples(labels_csv: str | Path, audio_dir: str | Path) -> list[LabeledExample]:
    labels_csv = Path(labels_csv)
    audio_dir = Path(audio_dir)

    df = pd.read_csv(labels_csv)
    if "file" not in df.columns or "label" not in df.columns:
        raise ValueError("labels.csv must contain columns: file,label")

    out: list[LabeledExample] = []
    for _, row in df.iterrows():
        p = (audio_dir / str(row["file"])).resolve()
        out.append(LabeledExample(path=p, label=int(row["label"])))
    return out


def featurize_examples(
    examples: list[LabeledExample],
    *,
    sr: int = 22050,
    duration: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray([e.label for e in examples], dtype=np.int64)
    X = np.stack([featurize_file(e.path, sr=sr, duration=duration) for e in examples], axis=0)
    return X, y


def stratified_split(y: np.ndarray, *, test_size: float = 0.2, seed: int = 42):
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    return train_idx, test_idx


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_name: str,
):
    bundle = create_model(model_name, input_dim=int(X_train.shape[1]))
    model = bundle.model
    model.fit(X_train, y_train)
    return bundle.name, model, bundle.save_path


def save_model(model: object, path: str | Path):
    model.save(str(path))  # type: ignore[attr-defined]


def load_model(path: str | Path):
    p = Path(path)
    if p.suffix == ".joblib":
        return SklearnBinaryClassifier.load(str(p))
    return TFBinaryClassifier.load(str(p))

