from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio_mono(path: str | Path, *, sr: int = 22050, duration: float | None = None):
    y, _sr = librosa.load(str(path), sr=sr, mono=True, duration=duration)
    return y, _sr


def logmel_stats(
    y: np.ndarray,
    sr: int,
    *,
    n_mels: int = 64,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    mels = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft, power=2.0
    )
    logmels = librosa.power_to_db(mels, ref=np.max)
    mean = logmels.mean(axis=1)
    std = logmels.std(axis=1)
    return np.concatenate([mean, std], axis=0).astype(np.float32)


def featurize_file(path: str | Path, *, sr: int = 22050, duration: float | None = None) -> np.ndarray:
    y, sr = load_audio_mono(path, sr=sr, duration=duration)
    return logmel_stats(y, sr)

