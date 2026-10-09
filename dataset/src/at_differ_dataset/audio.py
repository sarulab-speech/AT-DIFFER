"""Reading FSD50K clips and writing rendered WAV files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


def load_clip(path: Path) -> tuple[np.ndarray, int]:
    """Read a clip as mono float32 in [-1, 1] and return it with its sample rate."""
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    return audio, sample_rate


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write a 32-bit float WAV file.

    The audio is first written to a temporary file in the same directory and
    then renamed, so a file at ``path`` is always complete even if the process
    is interrupted.
    """
    tmp_path = path.with_name(f".{path.name}.tmp")
    sf.write(str(tmp_path), audio, sample_rate, format="WAV", subtype="FLOAT")
    os.replace(tmp_path, path)
