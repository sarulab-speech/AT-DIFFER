"""Render one AT-DIFFER audio clip from its source clips and audio structure.

The numerical steps (operation order and dtypes included) are exactly those
used to create the paper's dataset; changing any of them can break the
bit-identical reproduction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import soxr

TARGET_SAMPLE_RATE = 22050

# Silence trimming for simultaneous structures: a frame counts as silence when
# its RMS sits more than TRIM_TOP_DB below the clip's loudest frame.
TRIM_TOP_DB = 40.0
TRIM_FRAME_LENGTH = 1024
TRIM_HOP_LENGTH = 256
MAX_LEVEL_MATCH_GAIN = 10.0

# Number of source clips each audio structure takes.
REQUIRED_CLIP_COUNT = {"single": 1, "simultaneous": 2, "sequential": 2}


def render(
    audio_structure: str,
    structure_params: Mapping[str, Any],
    clips: Sequence[tuple[np.ndarray, int]],
) -> np.ndarray:
    """Render the audio for one metadata row.

    Args:
        audio_structure: ``single``, ``simultaneous``, or ``sequential``.
        structure_params: Parsed ``structure_params`` column
            (``{"gap_seconds": ...}`` for sequential, ``{}`` otherwise).
        clips: ``(audio, sample_rate)`` for each source clip, in the order of
            the ``source_clip_ids`` column.

    Returns:
        Mono float32 audio at ``TARGET_SAMPLE_RATE``.
    """
    if audio_structure not in REQUIRED_CLIP_COUNT:
        raise ValueError(f"Unknown audio structure: {audio_structure!r}")
    if len(clips) != REQUIRED_CLIP_COUNT[audio_structure]:
        raise ValueError(
            f"Expected {REQUIRED_CLIP_COUNT[audio_structure]} clips for "
            f"{audio_structure}, got {len(clips)}."
        )

    resampled = [resample(audio, sample_rate) for audio, sample_rate in clips]

    if audio_structure == "single":
        return render_single(resampled)
    if audio_structure == "sequential":
        return render_sequential(resampled, **structure_params)
    return render_simultaneous(resampled)


def resample(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample to ``TARGET_SAMPLE_RATE`` with soxr's high-quality setting.

    The paper's dataset was created with ``librosa.resample`` (default
    ``soxr_hq``), which also pads or trims the output to
    ``ceil(len(audio) * ratio)`` samples. For 44.1 kHz input, which covers
    every FSD50K clip used in AT-DIFFER, soxr's output already has that
    length, so the result is identical. For other rates the length can differ
    from librosa's by one sample.
    """
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio
    return soxr.resample(
        audio, sample_rate, TARGET_SAMPLE_RATE, quality="soxr_hq"
    ).astype(np.float32)


def render_single(clips: Sequence[np.ndarray]) -> np.ndarray:
    """Use the clip as it is."""
    return clips[0]


def render_sequential(clips: Sequence[np.ndarray], gap_seconds: float) -> np.ndarray:
    """Play the first clip, then ``gap_seconds`` of silence, then the second."""
    silence = _make_silence(gap_seconds)
    return np.concatenate([clips[0], silence, clips[1]])


def render_simultaneous(clips: Sequence[np.ndarray]) -> np.ndarray:
    """Mix two clips so that both are audible over the whole output.

    Leading and trailing silence is cut from each clip, both are truncated to
    the shorter one's length, the second is scaled to the first one's RMS, and
    the sum is peak-normalized.
    """
    a = _trim_silence(clips[0])
    b = _trim_silence(clips[1])
    length = min(len(a), len(b))
    a, b = a[:length], b[:length]
    return _normalize(a + _match_level(a, b))


def _trim_silence(audio: np.ndarray) -> np.ndarray:
    """Drop the leading and trailing frames that sit more than ``TRIM_TOP_DB``
    below the loudest frame. Returns ``audio`` unchanged when it is digital
    silence or too short to frame."""
    if len(audio) < TRIM_FRAME_LENGTH:
        return audio
    frames = np.lib.stride_tricks.sliding_window_view(audio, TRIM_FRAME_LENGTH)[
        ::TRIM_HOP_LENGTH
    ]
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    peak = rms.max()
    if peak == 0.0:
        return audio
    loud = np.flatnonzero(rms >= peak * 10 ** (-TRIM_TOP_DB / 20))
    start = int(loud[0]) * TRIM_HOP_LENGTH
    end = int(loud[-1]) * TRIM_HOP_LENGTH + TRIM_FRAME_LENGTH
    return audio[start:end]


def _match_level(reference: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """Scale ``audio`` to ``reference``'s RMS. The gain is limited to
    [1 / MAX_LEVEL_MATCH_GAIN, MAX_LEVEL_MATCH_GAIN] so that a near-silent clip
    is not blown up into noise."""
    if audio.size == 0:
        return audio
    ref_rms = float(np.sqrt(np.mean(np.square(reference, dtype=np.float64))))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if ref_rms == 0.0 or rms == 0.0:
        return audio
    gain = np.clip(ref_rms / rms, 1 / MAX_LEVEL_MATCH_GAIN, MAX_LEVEL_MATCH_GAIN)
    return (audio * gain).astype(np.float32)


def _make_silence(duration_seconds: float) -> np.ndarray:
    n_samples = int(duration_seconds * TARGET_SAMPLE_RATE)
    return np.zeros(n_samples, dtype=np.float32)


def _normalize(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio
    peak = np.abs(audio).max()
    if peak == 0.0:
        return audio
    return (audio / peak).astype(np.float32)
