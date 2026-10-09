import math

import numpy as np
import pytest

from at_differ_dataset.render import (
    TARGET_SAMPLE_RATE,
    TRIM_FRAME_LENGTH,
    render,
)


def tone(n_samples: int, freq: float = 440.0, amplitude: float = 0.5,
         sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    t = np.arange(n_samples) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def padded(sound: np.ndarray, before: int, after: int) -> np.ndarray:
    return np.concatenate(
        [np.zeros(before, np.float32), sound, np.zeros(after, np.float32)]
    )


def test_single_returns_clip_unchanged():
    clip = tone(10_000)
    out = render("single", {}, [(clip, TARGET_SAMPLE_RATE)])
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, clip)


def test_sequential_concatenates_with_silent_gap():
    a, b = tone(8_000, 440.0), tone(12_000, 880.0)
    gap = int(0.5 * TARGET_SAMPLE_RATE)
    out = render(
        "sequential", {"gap_seconds": 0.5},
        [(a, TARGET_SAMPLE_RATE), (b, TARGET_SAMPLE_RATE)],
    )
    assert out.dtype == np.float32
    assert len(out) == len(a) + gap + len(b)
    np.testing.assert_array_equal(out[: len(a)], a)
    assert not out[len(a) : len(a) + gap].any()
    np.testing.assert_array_equal(out[len(a) + gap :], b)


def test_simultaneous_trims_silence_truncates_and_normalizes():
    # a: 20,000 samples of sound inside 10,000 samples of silence on each side.
    # b: 30,000 samples of sound inside 2,000 samples of silence on each side.
    a = padded(tone(20_000, 440.0, 0.2), 10_000, 10_000)
    b = padded(tone(30_000, 660.0, 0.8), 2_000, 2_000)
    out = render(
        "simultaneous", {},
        [(a, TARGET_SAMPLE_RATE), (b, TARGET_SAMPLE_RATE)],
    )
    assert out.dtype == np.float32
    # The silence around the shorter sound is removed (up to framing), and the
    # mix is truncated to that sound's length.
    assert 20_000 <= len(out) <= 20_000 + 2 * TRIM_FRAME_LENGTH
    # The output starts with sound rather than with the leading silence.
    assert np.abs(out[:TRIM_FRAME_LENGTH]).max() > 0.1
    assert np.abs(out).max() == pytest.approx(1.0, abs=1e-6)


def test_input_not_at_target_rate_is_resampled():
    sample_rate = 44_100
    clip = tone(44_101, 440.0, sample_rate=sample_rate)
    out = render("single", {}, [(clip, sample_rate)])
    assert out.dtype == np.float32
    assert len(out) == math.ceil(len(clip) * TARGET_SAMPLE_RATE / sample_rate)
    # The 440 Hz tone is preserved after resampling.
    spectrum = np.abs(np.fft.rfft(out))
    peak_freq = np.argmax(spectrum) * TARGET_SAMPLE_RATE / len(out)
    assert peak_freq == pytest.approx(440.0, abs=2.0)


@pytest.mark.parametrize(
    ("audio_structure", "n_clips"),
    [("single", 2), ("simultaneous", 1), ("sequential", 1)],
)
def test_wrong_number_of_clips_is_rejected(audio_structure, n_clips):
    clips = [(tone(4_000), TARGET_SAMPLE_RATE)] * n_clips
    params = {"gap_seconds": 0.5} if audio_structure == "sequential" else {}
    with pytest.raises(ValueError):
        render(audio_structure, params, clips)
