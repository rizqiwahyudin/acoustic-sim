import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heimdall_ml import tonal  # noqa: E402

RATE = 48000
BLOCK = 256


class FirmwareTestSignal:
    """The input of MAX78002/tests/test_audio_features.c: the same LCG noise, sines and int16 rounding."""

    def __init__(self):
        self.lcg = 12345

    def _noise(self, count):
        values = np.empty(count)
        state = self.lcg
        for i in range(count):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            values[i] = ((state >> 8) / 16777216.0) * 2.0 - 1.0
        self.lcg = state
        return values

    def run(self, f0, right_hz, right_amplitude, seconds):
        total = int(seconds * RATE)
        count = -(-total // BLOCK) * BLOCK  # the C loop finishes the last block
        t = np.arange(count) / RATE
        noise = self._noise(2 * count)
        left = sum(0.08 / h * np.sin(2.0 * np.pi * f0 * h * t + h) for h in range(1, 7)) + 0.01 * noise[0::2]
        right = right_amplitude * np.sin(2.0 * np.pi * right_hz * t) + 0.01 * noise[1::2]

        def pcm(x):
            return np.sign(x) * np.floor(np.abs(32767.0 * x) + 0.5) / 32768.0

        return pcm(left), pcm(right)


def test_matches_the_firmware_host_test():
    signal = FirmwareTestSignal()
    signal.run(150.0, 1000.0, 0.0, 0.1)  # fed before the detector is enabled: only advances the noise
    left_detector, right_detector = tonal.TonalDetector(), tonal.TonalDetector()
    left, right = signal.run(150.0, 1000.0, 0.0, 1.0)
    left_detector.feed_48k(left)
    right_detector.feed_48k(right)
    # Printed by the C test: "left: f0 152.0 Hz, score 22.9 dB, 6 harmonics, -23.2 dBFS"
    tone = left_detector.tone()
    assert tone.f0_hz == 152.0
    assert tone.harmonics == 6
    assert tone.score_db == pytest.approx(22.9, abs=0.06)
    assert tone.level_db == pytest.approx(-23.2, abs=0.06)
    # "right (noise): f0 204.0 Hz, score 1.9 dB, 0 harmonics"; the C test also checks 27-32 frames
    tone = right_detector.tone()
    assert tone.f0_hz == 204.0
    assert tone.harmonics == 0
    assert tone.score_db == pytest.approx(1.9, abs=0.06)
    assert 27 <= right_detector.frames <= 32
    # "right (1 kHz, 0.5): -9.02 dBFS" and "right (13 kHz, 0.5): -52.77 dBFS", each after a reset
    for right_hz, level in ((1000.0, -9.02), (13000.0, -52.77)):
        detector = tonal.TonalDetector()
        detector.feed_48k(signal.run(150.0, right_hz, 0.5, 0.5)[1])
        assert detector.tone().level_db == pytest.approx(level, abs=0.02)


def test_streaming_matches_one_block():
    rng = np.random.default_rng(1)
    x = 0.1 * rng.standard_normal(RATE)
    whole, pieces = tonal.TonalDetector(), tonal.TonalDetector()
    whole.feed_48k(x)
    start = 0
    while start < len(x):
        size = int(rng.integers(1, 700))
        pieces.feed_48k(x[start:start + size])
        start += size
    assert pieces.frames == whole.frames == 30
    assert np.allclose(pieces.smoothed, whole.smoothed)


def test_decimator_is_the_streaming_decimator():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(4800)
    detector = tonal.TonalDetector()
    detector.feed_48k(x)
    reference = tonal.TonalDetector()
    reference.feed_16k(tonal.decimate_48k(x))
    assert reference.frames == detector.frames
    assert np.allclose(reference.smoothed, detector.smoothed)


def test_a_narrower_search_finds_a_tuned_pitch():
    t = np.arange(RATE) / RATE
    x = sum(0.05 / h * np.sin(2.0 * np.pi * 190.0 * h * t) for h in range(1, 9))
    x = x + 0.005 * np.random.default_rng(3).standard_normal(len(t))
    config = tonal.ToneConfig(min_hz=150.0, max_hz=260.0)
    reports = tonal.tone_reports(x, RATE, config=config)
    assert len(reports) == 9
    assert abs(reports[-1].f0_hz - 190.0) <= 4.0
    assert reports[-1].score_db > 10.0
