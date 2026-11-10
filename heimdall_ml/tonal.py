"""Reference port of the firmware tonal detector (MAX78002/audio_features.c).

The firmware runs it in float32; this port uses float64, so scores can differ by a few hundredths of a dB.
tests/test_ml_tonal.py checks it against the firmware host test (MAX78002/tests/test_audio_features.c).
"""
from dataclasses import dataclass

import numpy as np

CAPTURE_RATE_HZ = 48000
RATE_HZ = 16000
DECIMATION = 3
DECIMATOR_TAPS = 47
DECIMATOR_CUTOFF_HZ = 6500.0
FRAME = 1024
HOP = 512
BINS = FRAME // 2 + 1
BIN_HZ = RATE_HZ / FRAME


def decimator_taps() -> np.ndarray:
    """Hamming-windowed sinc low-pass at 6.5 kHz, unity gain at DC (audio_features_init)."""
    n = np.arange(DECIMATOR_TAPS)
    x = n - (DECIMATOR_TAPS - 1) / 2.0
    fc = DECIMATOR_CUTOFF_HZ / CAPTURE_RATE_HZ
    sinc = np.where(x == 0, 2.0 * fc, np.sin(2.0 * np.pi * fc * x) / (np.pi * np.where(x == 0, 1.0, x)))
    taps = sinc * (0.54 - 0.46 * np.cos(2.0 * np.pi * n / (DECIMATOR_TAPS - 1)))
    return taps / taps.sum()


TAPS = decimator_taps()
WINDOW = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(FRAME) / FRAME)
POWER_SCALE = 1.0 / (FRAME * np.sum(WINDOW ** 2))


def decimate_48k(samples: np.ndarray) -> np.ndarray:
    """48 kHz to 16 kHz as the firmware does it: filter from rest, keep inputs 2, 5, 8, ..."""
    x = np.asarray(samples, dtype=np.float64)
    return np.convolve(x, TAPS)[: len(x)][DECIMATION - 1::DECIMATION]


def frame_power(frame: np.ndarray) -> np.ndarray:
    """One-sided power per bin of a 1,024-sample frame, scaled so the bins add up to the mean square."""
    power = np.abs(np.fft.rfft(np.asarray(frame, dtype=np.float64) * WINDOW)) ** 2 * POWER_SCALE
    power[1:-1] *= 2.0
    return power


@dataclass(frozen=True)
class ToneConfig:
    """Search settings. The defaults are the firmware's (audio_features.h)."""
    min_hz: float = 80.0
    max_hz: float = 800.0
    step_hz: float = 2.0
    harmonics: int = 8
    top_hz: float = 4000.0
    present_db: float = 6.0
    floor_half_width: int = 8
    smoothing: float = 0.25
    submultiple_share: float = 0.8

    def candidates(self) -> np.ndarray:
        count = int(round((self.max_hz - self.min_hz) / self.step_hz)) + 1
        return self.min_hz + self.step_hz * np.arange(count)


@dataclass(frozen=True)
class Tone:
    f0_hz: float
    score_db: float
    harmonics: int
    level_db: float
    frames: int


@dataclass(frozen=True)
class ToneAnalysis:
    """Everything that audio_features_tone() computes, for plots and checks."""
    tone: Tone
    level_db: np.ndarray
    floor_db: np.ndarray
    candidates_hz: np.ndarray
    scores_db: np.ndarray


def _floor(level: np.ndarray, half: int) -> np.ndarray:
    """Running median over +-half bins; at the edges the upper middle of the shorter list (sort_median)."""
    floor = np.empty_like(level)
    width = 2 * half + 1
    if len(level) >= width:
        floor[half:len(level) - half] = np.sort(np.lib.stride_tricks.sliding_window_view(level, width),
                                                axis=1)[:, half]
    for k in list(range(min(half, len(level)))) + list(range(max(half, len(level) - half), len(level))):
        values = np.sort(level[max(0, k - half):k + half + 1])
        floor[k] = values[len(values) // 2]
    return floor


def _prominence(above: np.ndarray, hz: np.ndarray) -> np.ndarray:
    """The larger of the two bins around each frequency, above the floor."""
    k = np.floor(hz / BIN_HZ).astype(int)

    def at(j):
        return np.where((j >= 1) & (j < BINS), above[np.clip(j, 0, BINS - 1)], -1000.0)

    return np.maximum(at(k), at(k + 1))


def analyse(smoothed: np.ndarray, frames: int = 0, config: ToneConfig = ToneConfig()) -> ToneAnalysis:
    """audio_features_tone() on a smoothed power spectrum."""
    level = 10.0 * np.log10(np.asarray(smoothed, dtype=np.float64) + 1e-14)
    floor = _floor(level, config.floor_half_width)
    above = level - floor
    candidates = config.candidates()
    orders = np.arange(1, config.harmonics + 1)
    hz = candidates[:, None] * orders[None, :]
    used = hz <= config.top_hz
    peaks = np.clip(_prominence(above, hz), 0.0, 30.0)
    counts = used.sum(axis=1)
    scores = np.where(counts > 0, np.where(used, peaks, 0.0).sum(axis=1) / np.maximum(counts, 1), 0.0)
    chosen = int(np.argmax(scores))
    best = scores[chosen]
    best_hz = candidates[chosen]
    # A multiple of f0 explains the peaks as well as f0; prefer the lowest sub-multiple that scores as well.
    for divisor in (4, 3, 2):
        target = best_hz / divisor
        near = np.flatnonzero((candidates >= target * 0.97) & (candidates <= target * 1.03))
        if near.size:
            local = int(near[np.argmax(scores[near])])
            if scores[local] >= config.submultiple_share * best:
                chosen = local
                break
    f0 = float(candidates[chosen])
    orders_used = orders[f0 * orders <= config.top_hz]
    harmonics = int(np.sum(_prominence(above, f0 * orders_used) >= config.present_db))
    freqs = np.arange(BINS) * BIN_HZ
    band = (freqs >= 100.0) & (freqs <= config.top_hz)
    level_db = 10.0 * np.log10(np.sum(np.asarray(smoothed)[band]) + 1e-14)
    tone = Tone(f0, float(scores[chosen]), harmonics, float(level_db), frames)
    return ToneAnalysis(tone, level, floor, candidates, scores)


class TonalDetector:
    """One channel of audio_features.c: streaming decimation, 64 ms frames every 32 ms, smoothing, TONE."""

    def __init__(self, config: ToneConfig = ToneConfig()):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._history = np.zeros(DECIMATOR_TAPS - 1)
        self._inputs = 0
        self._ring = np.zeros(0)
        self._decimated = 0
        self.smoothed = None
        self.frames = 0

    def feed_48k(self, samples: np.ndarray) -> int:
        """Feed 48 kHz samples (full scale 1.0). Returns the number of new frames."""
        x = np.asarray(samples, dtype=np.float64)
        if x.size == 0:
            return 0
        filtered = np.convolve(np.concatenate([self._history, x]), TAPS, mode="valid")
        indices = self._inputs + np.arange(x.size)
        outputs = filtered[indices % DECIMATION == DECIMATION - 1]
        self._history = np.concatenate([self._history, x])[-(DECIMATOR_TAPS - 1):]
        self._inputs += x.size
        return self.feed_16k(outputs)

    def feed_16k(self, samples: np.ndarray) -> int:
        """Feed 16 kHz samples. A frame closes after 1,024 samples, then every 512 (as the firmware)."""
        before = self.frames
        self._ring = np.concatenate([self._ring, np.asarray(samples, dtype=np.float64)])
        start = self._decimated - (len(self._ring) - len(samples))
        self._decimated += len(samples)
        end = FRAME if self.frames == 0 else FRAME + HOP * self.frames
        while end <= self._decimated:
            frame = self._ring[end - FRAME - start:end - start]
            power = frame_power(frame)
            self.smoothed = power if self.smoothed is None else \
                self.smoothed + self.config.smoothing * (power - self.smoothed)
            self.frames += 1
            end += HOP
        self._ring = self._ring[max(0, end - FRAME - start):]
        return self.frames - before

    def analysis(self) -> ToneAnalysis | None:
        return None if self.smoothed is None else analyse(self.smoothed, self.frames, self.config)

    def tone(self) -> Tone | None:
        result = self.analysis()
        return None if result is None else result.tone


def tone_reports(samples: np.ndarray, rate: int, every: int = 3, skip: int = 4,
                 config: ToneConfig = ToneConfig()) -> list[Tone]:
    """A TONE report every `every` frames (3 frames = 96 ms, as `tone` = 100), after `skip` frames."""
    detector = TonalDetector(config)
    reports = []
    stream = to_16k(samples, rate)
    for start in range(0, len(stream), HOP):
        if detector.feed_16k(stream[start:start + HOP]) and detector.frames > skip \
                and (detector.frames - 1 - skip) % every == 0:
            reports.append(detector.tone())
    return reports


def to_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    """16 kHz audio: the firmware decimator for 48 kHz, a polyphase resampler for other rates."""
    x = np.asarray(samples, dtype=np.float64)
    if rate == CAPTURE_RATE_HZ:
        return decimate_48k(x)
    if rate == RATE_HZ:
        return x
    from math import gcd

    from scipy.signal import resample_poly
    common = gcd(RATE_HZ, rate)
    return resample_poly(x, RATE_HZ // common, rate // common)
