"""Where in a known clip each part of a recording is (for a clip that loops through the speaker).

With the clip time of every window, train and test can use different parts of the clip
(dataset.split_by_clip_time). The method is GCC-PHAT, a whitened cross-correlation that tolerates the
speaker, the room and the beam, applied to several excerpts that must agree.
"""
from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt


@dataclass(frozen=True)
class ClipAlignment:
    offset_s: float          # clip time at the first sample of the recording
    clip_length_s: float
    confidence: float        # weakest excerpt: correlation peak over the next-best peak
    spread_s: float          # largest disagreement of an excerpt with the result
    aligned: bool

    def clip_time(self, recording_time_s: float) -> float:
        return (self.offset_s + recording_time_s) % self.clip_length_s


def _prepare(samples: np.ndarray, rate: int, analysis_rate: int, band: tuple[float, float]) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    if rate != analysis_rate:
        common = gcd(rate, analysis_rate)
        x = resample_poly(x, analysis_rate // common, rate // common)
    sos = butter(4, band, btype="bandpass", fs=analysis_rate, output="sos")
    return sosfilt(sos, x)


def align_to_clip(recording: np.ndarray, rate: int, clip: np.ndarray, clip_rate: int, excerpt_s: float = 4.0,
                  excerpts: int = 3, band: tuple[float, float] = (200.0, 3800.0), analysis_rate: int = 8000,
                  guard_s: float = 0.05, min_confidence: float = 1.5, max_spread_s: float = 0.02) -> ClipAlignment:
    """Clip time at the start of the recording, from `excerpts` evenly spaced excerpts of `excerpt_s`."""
    rec = _prepare(recording, rate, analysis_rate, band)
    ref = _prepare(clip, clip_rate, analysis_rate, band)
    period = len(ref)
    size = min(int(excerpt_s * analysis_rate), period, len(rec))
    guard = int(guard_s * analysis_rate)
    spectrum = np.fft.rfft(ref)
    starts = np.linspace(0, len(rec) - size, max(1, excerpts)).astype(int)
    offsets, ratios = [], []
    for start in starts:
        cross = spectrum * np.conj(np.fft.rfft(rec[start:start + size], n=period))
        correlation = np.fft.irfft(cross / (np.abs(cross) + 1e-12), n=period)
        peak = int(np.argmax(correlation))
        distance = np.abs((np.arange(period) - peak + period // 2) % period - period // 2)
        others = correlation[distance > guard]
        ratios.append(correlation[peak] / max(float(others.max()), 1e-12) if others.size else np.inf)
        offsets.append((peak - start) % period)
    offsets = np.asarray(offsets, dtype=np.float64)
    reference = offsets[0]
    unwrapped = reference + (offsets - reference + period / 2) % period - period / 2
    offset = float(np.median(unwrapped)) % period
    spread = float(np.max(np.abs(unwrapped - np.median(unwrapped)))) / analysis_rate
    confidence = float(min(ratios))
    return ClipAlignment(offset / analysis_rate, period / analysis_rate, confidence, spread,
                         confidence >= min_confidence and spread <= max_spread_s)
