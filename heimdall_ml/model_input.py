"""The classifier's model input, built exactly as the firmware builds it (MAX78002 classifier_input.c).

The board takes Ch0 at 16 kHz (the decimator of heimdall_ml.tonal), keeps the last 16,384 samples as int16,
turns them into int8 and writes them into the CNN accelerator in the order of the SDK kws20 example. Train on
the int8 values of `window_int8`: then the network sees the same numbers on the board as in training. The C
test (MAX78002/tests/test_classifier_input.c) and tests/test_ml_model_input.py check the same checksums.

    x16 = tonal.to_16k(audio_48k, 48000)        # or a 16 kHz window exported by scripts/ml_data.py
    q = window_int8(x16, scale="gain", gain=1.0)   # int8, oldest sample first, 16,384 samples
    tensor = kws20_tensor(q)                       # (128 channels, 128 steps): [c, t] = sample 128 t + c

In ai8x-training, give the network (tensor + 128) / 256 through ai8x.normalize: that returns tensor / 128
(tensor itself in 8-bit mode), as the kws20 dataset does with its uint8 audio.
"""
from __future__ import annotations

import numpy as np

KWS20_SAMPLES = 16384
KWS20_ROWS = 128
SCALES = ("gain", "peak")

# Shared with MAX78002/tests/test_classifier_input.c.
CHECKSUM_GAIN_1 = 76886476
CHECKSUM_PEAK = 77422613


def _round_half_away(values: np.ndarray) -> np.ndarray:
    """C lroundf: nearest integer, halves away from zero (numpy rounds halves to even)."""
    v = np.asarray(values, dtype=np.float64)
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def to_int16(samples: np.ndarray) -> np.ndarray:
    """Samples in [-1, 1) as the firmware window stores them: float32(x) * 32768, rounded, clipped."""
    scaled = np.asarray(samples, dtype=np.float32) * np.float32(32768.0)
    return np.clip(_round_half_away(scaled), -32768, 32767).astype(np.int16)


def to_int8(window: np.ndarray, scale: str = "gain", gain: float = 1.0) -> np.ndarray:
    """int16 window -> int8. gain: round(s * gain / 256) (gain 1 = the SDK kws20 demo with factor 1).
    peak: the largest magnitude becomes 127 (silence stays 0)."""
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}")
    s = np.asarray(window, dtype=np.int16)
    if scale == "peak":
        peak = int(np.max(np.abs(s.astype(np.int32)))) if s.size else 0
        factor = np.float32(127.0) / np.float32(peak) if peak > 0 else np.float32(0.0)
    else:
        factor = np.float32(gain) / np.float32(256.0)
    products = s.astype(np.float32) * factor
    return np.clip(_round_half_away(products), -128, 127).astype(np.int8)


def window_int8(samples_16k: np.ndarray, scale: str = "gain", gain: float = 1.0,
                length: int = KWS20_SAMPLES) -> np.ndarray:
    """The last `length` samples of a 16 kHz signal as the board's int8 model input, oldest first."""
    x = np.asarray(samples_16k)
    if x.ndim != 1:
        raise ValueError("give one channel (a 1-D array)")
    if len(x) < length:
        raise ValueError(f"need at least {length} samples ({length / 16000:.3f} s at 16 kHz), got {len(x)}")
    return to_int8(to_int16(x[len(x) - length:]), scale, gain)


def kws20_offsets() -> np.ndarray:
    """Byte offset of each sample in the accelerator's input memory (16 groups of 1 KB)."""
    index = np.arange(KWS20_SAMPLES)
    row, column = index // KWS20_ROWS, index % KWS20_ROWS
    group = (column % 64) // 4
    word_row = (column >= 64).astype(int) + 2 * row
    return 1024 * group + 4 * word_row + column % 4


def kws20_layout(window: np.ndarray) -> np.ndarray:
    """The 16 KB the firmware copies into the accelerator for this int8 window."""
    q = np.asarray(window, dtype=np.int8)
    if q.shape != (KWS20_SAMPLES,):
        raise ValueError(f"need {KWS20_SAMPLES} int8 samples")
    out = np.zeros(KWS20_SAMPLES, dtype=np.uint8)
    out[kws20_offsets()] = q.view(np.uint8)
    return out


def kws20_tensor(window: np.ndarray) -> np.ndarray:
    """(128 channels, 128 steps) int8, the network input: [c, t] = sample 128 t + c.

    ai8x-training's kws20 dataset folds audio the same way (reshape to (-1, 128), then transpose), and
    kws20_layout puts sample 128 t + c in input channel c at step t of the accelerator."""
    q = np.asarray(window, dtype=np.int8)
    if q.shape != (KWS20_SAMPLES,):
        raise ValueError(f"need {KWS20_SAMPLES} int8 samples")
    return np.ascontiguousarray(q.reshape(KWS20_SAMPLES // KWS20_ROWS, KWS20_ROWS).T)


def softmax(outputs: np.ndarray, fraction_bits: int = 14) -> tuple[int, np.ndarray]:
    """Top class and probabilities of the network's 32-bit outputs (as classifier_softmax)."""
    o = np.asarray(outputs, dtype=np.int64)
    top = int(np.argmax(o))
    e = np.exp((o - o[top]).astype(np.float64) / float(1 << fraction_bits))
    return top, e / e.sum()


def checksum(data: np.ndarray) -> int:
    """sum of (byte + 1) * (index + 1) modulo 2^32, as classifier_checksum."""
    b = np.asarray(data, dtype=np.uint8).astype(np.uint64)
    return int(np.sum((b + 1) * (np.arange(len(b), dtype=np.uint64) + 1)) % (1 << 32))


def parity_signal(count: int) -> np.ndarray:
    """The signal both parity tests use: exact in float32, so C and Python start from the same int16."""
    i = np.arange(count, dtype=np.int64)
    return ((i * 7919) % 65536 - 32768).astype(np.float32) / np.float32(32768.0)
