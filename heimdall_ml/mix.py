"""Mixing for training data: the target sound over a background at a chosen signal-to-noise ratio."""
import numpy as np


def power_db(samples: np.ndarray) -> float:
    """Mean-square level in dB re full scale."""
    return float(10.0 * np.log10(np.mean(np.square(np.asarray(samples, dtype=np.float64))) + 1e-20))


def with_gain(samples: np.ndarray, gain_db: float) -> np.ndarray:
    return np.asarray(samples, dtype=np.float64) * 10.0 ** (gain_db / 20.0)


def background_cut(background: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """`length` samples of the background from a random start; a short background repeats."""
    background = np.asarray(background, dtype=np.float64)
    if len(background) == 0:
        raise ValueError("empty background")
    if len(background) < length:
        background = np.tile(background, -(-length // len(background)) + 1)
    start = int(rng.integers(0, len(background) - length + 1))
    return background[start:start + length]


def mix_at_snr(target: np.ndarray, background: np.ndarray, snr_db: float,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """target + background, with the background scaled so that the target is `snr_db` above it."""
    rng = rng or np.random.default_rng()
    target = np.asarray(target, dtype=np.float64)
    noise = background_cut(background, len(target), rng)
    return target + with_gain(noise, power_db(target) - snr_db - power_db(noise))


def mixed_windows(targets: list, backgrounds: list, snrs_db: list[float], seed: int = 0) -> list:
    """For each target window and SNR, a copy mixed with a random background window. A mix that would clip
    is scaled down as a whole, so its SNR stays."""
    from dataclasses import replace
    if not backgrounds:
        raise ValueError("no background windows to mix with")
    rng = np.random.default_rng(seed)
    result = []
    for target in targets:
        for snr in snrs_db:
            background = backgrounds[int(rng.integers(len(backgrounds)))]
            samples = mix_at_snr(target.samples, background.samples, snr, rng)
            peak = float(np.max(np.abs(samples)))
            if peak > 0.99:
                samples = samples * (0.99 / peak)
            name = f"{background.recording.name}_{int(round(background.start_s * 1000)):07d}"
            result.append(replace(target, samples=samples, snr_db=float(snr), background=name))
    return result
