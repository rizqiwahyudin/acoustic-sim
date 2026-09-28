"""Export the data the reveal video is drawn from.

* assets/array.json   -- the deployed 44-microphone geometry, 6x6 sector grid
                         and steering delays from data/heimdall_acoustic_contract.json
* assets/spectra.json -- 64-band log-mel spectrograms of audio/drone.wav and a
                         voice-dominated stretch of audio/crowd.wav

Run from the repository root:  python demo/heimdall-reveal/build_assets.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

VOICE_START_S = 88.0      # crowd.wav stretch with clearly voiced harmonics
DRONE_START_S = 5.0
CLIP_S = 3.0
MEL_BANDS = 64
DYNAMIC_RANGE_DB = 70.0


def export_array() -> None:
    contract = json.loads((ROOT / "data" / "heimdall_acoustic_contract.json").read_text())
    mics = np.asarray(contract["microphone_positions_mm"], dtype=float)
    max_delay = float(contract["max_delay_samples"])
    delays = np.asarray(contract["delay_fixpt_8_24"], dtype=float) / float(1 << 24) * max_delay
    centered = mics - mics.mean(axis=0)
    radius = np.hypot(centered[:, 0], centered[:, 1])
    angle = np.mod(np.arctan2(centered[:, 1], centered[:, 0]), 2 * math.pi)
    # Boot order: inner ring outwards, sweeping clockwise inside each ring.
    ignite_order = sorted(range(len(mics)), key=lambda i: (round(radius[i] / 20.0), -angle[i]))
    payload = {
        "source": "data/heimdall_acoustic_contract.json",
        "coordinate_system": contract["coordinate_system"],
        "microphones_mm": [[round(x, 3), round(y, 3)] for x, y in mics.tolist()],
        "rows": contract["rows"],
        "columns": contract["columns"],
        "azimuth_deg": contract["azimuth_deg"],
        "elevation_deg": contract["elevation_deg"],
        "sampling_rate_hz": contract["sampling_rate_hz"],
        "speed_of_sound_m_s": contract["speed_of_sound_m_s"],
        "delay_samples": np.round(delays, 4).tolist(),
        "ignite_order": ignite_order,
    }
    (ASSETS / "array.json").write_text(json.dumps(payload, separators=(",", ":")))


def mel_filterbank(sample_rate: int, n_fft: int, bands: int, fmin: float, fmax: float) -> np.ndarray:
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), bands + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    bank = np.zeros((bands, bins.size))
    for band in range(bands):
        low, center, high = edges[band:band + 3]
        rising = (bins - low) / max(center - low, 1e-9)
        falling = (high - bins) / max(high - center, 1e-9)
        bank[band] = np.clip(np.minimum(rising, falling), 0.0, None)
    return bank


def log_mel(path: Path, start_s: float) -> dict:
    sample_rate, samples = wavfile.read(path)
    samples = samples.astype(np.float64)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    first = int(start_s * sample_rate)
    clip = samples[first:first + int(CLIP_S * sample_rate)]
    n_fft = 512
    hop = sample_rate // 100
    _, _, spectrum = stft(clip, sample_rate, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    power = np.abs(spectrum) ** 2
    mel = mel_filterbank(sample_rate, n_fft, MEL_BANDS, 60.0, sample_rate / 2) @ power
    db = 10.0 * np.log10(mel + 1e-12)
    scaled = np.clip((db - (db.max() - DYNAMIC_RANGE_DB)) / DYNAMIC_RANGE_DB, 0.0, 1.0)
    quantized = np.round(scaled * 255).astype(np.uint8)
    return {
        "source": f"{path.relative_to(ROOT).as_posix()} @ {start_s:g}-{start_s + CLIP_S:g} s",
        "bands": MEL_BANDS,
        "frames": int(quantized.shape[1]),
        "data": quantized.ravel().tolist(),   # band-major, band 0 = lowest frequency
    }


def export_spectra() -> None:
    payload = {
        "drone": log_mel(ROOT / "audio" / "drone.wav", DRONE_START_S),
        "voice": log_mel(ROOT / "audio" / "crowd.wav", VOICE_START_S),
    }
    (ASSETS / "spectra.json").write_text(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    export_array()
    export_spectra()
    print(f"wrote {ASSETS / 'array.json'} and {ASSETS / 'spectra.json'}")
