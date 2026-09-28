"""Synthesise the teaser's ambient score from the scene's cue list.

    python demo/heimdall-teaser/soundtrack.py out/events.json out/soundtrack.wav

Slow and quiet: a drifting D-minor pad under rain, a glass pluck as each
microphone wakes, bell tones for the hex ripple, one low hit on the lock, a
heartbeat under the drone and the repository's own audio/drone.wav recording,
heard from far away. Everything shares one long hall reverb.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

SR = 48_000
ROOT = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(1467)

# Equal-tempered pitches (Hz).
D1, D2, A2, Bb2, D3, Eb3, E3, F3, G3, A3, Bb3 = 36.71, 73.42, 110.0, 116.54, 146.83, 155.56, 164.81, 174.61, 196.0, 220.0, 233.08
C4, D4, E4, F4, G4, A4, C5, D5, F5, G5, A5, C6, D6, F6, G6, A6 = (
    261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 523.25, 587.33, 698.46, 783.99, 880.0, 1046.5, 1174.66, 1396.91, 1567.98, 1760.0)

CHORDS = {
    "night": (D2, A2, D3, F3, A3),        # D minor, open
    "wake": (D2, Bb2, D3, F3, Bb3),       # Bb over D: the array comes alive
    "lock": (D2, Eb3, G3, Bb3),           # Eb over D: tension on the lock
    "drone": (D2, A2, F3, C4, E4),        # D minor 9
    "title": (D2, A2, E3, A3, D4),        # D sus2, left open
}
PLUCK_SCALE = (D4, F4, G4, A4, C5, D5, F5, G5, A5, C6, D6, F6, G6)
RING_NOTES = (D5, A4, F4, D4, A3)


def seconds(duration: float) -> np.ndarray:
    return np.arange(int(round(duration * SR))) / SR


def band(signal: np.ndarray, low: float | None, high: float | None, order: int = 4) -> np.ndarray:
    if low and high:
        sos = butter(order, [low, high], btype="band", fs=SR, output="sos")
    elif low:
        sos = butter(order, low, btype="high", fs=SR, output="sos")
    else:
        sos = butter(order, high, btype="low", fs=SR, output="sos")
    return sosfilt(sos, signal, axis=-1)


def envelope(t: np.ndarray, attack: float, decay: float) -> np.ndarray:
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / decay)


def ramp(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    x = np.clip((t - t0) / max(t1 - t0, 1e-6), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def glide(t: np.ndarray, f0: float, f1: float, tau: float) -> np.ndarray:
    frequency = f1 + (f0 - f1) * np.exp(-t / tau)
    return np.sin(2 * np.pi * np.cumsum(frequency) / SR)


def saw(frequency: float, t: np.ndarray, phase: float) -> np.ndarray:
    p = frequency * t + phase
    return 2 * (p - np.floor(p + 0.5))


def drift(n: int, rate: float) -> np.ndarray:
    """A smooth random curve in [0, 1] that wanders at about `rate` Hz."""
    knots = rng.random(int(n / SR * rate) + 3)
    x = np.arange(n) / SR * rate
    i = x.astype(int)
    w = (1 - np.cos(np.pi * (x - i))) / 2
    return knots[i] * (1 - w) + knots[i + 1] * w


def panned(signal: np.ndarray, pan: float | np.ndarray) -> np.ndarray:
    return np.stack([signal * np.sqrt(0.5 * (1 - pan)), signal * np.sqrt(0.5 * (1 + pan))])


def hall(length: float) -> np.ndarray:
    """Decorrelated stereo impulse response whose highs die away first."""
    t = seconds(length)
    channels = []
    for _ in range(2):
        noise = rng.standard_normal(t.size)
        low = band(noise, 90, 1200, 2) * np.exp(-t / (length / 6.9))
        high = band(noise, 1200, 8000, 2) * np.exp(-t / (0.45 * length / 6.9)) * 0.6
        ir = np.concatenate([np.zeros(int(0.028 * SR)), low + high])
        channels.append(ir / np.sqrt(np.sum(ir ** 2)))
    return np.stack(channels)


class Mix:
    def __init__(self, duration: float) -> None:
        self.n = int(round(duration * SR))
        self.dry = np.zeros((2, self.n))
        self.send = np.zeros((2, self.n))

    def add(self, signal: np.ndarray, at: float, gain: float = 1.0, pan: float = 0.0, send: float = 0.3) -> None:
        stereo = panned(signal, pan) if signal.ndim == 1 else signal
        start = int(round(at * SR))
        if start < 0:
            stereo, start = stereo[:, -start:], 0
        length = min(stereo.shape[1], self.n - start)
        if length <= 0:
            return
        self.dry[:, start:start + length] += gain * stereo[:, :length]
        self.send[:, start:start + length] += gain * send * stereo[:, :length]

    def render(self) -> np.ndarray:
        ir = hall(6.0)
        wet = np.stack([fftconvolve(self.send[c], ir[c])[: self.n] for c in range(2)])
        return self.dry + wet


# ── Instruments ──────────────────────────────────────────────────────────────

def chord(freqs: tuple[float, ...], t: np.ndarray) -> np.ndarray:
    out = np.zeros((2, t.size))
    for f in freqs:
        for cents in (-6.0, 0.0, 7.0):
            for c in range(2):   # a different phase per side keeps it wide
                out[c] += saw(f * 2 ** (cents / 1200), t, rng.random()) * (0.7 + 0.3 * drift(t.size, 0.15))
    return out / (3 * len(freqs))


def pad(t0: float, t1: float, sections: list[tuple[float, str]], brightness: list[tuple[float, float]]) -> np.ndarray:
    """Chords cross-fading at each section time, filter opening with `brightness`."""
    t = seconds(t1 - t0)
    at = t + t0
    out = np.zeros((2, t.size))
    for i, (start, name) in enumerate(sections):
        weight = ramp(at, start - 1.4, start + 1.4) if i else np.ones_like(at)
        if i + 1 < len(sections):
            weight = weight * (1 - ramp(at, sections[i + 1][0] - 1.4, sections[i + 1][0] + 1.4))
        live = weight > 1e-4
        if live.any():
            out[:, live] += chord(CHORDS[name], t[live]) * weight[live]
    dark = band(out, None, 300, 4)
    bright = band(out, None, 1500, 2) * 0.6
    b = np.interp(at, *zip(*brightness))
    return (dark * (1 - b) + bright * b) * ramp(at, t0, t0 + 3.5) * (1 - ramp(at, t1 - 1.6, t1))


def sub(t0: float, t1: float, level: list[tuple[float, float]]) -> np.ndarray:
    t = seconds(t1 - t0)
    return np.sin(2 * np.pi * D1 * t) * np.interp(t + t0, *zip(*level))


def shimmer(duration: float) -> np.ndarray:
    t = seconds(duration)
    out = np.zeros((2, t.size))
    for f in (D6, F6, A6, 2093.0, 2349.32, 2793.83):
        for c in range(2):
            vibrato = 1 + 0.002 * np.sin(2 * np.pi * 4.1 * t + 6 * rng.random())
            out[c] += np.sin(2 * np.pi * np.cumsum(f * vibrato) / SR) * drift(t.size, 0.7) ** 3
    return out / 6 * ramp(t, 0, 1.8) * (1 - ramp(t, duration - 1.8, duration))


def rain(duration: float) -> np.ndarray:
    t = seconds(duration)
    out = np.zeros((2, t.size))
    drop_t = seconds(0.012)
    for c in range(2):
        hiss = band(rng.standard_normal(t.size), 2000, 9000, 2) * (0.6 + 0.4 * drift(t.size, 0.3))
        drops = np.zeros(t.size)
        for i in rng.integers(0, t.size - drop_t.size, int(duration * 40)):
            tick = band(rng.standard_normal(drop_t.size), 2500, None, 2) * envelope(drop_t, 0.0003, 0.002)
            drops[i:i + drop_t.size] += tick * rng.random()
        out[c] = 0.25 * hiss + 0.8 * drops
    return out * ramp(t, 0, 2.5) * (1 - ramp(t, duration - 1.5, duration))


def pluck(freq: float) -> np.ndarray:
    """Struck glass: a pure tone with two quick inharmonic overtones."""
    t = seconds(3.0)
    return (np.sin(2 * np.pi * freq * t) * envelope(t, 0.002, 0.7)
            + 0.3 * np.sin(2 * np.pi * 2.76 * freq * t) * envelope(t, 0.001, 0.2)
            + 0.12 * np.sin(2 * np.pi * 5.4 * freq * t) * envelope(t, 0.0005, 0.06))


def swell(duration: float) -> np.ndarray:
    """A breath: filtered air and a soft fifth rising and falling away."""
    t = seconds(duration)
    shape = np.sin(np.pi * t / duration) ** 2
    out = np.zeros((2, t.size))
    for c in range(2):
        air = band(rng.standard_normal(t.size), 300, 4000, 2)
        voice = sum(np.sin(2 * np.pi * np.cumsum(f * (1 + 0.003 * np.sin(2 * np.pi * 5.2 * t + c))) / SR)
                    for f in (A4, D5))
        out[c] = (0.35 * air + 0.3 * voice) * shape
    return out


def whoosh(duration: float) -> np.ndarray:
    """The wavefront: rumble opening into air, travelling left to right, cut at the crossing."""
    t = seconds(duration)
    k = t / duration
    noise = rng.standard_normal(t.size)
    body = band(noise, 50, 400, 2) * (1 - k) + band(noise, 400, 2500, 2) * k
    amp = k ** 2.2 * (1 - ramp(t, duration - 0.08, duration))
    return panned(body * amp, -0.8 + 1.4 * k)


def ring(freq: float) -> np.ndarray:
    """A.T.-field bell: inharmonic partials and a slow beat."""
    t = seconds(4.5)
    partials = ((1.0, 1.0, 2.2), (2.32, 0.5, 1.2), (4.25, 0.28, 0.6), (6.63, 0.15, 0.3))
    out = sum(a * np.sin(2 * np.pi * freq * r * t) * envelope(t, 0.001, d) for r, a, d in partials)
    return out + 0.6 * np.sin(2 * np.pi * (freq + 1.3) * t) * envelope(t, 0.001, 2.0)


def riser(duration: float) -> np.ndarray:
    t = seconds(duration)
    k = t / duration
    air = band(rng.standard_normal(t.size), 800, 6000, 2)
    tone = np.sin(2 * np.pi * np.cumsum(220 * 2 ** (2 * k)) / SR)
    return (0.5 * air + 0.4 * tone) * k ** 2


def lock_hit() -> np.ndarray:
    """One low, slow hit: a sub drop under a dark brass-like chord that opens and closes."""
    t = seconds(5.0)
    low = glide(t, 58, 31, 0.25) * envelope(t, 0.004, 1.2)
    brass = sum(saw(f * 2 ** (cents / 1200), t, rng.random()) for f in (D1, D2, A2) for cents in (-5, 5)) / 6
    brass = band(brass, None, 220, 4) + band(brass, None, 1400, 2) * envelope(t, 0.02, 0.35)
    brass *= envelope(t, 0.012, 1.4)
    bell = ring(A5)
    return np.tanh(1.5 * (0.9 * low + 0.6 * brass)) + 0.25 * np.pad(bell, (0, t.size - bell.size))


def heartbeat(duration: float, bpm: float = 54.0) -> np.ndarray:
    out = np.zeros(int(round(duration * SR)))
    t = seconds(0.35)
    beat = glide(t, 62, 38, 0.06) * envelope(t, 0.003, 0.09)
    for start in np.arange(0.0, duration - 0.7, 60 / bpm):
        for offset, strength in ((0.0, 1.0), (0.26, 0.6)):
            i = int((start + offset) * SR)
            out[i:i + t.size] += beat[: out.size - i] * strength
    return band(out, None, 180, 2)


def recording(path: Path, start_s: float, duration: float) -> np.ndarray:
    rate, samples = wavfile.read(path)
    samples = samples.astype(np.float64)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples[int(start_s * rate): int((start_s + duration) * rate)]
    samples = band(resample_poly(samples, SR, rate), 150, 3000, 2)
    return samples / (np.max(np.abs(samples)) + 1e-12)


def far_drone(duration: float) -> np.ndarray:
    rotor = recording(ROOT / "audio" / "drone.wav", 5.0, duration)
    t = np.arange(rotor.size) / SR
    return rotor * ramp(t, 0, 2.2) * (1 - ramp(t, duration - 1.0, duration))


def chime() -> np.ndarray:
    t = seconds(3.0)
    return (np.sin(2 * np.pi * D6 * t) * envelope(t, 0.002, 1.2)
            + 0.6 * np.sin(2 * np.pi * A6 * t) * envelope(t, 0.002, 1.6) * (t > 0.12))


def boom() -> np.ndarray:
    t = seconds(5.0)
    low = glide(t, 50, 27, 0.4) * envelope(t, 0.006, 1.6)
    body = band(rng.standard_normal(t.size), 30, 300, 2) * envelope(t, 0.003, 0.5) * 0.7
    return np.tanh(1.4 * (low + body))


# ── Score ────────────────────────────────────────────────────────────────────

def main(events_path: str, out_path: str) -> None:
    events = json.loads(Path(events_path).read_text())
    duration = max(e.get("t1", e["t"]) for e in events)
    mix = Mix(duration)

    def time_of(kind: str, index: int = 0) -> float:
        return [e["t"] for e in events if e["type"] == kind][index]

    listen, beam, lock, drone = time_of("swell"), time_of("riser"), time_of("lock"), time_of("drone")
    cross = time_of("ring")
    plucks = sorted((e for e in events if e["type"] == "pluck"), key=lambda e: e["t"])

    for e in events:
        kind, at = e["type"], e["t"]
        until = e.get("t1", at)
        if kind == "pad" and e.get("key") == "title":
            mix.add(pad(at, until, [(at, "title")], [(at, 0.35), (until, 0.15)]), at, 0.9, send=0.5)
        elif kind == "pad":
            sections = [(at, "night"), (listen, "wake"), (beam, "lock"), (drone, "drone")]
            brightness = [(at, 0.0), (listen, 0.25), (cross, 0.55), (lock, 0.9), (lock + 1.5, 0.6), (drone, 0.45), (until, 0.2)]
            mix.add(pad(at, until, sections, brightness), at, 1.0, send=0.45)
            level = [(at, 0.0), (listen, 0.3), (lock, 1.0), (lock + 3.0, 0.5), (until, 0.0)]
            mix.add(sub(at, until, level), at, 0.22, send=0.0)
        elif kind == "shimmer":
            mix.add(shimmer(until - at), at, 0.12, send=0.9)
        elif kind == "rain":
            mix.add(rain(until - at), at, 0.05, send=0.25)
        elif kind == "pluck":
            rank = plucks.index(e)
            note = PLUCK_SCALE[round(rank / max(len(plucks) - 1, 1) * (len(PLUCK_SCALE) - 1))]
            mix.add(pluck(note), at, 0.045, pan=0.7 * e.get("pan", 0.0), send=0.7)
        elif kind == "swell":
            mix.add(swell(until - at), at, 0.08, send=0.8)
        elif kind == "whoosh":
            mix.add(whoosh(until - at), at, 0.22, send=0.35)
        elif kind == "ring":
            k = e.get("index", 0)
            mix.add(ring(RING_NOTES[k % len(RING_NOTES)]), at, 0.11 * 0.8 ** k, pan=0.35 * (-1) ** k, send=0.75)
        elif kind == "riser":
            mix.add(riser(until - at), at, 0.06, send=0.5)
        elif kind == "lock":
            mix.add(lock_hit(), at, 0.55, send=0.4)
        elif kind == "heartbeat":
            mix.add(heartbeat(until - at), at, 0.45, send=0.1)
        elif kind == "drone":
            mix.add(far_drone(until - at), at, 0.1, send=0.6)
        elif kind == "chime":
            mix.add(chime(), at, 0.06, pan=-0.2, send=0.8)
        elif kind == "boom":
            mix.add(boom(), at, 0.6, send=0.5)

    out = mix.render()
    t = np.arange(mix.n) / SR
    out *= ramp(t, 0, 0.3) * (1 - ramp(t, duration - 1.2, duration))
    out = np.tanh(1.1 * out / (np.max(np.abs(out)) + 1e-12)) / np.tanh(1.1)
    out *= 10 ** (-1 / 20)
    wavfile.write(out_path, SR, (out.T * 32767).astype(np.int16))
    print(f"wrote {out_path} ({duration:.0f} s, {SR} Hz stereo)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
