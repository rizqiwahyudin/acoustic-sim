"""Synthesise the reveal soundtrack from the scene's cue list.

    python demo/heimdall-reveal/soundtrack.py out/events.json out/soundtrack.wav

The cue list comes from window.audioEvents() in the page, so every hit, tick
and alarm lands on the frame that draws it. All sounds are generated here
except the two beds, which use the repository's own audio/drone.wav and
audio/crowd.wav recordings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

SR = 48_000
DURATION = 20.0
N = int(SR * DURATION)
ROOT = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(78002)


def seconds(duration: float) -> np.ndarray:
    return np.arange(int(duration * SR)) / SR


def band(signal: np.ndarray, low: float | None, high: float | None, order: int = 4) -> np.ndarray:
    if low and high:
        sos = butter(order, [low, high], btype="band", fs=SR, output="sos")
    elif low:
        sos = butter(order, low, btype="high", fs=SR, output="sos")
    else:
        sos = butter(order, high, btype="low", fs=SR, output="sos")
    return sosfilt(sos, signal)


def envelope(t: np.ndarray, attack: float, decay: float) -> np.ndarray:
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / decay)


def glide(t: np.ndarray, f0: float, f1: float, tau: float) -> np.ndarray:
    frequency = f1 + (f0 - f1) * np.exp(-t / tau)
    return np.sin(2 * np.pi * np.cumsum(frequency) / SR)


def square(t: np.ndarray, frequency: float, harmonics: int = 7) -> np.ndarray:
    out = np.zeros_like(t)
    for k in range(1, harmonics + 1, 2):
        out += np.sin(2 * np.pi * frequency * k * t) / k
    return out


def reverb(signal: np.ndarray, seconds_long: float = 2.2, wet: float = 0.35) -> np.ndarray:
    t = seconds(seconds_long)
    impulse = rng.standard_normal(t.size) * np.exp(-t / (seconds_long / 6.9))
    impulse = band(impulse, 120, 5000, order=2)
    impulse /= np.sqrt(np.sum(impulse ** 2)) + 1e-12
    tail = np.convolve(signal, impulse)[: signal.size + t.size]
    dry = np.pad(signal, (0, tail.size - signal.size))
    return dry + wet * tail


class Mix:
    def __init__(self) -> None:
        self.left = np.zeros(N)
        self.right = np.zeros(N)

    def add(self, signal: np.ndarray, at: float, gain: float = 1.0, pan: float = 0.0, width_ms: float = 0.0) -> None:
        start = int(round(at * SR))
        if start >= N:
            return
        signal = signal[: N - start] * gain
        left_gain = np.sqrt(0.5 * (1 - pan))
        right_gain = np.sqrt(0.5 * (1 + pan))
        self.left[start:start + signal.size] += signal * left_gain
        offset = int(width_ms * SR / 1000)
        end = min(N, start + offset + signal.size)
        self.right[start + offset:end] += signal[: end - start - offset] * right_gain


def hit(strength: float) -> np.ndarray:
    t = seconds(1.6)
    sub = glide(t, 62, 36, 0.18) * envelope(t, 0.003, 0.5)
    thump = np.sin(2 * np.pi * 118 * t) * envelope(t, 0.001, 0.09)
    click = band(rng.standard_normal(t.size), 1800, None) * envelope(t, 0.0005, 0.006)
    return np.tanh(1.8 * strength * (sub + 0.6 * thump + 0.5 * click))


def boom() -> np.ndarray:
    t = seconds(3.0)
    sub = glide(t, 55, 28, 0.35) * envelope(t, 0.004, 1.1)
    body = band(rng.standard_normal(t.size), 40, 400) * envelope(t, 0.002, 0.35) * 0.8
    crack = band(rng.standard_normal(t.size), 2500, None) * envelope(t, 0.0005, 0.02) * 0.6
    return reverb(np.tanh(1.6 * (sub + body + crack)), 2.6, 0.45)


def riser(duration: float) -> np.ndarray:
    t = seconds(duration)
    k = t / duration
    noise = rng.standard_normal(t.size)
    swept = np.zeros_like(t)
    for i, (lo, hi) in enumerate([(300, 900), (800, 2400), (2000, 6000)]):
        weight = np.clip(1 - np.abs(k * 3 - i - 0.5), 0, 1)
        swept += band(noise, lo, hi, order=2) * weight
    tone = glide(t, 180, 1400, duration / 1.2) * 0.25
    return (swept + tone) * (k ** 1.6)


def pad(duration: float) -> np.ndarray:
    t = seconds(duration)
    out = np.zeros_like(t)
    for frequency in (55.0, 82.41, 110.0, 130.81, 164.81):
        for detune in (-0.35, 0.35):
            phase = (frequency + detune) * t
            out += 2 * (phase - np.floor(phase + 0.5))
    out = band(out, None, 420, order=4)
    swell = np.minimum(1.0, t / 1.3) * np.minimum(1.0, (duration - t) / 0.03)
    return out / 10 * swell


def blip(frequency: float, length: float = 0.05, decay: float = 0.014) -> np.ndarray:
    t = seconds(length)
    return np.sin(2 * np.pi * frequency * t) * envelope(t, 0.0008, decay)


def typing(duration: float) -> np.ndarray:
    out = np.zeros(int(duration * SR))
    for start in np.arange(0, duration, 1 / 26):
        t = seconds(0.004)
        burst = band(rng.standard_normal(t.size), 3000, None, order=2) * envelope(t, 0.0002, 0.0012)
        i = int(start * SR)
        out[i:i + burst.size] += burst[: out.size - i]
    return out


def glitch() -> np.ndarray:
    t = seconds(0.14)
    noise = band(rng.standard_normal(t.size), 400, 7000, order=2)
    crushed = np.round(noise * 6) / 6
    return crushed * envelope(t, 0.001, 0.05)


def scan_hum(duration: float) -> np.ndarray:
    t = seconds(duration)
    carrier = band(rng.standard_normal(t.size), 2400, 4200, order=2)
    pulse = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 9.3 * t))
    hum = 0.4 * np.sin(2 * np.pi * 60 * t) + 0.2 * np.sin(2 * np.pi * 120 * t)
    fade = np.minimum(1.0, np.minimum(t, duration - t) / 0.02)
    return (0.6 * carrier * pulse + hum) * fade


def lock(tone: str) -> np.ndarray:
    base = 660 if tone == "amber" else 880
    t1 = seconds(0.07); t2 = seconds(0.12)
    first = square(t1, base) * envelope(t1, 0.001, 0.05)
    second = square(t2, base * 1.5) * envelope(t2, 0.001, 0.09)
    out = np.concatenate([first, np.zeros(int(0.015 * SR)), second])
    if tone == "red":
        t = seconds(out.size / SR)
        out += 0.5 * square(t, 110) * envelope(t, 0.002, 0.12)
    return out


def data_chatter(duration: float) -> np.ndarray:
    out = np.zeros(int(duration * SR))
    for start in np.arange(0, duration, 1 / 18):
        frequency = 1000 + 2500 * rng.random()
        b = blip(frequency, 0.028, 0.008)
        i = int(start * SR)
        out[i:i + b.size] += b[: out.size - i]
    return out


def chime() -> np.ndarray:
    t = seconds(1.2)
    return (np.sin(2 * np.pi * 1046.5 * t) * envelope(t, 0.002, 0.35)
            + 0.7 * np.sin(2 * np.pi * 1568.0 * t) * envelope(t, 0.002, 0.45) * (t > 0.09))


def klaxon(duration: float) -> np.ndarray:
    t = seconds(duration)
    out = np.zeros_like(t)
    for start in np.arange(0, duration, 0.6):
        for offset, frequency in ((0.0, 740.0), (0.3, 587.0)):
            seg_start = start + offset
            if seg_start >= duration:
                continue
            seg = seconds(min(0.28, duration - seg_start))
            tone = square(seg, frequency, 9) * np.minimum(1, np.minimum(seg, seg[-1] - seg + 1e-3) / 0.01)
            i = int(seg_start * SR)
            out[i:i + tone.size] += tone
    return np.tanh(1.4 * out)


def console_bed(duration: float) -> np.ndarray:
    """Mains hum, air and a low 120 BPM pulse that carries tension under the console."""
    t = seconds(duration)
    hum = sum(np.sin(2 * np.pi * f * t) * a for f, a in ((50, 0.3), (100, 0.15), (150, 0.08)))
    air = band(rng.standard_normal(t.size), 4000, 9000, order=2) * 0.06
    drone = pad(duration) * 1.4
    pulse = np.zeros_like(t)
    for start in np.arange(0.0, duration, 0.5):
        k = seconds(0.4)
        beat = glide(k, 70, 44, 0.05) * envelope(k, 0.002, 0.14)
        i = int(start * SR)
        pulse[i:i + beat.size] += beat[: pulse.size - i]
    fade = np.minimum(1.0, np.minimum(t, duration - t) / 0.05)
    return (hum + air + drone + 0.9 * pulse) * fade


def recording(path: Path, start_s: float, duration: float) -> np.ndarray:
    rate, samples = wavfile.read(path)
    samples = samples.astype(np.float64)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples[int(start_s * rate): int((start_s + duration) * rate)]
    samples = resample_poly(samples, SR, rate)
    samples = band(samples, 120, 6000, order=2)
    return samples / (np.max(np.abs(samples)) + 1e-12)


def main(events_path: str, out_path: str) -> None:
    events = json.loads(Path(events_path).read_text())
    mix = Mix()
    for event in events:
        kind = event["type"]; at = event["t"]; until = event.get("t1", at)
        if kind == "hit":
            mix.add(hit(event.get("strength", 1.0)), at, 0.6)
        elif kind == "boom":
            mix.add(boom(), at, 0.9, width_ms=11)
        elif kind == "riser":
            mix.add(riser(until - at), at, 0.25, width_ms=7)
        elif kind == "pad":
            mix.add(pad(until - at), at, 0.8, width_ms=13)
        elif kind == "mic":
            mix.add(blip(1400 + 22 * event["index"]), at, 0.2, pan=event.get("pan", 0.0) * 0.8)
        elif kind == "type":
            mix.add(typing(until - at), at, 0.12)
        elif kind == "cut":
            mix.add(glitch(), at, 0.28)
        elif kind == "tick":
            if event.get("strong"):
                mix.add(blip(880, 0.25, 0.08), at, 0.22)
            else:
                mix.add(blip(2100 + 110 * event.get("row", 0), 0.012, 0.003), at, 0.13)
        elif kind == "scanhum":
            mix.add(scan_hum(until - at), at, 0.2)
        elif kind == "lock":
            mix.add(lock(event.get("tone", "amber")), at, 0.42)
        elif kind == "data":
            mix.add(data_chatter(until - at), at, 0.14)
        elif kind == "clear":
            mix.add(chime(), at, 0.36, width_ms=9)
        elif kind == "release":
            t = seconds(0.16)
            mix.add(glide(t, 1300, 520, 0.05) * envelope(t, 0.001, 0.07), at, 0.2)
        elif kind == "move":
            mix.add(blip(1560, 0.05, 0.015), at, 0.2)
        elif kind == "alarm":
            mix.add(klaxon(until - at), at, 0.16, width_ms=5)
        elif kind == "end":
            mix.add(boom(), at, 0.8, width_ms=11)
            mix.add(pad(DURATION - at) * np.linspace(1, 0.0, int((DURATION - at) * SR)) ** 0.7, at, 0.65, width_ms=13)
        elif kind == "bed_console":
            mix.add(console_bed(until - at), at, 0.16, width_ms=9)
        elif kind == "bed_voice":
            voice = recording(ROOT / "audio" / "crowd.wav", 88.0, until - at)
            fade = np.minimum(1.0, np.minimum(np.arange(voice.size), voice.size - np.arange(voice.size)) / (0.15 * SR))
            mix.add(voice * fade, at, 0.42, pan=0.35)
        elif kind == "bed_drone":
            drone = recording(ROOT / "audio" / "drone.wav", 5.0, until - at)
            ramp = np.clip((np.arange(drone.size) / SR) / 2.7, 0.15, 1.0)
            tail = np.minimum(1.0, (drone.size - np.arange(drone.size)) / (0.05 * SR))
            mix.add(drone * ramp * tail, at, 0.6, pan=-0.3)

    stereo = np.stack([mix.left, mix.right], axis=1)
    stereo = np.tanh(stereo * 1.2) / np.tanh(1.2)
    fade = np.clip((DURATION - np.arange(N) / SR) / 0.6, 0, 1)[:, None]
    stereo *= fade
    stereo *= 10 ** (-1 / 20) / (np.max(np.abs(stereo)) + 1e-12)
    wavfile.write(out_path, SR, (stereo * 32767).astype(np.int16))
    print(f"wrote {out_path} ({DURATION:.0f} s, {SR} Hz stereo)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
