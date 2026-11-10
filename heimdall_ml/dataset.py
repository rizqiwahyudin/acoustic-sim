"""Labelled windows from the Heimdall recordings (WAV + `heimdall-recording-v2` record, written by the backend).

Windows are cut from 16 kHz audio made with the firmware decimator, so they match what the board classifies.
"""
import csv
import json
import os
import re
import wave
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import tonal

RECORDING_SCHEMA = "heimdall-recording-v2"
DATASET_ENV = "HEIMDALL_DATASET_DIR"
MANIFEST_FIELDS = ["file", "split", "label", "recording", "start_s", "length_s", "hop_s", "clip_time_s",
                   "source", "session", "room", "date", "distance_m", "level", "azimuth_deg", "elevation_deg",
                   "snr_db", "background"]


def default_root() -> Path:
    """The backend's dataset folder: HEIMDALL_DATASET_DIR, else <project>/datasets/heimdall_recordings."""
    configured = os.environ.get(DATASET_ENV)
    return Path(configured) if configured else Path(__file__).resolve().parents[2] / "datasets" / "heimdall_recordings"


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """16-bit PCM as float (full scale 1.0), shape (frames, channels)."""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM is supported")
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        return data.reshape(-1, handle.getnchannels()).astype(np.float64) / 32768.0, handle.getframerate()


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    """Mono 16-bit PCM; samples are clipped to full scale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(np.asarray(samples) * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


@dataclass(frozen=True)
class Recording:
    wav: Path
    record: dict

    @property
    def name(self) -> str:
        return self.wav.stem

    @property
    def label(self) -> dict:
        return self.record.get("label") or {}

    @property
    def label_class(self) -> str | None:
        return self.label.get("class")

    def get(self, key: str):
        """A label field, or `date` (from the start time) or `file`."""
        if key == "date":
            return str((self.record.get("recording") or {}).get("started_at") or "")[:10] or None
        if key == "file":
            return self.wav.name
        value = self.label.get(key)
        return None if value == "" else value

    def audio(self, channel: int = 0) -> tuple[np.ndarray, int]:
        data, rate = read_wav(self.wav)
        return data[:, channel], rate


def load_recordings(root: Path | None = None) -> list[Recording]:
    """Every `heimdall-recording-v2` record under root whose WAV exists, sorted by file name."""
    root = Path(root) if root else default_root()
    recordings = []
    for path in sorted(root.rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or record.get("schema") != RECORDING_SCHEMA:
            continue
        files = record.get("files") or {}
        wav = root / files["wav"] if files.get("wav") else path.with_suffix(".wav")
        if wav.exists():
            recordings.append(Recording(wav, record))
    return recordings


@dataclass(frozen=True)
class Window:
    recording: Recording
    start_s: float
    length_s: float
    hop_s: float
    samples: np.ndarray
    clip_time_s: float | None = None
    snr_db: float | None = None      # set on a mixed copy: target level over the background
    background: str = ""             # the background window of a mixed copy


def windows(recording: Recording, length_s: float = 1.0, hop_s: float = 0.25, channel: int = 0,
            clip_offset_s: float | None = None, clip_length_s: float | None = None) -> list[Window]:
    """Windows of `length_s` every `hop_s` from one channel at 16 kHz. With a clip alignment, each window
    also gets its start time in the clip (looped)."""
    audio, rate = recording.audio(channel)
    stream = tonal.to_16k(audio, rate)
    size = int(round(length_s * tonal.RATE_HZ))
    step = int(round(hop_s * tonal.RATE_HZ))
    result = []
    for start in range(0, len(stream) - size + 1, step):
        start_s = start / tonal.RATE_HZ
        clip_time = None
        if clip_offset_s is not None and clip_length_s:
            clip_time = (clip_offset_s + start_s) % clip_length_s
        result.append(Window(recording, start_s, length_s, hop_s, stream[start:start + size], clip_time))
    return result


def split_by(items: Iterable, key: str | Callable, test_values: Iterable) -> tuple[list, list]:
    """Train and test by a recording field (for example session, room or date): never one recording in both."""
    wanted = {str(value) for value in test_values}

    def value(item):
        recording = item.recording if isinstance(item, Window) else item
        return key(recording) if callable(key) else recording.get(key)

    train, test = [], []
    for item in items:
        (test if str(value(item)) in wanted else train).append(item)
    return train, test


def _overlap(start: float, length: float, low: float, high: float, period: float) -> float:
    return sum(max(0.0, min(start + length, high + shift) - max(start, low + shift))
               for shift in (-period, 0.0, period))


def split_by_clip_time(items: Iterable[Window], clip_length_s: float, test_from_s: float,
                       test_to_s: float) -> tuple[list[Window], list[Window]]:
    """Train and test by the part of the clip that a window holds. A window that is partly in the test part
    goes to neither, so no second of the clip is in both sets."""
    train, test = [], []
    for item in items:
        if item.clip_time_s is None:
            continue
        shared = _overlap(item.clip_time_s, item.length_s, test_from_s, test_to_s, clip_length_s)
        if shared >= item.length_s - 1e-9:
            test.append(item)
        elif shared <= 1e-9:
            train.append(item)
    return train, test


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "unlabelled"


def export(splits: dict[str, list[Window]], out_dir: Path,
           label: Callable[[Window], str] = lambda item: item.recording.label_class or "unlabelled") -> Path:
    """Write each window as a 16 kHz mono WAV in <out>/<split>/<label>/ and a manifest.csv for all of them."""
    out_dir = Path(out_dir)
    rows = []
    for split, items in splits.items():
        for item in items:
            name = _safe(label(item))
            suffix = "" if item.snr_db is None else f"_snr{item.snr_db:+g}_{_safe(item.background)}"
            relative = Path(_safe(split)) / name / \
                f"{_safe(item.recording.name)}_{int(round(item.start_s * 1000)):07d}{suffix}.wav"
            write_wav(out_dir / relative, item.samples, tonal.RATE_HZ)
            recording = item.recording
            rows.append({
                "file": relative.as_posix(), "split": split, "label": name, "recording": recording.wav.name,
                "start_s": round(item.start_s, 4), "length_s": item.length_s, "hop_s": item.hop_s,
                "clip_time_s": "" if item.clip_time_s is None else round(item.clip_time_s, 4),
                **{key: "" if recording.get(key) is None else recording.get(key)
                   for key in ("source", "session", "room", "date", "distance_m", "level", "azimuth_deg",
                               "elevation_deg")},
                "snr_db": "" if item.snr_db is None else item.snr_db,
                "background": item.background,
            })
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def read_manifest(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
