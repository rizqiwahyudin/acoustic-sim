import csv
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ml_data  # noqa: E402
from heimdall_ml import align, dataset, evaluate, mix  # noqa: E402

CLIP_RATE = 16000
CLIP_S = 6.0


def make_clip(seed=0):
    """A drone-like clip: a 190 Hz harmonic series with slow pitch drift and noise bursts (unique in time)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(CLIP_S * CLIP_RATE)) / CLIP_RATE
    phase = 2 * np.pi * np.cumsum(190.0 + 6.0 * np.sin(2 * np.pi * 0.4 * t)) / CLIP_RATE
    clip = sum(0.08 / h * np.sin(h * phase) for h in range(1, 9))
    bursts = rng.standard_normal(len(t)) * (rng.random(len(t) // 400 + 1).repeat(400)[:len(t)] > 0.7)
    return 0.6 * clip + 0.05 * bursts


def write_recording(root, name, samples_48k, label, started="2026-11-10T10:00:00.000+01:00"):
    day = "2026-11-10"
    stereo = np.stack([samples_48k, 0.5 * samples_48k], axis=1)
    pcm = np.clip(np.round(stereo * 32767), -32768, 32767).astype("<i2")
    (root / day).mkdir(parents=True, exist_ok=True)
    with wave.open(str(root / day / f"{name}.wav"), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(pcm.tobytes())
    record = {"schema": dataset.RECORDING_SCHEMA, "recording": {"started_at": started},
              "files": {"wav": f"{day}/{name}.wav", "json": f"{day}/{name}.json"},
              "audio": {"sample_rate_hz": 48000, "channels": 2, "duration_s": len(samples_48k) / 48000},
              "label": {"class": None, "source": "", "session": "", "room": "", "level": "", "distance_m": None,
                        **label}}
    (root / day / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")


def played(clip, offset_s, seconds, rng, noise=0.003):
    """The clip looping through a 'speaker and room', recorded at 48 kHz from `offset_s` into the clip."""
    looped = np.tile(np.repeat(clip, 3), 4)  # 16 kHz to 48 kHz, crude; the room filter smooths it
    start = int(offset_s * 48000)
    x = looped[start:start + int(seconds * 48000)]
    room = np.zeros(400)
    room[0], room[90], room[250] = 1.0, 0.4, 0.2
    return np.convolve(x, room)[:len(x)] * 0.5 + noise * rng.standard_normal(len(x))


def other_sound(seconds, rng):
    t = np.arange(int(seconds * 48000)) / 48000
    pitch = 120.0 + 40.0 * np.sin(2 * np.pi * 0.7 * t)
    voice = np.sin(2 * np.pi * np.cumsum(pitch) / 48000) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t))
    return 0.05 * voice + 0.01 * rng.standard_normal(len(t))


@pytest.fixture()
def dataset_root(tmp_path):
    rng = np.random.default_rng(5)
    clip = make_clip()
    root = tmp_path / "recordings"
    for i, (offset, session) in enumerate(((0.0, "s1"), (2.3, "s1"), (4.1, "s2"))):
        write_recording(root, f"heimdall-drone-{i}", played(clip, offset, 10.0, rng),
                        {"class": "drone", "source": "drone.wav", "session": session, "level": f"-{6 * i} dB",
                         "distance_m": 2.0})
    for i, session in enumerate(("s1", "s2")):
        write_recording(root, f"heimdall-speech-{i}", other_sound(10.0, rng),
                        {"class": "not_drone", "source": "speech", "session": session, "distance_m": 1.5})
    clip_path = tmp_path / "clip.wav"
    dataset.write_wav(clip_path, clip, CLIP_RATE)
    (root / "2026-11-10" / "notes.json").write_text("{\"not\": \"a record\"}", encoding="utf-8")
    return root, clip_path, clip


def test_load_recordings_and_windows(dataset_root):
    root, _, _ = dataset_root
    recordings = dataset.load_recordings(root)
    assert [r.wav.name for r in recordings] == [f"heimdall-drone-{i}.wav" for i in range(3)] + \
        [f"heimdall-speech-{i}.wav" for i in range(2)]
    first = recordings[0]
    assert first.label_class == "drone" and first.get("session") == "s1" and first.get("date") == "2026-11-10"
    items = dataset.windows(first, length_s=1.0, hop_s=0.25)
    assert len(items) == 37 and items[1].start_s == 0.25 and len(items[0].samples) == 16000
    train, test = dataset.split_by(recordings, "session", ["s2"])
    assert {r.get("session") for r in train} == {"s1"} and {r.get("session") for r in test} == {"s2"}


@pytest.mark.parametrize("offset", [0.0, 2.3, 4.1])
def test_align_finds_the_clip_time(dataset_root, offset):
    root, _, clip = dataset_root
    recording = next(r for r in dataset.load_recordings(root) if r.get("level") == f"-{6 * [0.0, 2.3, 4.1].index(offset)} dB")
    audio, rate = recording.audio(0)
    result = align.align_to_clip(audio, rate, clip, CLIP_RATE)
    assert result.aligned
    error = (result.offset_s - offset + CLIP_S / 2) % CLIP_S - CLIP_S / 2
    assert abs(error) < 0.01  # the room adds up to 5 ms of spread
    assert result.clip_time(3.0) == pytest.approx((result.offset_s + 3.0) % CLIP_S)


def test_align_rejects_another_sound(dataset_root):
    root, _, clip = dataset_root
    speech = next(r for r in dataset.load_recordings(root) if r.label_class == "not_drone")
    audio, rate = speech.audio(0)
    assert not align.align_to_clip(audio, rate, clip, CLIP_RATE).aligned


def test_split_by_clip_time_keeps_every_second_on_one_side():
    rec = dataset.Recording(Path("x.wav"), {})
    items = [dataset.Window(rec, 0.0, 1.0, 0.25, np.zeros(1), clip_time_s=t) for t in np.arange(0.0, 6.0, 0.25)]
    train, test = dataset.split_by_clip_time(items, 6.0, 4.0, 6.0)
    assert [w.clip_time_s for w in test] == [4.0, 4.25, 4.5, 4.75, 5.0]
    assert all(w.clip_time_s + 1.0 <= 4.0 for w in train)  # nothing that reaches into 4-6 s, also not by wrapping
    assert 5.5 not in [w.clip_time_s for w in train + test]  # 5.5-6.5 s wraps into the start: dropped


def test_mix_sets_the_snr():
    rng = np.random.default_rng(0)
    target = 0.1 * np.sin(np.arange(16000) / 5.0)
    background = rng.standard_normal(5000)
    mixed = mix.mix_at_snr(target, background, 6.0, np.random.default_rng(1))
    assert mix.power_db(target) - mix.power_db(mixed - target) == pytest.approx(6.0, abs=1e-9)


def test_evaluate_counts_detections_and_false_alarms():
    items = [evaluate.Scored(s, True, {"level": "0"}) for s in (5, 6, 7, 8)] + \
            [evaluate.Scored(s, False, {"level": "x"}) for s in (1, 2, 3, 9)]
    result = evaluate.evaluate(items, hop_s=0.25, threshold=4.5, by=("level",))
    assert result["detection_rate"] == 1.0
    assert result["false_alarms_per_hour"] == pytest.approx(3600.0)
    assert result["auc"] == pytest.approx(0.75)
    assert [row["value"] for row in result["groups"]["level"]] == ["0", "x"]
    strict = evaluate.evaluate(items, hop_s=0.25, false_alarms_per_hour_target=0.0)
    assert strict["false_alarms_per_hour"] == 0.0 and strict["detection_rate"] == 0.0
    assert "detection" in evaluate.report(result)


def test_command_line_export_baseline_evaluate(dataset_root, tmp_path, capsys):
    root, clip_path, _ = dataset_root
    out = tmp_path / "ml"
    ml_data.main(["--root", str(root), "--source", "drone.wav", "export", "--out", str(out), "--clip",
                  str(clip_path), "--test-clip", "4", "6", "--test-key", "session", "--test", "s2",
                  "--mix", "0", "6"])
    rows = dataset.read_manifest(out / "manifest.csv")
    targets = [r for r in rows if r["label"] == "drone"]
    assert {r["split"] for r in targets} == {"train", "test"}
    assert all(float(r["clip_time_s"]) >= 4.0 for r in targets if r["split"] == "test")
    assert any(r["snr_db"] == "6.0" for r in rows) and not any(r["snr_db"] for r in rows if r["split"] == "test")
    assert {r["session"] for r in rows if r["label"] == "not_drone" and r["split"] == "test"} == {"s2"}
    assert all((out / r["file"]).exists() for r in rows)
    with wave.open(str(out / rows[0]["file"]), "rb") as handle:
        assert (handle.getframerate(), handle.getnchannels(), handle.getnframes()) == (16000, 1, 16000)
    scores = tmp_path / "tonal.csv"
    ml_data.main(["baseline", "--manifest", str(out / "manifest.csv"), "--out", str(scores),
                  "--min-hz", "150", "--max-hz", "260"])
    with scores.open(newline="", encoding="utf-8") as handle:
        scored = list(csv.DictReader(handle))
    drone = [float(r["score"]) for r in scored if r["label"] == "drone" and not r["snr_db"]]
    other = [float(r["score"]) for r in scored if r["label"] == "not_drone"]
    assert np.median(drone) > np.median(other) + 3.0
    capsys.readouterr()
    ml_data.main(["evaluate", "--scores", str(scores), "--false-alarms-per-hour", "60", "--by", "level"])
    printed = capsys.readouterr().out
    assert "detection" in printed and "by level:" in printed
