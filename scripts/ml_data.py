"""Classifier data from the Heimdall recordings: list, align, export, baseline, evaluate and tensors.

    python scripts/ml_data.py list
    python scripts/ml_data.py align --clip audio/drone.wav
    python scripts/ml_data.py export --out ml_out --source drone.wav --clip audio/drone.wav --test-clip 21 30
    python scripts/ml_data.py baseline --manifest ml_out/manifest.csv --out ml_out/tonal.csv --min-hz 150 --max-hz 260
    python scripts/ml_data.py evaluate --scores ml_out/tonal.csv --false-alarms-per-hour 60 --by level,distance_m
    python scripts/ml_data.py export --out ml_cnn --length 1.024 --source drone.wav --clip audio/drone.wav --test-clip 21 30
    python scripts/ml_data.py tensors --manifest ml_cnn/manifest.csv --out ml_cnn/tensors

Recordings come from the backend's dataset folder (HEIMDALL_DATASET_DIR or datasets/heimdall_recordings).
`evaluate` reads any scores file with the manifest columns plus a `score` column, so every model is scored
the same way. `tensors` turns exported windows into the CNN's int8 input, built as the firmware builds it
(heimdall_ml.model_input, MAX78002/docs/CLASSIFIER.md).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heimdall_ml import align, dataset, evaluate, mix, model_input, tonal  # noqa: E402


def _is_target(recording, args) -> bool:
    if recording.label_class != args.target:
        return False
    return not args.source or args.source in str(recording.get("source") or "")


def _selected(recordings, args):
    """Target recordings (class and, if given, source) and every recording of another class."""
    return [r for r in recordings if _is_target(r, args) or (r.label_class and r.label_class != args.target)]


def command_list(args):
    for recording in dataset.load_recordings(args.root):
        duration = (recording.record.get("audio") or {}).get("duration_s", 0.0)
        fields = "  ".join(f"{key}={recording.get(key)}" for key in ("source", "session", "room", "distance_m",
                                                                    "level") if recording.get(key) is not None)
        print(f"{recording.wav.name:44s} {recording.label_class or '-':10s} {duration:6.1f} s  {fields}")


def _load_clip(path):
    data, rate = dataset.read_wav(Path(path))
    return data.mean(axis=1), rate


def _alignment(recording, clip, clip_rate, channel):
    audio, rate = recording.audio(channel)
    return align.align_to_clip(audio, rate, clip, clip_rate)


def command_align(args):
    clip, clip_rate = _load_clip(args.clip)
    for recording in _selected(dataset.load_recordings(args.root), args):
        if not _is_target(recording, args):
            continue
        result = _alignment(recording, clip, clip_rate, args.channel)
        print(f"{recording.wav.name:44s} offset {result.offset_s:7.3f} s of {result.clip_length_s:.3f} s  "
              f"confidence {result.confidence:5.2f}  spread {1000 * result.spread_s:5.1f} ms  "
              f"{'aligned' if result.aligned else 'NOT ALIGNED'}")


def _split_others(recordings, args):
    """Train and test recordings: by a field if given, else the last share of the recordings by duration."""
    if args.test_key:
        return dataset.split_by(recordings, args.test_key, args.test)
    total = sum((r.record.get("audio") or {}).get("duration_s", 0.0) for r in recordings)
    train, test, seen = [], [], 0.0
    for recording in recordings:
        (test if seen >= (1.0 - args.test_share) * total else train).append(recording)
        seen += (recording.record.get("audio") or {}).get("duration_s", 0.0)
    return train, test


def command_export(args):
    recordings = _selected(dataset.load_recordings(args.root), args)
    targets = [r for r in recordings if _is_target(r, args)]
    others = [r for r in recordings if not _is_target(r, args)]
    if not targets:
        raise SystemExit("no target recordings: check --target and --source")
    cut = dict(length_s=args.length, hop_s=args.hop, channel=args.channel)
    splits = {"train": [], "test": []}
    if args.clip:
        if not args.test_clip:
            raise SystemExit("--clip needs --test-clip FROM TO (seconds of the clip held out for the test)")
        clip, clip_rate = _load_clip(args.clip)
        for recording in targets:
            result = _alignment(recording, clip, clip_rate, args.channel)
            if not result.aligned:
                print(f"skip {recording.wav.name}: not aligned to the clip (confidence {result.confidence:.2f})")
                continue
            items = dataset.windows(recording, clip_offset_s=result.offset_s,
                                    clip_length_s=result.clip_length_s, **cut)
            train, test = dataset.split_by_clip_time(items, result.clip_length_s, *args.test_clip)
            splits["train"] += train
            splits["test"] += test
        train_others, test_others = _split_others(others, args)
    else:
        train_targets, test_targets = _split_others(targets, args)
        splits["train"] += [w for r in train_targets for w in dataset.windows(r, **cut)]
        splits["test"] += [w for r in test_targets for w in dataset.windows(r, **cut)]
        train_others, test_others = _split_others(others, args)
    background = {"train": [w for r in train_others for w in dataset.windows(r, **cut)],
                  "test": [w for r in test_others for w in dataset.windows(r, **cut)]}
    if args.mix:
        target_train = [w for w in splits["train"] if _is_target(w.recording, args)]
        splits["train"] += mix.mixed_windows(target_train, background["train"], args.mix, seed=args.seed)
    for name in splits:
        splits[name] += background[name]
    manifest = dataset.export(splits, Path(args.out))
    for name, items in splits.items():
        positives = sum(_is_target(w.recording, args) for w in items)
        print(f"{name}: {positives} target windows ({sum(w.snr_db is not None for w in items)} mixed), "
              f"{len(items) - positives} other windows")
    print(manifest)


def command_baseline(args):
    config = tonal.ToneConfig(min_hz=args.min_hz, max_hz=args.max_hz)
    rows = dataset.read_manifest(Path(args.manifest))
    base = Path(args.manifest).parent
    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) + ["score", "f0_hz"] if rows else ["score"])
        writer.writeheader()
        for row in rows:
            samples, rate = dataset.read_wav(base / row["file"])
            detector = tonal.TonalDetector(config)
            detector.feed_16k(tonal.to_16k(samples[:, 0], rate))
            tone = detector.tone()
            writer.writerow({**row, "score": "" if tone is None else round(tone.score_db, 4),
                             "f0_hz": "" if tone is None else tone.f0_hz})
    print(f"{len(rows)} windows scored: {out}")


def command_evaluate(args):
    rows = [row for row in dataset.read_manifest(Path(args.scores))
            if (not args.split or row.get("split") == args.split) and row.get("score") not in ("", None)]
    if not rows:
        raise SystemExit("no scored windows (check --split)")
    hop = float(rows[0].get("hop_s") or 0.25)
    items = [evaluate.Scored(float(row["score"]), row["label"] == args.target, row) for row in rows]
    by = tuple(key for key in (args.by or "").split(",") if key)
    result = evaluate.evaluate(items, hop, threshold=args.threshold,
                               false_alarms_per_hour_target=args.false_alarms_per_hour, by=by)
    print(evaluate.report(result))


def command_tensors(args):
    """<out>/<split>.npz: x (N, 128, 128) int8 [channel, step], y class index, classes, files, scale, gain."""
    rows = dataset.read_manifest(Path(args.manifest))
    if not rows:
        raise SystemExit("the manifest has no windows")
    classes = [name for name in (args.classes or f"{args.target},not_{args.target}").split(",") if name]
    if len(classes) < 2:
        raise SystemExit("--classes needs two or more names")
    base = Path(args.manifest).parent
    splits = {}
    for row in rows:
        samples, rate = dataset.read_wav(base / row["file"])
        if rate != tonal.RATE_HZ:
            raise SystemExit(f"{row['file']}: {rate} Hz, but the model input is {tonal.RATE_HZ} Hz")
        if len(samples) < model_input.KWS20_SAMPLES:
            raise SystemExit(f"{row['file']}: {len(samples)} samples, but the model takes "
                             f"{model_input.KWS20_SAMPLES}: export with --length 1.024")
        window = model_input.window_int8(samples[:, 0], args.scale, args.gain)
        split = splits.setdefault(row["split"], {"x": [], "y": [], "files": []})
        split["x"].append(model_input.kws20_tensor(window))
        # A label that is not in --classes goes to the last class.
        split["y"].append(classes.index(row["label"]) if row["label"] in classes else len(classes) - 1)
        split["files"].append(row["file"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        path = out / f"{name}.npz"
        y = np.array(split["y"], dtype=np.int64)
        np.savez_compressed(path, x=np.stack(split["x"]), y=y, classes=np.array(classes),
                            files=np.array(split["files"]), scale=args.scale, gain=args.gain)
        counts = ", ".join(f"{label} {int(np.sum(y == index))}" for index, label in enumerate(classes))
        print(f"{name}: {len(y)} windows ({counts}): {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, help="dataset folder (default: the backend's)")
    parser.add_argument("--target", default="drone", help="the label class to detect (default drone)")
    parser.add_argument("--source", default="", help="only target recordings whose label source contains this")
    parser.add_argument("--channel", type=int, default=0, help="0 = steered beam (default), 1 = Ch1")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="the recordings and their labels")
    p = commands.add_parser("align", help="where each target recording is in the clip")
    p.add_argument("--clip", required=True)
    p = commands.add_parser("export", help="labelled 16 kHz windows, split, in folders with a manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--length", type=float, default=1.0, help="window length in s (default 1.0)")
    p.add_argument("--hop", type=float, default=0.25, help="window step in s (default 0.25)")
    p.add_argument("--clip", help="the clip played by the speaker: split the target windows by clip time "
                   "(--test-key or --test-share then split only the other recordings)")
    p.add_argument("--test-clip", type=float, nargs=2, metavar=("FROM", "TO"),
                   help="seconds of the clip that only the test set gets")
    p.add_argument("--test-key", help="split recordings by this label field (session, room, date, level ...)")
    p.add_argument("--test", nargs="*", default=[], help="the values of --test-key that go to the test set")
    p.add_argument("--test-share", type=float, default=0.3,
                   help="without --test-key: the share of recordings (by duration) for the test set")
    p.add_argument("--mix", type=float, nargs="*", help="also mix the training targets over training "
                   "backgrounds at these SNRs in dB")
    p.add_argument("--seed", type=int, default=0)
    p = commands.add_parser("baseline", help="score every exported window with the tonal detector")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-hz", type=float, default=80.0)
    p.add_argument("--max-hz", type=float, default=800.0)
    p = commands.add_parser("evaluate", help="detection and false alarms of a scores file")
    p.add_argument("--scores", required=True)
    p.add_argument("--split", default="test", help="evaluate only this split (default test; '' = all)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--threshold", type=float)
    group.add_argument("--false-alarms-per-hour", type=float)
    p.add_argument("--by", default="", help="comma-separated manifest columns to break down by")
    p = commands.add_parser("tensors", help="exported windows as the CNN's int8 input (one .npz per split)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="folder for <split>.npz")
    p.add_argument("--classes", help="network outputs in order, comma-separated (default <target>,not_<target>); "
                   "other labels go to the last class")
    p.add_argument("--scale", choices=model_input.SCALES, default="gain",
                   help="int16 to int8 as MODEL_INPUT_SCALE in model_config.h (default gain)")
    p.add_argument("--gain", type=float, default=1.0, help="MODEL_INPUT_GAIN for --scale gain (default 1.0)")
    args = parser.parse_args(argv)
    {"list": command_list, "align": command_align, "export": command_export, "baseline": command_baseline,
     "evaluate": command_evaluate, "tensors": command_tensors}[args.command](args)


if __name__ == "__main__":
    main()
