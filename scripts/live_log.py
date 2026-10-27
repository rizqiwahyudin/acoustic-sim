"""Live (bank) logs: save, load, replay the firmware's detector with other settings, and analyse.

    python scripts/live_log.py <folder> [old|nofreeze] [--speaker AZ EL]
        analyses l7_maps.npz and/or l5_maps.npz in the folder; writes live_log.json. 'old': the log was made with
        the detector before 2026-10-23 (integer backgrounds, no level floor); 'nofreeze': before 2026-10-27 (no
        freeze_s). --speaker: the speaker's direction
        (deg), to score the firmware target and the pattern fit (bank_fit.py) against it.

A run is the maps of one Live start (B). The replay is exact only when the log has every map from the start (the
first map primes the backgrounds); the analysis checks this against the targets that the board reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bank_fit import BankFit, BankFitStream  # noqa: E402
from heimdall_acoustic import default_contract_path  # noqa: E402
from heimdall_bank import FREEZE_MAPS  # noqa: E402

ROWS = COLUMNS = 7
MAP_S = 0.05  # HEIMDALL_BANK_PERIOD_US
ON_TARGET_DEG = 12.5  # about one bank cell
FAR_JUMP_DEG = 20.0
OLD_FIRMWARE = {"frac_bits": 0, "floor": 0, "freeze_maps": 0}
NO_FREEZE = {"freeze_maps": 0}  # firmware before freeze_s (2026-10-27)
SETTINGS = {
    "firmware (1.5x, 8 fraction bits, floor 4, freeze 60 s)": {},
    "no freeze": NO_FREEZE,
    "old firmware (integer background, floor 0)": OLD_FIRMWARE,
    "floor 8 words": {"floor": 8},
    "ratio 2x": {"ratio": (2, 1)},
    "2 maps in a row": {"confirm": 2},
    "slow rise (shift 9 up, 5 down)": {"rise_shift": 9},
}


def save_log(path, runs):
    """runs: dicts with label, start_maps, maps, t, levels, targets [(maps, az, el, beam, second)], events {}."""
    meta = [{"label": r["label"], "start_maps": r["start_maps"], "events": r.get("events", {})} for r in runs]
    run = np.concatenate([np.full(len(r["maps"]), i, np.int32) for i, r in enumerate(runs)] or [np.zeros(0, np.int32)])
    targets = [(i, *t) for i, r in enumerate(runs) for t in r["targets"]]
    np.savez_compressed(
        path, meta=np.array(json.dumps(meta)), run=run,
        maps=np.concatenate([np.asarray(r["maps"], np.int64) for r in runs] or [np.zeros(0, np.int64)]),
        t=np.concatenate([np.asarray(r["t"], float) for r in runs] or [np.zeros(0)]),
        levels=np.concatenate([np.asarray(r["levels"], np.int64).reshape(-1, ROWS * COLUMNS) for r in runs]
                              or [np.zeros((0, ROWS * COLUMNS), np.int64)]),
        targets=np.asarray(targets, float).reshape(-1, 6))


def load_log(path):
    data = np.load(path)
    runs = []
    for i, meta in enumerate(json.loads(str(data["meta"]))):
        keep = data["run"] == i
        own = data["targets"][data["targets"][:, 0] == i]
        runs.append({**meta, "maps": data["maps"][keep], "t": data["t"][keep], "levels": data["levels"][keep],
                     "targets": [(int(m), az, el, int(b), int(s)) for _, m, az, el, b, s in own]})
    return runs


class Detector:
    """MAX78002/beam_bank.c with its settings as parameters. With the defaults it is the firmware, bit for bit
    (NO_FREEZE: before 2026-10-27, OLD_FIRMWARE: before 2026-10-23). frac_bits: fraction bits of the backgrounds;
    floor: a level must be above it; freeze_maps: a beam above its background keeps it this many maps in a row;
    confirm reports a target only when a candidate was found in that many maps in a row, each within one row and
    column of the one before; rise_shift makes a quiet beam's background rise more slowly than it falls."""

    def __init__(self, ratio=(3, 2), shift=5, held_shift=9, frac_bits=8, confirm=1, rise_shift=None, floor=4,
                 freeze_maps=FREEZE_MAPS):
        self.num, self.den = ratio
        self.shift, self.held_shift, self.frac, self.confirm = shift, held_shift, frac_bits, confirm
        self.rise_shift = shift if rise_shift is None else rise_shift
        self.floor, self.freeze_maps = floor, freeze_maps
        self.background = self.held = None
        self.streak, self.last = 0, -1

    def above(self, levels, reference):
        """level >= ratio x reference, reference in the background's scale."""
        return (levels > self.floor) & (levels * self.den * (1 << self.frac) >= reference * self.num)

    def update(self, levels):
        """One map; returns (best beam or -1 when no target is reported, the candidates' raw best beam or -1)."""
        levels = np.asarray(levels, np.int64)
        if self.background is None:
            self.background = levels << self.frac
            self.held = np.zeros(len(levels), np.int64)
            return -1, -1
        field = int(np.sort(levels)[len(levels) // 2]) << self.frac
        above = self.above(levels, self.background) & self.above(levels, field)
        grid = levels.reshape(ROWS, COLUMNS)
        best, best_ratio = -1, None
        for beam in np.flatnonzero(above):
            r, c = divmod(int(beam), COLUMNS)
            if any(0 <= r + dr < ROWS and 0 <= c + dc < COLUMNS and grid[r + dr, c + dc] > grid[r, c]
                   for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                continue
            ratio = Fraction(int(levels[beam]), max(1, int(self.background[beam]) >> self.frac))
            if best_ratio is None or ratio > best_ratio:
                best, best_ratio = int(beam), ratio
        difference = (levels << self.frac) - self.background
        shift = np.where(above, self.held_shift, np.where(difference > 0, self.rise_shift, self.shift))
        step = np.where(difference >= 0, difference >> shift, -((-difference) >> shift))
        frozen = above & (self.held < self.freeze_maps)
        self.held = np.where(frozen, self.held + 1, np.where(above, self.held, 0))
        self.background = self.background + np.where(frozen, 0, step)
        near = best >= 0 and self.last >= 0 and all(abs(a - b) <= 1 for a, b in zip(divmod(best, COLUMNS),
                                                                                   divmod(self.last, COLUMNS)))
        self.streak = 0 if best < 0 else (self.streak + 1 if near or self.streak == 0 else 1)
        self.last = best
        return (best if self.streak >= self.confirm else -1), best

    def threshold(self):
        """Per beam: the level a new sound must reach (ratio x background), in raw level words."""
        return self.background * self.num / self.den / (1 << self.frac)


def replay(run, **settings):
    """Targets (best beam or -1) and thresholds for every logged map of a run."""
    detector = Detector(**settings)
    best, thresholds = [], []
    for levels in run["levels"]:
        best.append(detector.update(levels)[0])
        thresholds.append(detector.threshold())
    return np.asarray(best), np.asarray(thresholds)


def complete(run):
    """True when the log has every map from the priming map on."""
    maps = np.asarray(run["maps"])
    return maps.size > 0 and maps[0] == run["start_maps"] + 1 and bool(np.all(np.diff(maps) == 1))


def agreement(run, best):
    """Share of the board's target reports that the replay also reports, on the same map with the same beam."""
    index = {int(m): i for i, m in enumerate(run["maps"])}
    hits = [best[index[m]] == beam for m, _, _, beam, _ in run["targets"] if m in index]
    return (float(np.mean(hits)) if hits else float("nan")), len(hits)


def rate(best, keep):
    keep = np.asarray(keep, bool)
    return float(np.mean(best[keep] >= 0)) if keep.any() else float("nan")


def analyse(folder, firmware=None, speaker=None):
    """firmware: the detector settings of the board that made the log ({} = now, NO_FREEZE = before 27 Oct,
    OLD_FIRMWARE = before 23 Oct). speaker: (azimuth, elevation) in degrees, when known."""
    folder = Path(folder)
    result = {}
    for name in ("l7", "l5"):
        path = folder / f"{name}_maps.npz"
        if path.exists():
            result[name] = analyse_runs(name, load_log(path), firmware or {}, speaker)
    (folder / "live_log.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def analyse_runs(name, runs, firmware, speaker=None):
    print(f"{name}: {len(runs)} runs")
    out = {"runs": [], "settings": {}}
    for run in runs:
        best, _ = replay(run, **firmware)
        share, count = agreement(run, best)
        full = complete(run)
        expected = int(run["maps"][-1] - run["start_maps"]) if len(run["maps"]) else 0
        out["runs"].append({"label": run["label"], "maps": len(run["maps"]), "expected": expected,
                            "complete": full, "board_targets": count, "replay_agrees": share})
        print(f"  {run['label']}: {len(run['maps'])}/{expected} maps, complete {full}, replay agrees with "
              f"{share:.0%} of {count} board targets")
    for label, settings in SETTINGS.items():
        out["settings"][label] = summary(name, runs, settings)
        print(f"  {label}: " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
                                         for k, v in out["settings"][label].items() if not isinstance(v, (list, dict))))
    out["directions"] = directions(runs)
    print("  most frequent target beams (firmware): " + ", ".join(
        f"beam {d['beam']} ({d['azimuth_deg']:+.0f}, {d['elevation_deg']:+.0f} deg) {d['share']:.0%}"
        for d in out["directions"][:6]))
    out["fit"] = fit_summary(runs, firmware, speaker)
    if out["fit"]:
        print("  pattern fit: " + ", ".join(f"{k} {v:.3g}" for k, v in out["fit"].items()))
    return out


def load_fit(contract_path=None):
    """The pattern fit and bank beam directions of the deployed contract; (None, None) when unusable."""
    try:
        contract = json.loads(Path(contract_path or default_contract_path()).read_text(encoding="utf-8"))
        fit = BankFit.from_contract(contract)
    except (OSError, ValueError, KeyError, TypeError):
        return None, None
    if fit is None:
        return None, None
    bank = contract["bank"]
    return fit, np.column_stack([bank["azimuth_deg"], bank["elevation_deg"]])


def fit_summary(runs, firmware=None, speaker=None, contract_path=None):
    """On the maps where the firmware reports a target: how often the pattern fit's direction (over the recent maps,
    as the backend shows it) agrees with the firmware's beam, far jumps per minute of each, and with a speaker
    direction the share of each on it."""
    fit, beams = load_fit(contract_path)
    if fit is None or not runs or np.shape(runs[0]["levels"])[1] != fit.beams:
        return {}
    firmware_dirs, fit_dirs, jumps, minutes = [], [], {"firmware": 0, "fit": 0}, 0.0
    for run in runs:
        best, _ = replay(run, **(firmware or {}))
        minutes += len(best) * MAP_S / 60.0
        stream, previous = BankFitStream(fit), None
        results = [stream.push(count, levels) for count, levels in zip(run["maps"], run["levels"])]
        for index in np.flatnonzero(best >= 0):
            result = results[index]
            if result is None:
                continue
            pair = (beams[best[index]], np.array([result.azimuth_deg, result.elevation_deg]))
            firmware_dirs.append(pair[0])
            fit_dirs.append(pair[1])
            if previous is not None:
                for key, now, before in (("firmware", pair[0], previous[0]), ("fit", pair[1], previous[1])):
                    jumps[key] += bool(np.hypot(*(now - before)) > FAR_JUMP_DEG)
            previous = pair
    if not firmware_dirs:
        return {}
    firmware_dirs, fit_dirs = np.array(firmware_dirs), np.array(fit_dirs)
    near = lambda a, b: np.all(np.abs(a - b) <= ON_TARGET_DEG, axis=1)  # noqa: E731
    out = {"target maps": float(len(fit_dirs)), "fit agrees with firmware": float(np.mean(near(fit_dirs, firmware_dirs))),
           "far jumps per min, firmware": jumps["firmware"] / minutes, "far jumps per min, fit": jumps["fit"] / minutes}
    if speaker is not None:
        speaker = np.asarray(speaker, float)
        out["on the speaker, firmware"] = float(np.mean(near(firmware_dirs, speaker)))
        out["on the speaker, fit"] = float(np.mean(near(fit_dirs, speaker)))
    return out


def summary(name, runs, settings):
    """l7: target rate by time since the start and since the burst, and the threshold rise after the burst.
    l5: per level, the share of maps with a target within one beam of the speaker; in silence, with any target."""
    if name == "l7":
        bins = ((0, 2), (2, 10), (10, 30), (30, 1e9))
        since_start = {f"{a:g}-{b:g} s": [] for a, b in bins}
        after = {f"{a:g}-{b:g} s": [] for a, b in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9))}
        rise, recovery = [], []
        for run in runs:
            best, thresholds = replay(run, **settings)
            t = np.asarray(run["t"]) - run["t"][0]
            on, off = run["events"].get("burst_on"), run["events"].get("burst_off")
            quiet = np.ones(len(t), bool) if on is None else (t < on - run["t"][0])
            for (a, b), key in zip(bins, since_start):
                since_start[key].append(rate(best, quiet & (t >= a) & (t < b)))
            if on is None:
                continue
            ts = t - (off - run["t"][0])
            for key in after:
                a, b = (float(v) for v in key[:-2].split("-"))
                after[key].append(rate(best, (ts >= a) & (ts < b)))
            before = np.median(thresholds[quiet & (t >= 2)], axis=0)
            excess = 20 * np.log10(np.median(thresholds / np.maximum(before, 1), axis=1))
            during = np.flatnonzero(t >= on - run["t"][0])
            if during.size == 0:
                continue
            peak = during[int(np.argmax(excess[during]))]
            rise.append(float(excess[peak]))
            back = np.flatnonzero((np.arange(len(t)) > peak) & (excess <= 1.0))
            recovery.append(float(ts[back[0]]) if back.size else float("nan"))
        return {**{f"rate {k} after start": float(np.nanmean(v)) for k, v in since_start.items()},
                **{f"rate {k} after the burst": float(np.nanmean(v)) for k, v in after.items() if v},
                **({"threshold rise after the burst (dB)": float(np.nanmean(rise)),
                    "back within 1 dB after (s)": float(np.nanmean(recovery))} if rise else {})}
    reference = reference_beam(runs)
    rates = {}
    for run in runs:
        best, _ = replay(run, **settings)
        start = run["events"].get("sound_maps")
        keep = np.asarray(run["maps"]) > (start if start is not None else run["start_maps"] + 40)
        label = run["label"].split(",")[0]
        if label == "silent" or reference < 0:
            rates.setdefault(f"any target, {label}", []).append(rate(best, keep))
        else:
            r0, c0 = divmod(reference, COLUMNS)
            hit = np.array([b >= 0 and abs(b // COLUMNS - r0) <= 1 and abs(b % COLUMNS - c0) <= 1 for b in best])
            rates.setdefault(f"speaker found, {label}", []).append(float(np.mean(hit[keep])) if keep.any() else
                                                                  float("nan"))
    return {k: float(np.nanmean(v)) for k, v in rates.items()}


def reference_beam(runs):
    """The beam that the board reported most often in the runs at the loudest level (label 'level 0 dB')."""
    beams = [t[3] for run in runs if run["label"].startswith("level 0 dB") for t in run["targets"]]
    return int(np.bincount(beams).argmax()) if beams else -1


def directions(runs):
    beams, az, el = {}, {}, {}
    for run in runs:
        for _, a, e, beam, _ in run["targets"]:
            beams[beam] = beams.get(beam, 0) + 1
            az.setdefault(beam, []).append(a)
            el.setdefault(beam, []).append(e)
    total = sum(beams.values()) or 1
    return [{"beam": b, "share": n / total, "azimuth_deg": float(np.mean(az[b])), "elevation_deg": float(np.mean(el[b]))}
            for b, n in sorted(beams.items(), key=lambda kv: -kv[1])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    parser.add_argument("old", nargs="?", choices=["old", "nofreeze"],
                        help="the log is from firmware before 23 Oct (old) or before 27 Oct (nofreeze)")
    parser.add_argument("--speaker", type=float, nargs=2, metavar=("AZ", "EL"))
    args = parser.parse_args()
    analyse(args.folder, {"old": OLD_FIRMWARE, "nofreeze": NO_FREEZE}.get(args.old, {}), args.speaker)
