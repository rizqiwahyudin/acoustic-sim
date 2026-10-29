"""Track-before-detect (TBD) study on the Live logs: does integrating the evidence over maps detect weaker
steady sources than the firmware, without more targets in quiet rooms?

    python scripts/tbd_study.py      (reads MAX78002/docs/data; about 10 minutes)

TrackDetector runs next to the firmware detector (Detector, bit-exact) and reports the union. Per beam b, map t:
    x_t(b) = 10 log10(level / max(background, field median)), 0 at or below the level floor
    S_t(b) = max(0, decay * max of S_{t-1} over the 3 x 3 neighbourhood of b + min(x_t(b), clip) - drift)
A TBD target is the beam with the largest S when S >= threshold (dB summed over maps). While S >= hold, the beam's
background takes the held update (freeze, then shift 9), so a weak steady source is not learnt. The defaults
(threshold 15, drift 0.75, hold 1.5) are the best of the settings tried on 2026-10-29.
Quiet maps: the silent runs of 2026-10-28, the seconds before each sound on 2026-10-26, the quiet parts of the
logs of 2026-10-23 (the afternoon log had people in the room and is reported apart). Weak sources: the 3 dB level
steps of 2026-10-28 (speaker at 2 m straight ahead, beam 24).
"""
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_log import COLUMNS, ROWS, Detector, load_log  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "MAX78002" / "docs" / "data"
SPEAKER_BEAM = 24


class TrackDetector(Detector):
    def __init__(self, threshold=15.0, drift=0.75, decay=1.0, clip=6.0, hold=1.5, **settings):
        super().__init__(**settings)
        self.threshold, self.drift, self.decay, self.clip = threshold, drift, decay, clip
        self.hold = threshold / 2 if hold is None else hold
        self.score = np.zeros(ROWS * COLUMNS)

    def update(self, levels):
        """(the union's target beam or -1, the firmware's target beam or -1)."""
        levels = np.asarray(levels, np.int64)
        if self.background is None:
            return super().update(levels)
        field = int(np.sort(levels)[len(levels) // 2]) << self.frac
        above = self.above(levels, self.background) & self.above(levels, field)
        grid = levels.reshape(ROWS, COLUMNS)
        firmware, best_ratio = -1, None
        for beam in np.flatnonzero(above):
            r, c = divmod(int(beam), COLUMNS)
            if any(0 <= r + dr < ROWS and 0 <= c + dc < COLUMNS and grid[r + dr, c + dc] > grid[r, c]
                   for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                continue
            ratio = Fraction(int(levels[beam]), max(1, int(self.background[beam]) >> self.frac))
            if best_ratio is None or ratio > best_ratio:
                firmware, best_ratio = int(beam), ratio
        reference = np.maximum(self.background, field).astype(float)
        x = np.where(levels > self.floor,
                     10 * np.log10(np.maximum((levels << self.frac) / np.maximum(reference, 1.0), 1e-6)), 0.0)
        padded = np.pad(self.score.reshape(ROWS, COLUMNS), 1)
        spread = np.max([padded[1 + dr:1 + dr + ROWS, 1 + dc:1 + dc + COLUMNS]
                         for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=0).ravel()
        self.score = np.maximum(0.0, self.decay * spread + np.minimum(x, self.clip) - self.drift)
        tracked = int(np.argmax(self.score)) if self.score.max() >= self.threshold else -1
        held = above | (self.score >= self.hold)
        difference = (levels << self.frac) - self.background
        step = np.where(difference >= 0, difference >> np.where(held, self.held_shift, self.shift),
                        -((-difference) >> np.where(held, self.held_shift, self.shift)))
        frozen = held & (self.held < self.freeze_maps)
        self.held = np.where(frozen, self.held + 1, np.where(held, self.held, 0))
        self.background = self.background + np.where(frozen, 0, step)
        return (firmware if firmware >= 0 else tracked), firmware


def near(beam, reference=SPEAKER_BEAM):
    return beam >= 0 and abs(beam // COLUMNS - reference // COLUMNS) <= 1 and \
        abs(beam % COLUMNS - reference % COLUMNS) <= 1


def replay(levels, make):
    detector = make()
    return np.array([detector.update(row)[0] for row in levels])


def onset(levels, t):
    top = levels.max(axis=1)
    early = (t > 0.5) & (t < 3.0)
    quiet = np.median(top[early]) if early.any() else np.median(top[:20])
    loud = top >= 10 * quiet
    hits = np.flatnonzero(loud[:-2] & loud[1:-1] & loud[2:])
    return int(hits[0]) if hits.size else len(top)


def quiet_sets():
    """{group: [(levels of one Live start, mask of its quiet maps)]}."""
    groups = {"quiet rooms": [], "23 Oct afternoon (people in the room)": []}
    silent = load_log(DATA / "bench_2026-10-28" / "silent-live" / "l7_maps.npz")
    silent += [r for r in load_log(DATA / "bench_2026-10-28" / "detection" / "l5_maps.npz") if r["label"] == "silent"]
    for run in silent:
        levels = np.asarray(run["levels"], np.int64)
        groups["quiet rooms"].append((levels, np.arange(len(levels)) >= 2))
    for folder in ("live-log_3", "live-log_4", "live-log_5"):
        for run in load_log(DATA / "bench_2026-10-26" / folder / "l7_maps.npz"):
            levels = np.asarray(run["levels"], np.int64)
            t = np.asarray(run["t"]) - run["t"][0]
            groups["quiet rooms"].append((levels, (t > 1.0) & (np.arange(len(levels)) < onset(levels, t) - 5)))
    for folder, group in (("live-log", "quiet rooms"), ("live-log-new-fw", "23 Oct afternoon (people in the room)")):
        for run in load_log(DATA / "bench_2026-10-23" / folder / "l7_maps.npz"):
            levels = np.asarray(run["levels"], np.int64)
            t = np.asarray(run["t"])
            mask = (t - t[0]) > 1.0
            events = run.get("events", {})
            if "burst_on" in events:
                mask &= (t < events["burst_on"]) | (t > events.get("burst_off", t[-1]) + 5.0)
            groups[group].append((levels, mask))
    return groups


def level_runs():
    runs = []
    for run in load_log(DATA / "bench_2026-10-28" / "detection" / "l5_maps.npz"):
        match = re.match(r"level (-?\d+) dB, trial", run["label"])
        if match:
            t = np.asarray(run["t"]) - run["t"][0]
            runs.append((int(match.group(1)), np.asarray(run["levels"], np.int64), t > 2.8))
    return runs


def detection_curve(make, runs):
    rates = {}
    for level, levels, mask in runs:
        rates.setdefault(level, []).append(np.mean([near(b) for b in replay(levels, make)[mask]]))
    curve = {level: float(np.mean(v)) for level, v in sorted(rates.items(), reverse=True)}
    points = list(curve.items())
    fifty = next((l1 + (l2 - l1) * (r1 - 0.5) / (r1 - r2)
                  for (l1, r1), (l2, r2) in zip(points, points[1:]) if r1 >= 0.5 > r2), None)
    return curve, fifty


VARIANTS = [("firmware", Detector)] + [
    (f"TBD {threshold:g} dB, drift {drift:g} dB, hold {hold:g} dB",
     lambda th=threshold, dr=drift, ho=hold: TrackDetector(th, dr, hold=ho))
    for threshold, drift, hold in ((15, 0.75, 1.5), (12, 0.75, 1.5), (12, 0.75, 6), (9, 0.5, 4.5))]

if __name__ == "__main__":
    groups, runs = quiet_sets(), level_runs()
    for name, make in VARIANTS:
        cells = []
        for group, sets in groups.items():
            maps = sum(int(mask.sum()) for _, mask in sets)
            hits = sum(int(np.sum(replay(levels, make)[mask] >= 0)) for levels, mask in sets)
            cells.append(f"{group}: {hits} of {maps} maps ({hits / (maps * 0.05 / 60):.1f}/min)")
        curve, fifty = detection_curve(make, runs)
        print(f"{name}: 50% at {fifty:.1f} dB; " + ", ".join(f"{l} dB {100 * r:.0f}%" for l, r in curve.items()
                                                            if -36 <= l <= -27))
        print("   targets in quiet: " + "; ".join(cells))
