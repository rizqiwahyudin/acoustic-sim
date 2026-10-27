"""The Live log replay (scripts/live_log.py) against the host copy of the firmware's bank tracker."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import live_log  # noqa: E402
from heimdall_bank import BankTracker  # noqa: E402


def maps(seed=1, count=400):
    """Quiet levels of 120-450 words with fluctuations, clicks in single beams and one loud sound everywhere."""
    rng = np.random.default_rng(seed)
    floor = rng.integers(120, 450, 49)
    levels = (floor * rng.uniform(0.8, 1.25, (count, 49))).astype(np.int64)
    for i in rng.integers(5, count, 25):
        levels[i, rng.integers(0, 49)] *= 3
    levels[200:206] *= 400
    return levels


@pytest.mark.parametrize("settings", [{}, {"freeze_maps": 0}, {"freeze_maps": 5}])
def test_default_replay_is_the_firmware_tracker(settings):
    zeros = [0.0] * 49
    tracker = BankTracker(7, 7, zeros, zeros, zeros[:7], zeros[:7], **settings)
    detector = live_log.Detector(**settings)
    levels = maps()
    levels[250:330, 24] *= 6  # a source held for 4 s
    for row in levels:
        expected = tracker.update([int(v) for v in row]).beams
        assert detector.update(row)[0] == (expected[0] if expected else -1)
    assert [int(v) for v in detector.background] == tracker.background


def test_integer_background_stops_short_of_the_level():
    """The old firmware's shift-5 step is zero within 31 words; fraction bits let the background reach the level."""
    whole, fine = live_log.Detector(**live_log.OLD_FIRMWARE), live_log.Detector()
    loud, quiet = np.full(49, 4000), np.full(49, 150)
    for detector in (whole, fine):
        detector.update(loud)
        for _ in range(400):
            detector.update(quiet)
    assert 150 + 20 <= whole.threshold()[0] / 1.5 <= 150 + 31
    assert abs(fine.threshold()[0] / 1.5 - 150) < 1


def test_confirm_needs_maps_in_a_row():
    detector = live_log.Detector(confirm=2)
    quiet, click = np.full(49, 200), np.full(49, 200)
    click[24] = 900
    detector.update(quiet)
    assert detector.update(click)[0] == -1
    assert detector.update(click)[0] == 24


def test_log_round_trip_and_checks(tmp_path):
    levels = maps(count=30)
    run = {"label": "start 1", "start_maps": 10, "maps": list(range(11, 41)), "t": list(np.arange(30) * 0.05),
           "levels": levels.tolist(), "targets": [(20, 1.0, 2.0, 24, 23)], "events": {"burst_on": 1.0}}
    live_log.save_log(tmp_path / "l7_maps.npz", [run])
    loaded = live_log.load_log(tmp_path / "l7_maps.npz")[0]
    assert loaded["label"] == "start 1" and loaded["events"] == {"burst_on": 1.0}
    assert np.array_equal(loaded["levels"], levels) and loaded["targets"] == [(20, 1.0, 2.0, 24, 23)]
    assert live_log.complete(loaded)
    assert not live_log.complete({**loaded, "maps": np.array([11, 12, 14])})


def test_fit_summary_scores_the_firmware_target_and_the_pattern_fit():
    fit, _ = live_log.load_fit()
    if fit is None:
        return
    quiet = np.full(fit.beams, 40.0)
    source = 2000.0 * fit.fingerprint(-15.0, 3.0) + 40.0
    levels = np.vstack([np.tile(quiet, (20, 1)), np.tile(source, (40, 1))]).round().astype(np.int64)
    run = {"label": "start 1", "start_maps": 0, "maps": np.arange(1, 61), "t": np.arange(60) * 0.05,
           "levels": levels, "targets": [], "events": {}}
    summary = live_log.fit_summary([run], speaker=(-15.0, 3.0))
    assert summary["target maps"] > 0
    assert summary["on the speaker, fit"] == 1.0
    assert summary["far jumps per min, fit"] == 0.0
