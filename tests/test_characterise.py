"""scripts/characterise.py: the noise models behind N1 and N2 (no hardware)."""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import characterise  # noqa: E402
from array_theory import band_directivity_index_db, band_frequencies, first_null_half_width, steering_cut_db  # noqa: E402


@pytest.fixture(scope="module")
def array():
    return characterise.Array()


def test_diffuse_noise_lies_between_independent_and_common(array):
    assert array.diffuse_sum([0]) == pytest.approx(1.0)
    everyone = list(range(array.contract.microphones))
    total = array.diffuse_sum(everyone)
    assert len(everyone) < total < len(everyone) ** 2


def test_steering_does_not_change_noise_at_broadside(array):
    common, diffuse = array.steered(0.0, 0.0)
    assert common == pytest.approx(0.0, abs=1e-9)
    assert diffuse == pytest.approx(0.0, abs=1e-9)


def test_common_noise_drops_when_steered_away(array):
    common, diffuse = array.steered(30.0, 0.0)
    assert common < -6.0
    assert abs(diffuse) < 3.0


def test_find_reaches_nested_keys():
    assert characterise.find({"a": {"b": {"gain_addresses": [1, 2]}}}, "gain_addresses") == [1, 2]
    assert characterise.find({"a": 1}, "missing") is None


def test_positions_and_direction_sets():
    az, el, r = characterise.position_angles(1.0, 1.0, 0.0)
    assert (az, el, r) == pytest.approx((45.0, 0.0, 2 ** 0.5))
    assert characterise.position_angles(0.0, 2.0, 2.0)[1] == pytest.approx(45.0)
    assert characterise.parse_positions("-30,0; 0,7.5") == [(-30.0, 0.0), (0.0, 7.5)]
    assert characterise.axis_spec("azimuth", 2.0, -1.0, 1.0, 24.0) == (-22.0, 26.0, 49, -1.0, -1.0, 1)
    assert characterise.axis_spec("elevation", 0.0, 0.0, 2.0, 48.0)[3:] == (-48.0, 48.0, 49)


def test_theory_template_peaks_at_the_speaker():
    contract = characterise.load_beam_contract()
    rel, db = steering_cut_db(contract, (10.0, 0.0), "azimuth", [2000.0], None)
    assert rel[int(np.argmax(db))] == pytest.approx(0.0, abs=0.05)


def test_live_prediction_is_exact_on_a_bank_beam_and_symmetric():
    contract = characterise.load_beam_contract()
    bank = characterise.find(json.loads(Path(characterise.default_contract_path()).read_text(encoding="utf-8")), "bank")
    az, el = characterise.bank_prediction(bank, contract, 0.0, 0.0, None)
    assert (az, el) == pytest.approx((0.0, 0.0), abs=1e-6)
    left = characterise.bank_prediction(bank, contract, -7.5, 0.0, None)
    right = characterise.bank_prediction(bank, contract, 7.5, 0.0, None)
    assert left[0] == pytest.approx(-right[0], abs=1e-6)
    assert 0.0 < right[0] < 15.0


def test_decay_time_of_a_synthetic_room():
    rate, rt60, stop = 48000, 0.5, 1.5
    t = np.arange(int(4.0 * rate)) / rate
    noise = np.random.default_rng(1).normal(size=t.size)
    envelope = np.where(t < stop, 1.0, 10 ** (-3 * (t - stop) / rt60))
    x = noise * envelope + 1e-4 * np.random.default_rng(2).normal(size=t.size)
    result = characterise.decay_time(x, 1000.0, rate, stop)
    assert result["method"] == "T20"
    assert result["rt60_s"] == pytest.approx(rt60, rel=0.1)


def test_l90_and_summaries():
    assert characterise.l90(list(range(101))) == pytest.approx(10.0)
    mean, se = characterise.summarise([1.0, 3.0])
    assert (mean, se) == pytest.approx((2.0, 1.0))


def test_band_directivity_and_the_diffuse_field_formula():
    contract = characterise.load_beam_contract()
    assert band_directivity_index_db(contract, band_frequencies(2000.0)) == pytest.approx(14.8, abs=0.15)
    assert band_directivity_index_db(contract, band_frequencies(1000.0)) == pytest.approx(10.0, abs=0.15)
    beam = 44 ** 2 / 30.0 * 0.1
    assert characterise.directivity_from_levels(44, 0.1, 1.0, beam) == pytest.approx(30.0)
    assert np.isnan(characterise.excess(-60.0, -50.0))


def test_highpass_for_the_aliasing_test():
    taps = characterise.highpass_taps(3500.0)
    response = lambda f: 20 * np.log10(abs(np.sum(taps * np.exp(-2j * np.pi * f / 48000 * np.arange(11)))))
    assert response(0.0) < -60
    assert response(500.0) < -25
    assert -2.0 < response(8000.0) < 1.0
    assert response(5000.0) > -6.0
    assert characterise.to_word(-1.0) == 0xFF000000
    assert characterise.to_word(0.5) == 0x00800000


def test_grating_lobes_appear_at_8k_not_at_5k():
    contract = characterise.load_beam_contract()
    for centre, expect in ((5000.0, False), (8000.0, True)):
        rel, db = steering_cut_db(contract, (0.0, 0.0), "azimuth", band_frequencies(centre), None)
        half = first_null_half_width(rel, db)
        lobes = [lobe for lobe in characterise.grating_lobes(rel, db, half, -6.0) if abs(lobe[0]) <= 60.0]
        assert bool(lobes) is expect
        if expect:
            assert all(35.0 < abs(angle) < 60.0 for angle, _ in lobes)


def test_arrival_fraction_and_latency_fit():
    assert characterise.arrival_fraction(-20.0, -20.0, -40.0) == pytest.approx(1.0)
    assert characterise.arrival_fraction(-40.0, -20.0, -40.0) == pytest.approx(0.0)
    halfway = 10 * np.log10((10 ** -2 + 10 ** -4) / 2)
    assert characterise.arrival_fraction(halfway, -20.0, -40.0) == pytest.approx(0.5)
    guards = np.array(characterise.P2_GUARDS_US, float)
    fractions = np.clip((guards - 1500.0) / 2000.0, 0, 1) + np.random.default_rng(3).normal(0, 0.03, guards.size)
    start, ramp = characterise.arrival_fit(guards, fractions)
    assert start + ramp == pytest.approx(3500.0, abs=250.0)
    assert np.isnan(characterise.arrival_fit(guards, np.full(guards.size, np.nan))).all()


def test_detection_threshold_interpolates_and_flags_censoring():
    levels = [0, -6, -12, -18, -24, -30]
    assert characterise.threshold_db(levels, [1, 1, 0.9, 0.6, 0.3, 0.1]) == (pytest.approx(-20.0), False)
    assert characterise.threshold_db(levels, [1, 1, 1, 1, 0.9, 0.8]) == (-30.0, True)
    assert characterise.threshold_db(levels, [0.2, 0.1, 0, 0, 0, 0]) == (None, False)


def test_scan_pass_detection_rule():
    flat = np.full((7, 7), -40.0)
    flat[3, 4] = -34.0
    assert characterise.scan_detects(flat, 24, 7)
    assert not characterise.scan_detects(flat, 0, 7)
    flat[3, 4] = -38.0
    assert not characterise.scan_detects(flat, 24, 7)
    assert not characterise.scan_alarm(flat)


def test_detector_models_tell_amplitude_from_power():
    played = np.arange(0, -49, -6.0)
    floor_db = -60.0
    floor = 10 ** (floor_db / 20)
    amplitude = 20 * np.log10(np.sqrt((10 ** (-20 / 20) * 10 ** (played / 20)) ** 2 + floor ** 2))
    power_word = 20 * np.log10(10 ** (-20 / 20) * 10 ** (played / 10) + floor)
    errors = np.full(played.size, 0.2)
    fits = characterise.detector_models(played, amplitude, errors, floor_db)
    assert fits["amplitude"]["chi2"] < 0.01 < fits["power"]["chi2"]
    fits = characterise.detector_models(played, power_word, errors, floor_db)
    assert fits["power"]["chi2"] < 0.01 < fits["amplitude"]["chi2"]
    slope, _, _, _ = characterise.level_slope(played[:4], 20 * np.log10(0.1 * 10 ** (played[:4] / 10)), errors[:4])
    assert slope == pytest.approx(2.0, abs=1e-6)


def test_tone_stats_and_map_hits():
    stats = characterise.tone_stats([(200.0, 12.0, 6, -40.0), (202.0, 14.0, 7, -41.0), (198.0, 10.0, 5, -39.0)])
    assert stats["reports"] == 3 and stats["f0_median_hz"] == 200.0 and stats["score_median_db"] == 12.0
    assert characterise.tone_stats([]) == {"reports": 0}
    targets = [(5, 1.0, 0.0, 12, 13), (6, 30.0, 0.0, 14, 12), (6, 2.0, 1.0, 12, 13)]
    assert characterise.map_hits(targets, (0.0, 0.0)) == {5, 6}


def test_parse_numbers_accepts_decimal_commas():
    assert characterise.parse_numbers("72,5", 1) == [72.5]
    assert characterise.parse_numbers(" 50 ", 1) == [50.0]
    for text in ("152.5 200 -10", "152,5 200 -10", "152.5, 200, -10", "152.5,200,-10", "152.5;200;-10"):
        assert characterise.parse_numbers(text, 3) == [152.5, 200.0, -10.0], text
    for text, count in (("152,5,200,10", 3), ("abc", 1), ("5 x 4", 1), ("", 1)):
        with pytest.raises(ValueError):
            characterise.parse_numbers(text, count)


def test_real_drift_ignores_reading_scatter():
    rng = np.random.default_rng(1)
    flagged = sum(characterise.real_drift(rng.normal(0.0, 0.3, 6), [0.3] * 6) for _ in range(2000))
    assert flagged / 2000 < 0.02
    assert characterise.real_drift(np.linspace(0.0, -1.0, 6), [0.1] * 6)
    assert not characterise.real_drift(np.linspace(0.0, -0.4, 6), [0.01] * 6)
    assert not characterise.real_drift([0.0], [0.1])


@pytest.mark.parametrize("scale", ["power", "amplitude"])
def test_d1_fit_finds_the_time_constant_and_the_scale(scale):
    tau, delay, loud, floor = 8.665, 0.4, 0.05, 0.0005
    e = characterise.D1_SCALES[scale]
    t = np.asarray(characterise.D1_SETTLES_US) / 1000
    words = (floor ** e + (loud ** e - floor ** e) * np.exp(-np.maximum(t - delay, 0) / tau)) ** (1 / e)
    fits = {name: characterise.fit_decay(t, words, loud, floor, k) for name, k in characterise.D1_SCALES.items()}
    assert fits[scale]["tau_ms"] == pytest.approx(tau, rel=1e-6)
    assert fits[scale]["delay_ms"] == pytest.approx(delay, abs=1e-6)
    other = next(name for name in fits if name != scale)
    assert fits[other] is None or fits[other]["residual"] > 10 * fits[scale]["residual"]


def test_d1_settle_within_interpolates_and_reports_none():
    t, excess = [1.0, 2.0, 4.0, 8.0], [20.0, 10.0, 4.0, 2.0]
    assert characterise.settle_within(t, excess, 3.0) == pytest.approx(6.0)
    assert characterise.settle_within(t, excess, 25.0) == 1.0
    assert characterise.settle_within(t, excess, 1.0) is None
    assert characterise.fit_decay(t, [0.05] * 4, 0.05, 0.0005, 1) is None


class ModelDetector:
    """A board whose scan detector word moves toward the steered sector's level with time constant tau."""

    def __init__(self, tau_ms, on, loud=0.05, floor=0.0005):
        self.tau, self.on, self.loud, self.floor = tau_ms, on, loud, floor
        self.clock, self.settle, self.sector, self.start, self.steered = 0.0, 1.25, None, floor, 0.0

    def word(self):
        target = self.loud if self.sector == self.on else self.floor
        return target + (self.start - target) * np.exp(-(self.clock - self.steered) / self.tau)

    def pump(self, seconds, until=None):
        self.clock += seconds * 1000

    def read(self, command):
        if command.startswith("S,"):
            self.start, self.sector, self.steered = self.word(), int(command[2:]), self.clock
            self.clock += self.settle
        return {"raw_level": int(self.word() * 2 ** 24)}


class ModelSession:
    def __init__(self, board):
        import csv
        import io
        from study_session import FIELDS, Results
        self.board, self.home, self.values, self.report = board, ({"rows": 7, "columns": 7},), {}, Results()
        self.writer = csv.DictWriter(io.StringIO(), FIELDS)

    def speaker_sector(self, azimuth, elevation):
        return 24

    def sound(self, name):
        pass

    def read(self, command):
        return self.board.read(command)

    def set(self, key, value):
        if key == "settle_us":
            self.board.settle = value / 1000

    @contextmanager
    def settings(self, **changes):
        for key, value in changes.items():
            self.set(key, value)
        yield


def test_d1_runs_on_a_model_detector(monkeypatch, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    board = ModelDetector(8.665, on=24)
    words = {3739: 0x9D61, 3740: 0, 3741: 0x0E39}
    registry = {"parameters": [{"group": "detector", "name": name, "address": address}
                               for name, address in (("TCONST", 3739), ("hold", 3740), ("decay", 3741))]}
    monkeypatch.setattr(characterise, "place", lambda *a: None)
    monkeypatch.setattr(characterise, "http", lambda method, path, body=None: (200, registry))
    monkeypatch.setattr(characterise, "read_words", lambda address, count: [words[address]])
    monkeypatch.setattr(characterise, "time", type("Clock", (), {"monotonic": staticmethod(lambda: board.clock / 1000)}))
    session = ModelSession(board)
    characterise.d1(session, argparse.Namespace(distance=2.0, seed=1, repeats=2), str(tmp_path), None)
    result = session.values["d1"]
    assert result["expected_tau_ms"] == pytest.approx(8.665, abs=0.01)
    assert result["fits"]["power"]["tau_ms"] == pytest.approx(8.665, rel=0.02)
    assert ("D1 time constant against the TCONST word", True) in [row[:2] for row in session.report.rows]
    assert {r["timing"] for r in result["readings"]} == {"firmware", "host"}
    assert (tmp_path / "d1_detector_decay.png").exists()


class LiveBoard:
    """Live frames every 50 ms: the peak beam (and the pattern fit when given) once the sound plays."""

    def __init__(self, fit):
        self.now, self.fit, self.sound_on = 0.0, fit, False

    def stop(self):
        pass

    def command(self, text):
        pass

    def pump(self, seconds, until=None):
        end = self.now + seconds
        while self.now < end - 1e-9:
            self.now += 0.05
            bank = {"target": {"azimuth_deg": 10.0, "elevation_deg": 0.0}, "fit": self.fit} if self.sound_on else {}
            if until is not None:
                until({"firmware_mode": "B", "bank": bank})


@pytest.mark.parametrize("fit", [{"azimuth_deg": 12.0, "elevation_deg": 1.0}, None])
def test_live_trial_reports_the_peak_beam_and_the_fit(monkeypatch, fit):
    board = LiveBoard(fit)
    session = type("Session", (), {"board": board, "sound": lambda self, s: setattr(board, "sound_on", s is not None)})()
    monkeypatch.setattr(characterise, "time", type("Clock", (), {"monotonic": staticmethod(lambda: board.now)}))
    result = characterise.live_trial(session, "A", 8.0)
    assert result["peak"] == (10.0, 0.0)
    assert result["fit"] == ((12.0, 1.0) if fit else None)
