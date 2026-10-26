from __future__ import annotations

import json

import numpy as np
import pytest

import dsp_registry
from bank_fit import BankFit, BankFitStream


def _contract(monkeypatch, folder):
    monkeypatch.setenv("HEIMDALL_DSP_EXPORT", folder)
    return json.loads((dsp_registry.selected_export_dir() / "beam_table_2d.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bank7():
    with pytest.MonkeyPatch.context() as patch:
        contract = _contract(patch, "SigmaStudioBank7x7")
    return contract, BankFit.from_contract(contract)


@pytest.mark.parametrize("folder", ["SigmaStudioBank7x7", "SigmaStudioBank"])
def test_every_beam_hears_its_own_direction_loudest(monkeypatch, folder):
    contract = _contract(monkeypatch, folder)
    bank = contract["bank"]
    fit = BankFit.from_contract(contract, step_deg=10.0)
    assert fit.beams == bank["rows"] * bank["columns"]
    for beam, (azimuth, elevation) in enumerate(zip(bank["azimuth_deg"], bank["elevation_deg"])):
        assert int(np.argmax(fit.fingerprint(azimuth, elevation))) == beam


@pytest.mark.parametrize("azimuth, elevation", [(-15.0, 3.0), (25.0, 3.0), (-31.0, -22.5), (7.3, 35.0)])
def test_a_map_with_gain_floor_and_noise_gives_its_direction(bank7, azimuth, elevation):
    _, fit = bank7
    rng = np.random.default_rng(1)
    clean = 900.0 * fit.fingerprint(azimuth, elevation) + 350.0
    result = fit.fit(clean + rng.normal(0.0, 5.0, clean.shape))
    assert abs(result.azimuth_deg - azimuth) <= 1.0
    assert abs(result.elevation_deg - elevation) <= 1.0
    assert result.quality > 0.99
    assert result.gain == pytest.approx(900.0, rel=0.05)
    assert result.floor == pytest.approx(350.0, rel=0.05)


def test_sidelobes_of_a_centred_source_do_not_move_the_answer(bank7):
    _, fit = bank7
    levels = fit.fingerprint(0.0, 0.0)
    corners = [0, 6, 42, 48]
    levels[corners] *= 3.0
    result = fit.fit(levels)
    assert abs(result.azimuth_deg) <= 4.0 and abs(result.elevation_deg) <= 4.0


def test_flat_or_malformed_maps_give_no_answer(bank7):
    _, fit = bank7
    assert fit.fit(np.full(fit.beams, 12.0)) is None
    assert fit.fit(np.ones(fit.beams - 1)) is None
    assert fit.fit([float("nan")] * fit.beams) is None


def test_unit_patterns_are_centred_unit_rows_that_rank_as_the_fit(bank7):
    _, fit = bank7
    unit = fit.unit_patterns()
    np.testing.assert_allclose(unit.sum(axis=1), 0.0, atol=1e-9)
    np.testing.assert_allclose((unit ** 2).sum(axis=1), 1.0)
    levels = fit.fingerprint(-12.0, 9.0) * 500 + 40
    best = int(np.argmax(np.maximum(unit @ (levels - levels.mean()), 0.0)))
    row, column = divmod(best, len(fit.azimuths))
    assert fit.azimuths[column] == -12.0 and abs(fit.elevations[row] - 9.0) <= 1.0


def test_noise_alone_fits_poorly(bank7):
    _, fit = bank7
    quality = [fit.fit(np.random.default_rng(seed).normal(100.0, 10.0, fit.beams)).quality for seed in range(20)]
    assert np.median(quality) < 0.5


def test_no_fit_without_per_microphone_bank_delays(monkeypatch):
    contract = _contract(monkeypatch, "SigmaStudioBank7x7")
    assert BankFit.from_contract({**contract, "bank": {**contract["bank"], "mic_delays": None}}) is None
    assert BankFit.from_contract({**contract, "bank": None}) is None
    with pytest.raises(ValueError):
        BankFit.from_contract(contract, scale="decibel")


def test_power_scale_also_finds_the_direction(bank7):
    contract, _ = bank7
    fit = BankFit.from_contract(contract, scale="power", step_deg=4.0)
    result = fit.fit(fit.fingerprint(20.0, -10.0))
    assert abs(result.azimuth_deg - 20.0) <= 2.0 and abs(result.elevation_deg + 10.0) <= 2.0


def test_stream_fits_the_mean_of_recent_maps_and_restarts_with_live(bank7):
    _, fit = bank7
    stream = BankFitStream(fit, maps=4)
    right, left = 500 * fit.fingerprint(25.0, 0.0), 500 * fit.fingerprint(-25.0, 0.0)
    for count in range(1, 5):
        result = stream.push(count, right)
    assert abs(result.azimuth_deg - 25.0) <= 1.0
    assert stream.push(4, left) is result
    assert stream.push(5, left).azimuth_deg > 0.0
    assert stream.push(1, left).azimuth_deg == pytest.approx(-25.0, abs=1.0)
    assert stream.push(2, [1.0] * 3) is stream.result
