from __future__ import annotations

from dataclasses import asdict
import json
from hashlib import sha256
import math
from pathlib import Path

import numpy as np
import pytest

from heimdall_acoustic import (
    ACOUSTIC_MODEL_VERSION,
    AcousticCache,
    AcousticCacheLevelProvider,
    AcousticGridSpec,
    AcousticScenario,
    _detector_bandpass_fir,
    _plane_wave_transfer_powers,
    _progress,
    _scenario_hash,
    acoustic_scenario,
    apply_detector_envelope,
    build_acoustic_grid,
    handheld_speaker_position,
    load_audio_48k,
    load_acoustic_cache,
    load_beam_contract,
    prepare_acoustic_cache,
    render_acoustic_audition,
    room_evasive_position,
    room_microphone_positions,
    speech_speaker_position,
    stationary_speaker_position,
)


BASE_DIR = Path(__file__).resolve().parents[1]


def test_deployment_contract_dimensions_delays_and_fir():
    contract = load_beam_contract()
    assert contract.sampling_rate_hz == 48000
    assert contract.max_delay_samples == 68
    assert (contract.rows, contract.columns, contract.sectors, contract.microphones) == (7, 7, 49, 44)
    assert contract.delay_fixpt.shape == (49, 44)
    assert contract.delay_samples.shape == (49, 44)
    assert np.all(contract.delay_samples >= 0.0)
    assert np.max(contract.delay_samples) <= 68.0
    np.testing.assert_array_equal(contract.delay_fixpt[24], np.zeros(44, dtype=np.uint32))
    assert len(contract.fir_stages) == 2
    assert all(stage.size == 11 for stage in contract.fir_stages)


def test_deployment_contract_rejects_mismatched_firmware_header(tmp_path):
    header = tmp_path / "beam_table_2d.h"
    header.write_text("not the deployed table", encoding="ascii")
    with pytest.raises(ValueError, match="does not match firmware"):
        load_beam_contract(firmware_header_path=header)


def test_default_acoustic_grid_reuses_exact_deployment_contract():
    deployment = load_beam_contract()
    grid, mode = build_acoustic_grid(deployment, AcousticGridSpec())
    assert grid is deployment
    assert mode == "deployment_contract"


def test_custom_acoustic_grid_generates_quantized_delays_without_mutating_deployment():
    deployment = load_beam_contract()
    spec = AcousticGridSpec(
        rows=20,
        columns=20,
        azimuth_min_deg=-80.0,
        azimuth_max_deg=80.0,
        elevation_min_deg=-70.0,
        elevation_max_deg=70.0,
    )
    grid, mode = build_acoustic_grid(deployment, spec)
    assert mode == "exploratory_simulation"
    assert (grid.rows, grid.columns, grid.sectors) == (20, 20, 400)
    assert grid.delay_fixpt.shape == (400, 44)
    assert np.max(grid.delay_samples) <= deployment.max_delay_samples
    assert deployment.delay_fixpt.shape == (49, 44)


def test_vertical_array_room_transform_and_broadside_geometry():
    contract = load_beam_contract()
    scenario = AcousticScenario()
    microphones = room_microphone_positions(contract, scenario)
    assert microphones.shape == (3, 44)
    np.testing.assert_allclose(microphones.mean(axis=1), scenario.array_center_m, atol=1e-12)
    np.testing.assert_allclose(microphones[1], np.full(44, 0.5), atol=1e-12)
    assert np.ptp(microphones[0]) > 0.4
    assert np.ptp(microphones[2]) > 0.4


def test_scenario_source_positions():
    scenario = AcousticScenario()
    stationary = stationary_speaker_position(scenario)
    np.testing.assert_allclose(stationary, (10.0, 2.5, 1.0))
    speech = speech_speaker_position(scenario)
    offset = speech - np.asarray(scenario.array_center_m)
    assert math.hypot(offset[0], offset[1]) == pytest.approx(12.0)
    assert math.degrees(math.atan2(offset[0], offset[1])) == pytest.approx(25.0)
    assert speech[2] == pytest.approx(1.5)
    stationary_config = acoustic_scenario("stationary_2m")
    assert stationary_config.crowd_talkers == 0
    assert not stationary_config.speech_enabled


def test_stationary_scenario_is_drone_only_calibration_reference():
    scenario = acoustic_scenario("stationary_2m")
    assert scenario.scenario == "stationary_2m"
    assert scenario.drone_waypoints == 1
    assert scenario.crowd_talkers == 0
    assert not scenario.speech_enabled


def test_handheld_scenario_variants_share_motion_but_not_interference():
    drone_only = acoustic_scenario("handheld_2m_drone")
    mixed = acoustic_scenario("handheld_2m_mixed")
    assert drone_only.drone_waypoints == mixed.drone_waypoints == 16
    assert drone_only.crowd_talkers == 0
    assert not drone_only.speech_enabled
    assert mixed.crowd_talkers == 12
    assert mixed.speech_enabled


def test_handheld_route_is_closed_slow_and_fixed_at_two_meter_depth():
    scenario = acoustic_scenario("handheld_2m_drone")
    points = np.stack([
        handheld_speaker_position(index * 0.01, scenario)
        for index in range(1201)
    ])
    np.testing.assert_allclose(points[0], points[-1], atol=1e-12)
    np.testing.assert_allclose(points[:, 1], np.full(points.shape[0], 2.5))
    assert np.min(points[:, 0]) == pytest.approx(9.25)
    assert np.max(points[:, 0]) == pytest.approx(10.75)
    assert np.min(points[:, 2]) == pytest.approx(1.2)
    assert np.max(points[:, 2]) == pytest.approx(2.0)
    speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) / 0.01
    assert np.max(speeds) < 0.6


def test_room_evasive_route_is_closed_bounded_and_continuous():
    scenario = AcousticScenario()
    np.testing.assert_allclose(room_evasive_position(0.0, scenario), room_evasive_position(15.0, scenario))
    points = np.stack([room_evasive_position(index * 0.01, scenario) for index in range(1501)])
    assert np.all((points[:, 0] > 0.3) & (points[:, 0] < 19.7))
    assert np.all((points[:, 1] > 0.3) & (points[:, 1] < 39.7))
    assert np.all((points[:, 2] >= 1.5) & (points[:, 2] <= 3.0))
    speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) / 0.01
    assert np.max(speeds) < 20.0


def test_room_evasive_route_repeats_twice_in_thirty_second_scene():
    scenario = AcousticScenario()
    for elapsed in (0.0, 2.5, 7.5, 14.9):
        np.testing.assert_allclose(
            room_evasive_position(elapsed, scenario),
            room_evasive_position(elapsed + 15.0, scenario),
            atol=1e-12,
        )


def test_audio_loader_resamples_and_normalizes():
    samples = load_audio_48k(BASE_DIR / "audio" / "drone.wav", 0.1)
    assert samples.shape == (4800,)
    assert np.all(np.isfinite(samples))
    assert np.sqrt(np.mean(np.square(samples))) == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("sector", [0, 24, 48])
def test_exact_deployed_delays_peak_at_matching_plane_wave(sector):
    contract = load_beam_contract()
    row, column = divmod(sector, contract.columns)
    azimuth = math.radians(contract.azimuth_deg[column])
    elevation = math.radians(contract.elevation_deg[row])
    direction = np.asarray((
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
        math.cos(elevation) * math.cos(azimuth),
    ))
    response = _plane_wave_transfer_powers(contract, direction[None, :], 512)
    frequency_bin = int(round(4000.0 / (48000.0 / 512.0)))
    assert int(np.argmax(response[:, frequency_bin])) == sector


def test_detector_envelope_obeys_rise_and_decay_limits():
    levels = np.asarray(((0.001,), (1.0,), (0.001,)), dtype=np.float64)
    filtered = apply_detector_envelope(levels, step_s=0.001, rise_db_s=1000.0, decay_db_s=500.0)
    output_db = 20.0 * np.log10(filtered[:, 0])
    assert output_db[1] - output_db[0] == pytest.approx(1.0)
    assert output_db[2] - output_db[1] == pytest.approx(-0.5)


def test_intended_detector_filter_is_one_to_four_kilohertz_bandpass():
    scenario = AcousticScenario()
    detector_fir = _detector_bandpass_fir(scenario, 48000)
    frequencies = np.asarray((500.0, 2000.0, 8000.0))
    response = np.asarray([
        abs(np.sum(detector_fir * np.exp(
            -2j * np.pi * frequency * np.arange(detector_fir.size) / 48000.0
        )))
        for frequency in frequencies
    ])
    response_db = 20.0 * np.log10(np.maximum(response, 1e-12))
    assert response_db[0] < -25.0
    assert response_db[1] > -1.0
    assert response_db[2] < -40.0


@pytest.mark.parametrize("kwargs", [
    {"detector_highpass_hz": 4000.0, "detector_lowpass_hz": 1000.0},
    {"detector_fir_taps": 128},
])
def test_scenario_rejects_invalid_detector_filter(kwargs):
    with pytest.raises(ValueError, match="detector"):
        AcousticScenario(**kwargs).validate()


def test_preparation_progress_is_monotonic_across_stages():
    updates = []
    started = 0.0
    _progress(updates.append, "room-rirs", 9, 9, started)
    _progress(updates.append, "spectral-beamforming", 0, 3, started)
    _progress(updates.append, "spectral-beamforming", 3, 3, started)
    _progress(updates.append, "detector", 1, 1, started)
    _progress(updates.append, "cache-write", 1, 1, started)
    assert [update["overall_percent"] for update in updates] == [90.0, 90.0, 98.0, 99.0, 100.0]


def test_cache_provider_loops_by_monotonic_time(tmp_path):
    levels = np.asarray(((1, 2), (3, 4), (5, 6)), dtype=np.uint32)
    levels_path = tmp_path / "levels_raw.npy"
    np.save(levels_path, levels)
    manifest = {"level_step_s": 0.1, "scenario": {"scenario": "test"}, "calibrated": False,
                "schema": ACOUSTIC_MODEL_VERSION}
    now = [10.0]
    cache = AcousticCache("cache", tmp_path, manifest, np.load(levels_path, mmap_mode="r"))
    provider = AcousticCacheLevelProvider(cache, clock=lambda: now[0])
    assert provider.level_for_sector(1) == 2
    now[0] = 10.11
    assert provider.level_for_sector(1) == 4
    now[0] = 10.31
    assert provider.level_for_sector(1) == 2


def test_cache_provider_reports_synchronized_stationary_truth(tmp_path):
    levels_path = tmp_path / "levels_raw.npy"
    np.save(levels_path, np.ones((20, 49), dtype=np.uint32))
    scenario = acoustic_scenario("stationary_2m")
    manifest = {"level_step_s": 0.005, "scenario": asdict(scenario),
                "calibrated": False, "schema": ACOUSTIC_MODEL_VERSION}
    cache = AcousticCache("cache", tmp_path, manifest, np.load(levels_path, mmap_mode="r"))
    provider = AcousticCacheLevelProvider(cache, clock=lambda: 10.0)
    truth = provider.metadata()["emulator_truth"]
    assert truth["frame"] == 0
    assert truth["azimuth_deg"] == pytest.approx(0.0)
    assert truth["elevation_deg"] == pytest.approx(0.0)
    assert truth["range_m"] == pytest.approx(2.0)
    assert truth["room_position_m"] == pytest.approx([10.0, 2.5, 1.0])
    assert truth["source_active"]


def test_cache_provider_truth_uses_conference_waypoints_loop_and_dropout(tmp_path):
    levels_path = tmp_path / "levels_raw.npy"
    np.save(levels_path, np.ones((6000, 49), dtype=np.uint32))
    scenario = acoustic_scenario("conference_evasive")
    manifest = {"level_step_s": 0.005, "scenario": asdict(scenario),
                "calibrated": False, "schema": ACOUSTIC_MODEL_VERSION}
    now = [10.0]
    cache = AcousticCache("cache", tmp_path, manifest, np.load(levels_path, mmap_mode="r"))
    provider = AcousticCacheLevelProvider(cache, clock=lambda: now[0])
    start = provider.metadata()["emulator_truth"]
    now[0] = 25.0
    repeated = provider.metadata()["emulator_truth"]
    assert repeated["room_position_m"] == pytest.approx(start["room_position_m"])
    now[0] = 15.8
    assert not provider.metadata()["emulator_truth"]["source_active"]


def test_moving_drone_cache_can_disable_speech(tmp_path):
    scenario = AcousticScenario(
        duration_s=0.05,
        reflection_order=0,
        drone_waypoints=2,
        crowd_talkers=0,
        speech_enabled=False,
    )
    cache = prepare_acoustic_cache(scenario, cache_root=tmp_path)
    assert cache.levels_raw.shape == (10, 49)
    assert "1000-4000 Hz" in cache.manifest["detector_filter"]
    assert "not applied" in cache.manifest["exported_fir_status"]


def test_room_cache_is_reused_across_acoustic_grids(monkeypatch, tmp_path):
    scenario = AcousticScenario(
        scenario="stationary_2m",
        duration_s=0.05,
        reflection_order=0,
        drone_waypoints=1,
        crowd_talkers=0,
        speech_enabled=False,
    )
    cache_root = tmp_path / "levels"
    room_root = tmp_path / "rooms"
    first = prepare_acoustic_cache(
        scenario,
        cache_root=cache_root,
        room_cache_root=room_root,
        grid_spec=AcousticGridSpec(rows=3, columns=4),
    )

    def unexpected_rir(*_args, **_kwargs):
        raise AssertionError("room RIRs were recomputed for a grid-only change")

    monkeypatch.setattr("heimdall_acoustic._room_rirs", unexpected_rir)
    second = prepare_acoustic_cache(
        scenario,
        cache_root=cache_root,
        room_cache_root=room_root,
        grid_spec=AcousticGridSpec(rows=4, columns=5),
    )
    assert first.levels_raw.shape == (10, 12)
    assert second.levels_raw.shape == (10, 20)
    assert first.manifest["room_cache_id"] == second.manifest["room_cache_id"]
    assert first.manifest["grid_mode"] == "exploratory_simulation"
    assert second.manifest["grid"] == asdict(AcousticGridSpec(rows=4, columns=5))


def test_acoustic_audition_renders_comparable_clips(tmp_path):
    room_id = "room-test"
    room_directory = tmp_path / room_id
    room_directory.mkdir()
    transfer = np.ones((1, 44, 257), dtype=np.complex128)
    transfer_path = room_directory / "microphone_transfer.npy"
    np.save(transfer_path, transfer)
    room_manifest = {
        "schema": "heimdall-room-v1",
        "cache_id": room_id,
        "scenario": asdict(acoustic_scenario("stationary_2m")),
        "signal_fft_size": 512,
        "source_positions_m": [[10.0, 2.5, 1.0]],
        "speech_source_index": None,
        "transfer_shape": list(transfer.shape),
        "transfer_dtype": "complex128",
        "transfer_sha256": sha256(transfer_path.read_bytes()).hexdigest(),
    }
    (room_directory / "manifest.json").write_text(json.dumps(room_manifest), encoding="utf-8")
    levels = np.ones((10, 49), dtype=np.uint32)
    levels_path = tmp_path / "levels.npy"
    np.save(levels_path, levels)
    cache = AcousticCache("level-test", tmp_path, {
        "schema": ACOUSTIC_MODEL_VERSION,
        "scenario": asdict(acoustic_scenario("stationary_2m")),
        "grid": asdict(AcousticGridSpec()),
        "room_cache_id": room_id,
    }, np.load(levels_path, mmap_mode="r"))
    rendered = render_acoustic_audition(
        cache, start_s=0.0, duration_s=0.25, selected_sector=24,
        room_cache_root=tmp_path,
    )
    assert rendered["sample_rate_hz"] == 48000
    assert set(rendered["clips"]) == {
        "generated", "single_mic", "unsteered", "truth", "selected",
    }
    assert all(clip.shape == (12000,) for clip in rendered["clips"].values())
    assert all(np.all(np.isfinite(clip)) for clip in rendered["clips"].values())


def test_cache_hash_changes_with_scenario(tmp_path):
    contract = tmp_path / "contract.json"
    drone = tmp_path / "drone.wav"
    crowd = tmp_path / "crowd.wav"
    contract.write_text("contract", encoding="ascii")
    drone.write_bytes(b"drone")
    crowd.write_bytes(b"crowd")
    first = _scenario_hash(contract, AcousticScenario(), drone, crowd)
    second = _scenario_hash(contract, AcousticScenario(speech_spl_db=73.0), drone, crowd)
    assert first != second


def test_cache_loader_rejects_corrupted_levels(tmp_path):
    cache_id = "cache"
    directory = tmp_path / cache_id
    directory.mkdir()
    levels_path = directory / "levels_raw.npy"
    np.save(levels_path, np.asarray(((1, 2),), dtype=np.uint32))
    manifest = {
        "schema": ACOUSTIC_MODEL_VERSION,
        "cache_id": cache_id,
        "levels_shape": [1, 2],
        "levels_sha256": sha256(levels_path.read_bytes()).hexdigest(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_acoustic_cache(cache_id, tmp_path).levels_raw[0, 1] == 2
    with levels_path.open("ab") as cache_file:
        cache_file.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_acoustic_cache(cache_id, tmp_path)


def test_scenario_validation_rejects_unsupported_configuration():
    with pytest.raises(ValueError):
        AcousticScenario(room_dimensions_m=(10.0, 10.0, 3.0)).validate()
    with pytest.raises(ValueError):
        AcousticScenario(scenario="unknown").validate()
