from __future__ import annotations

import math
import time

import pytest

from heimdall_protocol import HeimdallState, ProtocolError, parse_line
from heimdall_transport import (
    EmulatorHeimdallTransport,
    HeimdallTransport,
    TRACK_FAILURE_LIMIT,
    validate_command,
)


class DeterministicAdaptiveEmulator(EmulatorHeimdallTransport):
    def __init__(self, rows=7, columns=7):
        super().__init__(rows=rows, columns=columns, sector_time_s=0.0)
        self.test_levels = [100] * (rows * columns)
        self.measured_sectors = []
        self.running = True
        self.connected = True
        self._emit_configuration()

    def _level_for_sector(self, sector):
        self.measured_sectors.append(sector)
        return self.test_levels[sector]


def test_dynamic_configuration_and_measurement_state():
    state = HeimdallState()
    state.apply_line("READY,2D,2,3,6,44")
    state.apply_line("AZIMUTH,3,-30.0000,0.0000,30.0000")
    state.apply_line("ELEVATION,2,-10.0000,10.0000")
    state.apply_line("P,1,2,30.0000,10.0000,16777216")

    snapshot = state.snapshot()
    assert snapshot["configuration"]["sectors"] == 6
    assert snapshot["levels_raw"][1][2] == 16777216
    assert snapshot["levels_linear"][1][2] == 1.0
    assert snapshot["levels_db"][1][2] == 0.0


def test_transport_reports_target_update_rate_and_age():
    now = [10.0]
    transport = HeimdallTransport(clock=lambda: now[0])
    transport.apply_line("READY,2D,2,2,4,44")
    transport.apply_line("SCAN_STARTED,G")
    transport.apply_line("TARGET_ACQUIRED,0,0,0,-10.0000,-10.0000,1000")
    now[0] = 10.02
    transport.apply_line("TARGET_UPDATED,0,0,0,-10.0000,-10.0000,1100")
    now[0] = 10.04
    transport.apply_line("TARGET_UPDATED,1,0,1,10.0000,-10.0000,1200")

    snapshot = transport.snapshot()
    assert snapshot["target_update_rate_hz"] == pytest.approx(50.0)
    assert snapshot["target_update_age_s"] == pytest.approx(0.0)

    now[0] = 10.14
    assert transport.snapshot()["target_update_age_s"] == pytest.approx(0.1)
    transport.apply_line("SCAN_STOPPED")
    assert transport.snapshot()["target_update_rate_hz"] is None
    assert transport.snapshot()["target_update_age_s"] is None


@pytest.mark.parametrize(
    "line",
    [
        "",
        "READY,2D,2,3,5,44",
        "AZIMUTH,3,-30,0",
        "P,0,0,NaN,0,1",
        "P,-1,0,0,0,1",
        "STEER_ERR,1,2,nope",
        "UNKNOWN,1",
    ],
)
def test_malformed_records_are_rejected(line):
    with pytest.raises(ProtocolError):
        parse_line(line)


def test_out_of_bounds_measurement_is_rejected_after_ready():
    state = HeimdallState()
    state.apply_line("READY,2D,2,3,6,44")
    with pytest.raises(ProtocolError):
        state.apply_line("P,2,0,0,0,1")


def test_target_and_timing_records():
    state = HeimdallState()
    state.apply_line("READY,2D,7,7,49,44")
    state.apply_line("TARGET_ACQUIRED,24,3,3,0.0000,0.0000,1000")
    state.apply_line("TIMING,G,49,100000,110000,2000,2500")
    snapshot = state.snapshot()
    assert snapshot["target"]["sector"] == 24
    assert snapshot["last_timing"]["wall_us"] == 110000
    state.apply_line("TARGET_LOST,24,3,3,0.0000,0.0000,10")
    assert state.snapshot()["target"] is None


def test_one_shot_scan_returns_to_idle_on_completion():
    state = HeimdallState()
    state.apply_line("READY,2D,2,2,4,44")
    state.apply_line("SCAN_STARTED,F")
    assert state.snapshot()["mode"] == "F"
    state.apply_line("SCAN_DONE")
    assert state.snapshot()["mode"] == "IDLE"


def test_information_refresh_preserves_active_mode():
    state = HeimdallState()
    state.apply_line("READY,2D,2,2,4,44")
    state.apply_line("SCAN_STARTED,G")
    state.apply_line("READY,2D,2,2,4,44")
    assert state.snapshot()["mode"] == "G"


def test_stop_and_direct_steer_clear_tracking_state():
    state = HeimdallState()
    state.apply_line("READY,2D,7,7,49,44")
    state.apply_line("TARGET_ACQUIRED,24,3,3,0.0000,0.0000,1000")
    state.apply_line("SCAN_STOPPED")
    assert state.snapshot()["target"] is None
    state.apply_line("TARGET_ACQUIRED,24,3,3,0.0000,0.0000,1000")
    record = state.apply_line("STEER_OK,48,6,6,60.0000,51.4286")
    assert record["sector"] == 48
    state.apply_line("TIMING,S,2800")
    assert state.snapshot()["last_steer"]["azimuth_deg"] == 60.0
    assert state.snapshot()["target"] is None


def test_command_validation():
    assert validate_command("S,399", sectors=400) == "S,399"
    assert validate_command("I") == "I"
    assert validate_command("M") == "M"
    with pytest.raises(ValueError):
        validate_command("S,400", sectors=400)
    with pytest.raises(ValueError):
        validate_command("s,1")


def test_emulator_uses_the_same_protocol_state_path():
    emulator = EmulatorHeimdallTransport(
        rows=3,
        columns=4,
        azimuth_min_deg=-40.0,
        azimuth_max_deg=40.0,
        elevation_min_deg=-30.0,
        elevation_max_deg=30.0,
        sector_time_s=0.0,
    )
    emulator.start()
    try:
        emulator.send_command("F")
        deadline = time.monotonic() + 2.0
        while emulator.snapshot()["scan_count"] < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        snapshot = emulator.snapshot()
        assert snapshot["connected"]
        assert snapshot["configuration"]["rows"] == 3
        assert snapshot["configuration"]["columns"] == 4
        assert snapshot["azimuth_deg"] == [-30.0, -10.0, 10.0, 30.0]
        assert snapshot["elevation_deg"] == [-20.0, 0.0, 20.0]
        assert snapshot["scan_count"] == 1
        assert all(value is not None for row in snapshot["levels_raw"] for value in row)
    finally:
        emulator.stop()
        emulator.join(timeout=1.0)


def test_acoustic_level_provider_uses_the_same_emulator_policy():
    class Provider:
        def level_for_sector(self, sector):
            return 400 if sector == 5 else 100

        def metadata(self):
            return {"acoustic_cache_id": "test-cache", "acoustic_calibrated": False}

    emulator = EmulatorHeimdallTransport(
        rows=2, columns=3, sector_time_s=0.0, level_provider=Provider()
    )
    emulator.start()
    try:
        emulator.send_command("G")
        deadline = time.monotonic() + 1.0
        while emulator.snapshot()["target"] is None and time.monotonic() < deadline:
            time.sleep(0.005)
        snapshot = emulator.snapshot()
        assert snapshot["transport"] == "acoustic-emulator"
        assert snapshot["target"]["sector"] == 5
        assert snapshot["acoustic_cache_id"] == "test-cache"
        assert not snapshot["acoustic_calibrated"]
        assert "emulator_flight_profile" not in snapshot
    finally:
        emulator.stop()
        emulator.join(timeout=1.0)


def test_large_emulator_stops_during_an_active_scan():
    emulator = EmulatorHeimdallTransport(rows=20, columns=20, sector_time_s=0.01)
    emulator.start()
    emulator.send_command("F")
    time.sleep(0.03)
    emulator.stop()
    emulator.join(timeout=0.5)
    assert not emulator.is_alive()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 0},
        {"columns": 21},
        {"azimuth_min_deg": 10.0, "azimuth_max_deg": 10.0},
        {"elevation_min_deg": -91.0},
    ],
)
def test_emulator_rejects_invalid_grid_configuration(kwargs):
    with pytest.raises(ValueError):
        EmulatorHeimdallTransport(**kwargs)


@pytest.mark.parametrize("flight_profile", ["crossing", "approach", "orbit", "patrol", "evasive", "legacy"])
def test_emulator_flight_profiles_produce_finite_target_states(flight_profile):
    emulator = EmulatorHeimdallTransport(flight_profile=flight_profile)
    for elapsed in (0.0, 3.0, 10.0, 25.0):
        azimuth, elevation, target_range, attenuation = emulator._target_state(elapsed)
        assert all(math.isfinite(value) for value in (azimuth, elevation, target_range, attenuation))
        assert -90.0 < azimuth < 90.0
        assert -90.0 < elevation < 90.0
        assert target_range > 0.0
        assert 0.0 <= attenuation <= 1.0


def test_emulator_flight_speed_scales_trajectory_time():
    normal = EmulatorHeimdallTransport(flight_profile="crossing", flight_speed=1.0)
    fast = EmulatorHeimdallTransport(flight_profile="crossing", flight_speed=2.0)
    assert fast._target_state(4.0) == pytest.approx(normal._target_state(8.0))


def test_emulator_flight_clock_and_snapshot_metadata_are_deterministic():
    now = [100.0]
    emulator = EmulatorHeimdallTransport(
        flight_profile="orbit", flight_speed=1.5, clock=lambda: now[0]
    )
    initial = emulator._target_state()
    now[0] = 104.0
    assert emulator._target_state() == pytest.approx(emulator._target_state(4.0))
    assert emulator._target_state() != pytest.approx(initial)
    snapshot = emulator.snapshot()
    assert snapshot["emulator_flight_profile"] == "orbit"
    assert snapshot["emulator_flight_speed"] == 1.5


@pytest.mark.parametrize("flight_profile", ["crossing", "approach", "orbit", "patrol", "evasive"])
def test_emulator_flight_trajectory_has_no_position_jumps(flight_profile):
    emulator = EmulatorHeimdallTransport(flight_profile=flight_profile)

    def cartesian(state):
        azimuth, elevation, target_range, _ = state
        azimuth_rad = math.radians(azimuth)
        elevation_rad = math.radians(elevation)
        horizontal = target_range * math.cos(elevation_rad)
        return (
            horizontal * math.cos(azimuth_rad),
            horizontal * math.sin(azimuth_rad),
            target_range * math.sin(elevation_rad),
        )

    previous = cartesian(emulator._target_state(0.0))
    for step in range(1, 6001):
        current = cartesian(emulator._target_state(step * 0.01))
        assert math.dist(previous, current) < 0.35
        previous = current


def test_emulator_approach_range_changes_detector_level():
    emulator = EmulatorHeimdallTransport(rows=1, columns=1, flight_profile="approach")
    emulator.random.uniform = lambda _low, _high: 0.0
    far_azimuth, far_elevation, far_range, _ = emulator._target_state(0.0)
    near_time = math.pi / 0.38
    near_azimuth, near_elevation, near_range, _ = emulator._target_state(near_time)
    assert near_range < far_range
    emulator.started_at = time.monotonic()
    emulator.azimuth_deg = [far_azimuth]
    emulator.elevation_deg = [far_elevation]
    far_level = emulator._level_for_sector(0)
    emulator.started_at = time.monotonic() - near_time
    emulator.azimuth_deg = [near_azimuth]
    emulator.elevation_deg = [near_elevation]
    near_level = emulator._level_for_sector(0)
    assert near_level > far_level


def test_emulator_evasive_profile_contains_short_attenuation_event():
    emulator = EmulatorHeimdallTransport(flight_profile="evasive")
    assert emulator._target_state(10.0)[3] == 1.0
    assert emulator._target_state(12.0)[3] == 0.0
    assert emulator._target_state(13.0)[3] == 1.0


def test_emulator_evasive_dropout_forces_loss_and_reacquisition():
    now = [0.0]
    emulator = EmulatorHeimdallTransport(
        flight_profile="evasive", sector_time_s=0.0, clock=lambda: now[0]
    )
    emulator.random.uniform = lambda _low, _high: 0.0
    emulator.running = True
    emulator.connected = True
    emulator._emit_configuration()
    emulator._handle_command("G")
    for _ in range(3):
        emulator._adaptive_cycle()
    assert emulator.adaptive_state == "TRACK"
    assert emulator.snapshot()["target"] is not None

    now[0] = 12.0
    for _ in range(TRACK_FAILURE_LIMIT):
        emulator._adaptive_cycle()
    assert emulator.adaptive_state == "REACQUIRE"
    emulator._adaptive_cycle()
    assert emulator.adaptive_state == "SEARCH"
    assert emulator.snapshot()["target"] is None

    now[0] = 13.0
    for _ in range(3):
        emulator._adaptive_cycle()
    assert emulator.adaptive_state == "TRACK"
    assert emulator.snapshot()["target"] is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"flight_profile": "unknown"},
        {"flight_speed": 0.1},
        {"flight_speed": 3.1},
    ],
)
def test_emulator_rejects_invalid_flight_configuration(kwargs):
    with pytest.raises(ValueError):
        EmulatorHeimdallTransport(**kwargs)


def test_emulator_steer_measures_and_repeat_measure_does_not_change_sector():
    emulator = EmulatorHeimdallTransport(rows=7, columns=7, sector_time_s=0.0)
    emulator.start()
    try:
        emulator.send_command("S,48")
        deadline = time.monotonic() + 1.0
        while emulator.snapshot()["levels_raw"][6][6] is None and time.monotonic() < deadline:
            time.sleep(0.005)
        first = emulator.snapshot()
        assert first["last_steer"]["sector"] == 48
        assert first["levels_raw"][6][6] is not None
        emulator.send_command("M")
        first_sequence = first["sequence"]
        while emulator.snapshot()["sequence"] == first_sequence and time.monotonic() < deadline:
            time.sleep(0.005)
        assert emulator.snapshot()["last_steer"]["sector"] == 48
    finally:
        emulator.stop()
        emulator.join(timeout=1.0)


def test_emulator_adaptive_uses_confirmation_and_fifty_local_tracking_passes():
    emulator = DeterministicAdaptiveEmulator()
    center = 24
    emulator.test_levels[center] = 400
    emulator._handle_command("G")

    emulator._adaptive_cycle()
    assert emulator.adaptive_state == "CONFIRM"
    assert emulator.snapshot()["scan_count"] == 1
    assert emulator.measured_sectors == list(range(49))

    emulator._adaptive_cycle()
    assert emulator.adaptive_state == "CONFIRM"
    emulator._adaptive_cycle()
    assert emulator.adaptive_state == "TRACK"
    assert emulator.target_sector == center
    assert emulator.snapshot()["target"]["sector"] == center
    assert emulator.measured_sectors[49:] == emulator._cross(center) * 2

    emulator.measured_sectors.clear()
    for _ in range(49):
        emulator._adaptive_cycle()
    assert emulator.adaptive_state == "TRACK"
    assert emulator.tracking_passes == 49
    assert emulator.snapshot()["scan_count"] == 1
    assert emulator.measured_sectors == emulator._cross(center) * 49

    emulator._adaptive_cycle()
    assert emulator.adaptive_state == "SEARCH"
    assert emulator.snapshot()["scan_count"] == 1
    emulator._adaptive_cycle()
    assert emulator.snapshot()["scan_count"] == 2


def test_emulator_adaptive_flat_levels_repeat_search_without_locking():
    emulator = DeterministicAdaptiveEmulator(rows=3, columns=4)
    emulator._handle_command("G")

    emulator._adaptive_cycle()
    assert emulator.adaptive_state == "SEARCH"
    assert emulator.snapshot()["target"] is None
    assert emulator.snapshot()["scan_count"] == 1

    emulator.measured_sectors.clear()
    emulator._adaptive_cycle()
    assert emulator.adaptive_state == "SEARCH"
    assert emulator.snapshot()["target"] is None
    assert emulator.snapshot()["scan_count"] == 2
    assert emulator.measured_sectors == list(range(12))


def test_emulator_adaptive_matches_move_hysteresis_and_reacquisition():
    emulator = DeterministicAdaptiveEmulator()
    center = 24
    neighbor = 25
    emulator.test_levels[center] = 400
    emulator._handle_command("G")
    for _ in range(3):
        emulator._adaptive_cycle()
    assert emulator.target_sector == center

    emulator.test_levels[center] = 100
    emulator.test_levels[neighbor] = 500
    emulator._adaptive_cycle()
    assert emulator.target_sector == center
    emulator._adaptive_cycle()
    assert emulator.target_sector == neighbor

    emulator.test_levels = [0] * 49
    for _ in range(3):
        emulator._adaptive_cycle()
    assert emulator.adaptive_state == "REACQUIRE"
    emulator.measured_sectors.clear()
    emulator._adaptive_cycle()
    assert emulator.adaptive_state == "SEARCH"
    assert emulator.snapshot()["target"] is None
    assert emulator.measured_sectors == emulator._neighborhood(neighbor)