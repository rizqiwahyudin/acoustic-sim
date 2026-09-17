from __future__ import annotations

import time

import pytest

from heimdall_protocol import HeimdallState, ProtocolError, parse_line
from heimdall_transport import EmulatorHeimdallTransport, validate_command


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