"""The telemetry log (heimdall_telemetry.py), its reader (scripts/telemetry_log.py) and the HEALTH record."""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sim_server  # noqa: E402
import telemetry_log  # noqa: E402
from heimdall_protocol import HeimdallState  # noqa: E402
from heimdall_telemetry import TelemetryLog  # noqa: E402
from test_heimdall_bank import ScriptedBankEmulator, bank_program  # noqa: E402


def test_log_keeps_received_sent_and_notes(tmp_path):
    log = TelemetryLog(tmp_path / "a" / "session.log", "serial:COM9", clock=lambda: 12.5)
    log.received("SCAN_STARTED,B")
    log.sent("B")
    log.note("mark drone   at 4 m")
    log.close()
    lines = (tmp_path / "a" / "session.log").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("12.500000\t#\theimdall-telemetry-v1 transport=serial:COM9")
    assert lines[1:4] == ["12.500000\t<\tSCAN_STARTED,B", "12.500000\t>\tB", "12.500000\t#\tmark drone at 4 m"]
    assert lines[-1].endswith("closed")
    log.received("LEVELS,B,1,1,5")
    assert log.status()["open"] is False and log.status()["lines"] == 5


def test_health_counts_the_levels_lines_the_host_missed():
    state = HeimdallState()
    for line in ("READY,2D,7,7,49,44", "BANK,5,5,12000", "SCAN_STARTED,B", "HEALTH,1000,5210,1830,10,0,0"):
        state.apply_line(line)
    for _ in range(3):
        state.apply_line("LEVELS,B,5,5," + ",".join(["1000"] * 25))
    state.apply_line("HEALTH,2000,5102,2410,15,1,0")
    health = state.snapshot()["health"]
    assert health["bank_maps_missed"] == 2
    assert (health["passes"], health["max_pass_us"], health["audio_overruns"]) == (5102, 2410, 1)


def test_logged_emulator_session_reads_back_as_live_runs(tmp_path):
    path = tmp_path / "session.log"
    emulator = ScriptedBankEmulator(bank_program())
    emulator.telemetry = TelemetryLog(path, "emulator")
    emulator.send_command("B")
    emulator._handle_command(emulator.commands.get_nowait())
    emulator._stare_cycle()
    emulator.map_levels[6] = 6000
    for _ in range(4):
        emulator._stare_cycle()
    emulator.telemetry.note("mark speaker on")
    emulator.send_command("X")
    emulator._handle_command(emulator.commands.get_nowait())
    emulator.telemetry.close()

    session = telemetry_log.read_session(path)
    assert session["sent"] == {"B": 1, "X": 1}
    [run] = session["runs"]
    assert run["maps"] == [1, 2, 3, 4, 5]
    assert len(run["levels"][0]) == 25
    assert [t[0] for t in run["targets"]] == [2, 3, 4, 5]
    assert "speaker on" in run["events"]
    assert "Live start 1" in telemetry_log.summary(session)


def test_status_reports_the_log_and_marks_go_into_it(tmp_path):
    class LoggedTransport:
        telemetry = TelemetryLog(tmp_path / "s.log", "serial:COM9")

        def snapshot(self):
            return {"connected": True, "transport": "serial:COM9", "configuration": None, "scan_count": 0,
                    "protocol_errors": 0}

        def send_command(self, command):
            pass

        def stop(self):
            pass

    sim_server._heimdall_transport = LoggedTransport()
    try:
        with TestClient(sim_server.app) as client:
            assert client.get("/hw_status").json()["telemetry"]["path"].endswith("s.log")
            assert client.post("/hw_telemetry/mark", json={"text": "hover 4 m"}).status_code == 200
        assert "mark hover 4 m" in (tmp_path / "s.log").read_text(encoding="utf-8")
    finally:
        sim_server._replace_heimdall_transport(None)
    assert (tmp_path / "s.log").read_text(encoding="utf-8").rstrip().endswith("closed")
