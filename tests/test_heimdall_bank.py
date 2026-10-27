from __future__ import annotations

import json
from pathlib import Path

import pytest

from bank_fit import BankFit
from heimdall_bank import FRACTION_BITS, BankTracker, LOST_MAPS
from heimdall_protocol import HeimdallState, ProtocolError, parse_line
from heimdall_transport import DeployedProgram, EmulatorHeimdallTransport, validate_command

BANK_CONTRACT = Path(__file__).resolve().parents[2] / "MAX78002" / "SigmaStudioBank" / "beam_table_2d.json"
SINES = [-0.6428, -0.3214, 0.0, 0.3214, 0.6428]


def bank_angles():
    import math
    azimuth, elevation = [], []
    for row in range(5):
        el = math.asin(SINES[row])
        for column in range(5):
            elevation.append(math.degrees(el))
            azimuth.append(math.degrees(math.asin(max(-1.0, min(1.0, SINES[column] / math.cos(el))))))
    return azimuth, elevation


def bank_program(selectors=True):
    azimuth, elevation = bank_angles()
    return DeployedProgram(name="Bank", style="per_mic", table_words=3571, selectors=selectors,
                           bank_rows=5, bank_columns=5, bank_rate_hz=12000,
                           bank_azimuth_deg=tuple(azimuth), bank_elevation_deg=tuple(elevation))


class ScriptedBankEmulator(EmulatorHeimdallTransport):
    def __init__(self, program, bank_fit=None):
        super().__init__(rows=7, columns=7, azimuth_min_deg=-40, azimuth_max_deg=40,
                         elevation_min_deg=-40, elevation_max_deg=40, sector_time_s=0.0, program=program,
                         bank_fit=bank_fit)
        self.map_levels = [1000] * 25
        self.beam = 0
        self.running = True
        self.connected = True
        self._emit_configuration()

    def _level_for_direction(self, azimuth_deg, elevation_deg):
        level = self.map_levels[self.beam % 25]
        self.beam += 1
        return level

    def _level_for_sector(self, sector):
        return 5000


def test_bank_records_parse_and_reach_the_snapshot():
    state = HeimdallState()
    state.apply_line("READY,2D,7,7,49,44")
    state.apply_line("BANK,5,5,12000")
    state.apply_line("OUTPUTS,44,25")
    state.apply_line("SCAN_STARTED,B")
    state.apply_line("LEVELS,B,5,5," + ",".join(["16777216"] + ["1677722"] * 24))
    state.apply_line("BANK_TARGET,12,13,1.5000,-2.0000,24,16777216")
    state.apply_line("CH1,2,13,0")
    snapshot = state.snapshot()
    assert snapshot["mode"] == "B"
    assert snapshot["outputs"] == {"microphones": 44, "bank_beams": 25}
    assert snapshot["ch1"] == {"source": 2, "index": 13, "code": 0}
    bank = snapshot["bank"]
    assert (bank["rows"], bank["columns"], bank["rate_hz"], bank["maps"]) == (5, 5, 12000, 1)
    assert bank["levels_db"][0][0] == 0.0
    assert bank["levels_db"][4][4] == pytest.approx(-20.0, abs=0.01)
    assert bank["target"]["beam"] == 12 and bank["target"]["second_beam"] == 13
    assert bank["target"]["sector"] == 24
    state.apply_line("SCAN_STOPPED")
    assert state.snapshot()["bank"]["target"] is None
    # INFO re-announces the bank; until then it is unknown.
    state.apply_line("READY,2D,7,7,49,44")
    assert state.snapshot()["bank"] is None and state.snapshot()["outputs"] is None


def test_bank_fit_record_reaches_the_snapshot_until_the_stop():
    state = HeimdallState()
    state.apply_line("READY,2D,7,7,49,44")
    state.apply_line("BANK,5,5,12000")
    state.apply_line("SCAN_STARTED,B")
    state.apply_line("LEVELS,B,5,5," + ",".join(["1000"] * 25))
    assert state.snapshot()["bank"]["fit"] is None
    state.apply_line("BANK_FIT,12.5000,-3.2500,812,31,4870")
    assert state.snapshot()["bank"]["fit"] == {"azimuth_deg": 12.5, "elevation_deg": -3.25, "quality": 0.812,
                                               "sector": 31, "elapsed_us": 4870}
    state.apply_line("SCAN_STOPPED")
    assert state.snapshot()["bank"]["fit"] is None


@pytest.mark.parametrize("line", [
    "BANK,0,5,12000",
    "LEVELS,B,2,2,1,2,3",
    "LEVELS,X,1,1,5",
    "BANK_TARGET,1,2,0,0,3",
    "BANK_FIT,1,2,500,3",
    "BANK_FIT,1,2,1001,3,0",
    "CH1,3,0,0",
    "SCAN_STARTED,Z",
])
def test_malformed_bank_records_are_rejected(line):
    with pytest.raises(ProtocolError):
        parse_line(line)


def test_bank_levels_need_a_matching_bank_record():
    state = HeimdallState()
    state.apply_line("READY,2D,7,7,49,44")
    with pytest.raises(ProtocolError):
        state.apply_line("LEVELS,B,1,1,5")
    state.apply_line("BANK,5,5,12000")
    with pytest.raises(ProtocolError):
        state.apply_line("LEVELS,B,1,1,5")
    with pytest.raises(ProtocolError):
        state.apply_line("BANK_TARGET,0,0,0,0,49,1")


def test_stare_and_output_commands_validate():
    assert validate_command("B") == "B"
    assert validate_command("O,0,0") == "O,0,0"
    assert validate_command("O,2,24") == "O,2,24"
    for command in ("O,3,0", "O,1", "O,1,1234", "b"):
        with pytest.raises(ValueError):
            validate_command(command)


def tracker(freeze_maps=0):
    azimuth, elevation = bank_angles()
    sector_angles = [-34.3, -22.9, -11.4, 0.0, 11.4, 22.9, 34.3]
    return BankTracker(5, 5, azimuth, elevation, sector_angles, sector_angles, freeze_maps=freeze_maps)


def test_tracker_primes_then_finds_a_single_source():
    bank = tracker()
    assert bank.update([1000] * 25).beams == []
    levels = [1000] * 25
    levels[18] = 4000
    levels[17] = 2000
    result = bank.update(levels)
    assert result.beams == [18]
    azimuth, elevation = bank_angles()
    assert azimuth[17] < result.azimuth_deg < azimuth[18]
    assert result.elevation_deg == pytest.approx(elevation[18], abs=0.5)
    assert result.sector == bank.nearest_sector(result.azimuth_deg, result.elevation_deg)


def test_tracker_ignores_a_field_wide_rise_and_keeps_two_peaks():
    bank = tracker()
    bank.update([1000] * 25)
    assert bank.update([3000] * 25).beams == []
    bank = tracker()
    bank.update([1000] * 25)
    levels = [1000] * 25
    levels[0], levels[24], levels[12] = 3000, 5000, 4000
    result = bank.update(levels)
    assert result.beams == [24, 12]
    assert result.levels == [5000, 4000]


def test_tracker_background_follows_slowly_while_held():
    bank = tracker()
    bank.update([1000] * 25)
    levels = [1000] * 25
    levels[12] = 4000
    for _ in range(20):
        assert bank.update(levels).beams == [12]
    assert bank.background[12] >> FRACTION_BITS < 1200
    assert bank.background[0] >> FRACTION_BITS == 1000


def test_tracker_freeze_keeps_a_held_background_then_adapts_slowly():
    bank = tracker(freeze_maps=20)
    bank.update([1000] * 25)
    levels = [1000] * 25
    levels[12] = 4000
    for _ in range(20):
        assert bank.update(levels).beams == [12]
        assert bank.background[12] == 1000 << FRACTION_BITS
    bank.update(levels)
    assert 1000 < bank.background[12] >> FRACTION_BITS < 1100
    # A quiet map ends the hold, so the next hold starts a new freeze.
    assert bank.update([1000] * 25).beams == []
    assert bank.held_maps[12] == 0


def test_emulator_stare_takes_the_freeze_from_the_settings():
    emulator = ScriptedBankEmulator(bank_program())
    emulator._handle_command("B")
    assert emulator.bank_tracker.freeze_maps == 1200
    emulator.parameter_service._settings["freeze_s"] = 0
    emulator._handle_command("B")
    assert emulator.bank_tracker.freeze_maps == 0


def test_target_lost_clears_the_bank_target_and_fit():
    state = HeimdallState()
    state.apply_line("READY,2D,7,7,49,44")
    state.apply_line("BANK,5,5,12000")
    state.apply_line("SCAN_STARTED,B")
    state.apply_line("BANK_FIT,12.5000,-3.2500,812,31,4870")
    state.apply_line("BANK_TARGET,12,13,1.5000,-2.0000,24,16777216")
    state.apply_line("TARGET_ACQUIRED,24,3,3,0.0000,0.0000,1000")
    bank = state.snapshot()["bank"]
    assert bank["target"] is not None and bank["fit"] is not None
    state.apply_line("TARGET_LOST,24,3,3,0.0000,0.0000,10")
    bank = state.snapshot()["bank"]
    assert bank["target"] is None and bank["fit"] is None
    assert state.snapshot()["mode"] == "B"


def test_emulator_stare_reports_maps_target_and_listening_beam():
    emulator = ScriptedBankEmulator(bank_program())
    snapshot = emulator.snapshot()
    assert snapshot["bank"]["rows"] == 5 and snapshot["outputs"]["bank_beams"] == 25
    emulator._handle_command("B")
    emulator._stare_cycle()
    assert emulator.snapshot()["bank"]["target"] is None
    emulator.map_levels[6] = 6000
    emulator.map_levels[18] = 4000
    emulator._stare_cycle()
    snapshot = emulator.snapshot()
    assert snapshot["mode"] == "B"
    assert snapshot["bank"]["maps"] == 2
    assert snapshot["bank"]["target"]["beam"] == 6
    assert snapshot["bank"]["target"]["second_beam"] == 18
    assert snapshot["ch1"] == {"source": 2, "index": 18, "code": 0}
    assert snapshot["target"]["sector"] == snapshot["bank"]["target"]["sector"]
    assert emulator.protocol_errors == 0

    emulator.map_levels = [1000] * 25
    for _ in range(LOST_MAPS):
        emulator._stare_cycle()
    assert emulator.snapshot()["target"] is None
    assert emulator.snapshot()["bank"]["target"] is None


def test_emulator_without_a_bank_refuses_stare_and_outputs():
    emulator = ScriptedBankEmulator(DeployedProgram())
    emulator._handle_command("B")
    assert emulator.mode == "IDLE"
    emulator._handle_command("O,1,3")
    assert emulator.snapshot()["ch1"]["code"] == -1
    emulator = ScriptedBankEmulator(bank_program())
    emulator._handle_command("O,1,43")
    assert emulator.snapshot()["ch1"] == {"source": 1, "index": 43, "code": 0}
    emulator._handle_command("O,2,25")
    assert emulator.snapshot()["ch1"]["code"] == -1


@pytest.mark.skipif(not BANK_CONTRACT.exists(), reason="bank export not present")
def test_deployed_program_reads_the_bank_contract():
    program = DeployedProgram.from_contract(json.loads(BANK_CONTRACT.read_text(encoding="ascii")))
    assert (program.bank_rows, program.bank_columns, program.bank_rate_hz) == (5, 5, 12000)
    assert len(program.bank_azimuth_deg) == 25 and program.selectors
    assert program.configuration_lines() == ["MOUNT,cw90", "BANK,5,5,12000", "OUTPUTS,44,25"]


def fit_emulator():
    contract = json.loads(BANK_CONTRACT.read_text(encoding="ascii"))
    fit = BankFit.from_contract(contract)
    emulator = ScriptedBankEmulator(DeployedProgram.from_contract(contract), bank_fit=fit)
    lines = []
    apply_line = emulator.apply_line
    emulator.apply_line = lambda line: lines.append(line) or apply_line(line)
    return emulator, fit, lines


def stare_at(emulator, fit, azimuth_deg, elevation_deg):
    emulator._handle_command("B")
    emulator._stare_cycle()
    emulator.map_levels = [int(round(v)) for v in 4000 * fit.fingerprint(azimuth_deg, elevation_deg) + 300]
    for _ in range(3):
        emulator._stare_cycle()


@pytest.mark.skipif(not BANK_CONTRACT.exists(), reason="bank export not present")
def test_emulator_stare_reports_the_pattern_fit_and_steers_to_it():
    emulator, fit, lines = fit_emulator()
    stare_at(emulator, fit, 20.0, -5.0)
    fits = [line for line in lines if line.startswith("BANK_FIT,")]
    targets = [line for line in lines if line.startswith("BANK_TARGET,")]
    assert len(fits) == len(targets) >= 1
    # As the firmware: BANK_FIT comes just before the map's BANK_TARGET, which carries the fitted sector.
    assert lines.index(fits[0]) + 1 == lines.index(targets[0])
    _, azimuth, elevation, quality, sector, _ = fits[-1].split(",")
    assert abs(float(azimuth) - 20.0) <= 1.0 and abs(float(elevation) + 5.0) <= 1.0 and int(quality) >= 990
    assert targets[-1].split(",")[5] == sector
    snapshot = emulator.snapshot()
    assert snapshot["bank"]["fit"]["sector"] == int(sector) == snapshot["target"]["sector"]
    assert emulator.protocol_errors == 0


@pytest.mark.skipif(not BANK_CONTRACT.exists(), reason="bank export not present")
def test_emulator_stare_with_the_fit_off_steers_to_the_best_beam():
    emulator, fit, lines = fit_emulator()
    emulator.parameter_service.set(emulator, "fit", 0)
    stare_at(emulator, fit, 20.0, -5.0)
    assert not [line for line in lines if line.startswith("BANK_FIT,")]
    assert [line for line in lines if line.startswith("BANK_TARGET,")]
    assert emulator.snapshot()["bank"]["fit"] is None
