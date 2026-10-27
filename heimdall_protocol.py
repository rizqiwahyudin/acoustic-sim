"""Strict parser and state model for the Heimdall firmware protocol."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
import math
from typing import Any


class ProtocolError(ValueError):
    """Raised when a firmware line violates the Heimdall protocol."""


# Firmware runtime settings (MAX78002 heimdall_settings.c): name -> (minimum, maximum, default).
FIRMWARE_SETTINGS = {
    "settle_us": (50, 20000, 1250),
    "guard_us": (0, 20000, 2000),
    "window_us": (1000, 100000, 5000),
    "measure": (0, 1, 1),
    "band": (0, 1, 1),
    "band_low_hz": (100, 5000, 500),
    "band_high_hz": (500, 12000, 3500),
    "channel": (0, 1, 0),
    "map_record": (0, 1, 0),
    "tone": (0, 10000, 0),
    "fit": (0, 1, 1),
    "freeze_s": (0, 600, 60),
}

# How the board is turned, seen from behind (beamforming_mathematics/array_mount.py).
ARRAY_MOUNTS = ("upright", "cw90", "180", "ccw90")


@dataclass(frozen=True)
class BeamConfiguration:
    rows: int
    columns: int
    sectors: int
    microphones: int


def _integer(text: str, name: str, minimum: int = 0, maximum: int = 0xFFFFFFFF) -> int:
    if not text or not text.isdecimal():
        raise ProtocolError(f"{name} must be an unsigned decimal integer")
    value = int(text)
    if not minimum <= value <= maximum:
        raise ProtocolError(f"{name} is out of range")
    return value


def _number(text: str, name: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise ProtocolError(f"{name} must be numeric") from error
    if not math.isfinite(value):
        raise ProtocolError(f"{name} must be finite")
    return value


def _signed_integer(text: str, name: str) -> int:
    try:
        return int(text)
    except ValueError as error:
        raise ProtocolError(f"{name} must be an integer") from error


def parse_line(line: str) -> dict[str, Any]:
    """Parse one newline-stripped firmware record into a typed dictionary."""
    if not isinstance(line, str):
        raise ProtocolError("record must be text")
    line = line.strip()
    if not line:
        raise ProtocolError("record is empty")

    parts = line.split(",")
    record_type = parts[0]

    if record_type == "READY" and len(parts) == 6:
        if parts[1] != "2D":
            raise ProtocolError("unsupported READY protocol")
        rows = _integer(parts[2], "rows", 1, 255)
        columns = _integer(parts[3], "columns", 1, 255)
        sectors = _integer(parts[4], "sectors", 1, 65535)
        microphones = _integer(parts[5], "microphones", 1, 255)
        if sectors != rows * columns:
            raise ProtocolError("sector count does not match rows and columns")
        return {
            "type": "ready",
            "configuration": BeamConfiguration(rows, columns, sectors, microphones),
        }

    if record_type in {"AZIMUTH", "ELEVATION"} and len(parts) >= 3:
        count = _integer(parts[1], "angle count", 1, 255)
        if len(parts) != count + 2:
            raise ProtocolError("angle count does not match payload")
        return {
            "type": record_type.lower(),
            "angles_deg": [_number(value, "angle") for value in parts[2:]],
        }

    if record_type == "DSP" and len(parts) == 7:
        if not parts[1]:
            raise ProtocolError("DSP program name is empty")
        if parts[2] not in {"per_mic", "lut"}:
            raise ProtocolError("unknown steering style")
        return {
            "type": "dsp",
            "program": parts[1],
            "style": parts[2],
            "steer_words": _integer(parts[3], "steer words", 1, 255),
            "read": _integer(parts[4], "signature words read", 0, 65535),
            "matched": _integer(parts[5], "signature words matched", 0, 65535),
            "code": _signed_integer(parts[6], "error code"),
        }

    if record_type == "DSP_LOAD" and len(parts) == 6:
        return {
            "type": "dsp_load",
            "steps": _integer(parts[1], "load steps", 0, 65535),
            "writes": _integer(parts[2], "load writes", 0, 65535),
            "bytes": _integer(parts[3], "load bytes", 0, 0xFFFFFFFF),
            "elapsed_us": _integer(parts[4], "load time", 0, 0xFFFFFFFF),
            "code": _signed_integer(parts[5], "error code"),
        }

    if record_type == "DSP_RESET" and len(parts) == 3:
        return {
            "type": "dsp_reset",
            "code": _signed_integer(parts[1], "error code"),
            "elapsed_us": _integer(parts[2], "reset time", 0, 0xFFFFFFFF),
        }

    if record_type == "DSP_CORE" and len(parts) == 3:
        if len(parts[2]) != 4:
            raise ProtocolError("DSP panic code is 4 hex digits")
        try:
            panic_code = int(parts[2], 16)
        except ValueError as error:
            raise ProtocolError("DSP panic code is 4 hex digits") from error
        return {
            "type": "dsp_core",
            "status": _integer(parts[1], "core status", 0, 255),
            "panic_code": panic_code,
        }

    if record_type == "TABLES" and len(parts) == 5:
        return {
            "type": "tables",
            "words": _integer(parts[1], "table words", 0, 65535),
            "written": _integer(parts[2], "written table words", 0, 65535),
            "verified": _integer(parts[3], "verified table words", 0, 65535),
            "code": _signed_integer(parts[4], "error code"),
        }

    if record_type == "TAPER" and len(parts) == 5:
        return {
            "type": "taper",
            "sigma": _integer(parts[1], "taper sigma", 0, 100000) / 1000.0,
            "written": _integer(parts[2], "written gains", 0, 255),
            "verified": _integer(parts[3], "verified gains", 0, 255),
            "code": _signed_integer(parts[4], "error code"),
        }

    if record_type == "P" and len(parts) == 6:
        return {
            "type": "measurement",
            "row": _integer(parts[1], "row", 0, 254),
            "column": _integer(parts[2], "column", 0, 254),
            "azimuth_deg": _number(parts[3], "azimuth"),
            "elevation_deg": _number(parts[4], "elevation"),
            "raw_level": _integer(parts[5], "raw level"),
        }

    if record_type == "SCAN_DONE" and len(parts) == 1:
        return {"type": "scan_done"}

    if record_type == "SCAN_STOPPED" and len(parts) == 1:
        return {"type": "scan_stopped"}

    if record_type == "SCAN_STARTED" and len(parts) == 2 and parts[1] in {"F", "C", "G", "B"}:
        return {"type": "scan_started", "mode": parts[1]}

    if record_type == "BANK" and len(parts) == 4:
        return {
            "type": "bank",
            "rows": _integer(parts[1], "bank rows", 1, 20),
            "columns": _integer(parts[2], "bank columns", 1, 20),
            "rate_hz": _integer(parts[3], "bank rate", 1),
        }

    if record_type == "OUTPUTS" and len(parts) == 3:
        return {
            "type": "outputs",
            "microphones": _integer(parts[1], "microphones", 1, 255),
            "bank_beams": _integer(parts[2], "bank beams", 0, 400),
        }

    if record_type == "MOUNT" and len(parts) == 2 and parts[1] in ARRAY_MOUNTS:
        return {"type": "mount", "mount": parts[1]}

    if record_type == "LEVELS" and len(parts) >= 5 and parts[1] in {"B", "S"}:
        rows = _integer(parts[2], "rows", 1, 255)
        columns = _integer(parts[3], "columns", 1, 255)
        if len(parts) != 4 + rows * columns:
            raise ProtocolError("level count does not match rows and columns")
        return {
            "type": "levels",
            "source": parts[1],
            "rows": rows,
            "columns": columns,
            "raw_levels": [_integer(value, "raw level") for value in parts[4:]],
        }

    if record_type == "SETTINGS" and len(parts) >= 2:
        values = {}
        for item in parts[1:]:
            key, separator, value = item.partition("=")
            if not separator or not key:
                raise ProtocolError("SETTINGS items are key=value")
            values[key] = _signed_integer(value, key)
        return {"type": "settings", "values": values}

    if record_type == "MAP" and len(parts) == 2:
        if len(parts[1]) != 16 or any(c not in "0123456789abcdef" for c in parts[1]):
            raise ProtocolError("MAP must carry 16 lower-case hex digits")
        return {"type": "map", "hash": parts[1]}

    if record_type == "SET_OK" and len(parts) == 3 and parts[1]:
        return {"type": "set_ok", "key": parts[1], "value": _signed_integer(parts[2], "value")}

    if record_type == "PW_OK" and len(parts) == 3:
        return {"type": "pw_ok", "address": _integer(parts[1], "address", 0, 0xFFFF),
                "count": _integer(parts[2], "count", 1, 5)}

    if record_type == "PV" and 3 <= len(parts) <= 18:
        words = []
        for value in parts[2:]:
            if len(value) != 8:
                raise ProtocolError("PV words are 8 hex digits")
            try:
                word = int(value, 16)
            except ValueError as error:
                raise ProtocolError("PV words are 8 hex digits") from error
            words.append(word - (1 << 32) if word >= (1 << 31) else word)
        return {"type": "pv", "address": _integer(parts[1], "address", 0, 0xFFFF), "words": words}

    if record_type == "TONE" and len(parts) == 6:
        return {
            "type": "tone",
            "channel": _integer(parts[1], "channel", 0, 1),
            "f0_hz": _number(parts[2], "f0"),
            "score_db": _number(parts[3], "score"),
            "harmonics": _integer(parts[4], "harmonics", 0, 64),
            "level_db": _number(parts[5], "level"),
        }

    if record_type == "CLASS" and len(parts) == 4 and parts[2]:
        return {"type": "class", "channel": _integer(parts[1], "channel", 0, 1),
                "label": parts[2], "confidence": _number(parts[3], "confidence")}

    if record_type == "DIRSET" and len(parts) == 7:
        return {
            "type": "dirset",
            "rows": _integer(parts[1], "rows", 1, 255),
            "columns": _integer(parts[2], "columns", 1, 255),
            "transferred": _integer(parts[3], "transferred", 0, 65535),
            "verified": _integer(parts[4], "verified", 0, 65535),
            "code": _signed_integer(parts[5], "error code"),
            "elapsed_us": _integer(parts[6], "elapsed"),
        }

    if record_type == "REC_META" and len(parts) == 4:
        return {"type": "recording_meta", "sector": _signed_integer(parts[1], "sector"),
                "ch1_source": _integer(parts[2], "Ch1 source", 0, 2),
                "ch1_index": _integer(parts[3], "Ch1 index", 0, 399)}

    if record_type == "BANK_FIT" and len(parts) == 6:
        return {
            "type": "bank_fit",
            "azimuth_deg": _number(parts[1], "azimuth"),
            "elevation_deg": _number(parts[2], "elevation"),
            "quality": _integer(parts[3], "quality", 0, 1000) / 1000.0,
            "sector": _integer(parts[4], "sector", 0, 65535),
            "elapsed_us": _integer(parts[5], "elapsed"),
        }

    if record_type == "BANK_TARGET" and len(parts) == 7:
        return {
            "type": "bank_target",
            "beam": _integer(parts[1], "beam", 0, 399),
            "second_beam": _integer(parts[2], "second beam", 0, 399),
            "azimuth_deg": _number(parts[3], "azimuth"),
            "elevation_deg": _number(parts[4], "elevation"),
            "sector": _integer(parts[5], "sector", 0, 65535),
            "raw_level": _integer(parts[6], "raw level"),
        }

    if record_type == "CH1" and len(parts) == 4:
        return {
            "type": "ch1",
            "source": _integer(parts[1], "Ch1 source", 0, 2),
            "index": _integer(parts[2], "Ch1 index", 0, 399),
            "code": _signed_integer(parts[3], "error code"),
        }

    if record_type == "STEER_OK" and len(parts) == 6:
        return {
            "type": "steer_ok",
            **_parse_sector_fields(parts[1:6]),
        }

    if record_type == "STEER_ERR" and len(parts) == 4:
        return {
            "type": "steer_error",
            "sector": _integer(parts[1], "sector", 0, 65535),
            "microphone": _integer(parts[2], "microphone", 0, 255),
            "code": _signed_integer(parts[3], "error code"),
        }

    if record_type == "TIMING" and len(parts) == 3 and parts[1] == "S":
        return {"type": "timing", "mode": "S", "elapsed_us": _integer(parts[2], "elapsed")}

    if record_type == "TIMING" and len(parts) == 7 and parts[1] in {"F", "C", "G"}:
        return {
            "type": "timing",
            "mode": parts[1],
            "sector_count": _integer(parts[2], "sector count", 0, 65535),
            "core_us": _integer(parts[3], "core time"),
            "wall_us": _integer(parts[4], "wall time"),
            "min_sector_us": _integer(parts[5], "minimum sector time"),
            "max_sector_us": _integer(parts[6], "maximum sector time"),
        }

    if record_type in {"TARGET_ACQUIRED", "TARGET_UPDATED", "TARGET_LOST"} and len(parts) == 7:
        return {
            "type": record_type.lower(),
            **_parse_sector_fields(parts[1:6]),
            "raw_level": _integer(parts[6], "raw level"),
        }

    if record_type == "AUDIO_READY" and len(parts) == 4:
        return {
            "type": "audio_ready",
            "sample_rate_hz": _integer(parts[1], "sample rate", 1),
            "bits_per_sample": _integer(parts[2], "bits per sample", 1, 32),
            "channels": _integer(parts[3], "channels", 1, 2),
        }

    if record_type == "REC_STARTED" and len(parts) == 4:
        return {
            "type": "recording_started",
            "sample_rate_hz": _integer(parts[1], "sample rate", 1),
            "bits_per_sample": _integer(parts[2], "bits per sample", 1, 32),
            "channels": _integer(parts[3], "channels", 1, 2),
        }

    if record_type == "REC_STOPPED" and len(parts) == 8:
        return {
            "type": "recording_stopped",
            "samples": _integer(parts[1], "samples"),
            "bytes": _integer(parts[2], "bytes"),
            "minimum": _signed_integer(parts[3], "minimum"),
            "maximum": _signed_integer(parts[4], "maximum"),
            "dma_completions": _integer(parts[5], "DMA completions"),
            "overruns": _integer(parts[6], "overruns"),
            "sum_squares": _integer(parts[7], "sum of squares", 0, 0xFFFFFFFFFFFFFFFF),
        }

    if record_type == "REC_ERROR" and len(parts) == 3:
        return {
            "type": "recording_error",
            "stage": parts[1],
            "code": _signed_integer(parts[2], "error code"),
        }

    if record_type == "AUDIO_BEGIN" and len(parts) in {7, 8}:
        crc32 = parts[-1]
        if len(crc32) != 8 or any(character not in "0123456789abcdefABCDEF" for character in crc32):
            raise ProtocolError("audio CRC32 must contain eight hexadecimal digits")
        return {
            "type": "audio_begin",
            "version": _integer(parts[1], "audio version", 1, 255),
            "bytes": _integer(parts[2], "audio bytes", 1),
            "sample_rate_hz": _integer(parts[3], "sample rate", 1),
            "bits_per_sample": _integer(parts[4], "bits per sample", 1, 32),
            "channels": _integer(parts[5], "channels", 1, 2),
            "ack_interval": 0 if len(parts) == 7 else _integer(parts[6], "ACK interval", 1),
            "crc32": int(crc32, 16),
        }

    if record_type == "AUDIO_END" and len(parts) == 2:
        crc32 = parts[1]
        if len(crc32) != 8 or any(character not in "0123456789abcdefABCDEF" for character in crc32):
            raise ProtocolError("audio CRC32 must contain eight hexadecimal digits")
        return {"type": "audio_end", "crc32": int(crc32, 16)}

    if record_type == "ERR" and len(parts) >= 2:
        return {"type": "error", "reason": parts[1], "details": parts[2:]}

    raise ProtocolError(f"unknown or malformed record: {record_type}")


def _parse_sector_fields(parts: list[str]) -> dict[str, Any]:
    if len(parts) != 5:
        raise ProtocolError("sector record has the wrong field count")
    return {
        "sector": _integer(parts[0], "sector", 0, 65535),
        "row": _integer(parts[1], "row", 0, 254),
        "column": _integer(parts[2], "column", 0, 254),
        "azimuth_deg": _number(parts[3], "azimuth"),
        "elevation_deg": _number(parts[4], "elevation"),
    }


def raw_8_24_to_linear(raw_level: int) -> float:
    """Interpret the exact four readback bytes as signed 8.24 fixed point."""
    signed = raw_level if raw_level < 0x80000000 else raw_level - 0x100000000
    return signed / 16777216.0


def linear_to_db(level: float, floor_db: float = -120.0) -> float:
    if level <= 0.0:
        return floor_db
    return max(floor_db, 20.0 * math.log10(level))


class HeimdallState:
    """Mutable protocol state. Callers provide synchronization around access."""

    def __init__(self) -> None:
        self.sequence = 0
        self.configuration: BeamConfiguration | None = None
        self.azimuth_deg: list[float] = []
        self.elevation_deg: list[float] = []
        self.levels_raw: list[list[int | None]] = []
        self.scan_count = 0
        self.measurement_count = 0
        self.last_measurement: dict[str, int] | None = None
        self.mode = "IDLE"
        self.target: dict[str, Any] | None = None
        self.last_steer: dict[str, Any] | None = None
        self.last_timing: dict[str, Any] | None = None
        self.last_error: dict[str, Any] | None = None
        self.last_record: dict[str, Any] | None = None
        self.audio_format: dict[str, int] | None = None
        self.recording_state = "idle"
        self.last_recording: dict[str, Any] | None = None
        self.audio_transfer: dict[str, Any] | None = None
        self.taper: dict[str, Any] | None = None
        self.dsp: dict[str, Any] | None = None
        self.pending_dsp_load: dict[str, Any] | None = None
        self.pending_dsp_reset: dict[str, Any] | None = None
        self.tables: dict[str, Any] | None = None
        # TABLES records (each I reloads the taper and tables) and program loads, counted.
        self.boot_checks = 0
        self.dsp_loads = 0
        self.bank: dict[str, Any] | None = None
        self.bank_levels_raw: list[int] | None = None
        self.bank_maps = 0
        self.bank_target: dict[str, Any] | None = None
        self.bank_fit: dict[str, Any] | None = None
        self.outputs: dict[str, Any] | None = None
        self.array_mount: str | None = None
        self.ch1: dict[str, Any] | None = None
        self.settings: dict[str, int] = {}
        self.map_hash: str | None = None
        self.tones: dict[int, dict[str, Any]] = {}
        self.classes: dict[int, dict[str, Any]] = {}
        self.direction_set: dict[str, Any] | None = None
        self.recording_meta: dict[str, Any] | None = None

    def apply_line(self, line: str) -> dict[str, Any]:
        record = parse_line(line)
        record_type = record["type"]

        if record_type == "ready":
            configuration = record["configuration"]
            # The mount, bank and output lines follow READY on every INFO.
            self.bank = None
            self.bank_levels_raw = None
            self.bank_target = None
            self.bank_fit = None
            self.outputs = None
            self.array_mount = None
            if configuration != self.configuration:
                self.configuration = configuration
                self.azimuth_deg = []
                self.elevation_deg = []
                self.levels_raw = [
                    [None for _ in range(configuration.columns)]
                    for _ in range(configuration.rows)
                ]
                self.scan_count = 0
                self.target = None
                self.last_steer = None
                self.mode = "IDLE"
        elif record_type == "azimuth":
            self._require_configuration()
            if len(record["angles_deg"]) != self.configuration.columns:
                raise ProtocolError("azimuth count does not match READY")
            self.azimuth_deg = record["angles_deg"]
        elif record_type == "elevation":
            self._require_configuration()
            if len(record["angles_deg"]) != self.configuration.rows:
                raise ProtocolError("elevation count does not match READY")
            self.elevation_deg = record["angles_deg"]
        elif record_type == "measurement":
            self._validate_grid_position(record)
            self.levels_raw[record["row"]][record["column"]] = record["raw_level"]
            self._learn_angles(record)
            self.measurement_count += 1
            self.last_measurement = {"count": self.measurement_count, "row": record["row"],
                                     "column": record["column"], "raw_level": record["raw_level"]}
        elif record_type == "scan_started":
            self.mode = record["mode"]
            self.target = None
            self.last_steer = None
            self.bank_target = None
            self.bank_fit = None
        elif record_type == "bank":
            self.bank = {key: record[key] for key in ("rows", "columns", "rate_hz")}
        elif record_type == "outputs":
            self.outputs = {key: record[key] for key in ("microphones", "bank_beams")}
        elif record_type == "mount":
            self.array_mount = record["mount"]
        elif record_type == "levels":
            if record["source"] == "S":
                self._require_configuration()
                if (record["rows"], record["columns"]) != (self.configuration.rows, self.configuration.columns):
                    raise ProtocolError("map levels do not match READY dimensions")
                columns = record["columns"]
                raw = record["raw_levels"]
                self.levels_raw = [raw[row * columns:(row + 1) * columns] for row in range(record["rows"])]
            else:
                if self.bank is None or (record["rows"], record["columns"]) != (self.bank["rows"], self.bank["columns"]):
                    raise ProtocolError("bank levels do not match the BANK record")
                self.bank_levels_raw = record["raw_levels"]
                self.bank_maps += 1
        elif record_type == "settings":
            self.settings = dict(record["values"])
        elif record_type == "set_ok":
            self.settings[record["key"]] = record["value"]
        elif record_type == "map":
            self.map_hash = record["hash"]
        elif record_type == "tone":
            self.tones[record["channel"]] = {k: v for k, v in record.items() if k != "type"}
        elif record_type == "class":
            self.classes[record["channel"]] = {k: v for k, v in record.items() if k != "type"}
        elif record_type == "dirset":
            self.direction_set = {k: v for k, v in record.items() if k != "type"}
        elif record_type == "recording_meta":
            self.recording_meta = {k: v for k, v in record.items() if k != "type"}
        elif record_type == "bank_target":
            self._require_configuration()
            if record["sector"] >= self.configuration.sectors:
                raise ProtocolError("bank target sector is outside READY dimensions")
            self.bank_target = record
        elif record_type == "bank_fit":
            self.bank_fit = {key: value for key, value in record.items() if key != "type"}
        elif record_type == "ch1":
            self.ch1 = {key: record[key] for key in ("source", "index", "code")}
        elif record_type == "scan_done":
            self.scan_count += 1
            if self.mode == "F":
                self.mode = "IDLE"
        elif record_type == "scan_stopped":
            self.mode = "IDLE"
            self.target = None
            self.bank_target = None
            self.bank_fit = None
        elif record_type == "steer_ok":
            self._validate_grid_position(record)
            self.mode = "IDLE"
            self.target = None
            self.last_steer = record
            self._learn_angles(record)
        elif record_type == "timing":
            self.last_timing = record
        elif record_type == "dsp_reset":
            self.pending_dsp_reset = {key: record[key] for key in ("code", "elapsed_us")}
        elif record_type == "dsp_load":
            # The MCU loaded the program; the DSP check that follows reports the result. A failed
            # check is retried, so a later load replaces this one and counts as another attempt.
            attempts = self.pending_dsp_load["attempts"] + 1 if self.pending_dsp_load else 1
            self.pending_dsp_load = {
                **{key: record[key] for key in ("steps", "writes", "bytes", "elapsed_us", "code")},
                "reset": self.pending_dsp_reset,
                "attempts": attempts,
            }
            self.pending_dsp_reset = None
        elif record_type == "dsp":
            # Every program check starts afresh: on a mismatch the firmware writes nothing.
            if self.pending_dsp_load is not None:
                self.dsp_loads += 1
            self.dsp = {
                **{key: record[key] for key in ("program", "style", "steer_words", "read", "matched", "code")},
                "matches": record["code"] == 0 and 0 < record["read"] == record["matched"],
                "load": self.pending_dsp_load,
                "core": None,
            }
            self.pending_dsp_load = None
            self.pending_dsp_reset = None
            self.taper = None
            self.tables = None
        elif record_type == "dsp_core":
            # Follows a failed check: status 0 with a panic code means the DSP core stopped.
            if self.dsp is not None:
                self.dsp["core"] = {key: record[key] for key in ("status", "panic_code")}
        elif record_type == "tables":
            self.boot_checks += 1
            self.tables = {
                **{key: record[key] for key in ("words", "written", "verified", "code")},
                "loaded": record["code"] == 0 and record["written"] == record["verified"] == record["words"],
            }
        elif record_type == "taper":
            microphones = self.configuration.microphones if self.configuration else record["written"]
            self.taper = {
                **{key: record[key] for key in ("sigma", "written", "verified", "code")},
                "applied": record["code"] == 0 and record["written"] == record["verified"] == microphones,
            }
        elif record_type.startswith("target_"):
            self._validate_grid_position(record)
            self._learn_angles(record)
            self.target = None if record_type == "target_lost" else record
            if record_type == "target_lost":
                # Live: the last bank target and fit belong to the lost target.
                self.bank_target = None
                self.bank_fit = None
        elif record_type in {"error", "steer_error"}:
            self.last_error = record
        elif record_type == "audio_ready":
            self.audio_format = {
                "sample_rate_hz": record["sample_rate_hz"],
                "bits_per_sample": record["bits_per_sample"],
                "channels": record["channels"],
            }
        elif record_type == "recording_started":
            self.recording_state = "recording"
            self.last_recording = record
            self.audio_transfer = None
            self.recording_meta = None
        elif record_type == "recording_stopped":
            self.recording_state = "ready"
            self.last_recording = record
        elif record_type == "recording_error":
            self.recording_state = "error"
            self.last_recording = record
            self.last_error = record
        elif record_type == "audio_begin":
            self.recording_state = "downloading"
            self.audio_transfer = record
        elif record_type == "audio_end":
            self.recording_state = "downloaded"
            if self.audio_transfer is not None:
                self.audio_transfer = {**self.audio_transfer, "end_crc32": record["crc32"]}

        self.sequence += 1
        self.last_record = record
        return record

    def snapshot(self) -> dict[str, Any]:
        configuration = self.configuration
        levels_linear = [
            [None if value is None else raw_8_24_to_linear(value) for value in row]
            for row in self.levels_raw
        ]
        levels_db = [
            [None if value is None else linear_to_db(value) for value in row]
            for row in levels_linear
        ]
        return {
            "sequence": self.sequence,
            "configuration": None if configuration is None else {
                "rows": configuration.rows,
                "columns": configuration.columns,
                "sectors": configuration.sectors,
                "microphones": configuration.microphones,
            },
            "azimuth_deg": self.azimuth_deg.copy(),
            "elevation_deg": self.elevation_deg.copy(),
            "levels_raw": deepcopy(self.levels_raw),
            "levels_linear": levels_linear,
            "levels_db": levels_db,
            "scan_count": self.scan_count,
            "last_measurement": None if self.last_measurement is None else {
                **self.last_measurement,
                "level_db": linear_to_db(raw_8_24_to_linear(self.last_measurement["raw_level"])),
            },
            "mode": self.mode,
            "target": deepcopy(self.target),
            "last_steer": deepcopy(self.last_steer),
            "last_timing": deepcopy(self.last_timing),
            "last_error": deepcopy(self.last_error),
            "last_record": None if self.last_record is None else {
                key: asdict(value) if is_dataclass(value) else deepcopy(value)
                for key, value in self.last_record.items()
            },
            "audio_format": deepcopy(self.audio_format),
            "recording_state": self.recording_state,
            "last_recording": deepcopy(self.last_recording),
            "audio_transfer": deepcopy(self.audio_transfer),
            "taper": deepcopy(self.taper),
            "dsp": deepcopy(self.dsp),
            "tables": deepcopy(self.tables),
            "boot_checks": self.boot_checks,
            "dsp_loads": self.dsp_loads,
            "bank": self._bank_snapshot(),
            "outputs": deepcopy(self.outputs),
            "array_mount": self.array_mount,
            "ch1": deepcopy(self.ch1),
            "settings": dict(self.settings),
            "map_hash": self.map_hash,
            "tones": {str(channel): dict(tone) for channel, tone in self.tones.items()},
            "classes": {str(channel): dict(item) for channel, item in self.classes.items()},
            "direction_set": deepcopy(self.direction_set),
            "recording_meta": deepcopy(self.recording_meta),
        }

    def _bank_snapshot(self) -> dict[str, Any] | None:
        if self.bank is None:
            return None
        rows, columns = self.bank["rows"], self.bank["columns"]
        raw = self.bank_levels_raw
        db = None if raw is None else [
            [linear_to_db(raw_8_24_to_linear(raw[row * columns + column])) for column in range(columns)]
            for row in range(rows)
        ]
        return {
            **self.bank,
            "levels_raw": None if raw is None else list(raw),
            "levels_db": db,
            "maps": self.bank_maps,
            "target": deepcopy(self.bank_target),
            # The firmware's pattern fit (BANK_FIT) of the latest map with a target.
            "fit": deepcopy(self.bank_fit),
        }

    def _require_configuration(self) -> None:
        if self.configuration is None:
            raise ProtocolError("READY must precede this record")

    def _validate_grid_position(self, record: dict[str, Any]) -> None:
        self._require_configuration()
        if record["row"] >= self.configuration.rows or record["column"] >= self.configuration.columns:
            raise ProtocolError("grid position is outside READY dimensions")
        expected_sector = record["row"] * self.configuration.columns + record["column"]
        if "sector" in record and record["sector"] != expected_sector:
            raise ProtocolError("sector does not match row and column")

    def _learn_angles(self, record: dict[str, Any]) -> None:
        if not self.azimuth_deg:
            self.azimuth_deg = [math.nan] * self.configuration.columns
        if not self.elevation_deg:
            self.elevation_deg = [math.nan] * self.configuration.rows
        self.azimuth_deg[record["column"]] = record["azimuth_deg"]
        self.elevation_deg[record["row"]] = record["elevation_deg"]