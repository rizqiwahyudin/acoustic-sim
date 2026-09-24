"""Strict parser and state model for the Heimdall firmware protocol."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any


class ProtocolError(ValueError):
    """Raised when a firmware line violates the Heimdall protocol."""


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

    if record_type == "SCAN_STARTED" and len(parts) == 2 and parts[1] in {"F", "C", "G"}:
        return {"type": "scan_started", "mode": parts[1]}

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

    def apply_line(self, line: str) -> dict[str, Any]:
        record = parse_line(line)
        record_type = record["type"]

        if record_type == "ready":
            configuration = record["configuration"]
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
        elif record_type == "scan_started":
            self.mode = record["mode"]
            self.target = None
            self.last_steer = None
        elif record_type == "scan_done":
            self.scan_count += 1
            if self.mode == "F":
                self.mode = "IDLE"
        elif record_type == "scan_stopped":
            self.mode = "IDLE"
            self.target = None
        elif record_type == "steer_ok":
            self._validate_grid_position(record)
            self.mode = "IDLE"
            self.target = None
            self.last_steer = record
            self._learn_angles(record)
        elif record_type == "timing":
            self.last_timing = record
        elif record_type.startswith("target_"):
            self._validate_grid_position(record)
            self._learn_angles(record)
            self.target = None if record_type == "target_lost" else record
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
            "mode": self.mode,
            "target": deepcopy(self.target),
            "last_steer": deepcopy(self.last_steer),
            "last_timing": deepcopy(self.last_timing),
            "last_error": deepcopy(self.last_error),
            "last_record": deepcopy(self.last_record),
            "audio_format": deepcopy(self.audio_format),
            "recording_state": self.recording_state,
            "last_recording": deepcopy(self.last_recording),
            "audio_transfer": deepcopy(self.audio_transfer),
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