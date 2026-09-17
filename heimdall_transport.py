"""Real and emulated transports for the Heimdall firmware protocol."""

from __future__ import annotations

import math
from pathlib import Path
import queue
import random
import re
import threading
import time
from typing import Any

from heimdall_protocol import HeimdallState, ProtocolError


class HeimdallTransport:
    """Thread-safe shared state exposed by all Heimdall transports."""

    def __init__(self) -> None:
        self.state = HeimdallState()
        self.lock = threading.Lock()
        self.running = False
        self.connected = False
        self.transport_name = "disconnected"
        self.protocol_errors = 0

    def apply_line(self, line: str) -> None:
        try:
            with self.lock:
                self.state.apply_line(line)
        except ProtocolError:
            self.protocol_errors += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            snapshot = self.state.snapshot()
        snapshot.update({
            "connected": self.connected,
            "transport": self.transport_name,
            "protocol_errors": self.protocol_errors,
        })
        return snapshot

    def send_command(self, command: str) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        self.running = False


class SerialHeimdallTransport(HeimdallTransport, threading.Thread):
    def __init__(self, port: str, baud: int = 921600) -> None:
        HeimdallTransport.__init__(self)
        threading.Thread.__init__(self, daemon=True)
        import serial

        self.serial = serial.Serial(port, baud, timeout=0.05, write_timeout=0.5)
        self.port = port
        self.baud = baud
        self.transport_name = f"serial:{port}"
        self.write_lock = threading.Lock()
        self.rx_buffer = bytearray()

    def run(self) -> None:
        self.running = True
        self.connected = True
        self.send_command("I")
        try:
            while self.running:
                data = self.serial.read(max(1, self.serial.in_waiting))
                if not data:
                    continue
                self.rx_buffer.extend(data)
                if len(self.rx_buffer) > 65536:
                    self.rx_buffer.clear()
                    self.protocol_errors += 1
                    continue
                while b"\n" in self.rx_buffer:
                    raw_line, _, remainder = self.rx_buffer.partition(b"\n")
                    self.rx_buffer = bytearray(remainder)
                    try:
                        line = raw_line.decode("ascii", errors="strict").rstrip("\r")
                    except UnicodeError:
                        self.protocol_errors += 1
                        continue
                    self.apply_line(line)
        except OSError:
            self.protocol_errors += 1
        finally:
            self.connected = False
            self.running = False
            with self.lock:
                self.state.sequence += 1

    def send_command(self, command: str) -> None:
        command = validate_command(command)
        with self.write_lock:
            self.serial.write((command + "\n").encode("ascii"))
            self.serial.flush()

    def stop(self) -> None:
        self.running = False
        self.serial.close()


class EmulatorHeimdallTransport(HeimdallTransport, threading.Thread):
    """Protocol-faithful moving-target emulator using the real parsed state path."""

    def __init__(self, rows: int = 7, columns: int = 7,
                 azimuth_min_deg: float = -70.0, azimuth_max_deg: float = 70.0,
                 elevation_min_deg: float = -60.0, elevation_max_deg: float = 60.0,
                 sector_time_s: float = 0.0028) -> None:
        HeimdallTransport.__init__(self)
        threading.Thread.__init__(self, daemon=True)
        if not 1 <= rows <= 20 or not 1 <= columns <= 20:
            raise ValueError("emulator grid must be between 1x1 and 20x20")
        if not -90.0 <= azimuth_min_deg < azimuth_max_deg <= 90.0:
            raise ValueError("azimuth limits must satisfy -90 <= minimum < maximum <= 90")
        if not -90.0 <= elevation_min_deg < elevation_max_deg <= 90.0:
            raise ValueError("elevation limits must satisfy -90 <= minimum < maximum <= 90")
        self.rows = rows
        self.columns = columns
        self.azimuth_deg = _cell_centers(azimuth_min_deg, azimuth_max_deg, columns)
        self.elevation_deg = _cell_centers(elevation_min_deg, elevation_max_deg, rows)
        self.sector_time_s = sector_time_s
        self.commands: queue.Queue[str] = queue.Queue()
        self.mode = "IDLE"
        self.transport_name = "emulator"
        self.random = random.Random(78002)
        self.started_at = time.monotonic()
        self.current_sector: int | None = None

    def run(self) -> None:
        self.running = True
        self.connected = True
        self._emit_configuration()
        try:
            while self.running:
                try:
                    command = self.commands.get(timeout=0.01 if self.mode == "IDLE" else 0.0)
                    self._handle_command(command)
                except queue.Empty:
                    if self.mode == "C":
                        self._scan_pass("C")
                    elif self.mode == "G":
                        self._adaptive_cycle()
        finally:
            self.connected = False
            self.running = False

    def send_command(self, command: str) -> None:
        self.commands.put(validate_command(command, self.rows * self.columns))

    def _handle_command(self, command: str) -> None:
        if command == "I":
            self._emit_configuration()
        elif command == "F":
            self.mode = "F"
            self.apply_line("SCAN_STARTED,F")
            self._scan_pass("F")
            self.mode = "IDLE"
        elif command == "C":
            self.mode = "C"
            self.apply_line("SCAN_STARTED,C")
        elif command == "G":
            self.mode = "G"
            self.apply_line("SCAN_STARTED,G")
        elif command == "X":
            self.mode = "IDLE"
            self.apply_line("SCAN_STOPPED")
        elif command == "M":
            if self.current_sector is None:
                self.apply_line("ERR,NO_ACTIVE_BEAM")
            else:
                self._emit_measurement(self.current_sector, delay=False)
        elif command.startswith("S,"):
            self.mode = "IDLE"
            sector = int(command[2:])
            self.current_sector = sector
            self._emit_steer_ok(sector)
            self._emit_measurement(sector, delay=False)

    def _emit_configuration(self) -> None:
        sectors = self.rows * self.columns
        self.apply_line(f"READY,2D,{self.rows},{self.columns},{sectors},44")
        self.apply_line("AZIMUTH," + str(self.columns) + "," + ",".join(_format_angle(v) for v in self.azimuth_deg))
        self.apply_line("ELEVATION," + str(self.rows) + "," + ",".join(_format_angle(v) for v in self.elevation_deg))

    def _scan_pass(self, mode: str) -> None:
        started = time.monotonic()
        measured = 0
        for sector in range(self.rows * self.columns):
            if self._consume_interrupt():
                return
            self._emit_measurement(sector)
            measured += 1
        elapsed_us = max(1, int((time.monotonic() - started) * 1_000_000))
        sector_us = elapsed_us // max(1, measured)
        self.apply_line("SCAN_DONE")
        self.apply_line(f"TIMING,{mode},{measured},{elapsed_us},{elapsed_us},{sector_us},{sector_us}")

    def _adaptive_cycle(self) -> None:
        self._scan_pass("G")
        if self.mode != "G":
            return
        target_sector = self._nearest_target_sector()
        row, column = divmod(target_sector, self.columns)
        level = self._level_for_sector(target_sector)
        self.apply_line(
            f"TARGET_ACQUIRED,{target_sector},{row},{column},"
            f"{_format_angle(self.azimuth_deg[column])},{_format_angle(self.elevation_deg[row])},{level}"
        )
        for _ in range(12):
            if self._consume_interrupt():
                return
            target_sector = self._nearest_target_sector()
            for sector in self._cross(target_sector):
                if self._consume_interrupt():
                    return
                self._emit_measurement(sector)
            row, column = divmod(target_sector, self.columns)
            level = self._level_for_sector(target_sector)
            self.apply_line(
                f"TARGET_UPDATED,{target_sector},{row},{column},"
                f"{_format_angle(self.azimuth_deg[column])},{_format_angle(self.elevation_deg[row])},{level}"
            )

    def _consume_interrupt(self) -> bool:
        if not self.running:
            return True
        try:
            command = self.commands.get_nowait()
        except queue.Empty:
            return False
        self._handle_command(command)
        return self.mode not in {"C", "G"}

    def _emit_measurement(self, sector: int, delay: bool = True) -> None:
        self.current_sector = sector
        row, column = divmod(sector, self.columns)
        level = self._level_for_sector(sector)
        self.apply_line(
            f"P,{row},{column},{_format_angle(self.azimuth_deg[column])},"
            f"{_format_angle(self.elevation_deg[row])},{level}"
        )
        if delay and self.sector_time_s > 0:
            time.sleep(self.sector_time_s)

    def _emit_steer_ok(self, sector: int) -> None:
        row, column = divmod(sector, self.columns)
        self.apply_line(
            f"STEER_OK,{sector},{row},{column},{_format_angle(self.azimuth_deg[column])},"
            f"{_format_angle(self.elevation_deg[row])}"
        )
        self.apply_line(f"TIMING,S,{int(self.sector_time_s * 1_000_000)}")

    def _target_direction(self) -> tuple[float, float]:
        elapsed = time.monotonic() - self.started_at
        return 48.0 * math.sin(elapsed * 0.22), 34.0 * math.sin(elapsed * 0.15 + 0.8)

    def _nearest_target_sector(self) -> int:
        target_azimuth, target_elevation = self._target_direction()
        column = min(range(self.columns), key=lambda index: abs(self.azimuth_deg[index] - target_azimuth))
        row = min(range(self.rows), key=lambda index: abs(self.elevation_deg[index] - target_elevation))
        return row * self.columns + column

    def _level_for_sector(self, sector: int) -> int:
        row, column = divmod(sector, self.columns)
        target_azimuth, target_elevation = self._target_direction()
        azimuth_error = self.azimuth_deg[column] - target_azimuth
        elevation_error = self.elevation_deg[row] - target_elevation
        distance_squared = azimuth_error * azimuth_error + elevation_error * elevation_error
        linear = 0.015 + 0.85 * math.exp(-distance_squared / (2.0 * 18.0 * 18.0))
        linear += self.random.uniform(-0.004, 0.004)
        return max(0, min(0x7FFFFFFF, round(linear * 16777216.0)))

    def _cross(self, center: int) -> list[int]:
        row, column = divmod(center, self.columns)
        sectors = [center]
        if column > 0:
            sectors.append(center - 1)
        if column + 1 < self.columns:
            sectors.append(center + 1)
        if row > 0:
            sectors.append(center - self.columns)
        if row + 1 < self.rows:
            sectors.append(center + self.columns)
        return sectors


def validate_command(command: str, sectors: int = 65535) -> str:
    command = command.strip()
    if command in {"F", "C", "G", "X", "I", "M"}:
        return command
    match = re.fullmatch(r"S,(\d+)", command)
    if match and int(match.group(1)) < sectors:
        return command
    raise ValueError("invalid Heimdall command")


def _cell_centers(minimum: float, maximum: float, count: int) -> list[float]:
    step = (maximum - minimum) / count
    return [minimum + (index + 0.5) * step for index in range(count)]


def _format_angle(value: float) -> str:
    return f"{value:.4f}"