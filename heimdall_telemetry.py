"""Telemetry log: every line from the board and every command to it, with the host time.

One text file per connection, <directory>/<YYYY-MM-DD>/<HHMMSS>_<transport>.log, one entry per line:
<unix time in s>\t<direction>\t<text>, where the direction is '<' from the board, '>' to the board and
'#' a note from the host (start, marks, audio downloads, end). Audio payload bytes are not logged; their
AUDIO_BEGIN and AUDIO_END lines are. scripts/telemetry_log.py reads a log back.
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

FORMAT = "heimdall-telemetry-v1"


class TelemetryLog:
    def __init__(self, path: str | Path, transport_name: str, clock=time.time) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.lines = 0
        self._lock = threading.Lock()
        self._file = open(self.path, "w", encoding="utf-8", newline="\n", buffering=1)
        self.note(f"{FORMAT} transport={transport_name} started={datetime.now().isoformat(timespec='seconds')}")

    @classmethod
    def create(cls, directory: str | Path, transport_name: str) -> "TelemetryLog":
        now = datetime.now()
        name = re.sub(r"[^A-Za-z0-9]+", "_", transport_name).strip("_") or "board"
        return cls(Path(directory) / f"{now:%Y-%m-%d}" / f"{now:%H%M%S}_{name}.log", transport_name)

    def received(self, line: str) -> None:
        self._write("<", line)

    def sent(self, command: str) -> None:
        self._write(">", command)

    def note(self, text: str) -> None:
        self._write("#", " ".join(text.split()))

    def _write(self, direction: str, text: str) -> None:
        with self._lock:
            if self._file is None:
                return
            self._file.write(f"{self.clock():.6f}\t{direction}\t{text}\n")
            self.lines += 1

    def close(self) -> None:
        self.note("closed")
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            open_ = self._file is not None
            lines = self.lines
        size = self.path.stat().st_size if self.path.exists() else 0
        return {"path": str(self.path), "lines": lines, "bytes": size, "open": open_}
