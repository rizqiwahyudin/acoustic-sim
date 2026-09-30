"""Heimdall DSP register map, parsed from a SigmaStudio export (read-only).

The export JSON lists every module parameter with its data-memory address,
size and default. This module turns it into a register map for the Study
screen, decides which parameters the GUI may write (an allow-list), and plans
writes as safeload commands.

SigmaStudio places parameters at compile time. Addresses of blocks of one kind
are NOT guaranteed to be contiguous or to increase by one, so the planner only
batches words whose addresses are actually consecutive, at most five per
command (the ADAU1467 safeload block size), and otherwise writes single words.

The PW/PR/SET device commands are a proposal (docs/study-protocol.md); the
firmware does not implement them yet. The emulator does, through
DspParameterService, so the GUI can be developed end to end without hardware.
"""

from __future__ import annotations

import json
import os
import re
import threading
from hashlib import sha256
from pathlib import Path
from typing import Any

SCHEMA = "heimdall-dsp-registry-v1"
MAX_WORDS_PER_COMMAND = 5
DEFAULT_EXPORT = (Path(__file__).resolve().parent.parent / "MAX78002" / "SigmaStudioExport"
                  / "MICCANVAS_SS+_BEAMFORM_Golden_Image_ADAU1467_0.json")

# First microphone (DSP index) behind channel 0 of each mute block, from the
# Golden Image schematic. Confirm with the DSP owner when the project changes.
MUTE_BLOCK_FIRST_MIC = {
    "MultipleControlMute_0": 0,
    "MultipleControlMute_3": 16,
    "MultipleControlMute_0_1": 20,
    "MultipleControlMute_2": 36,
}

GROUPS = {
    "gains": "Microphone gains",
    "fir1": "FIR stage 1",
    "fir2": "FIR stage 2",
    "mutes": "Channel mutes",
    "detector": "Level detector",
    "output": "Output level",
    "delays": "Fractional delays",
    "system": "Safeload registers",
    "other": "Other parameters",
}

TYPES = {"FixPoint8d24": "8.24", "Integer32": "int32"}


def export_path() -> Path:
    override = os.environ.get("HEIMDALL_DSP_EXPORT")
    return Path(override) if override else DEFAULT_EXPORT


def _classify(algo: str, param: str) -> tuple[str, int | None]:
    match = re.fullmatch(r"(Gain|FIR_Stage1|FIR_Stage2|Delay)_DSP(\d+)", algo)
    if match:
        kind = {"Gain": "gains", "FIR_Stage1": "fir1", "FIR_Stage2": "fir2", "Delay": "delays"}[match.group(1)]
        return kind, int(match.group(2))
    if algo in MUTE_BLOCK_FIRST_MIC:
        channel = re.search(r"(\d+)$", param)
        return "mutes", MUTE_BLOCK_FIRST_MIC[algo] + int(channel.group(1)) if channel else None
    if algo.startswith("SingleBandwNumericalDisplay"):
        return "detector", None
    if algo == "Beamformer_Output_Gain" or algo.startswith("SingleVolumeControl"):
        return "output", None
    if algo == "Safeload":
        return "system", None
    return "other", None


def _policy(group: str, param: str) -> tuple[bool, str]:
    if group in ("gains", "fir1", "fir2", "mutes", "detector", "output"):
        return True, ""
    if group == "delays":
        if param == "MaxDelay":
            return False, "fixed by the program"
        return False, "written by the scan engine on every steer"
    if group == "system":
        return False, "used by the firmware for every write"
    return False, "not in the Study allow-list"


def _words(data: str) -> list[int]:
    raw = [int(item, 16) for item in data.split(",") if item.strip()] if data else []
    return [int.from_bytes(bytes(raw[i:i + 4]), "big", signed=True) for i in range(0, len(raw) - 3, 4)]


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    """Parse the export; raises FileNotFoundError or ValueError."""
    path = Path(path) if path else export_path()
    raw = path.read_bytes()
    digest = sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    data = json.loads(raw)
    try:
        schematic = data["Processor"]["Core"][0]["Schematic"][0]
        modules = schematic["ModuleList"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("not a SigmaStudio ADAU146x export") from error

    parameters = []
    for module in modules:
        algo = str(module.get("AlgoName", ""))
        for param in module.get("ModuleParameter", []) or []:
            group, mic = _classify(algo, str(param.get("Name", "")))
            writable, reason = _policy(group, str(param.get("Name", "")))
            words = max(1, int(param.get("Size", 4)) // 4)
            defaults = _words(str(param.get("Data", "")))
            parameters.append({
                "block": algo,
                "caption": module.get("Caption"),
                "name": param.get("Name"),
                "group": group,
                "mic": mic,
                "address": int(param["Address"]),
                "words": words,
                "type": TYPES.get(param.get("DataType"), str(param.get("DataType"))),
                "default_words": defaults[:words] + [0] * max(0, words - len(defaults)),
                "writable": writable,
                "reason": reason,
            })
    parameters.sort(key=lambda item: item["address"])

    covered = set()
    for item in parameters:
        covered.update(range(item["address"], item["address"] + item["words"]))
    states = []
    for name, info in (schematic.get("AddressMap") or {}).items():
        try:
            address = int(info["Address"])
        except (KeyError, TypeError, ValueError):
            continue
        if address not in covered:
            states.append({"name": name, "address": address})
    states.sort(key=lambda item: item["address"])

    groups = []
    for key, label in GROUPS.items():
        members = [item for item in parameters if item["group"] == key]
        if not members:
            continue
        addresses = [a for item in members for a in range(item["address"], item["address"] + item["words"])]
        groups.append({
            "key": key,
            "label": label,
            "parameters": len(members),
            "words": len(addresses),
            "writable_words": sum(item["words"] for item in members if item["writable"]),
            "min_address": min(addresses),
            "max_address": max(addresses),
            "contiguous": max(addresses) - min(addresses) + 1 == len(set(addresses)),
        })

    return {
        "schema": SCHEMA,
        "export": {"file": path.name, "sha256": digest},
        "parameters": parameters,
        "states": states,
        "groups": groups,
        "writable_words": sum(item["words"] for item in parameters if item["writable"]),
    }


def word_index(registry: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Address → {param, offset} for every parameter word."""
    index = {}
    for item in registry["parameters"]:
        for offset in range(item["words"]):
            index[item["address"] + offset] = {"param": item, "offset": offset}
    return index


def checksum(body: str) -> int:
    value = 0
    for char in body.encode("ascii"):
        value ^= char
    return value


def format_pw(address: int, words: list[int]) -> str:
    body = "PW,{},{}".format(address, ",".join(f"{word & 0xFFFFFFFF:08X}" for word in words))
    return f"{body}*{checksum(body):02X}"


def plan_writes(writes: list[dict[str, int]], registry: dict[str, Any], baud: int = 921600) -> dict[str, Any]:
    """Validate writes and group them into safeload commands by real address."""
    index = word_index(registry)
    merged: dict[int, int] = {}
    for write in writes:
        address = int(write["address"])
        word = int(write["word"])
        entry = index.get(address)
        if entry is None:
            raise ValueError(f"address {address} is not a parameter in the loaded program")
        if not entry["param"]["writable"]:
            raise ValueError(f"address {address} ({entry['param']['block']}.{entry['param']['name']}) is read-only: {entry['param']['reason']}")
        if not -(1 << 31) <= word < (1 << 32):
            raise ValueError(f"word for address {address} does not fit in 32 bits")
        merged[address] = word
    runs: list[tuple[int, list[int]]] = []
    for address in sorted(merged):
        if runs and runs[-1][0] + len(runs[-1][1]) == address and len(runs[-1][1]) < MAX_WORDS_PER_COMMAND:
            runs[-1][1].append(merged[address])
        else:
            runs.append((address, [merged[address]]))
    commands = [format_pw(address, words) for address, words in runs]
    # Per command: the line on the UART, a PW_OK reply, SPI safeload and host turnaround.
    per_byte_ms = 10_000 / baud
    estimate_ms = sum((len(command) + 2 + 7) * per_byte_ms + 1.2 for command in commands)
    return {
        "commands": commands,
        "command_count": len(commands),
        "word_count": len(merged),
        "estimate_ms": round(estimate_ms, 1),
        "runs": [{"address": address, "words": len(words)} for address, words in runs],
    }


class DspParameterService:
    """Parameter memory and settings per transport.

    The emulator keeps a memory seeded from the export defaults and answers
    reads and writes like the proposed firmware commands. Serial hardware is
    reported as unsupported until the firmware implements protocol v3.
    """

    def __init__(self, loader=load_registry):
        self._loader = loader
        self._registry = None
        self._registry_error = None
        self._lock = threading.RLock()
        self._memory: dict[int, int] = {}
        self._settings = {"settle_us": 1250}
        self._owner = None

    def registry(self) -> dict[str, Any]:
        with self._lock:
            if self._registry is None and self._registry_error is None:
                try:
                    self._registry = self._loader()
                except (OSError, ValueError) as error:
                    self._registry_error = str(error)
            if self._registry is None:
                raise FileNotFoundError(self._registry_error or "register map unavailable")
            return self._registry

    def _bind(self, transport) -> None:
        if transport is not self._owner:
            self._owner = transport
            self._memory = {}
            self._settings = {"settle_us": 1250}
            registry = self.registry()
            for item in registry["parameters"]:
                for offset, word in enumerate(item["default_words"]):
                    self._memory[item["address"] + offset] = word

    @staticmethod
    def supports(transport) -> tuple[bool, str]:
        if transport is None:
            return False, "No device is connected."
        name = str(getattr(transport, "transport_name", ""))
        if name in ("emulator", "acoustic-emulator"):
            return True, ""
        return False, "This firmware does not accept parameter writes yet (protocol v3, PW/PR/SET)."

    def status(self, transport) -> dict[str, Any]:
        supported, reason = self.supports(transport)
        result = {"supported": supported, "reason": reason, "idle": False, "firmware_mode": None,
                  "modified_words": 0, "settings": dict(self._settings)}
        try:
            registry = self.registry()
            result["export_sha256"] = registry["export"]["sha256"]
        except FileNotFoundError as error:
            result.update(supported=False, reason=str(error))
            return result
        if transport is not None:
            mode = transport.snapshot().get("mode")
            result["firmware_mode"] = mode
            result["idle"] = mode == "IDLE"
        if supported:
            with self._lock:
                self._bind(transport)
                result["modified_words"] = self._modified_count()
                result["settings"] = dict(self._settings)
        return result

    def _modified_count(self) -> int:
        count = 0
        for item in self._registry["parameters"]:
            for offset, word in enumerate(item["default_words"]):
                if self._memory.get(item["address"] + offset, word) != word:
                    count += 1
        return count

    def _require(self, transport, write: bool) -> None:
        supported, reason = self.supports(transport)
        if not supported:
            raise PermissionError(reason)
        if write and transport.snapshot().get("mode") != "IDLE":
            raise RuntimeError("The device is busy. Stop the scan before writing parameters.")

    def memory(self, transport) -> dict[int, int]:
        self._require(transport, write=False)
        with self._lock:
            self._bind(transport)
            return dict(self._memory)

    def read(self, transport, address: int, count: int = 1) -> list[int]:
        self._require(transport, write=False)
        if not 1 <= count <= 16:
            raise ValueError("read between 1 and 16 words")
        with self._lock:
            self._bind(transport)
            return [self._memory.get(address + i, 0) for i in range(count)]

    def write(self, transport, writes: list[dict[str, int]]) -> dict[str, Any]:
        self._require(transport, write=True)
        registry = self.registry()
        plan = plan_writes(writes, registry)
        with self._lock:
            self._bind(transport)
            for write in writes:
                word = int(write["word"])
                self._memory[int(write["address"])] = word - (1 << 32) if word >= (1 << 31) else word
            readback = {int(w["address"]): self._memory[int(w["address"])] for w in writes}
            plan["modified_words"] = self._modified_count()
        plan["readback"] = readback
        return plan

    def set(self, transport, key: str, value: float) -> dict[str, Any]:
        self._require(transport, write=True)
        if key != "settle_us":
            raise ValueError(f"unknown setting {key!r}")
        if not 50 <= float(value) <= 20000:
            raise ValueError("settle_us must be between 50 and 20000")
        with self._lock:
            self._bind(transport)
            self._settings[key] = int(round(float(value)))
            return dict(self._settings)
