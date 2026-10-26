"""Heimdall DSP register map, parsed from a SigmaStudio export (read-only).

The export JSON lists every module parameter with its data-memory address,
size and default. This module turns it into a register map for the Study
screen, decides which parameters the GUI may write (an allow-list), and plans
writes as safeload commands.

SigmaStudio places parameters at compile time. Addresses of blocks of one kind
are NOT guaranteed to be contiguous or to increase by one, so the planner only
batches words whose addresses are actually consecutive, at most five per
command (the ADAU1467 safeload block size), and otherwise writes single words.

The PW/PR/SET device commands are described in MAX78002/docs/REFERENCE.md (section 4.12).
The emulator implements them too, through
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
FIRMWARE_DIR = Path(__file__).resolve().parent.parent / "MAX78002"
DEFAULT_EXPORT_FOLDER = "SigmaStudioExport"
EXPORT_SUFFIX = "_ADAU1467_0.json"
BANK_PREFIX = "Bank_"

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
    "bank_band": "Bank band-pass (Live detectors)",
    "bank_alias": "Bank anti-alias low-pass",
    "bank_ch1": "Bank Ch1 low-pass",
    "bank_detector": "Bank level detectors",
    "delays": "Fractional delays",
    "steering": "Lookup-table steering",
    "system": "Safeload registers",
    "other": "Other parameters",
}
WRITABLE_GROUPS = ("gains", "fir1", "fir2", "mutes", "detector", "output", "bank_band", "bank_alias",
                   "bank_ch1", "bank_detector")

TYPES = {"FixPoint8d24": "8.24", "Integer32": "int32"}


def selected_export_dir() -> Path:
    """The export folder the firmware is built for, chosen exactly as make chooses it.

    HEIMDALL_DSP_EXPORT from the environment wins, else its default in
    MAX78002/project.mk. A relative value is a folder inside MAX78002.
    """
    value = os.environ.get("HEIMDALL_DSP_EXPORT", "").strip()
    if not value:
        try:
            project = (FIRMWARE_DIR / "project.mk").read_text(encoding="utf-8")
        except OSError:
            project = ""
        match = re.search(r"^\s*HEIMDALL_DSP_EXPORT\s*\??=\s*(\S+)", project, re.MULTILINE)
        value = match.group(1) if match else DEFAULT_EXPORT_FOLDER
    return FIRMWARE_DIR / value


def export_path() -> Path:
    """The single SigmaStudio export JSON in the selected export folder."""
    folder = selected_export_dir()
    exports = sorted(folder.glob(f"*{EXPORT_SUFFIX}"))
    if len(exports) != 1:
        names = ", ".join(path.name for path in exports) or "none"
        raise FileNotFoundError(f"expected exactly one *{EXPORT_SUFFIX} in {folder}, found: {names}")
    return exports[0]


def _scan_detector(labels: list[str], contract_choice: str | None = None) -> str | None:
    """The detector the firmware reads during a scan: the generator's choice from the contract,
    else its rule (*_Beam1, or the only detector besides the bank's)."""
    if contract_choice in labels:
        return contract_choice
    labels = [label for label in labels if not label.startswith(BANK_PREFIX)]
    beam1 = [label for label in labels if label.endswith("_Beam1")]
    if len(beam1) == 1:
        return beam1[0]
    return labels[0] if len(labels) == 1 else None


def _classify(algo: str, param: str, scan_detector: str | None) -> tuple[str, int | None, int | None]:
    """(group, microphone, beam). Beam 1 delays are Delay_DSPxx; beams 2-4 add _1.._3.
    A *_Beam gain or FIR stage sits once after the beam sum and has no microphone."""
    match = re.fullmatch(r"(Gain|FIR_Stage1|FIR_Stage2)_(?:DSP(\d+)|Beam)", algo)
    if match:
        kind = {"Gain": "gains", "FIR_Stage1": "fir1", "FIR_Stage2": "fir2"}[match.group(1)]
        return kind, int(match.group(2)) if match.group(2) else None, None
    match = re.fullmatch(r"Delay_DSP(\d+)(?:_(\d))?", algo)
    if match:
        return "delays", int(match.group(1)), 1 + int(match.group(2) or 0)
    match = re.fullmatch(r"LUT_DSP(\d+)", algo)
    if match:
        return "steering", int(match.group(1)), 1
    if algo == "Sector_Index":
        return "steering", None, 1
    if algo in MUTE_BLOCK_FIRST_MIC:
        channel = re.search(r"(\d+)$", param)
        return "mutes", MUTE_BLOCK_FIRST_MIC[algo] + int(channel.group(1)) if channel else None, None
    if algo == scan_detector:
        return "detector", None, 1
    if algo.startswith("Beamformer_Output_Gain") or algo.startswith("SingleVolumeControl"):
        return "output", None, None
    if re.fullmatch(r"Bank_BandPass_\d+_\d+", algo):
        return "bank_band", None, None
    match = re.fullmatch(r"Bank_LowPass_DSP(\d+)", algo)
    if match:
        return "bank_alias", int(match.group(1)), None
    if algo == "Bank_Ch1_LowPass":
        return "bank_ch1", None, None
    if re.fullmatch(r"Bank_Beam_\d+_\d+", algo):
        return "bank_detector", None, None
    if algo == "Safeload":
        return "system", None, None
    return "other", None, None


def _policy(group: str, param: str, beam: int | None) -> tuple[bool, str]:
    if group in WRITABLE_GROUPS:
        return True, ""
    if group == "delays":
        if param == "MaxDelay":
            return False, "fixed by the program"
        if beam != 1:
            return False, f"beam {beam} is not steered by the firmware yet"
        return False, "written by the scan engine on every steer"
    if group == "steering":
        if param == "TableValues":
            return False, "loaded by the firmware at boot and on I"
        return False, "written by the scan engine on every steer"
    if group == "system":
        return False, "used by the firmware for every write"
    return False, "not in the Study allow-list"


def _words(data: str) -> list[int]:
    raw = [int(item, 16) for item in data.split(",") if item.strip()] if data else []
    return [int.from_bytes(bytes(raw[i:i + 4]), "big", signed=True) for i in range(0, len(raw) - 3, 4)]


def _signed_8_24(value: float) -> int:
    """An 8.24 word as the generator rounds it (beam_bank.to_8_24), signed."""
    word = int(round(float(value) * (1 << 24))) & 0xFFFFFFFF
    return word - (1 << 32) if word >= (1 << 31) else word


def _boot_words(contract: dict) -> tuple[dict[int, int], list[int]]:
    """({address: word} of the gains the firmware writes at boot and on I (the taper), the bank
    band-pass taps it loads with its tables). Both differ from the compiled export when the
    generator chose so; they are what "flashed" means for these words."""
    gains = {}
    addresses, words = contract.get("gain_addresses"), contract.get("gain_fixpt_8_24")
    if isinstance(addresses, list) and isinstance(words, list) and len(addresses) == len(words):
        gains = {int(a): (int(w) - (1 << 32) if int(w) >= (1 << 31) else int(w)) for a, w in zip(addresses, words)}
    band_pass = ((contract.get("bank") or {}).get("band_pass")) or []
    return gains, [_signed_8_24(v) for v in band_pass]


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

    contract = {}
    contract_path = path.parent / "beam_table_2d.json"
    if contract_path.exists():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except ValueError:
            contract = {}

    # AddressMap is keyed by the block's internal name, not its label; a
    # detector's TCONST address links the two (as in the beam table generator).
    address_map = schematic.get("AddressMap") or {}
    rms_by_tconst = {
        str(info.get("Address")): address_map[name[:-len("TCONST")] + "rms"]
        for name, info in address_map.items()
        if name.endswith("TCONST") and name[:-len("TCONST")] + "rms" in address_map
    }
    detectors = {}
    for module in modules:
        for param in module.get("ModuleParameter", []) or []:
            rms = rms_by_tconst.get(str(param.get("Address")))
            if param.get("Name") == "TCONST" and rms is not None:
                detectors[str(module.get("AlgoName", ""))] = int(rms["Address"])
    scan_detector = _scan_detector(sorted(detectors), contract.get("level_detector"))
    boot_gains, boot_band_pass = _boot_words(contract)
    allowed = None
    try:
        allowed = {start + k for start, count in contract["study"]["writable_ranges"] for k in range(int(count))}
    except (KeyError, TypeError, ValueError):
        pass

    parameters = []
    for module in modules:
        algo = str(module.get("AlgoName", ""))
        for param in module.get("ModuleParameter", []) or []:
            group, mic, beam = _classify(algo, str(param.get("Name", "")), scan_detector)
            writable, reason = _policy(group, str(param.get("Name", "")), beam)
            words = max(1, int(param.get("Size", 4)) // 4)
            address = int(param["Address"])
            if writable and allowed is not None and not all(address + k in allowed for k in range(words)):
                writable, reason = False, "not in the firmware's allow-list (regenerate the export's table)"
            defaults = _words(str(param.get("Data", "")))
            defaults = defaults[:words] + [0] * max(0, words - len(defaults))
            compiled = list(defaults)
            boot = None
            if group == "bank_band" and len(boot_band_pass) == words:
                defaults, boot = list(boot_band_pass), "tables"
            elif group == "gains" and words == 1 and address in boot_gains:
                defaults, boot = [boot_gains[address]], "taper"
            parameters.append({
                "block": algo,
                "caption": module.get("Caption"),
                "name": param.get("Name"),
                "group": group,
                "mic": mic,
                "beam": beam,
                "address": address,
                "words": words,
                "type": TYPES.get(param.get("DataType"), str(param.get("DataType"))),
                "default_words": defaults,
                "compiled_words": compiled if compiled != defaults else None,
                "boot": boot,
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

    map_hash = None
    try:
        map_hash = contract["study"]["map_hash"]
    except (KeyError, TypeError):
        pass

    return {
        "schema": SCHEMA,
        "export": {"file": path.name, "sha256": digest},
        "map_hash": map_hash,
        "parameters": parameters,
        "states": states,
        "groups": groups,
        "writable_words": sum(item["words"] for item in parameters if item["writable"]),
        "level_readback_address": detectors.get(scan_detector),
        "bank": None if not contract.get("bank") else {
            key: contract["bank"].get(key) for key in ("rows", "columns", "sample_rate_hz", "band_hz",
                                                       "level_time_constant_ms")},
        "level_time_constant_ms": contract.get("level_time_constant_ms"),
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


def setting_order(current: dict[str, int], changes: dict[str, int]) -> list[tuple[str, int]]:
    """The changes in an order that keeps band_low_hz below band_high_hz after each SET."""
    high_first = changes.get("band_low_hz", 0) >= current.get("band_high_hz", 3500)
    first = "band_high_hz" if high_first else "band_low_hz"
    return sorted(changes.items(), key=lambda item: item[0] != first)


class DspParameterService:
    """Parameter memory and settings per transport.

    The emulator keeps a memory seeded from the export defaults. Serial hardware
    uses the firmware PW/PR/SET commands after its register-map hash is verified.
    """

    SETTINGS = {
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
    }

    def __init__(self, loader=load_registry):
        self._loader = loader
        self._registry = None
        self._registry_error = None
        self._lock = threading.RLock()
        self._memory: dict[int, int] = {}
        self._settings = self._default_settings()
        self._owner = None
        self._marks = (0, 0)
        self._lost: dict[int, int] = {}

    @classmethod
    def _default_settings(cls) -> dict[str, int]:
        return {key: limits[2] for key, limits in cls.SETTINGS.items()}

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
            self._lost = {}
            self._marks = self._reload_marks(transport)
            self._settings = self._default_settings()
            registry = self.registry()
            for item in registry["parameters"]:
                for offset, word in enumerate(item["default_words"]):
                    self._memory[item["address"] + offset] = word
            return
        marks = self._reload_marks(transport)
        if marks != self._marks:
            # A program load puts back every word; I puts back the taper and the tables.
            everything = marks[1] != self._marks[1]
            self._marks = marks
            for item in self.registry()["parameters"]:
                if not item["writable"] or not (everything or item.get("boot")):
                    continue
                for offset, word in enumerate(item["default_words"]):
                    address = item["address"] + offset
                    if self._memory.get(address, word) != word:
                        self._lost[address] = self._memory[address]
                        self._memory[address] = word

    @staticmethod
    def _reload_marks(transport) -> tuple[int, int]:
        try:
            snapshot = transport.snapshot()
        except Exception:  # a transport without state cannot report reloads
            return 0, 0
        return int(snapshot.get("boot_checks") or 0), int(snapshot.get("dsp_loads") or 0)

    def supports(self, transport) -> tuple[bool, str]:
        if transport is None:
            return False, "No device is connected."
        name = str(getattr(transport, "transport_name", ""))
        if name in ("emulator", "acoustic-emulator"):
            return True, ""
        if name.startswith("serial:"):
            expected = self.registry().get("map_hash")
            actual = transport.snapshot().get("map_hash")
            if not expected:
                return False, "The selected export has no register-map hash."
            if not actual:
                return False, "The firmware has not reported its register-map hash."
            if actual != expected:
                return False, f"Firmware map {actual} does not match export map {expected}."
            return True, ""
        return False, "This transport does not support PW/PR/SET."

    def status(self, transport) -> dict[str, Any]:
        supported, reason = self.supports(transport)
        result = {"supported": supported, "reason": reason, "idle": False, "firmware_mode": None,
                  "modified_words": 0, "settings": dict(self._settings)}
        try:
            registry = self.registry()
            result["export_sha256"] = registry["export"]["sha256"]
            result["map_hash"] = registry.get("map_hash")
        except FileNotFoundError as error:
            result.update(supported=False, reason=str(error))
            return result
        if transport is not None:
            snapshot = transport.snapshot()
            mode = snapshot.get("mode")
            result["firmware_mode"] = mode
            result["idle"] = mode == "IDLE"
        if supported:
            with self._lock:
                self._bind(transport)
                result["modified_words"] = self._modified_count()
                result["settings"] = dict(self._settings)
                result["lost"] = self.lost()
            if str(getattr(transport, "transport_name", "")).startswith("serial:"):
                result["settings"].update(snapshot.get("settings") or {})
        return result

    def lost(self) -> list[dict[str, int]]:
        """Edited words that a boot check (I) or a program load put back, with the edited value."""
        with self._lock:
            return [{"address": address, "word": word} for address, word in sorted(self._lost.items())]

    def forget_lost(self) -> None:
        with self._lock:
            self._lost = {}

    def _modified_count(self) -> int:
        count = 0
        for item in self._registry["parameters"]:
            for offset, word in enumerate(item["default_words"]):
                if self._memory.get(item["address"] + offset, word) != word:
                    count += 1
        return count

    def settings(self, transport) -> dict[str, int]:
        """The settings for this transport, without the status bookkeeping."""
        with self._lock:
            try:
                self._bind(transport)
            except FileNotFoundError:
                pass
            return dict(self._settings)

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
        if str(getattr(transport, "transport_name", "")).startswith("serial:"):
            record = transport.request(f"PR,{address},{count}", {"pv"})
            if record["address"] != address or len(record["words"]) != count:
                raise RuntimeError("firmware returned a mismatched PR response")
            return record["words"]
        with self._lock:
            self._bind(transport)
            return [self._memory.get(address + i, 0) for i in range(count)]

    def write(self, transport, writes: list[dict[str, int]]) -> dict[str, Any]:
        self._require(transport, write=True)
        registry = self.registry()
        plan = plan_writes(writes, registry)
        if str(getattr(transport, "transport_name", "")).startswith("serial:"):
            readback = {}
            for command, run in zip(plan["commands"], plan["runs"]):
                response = transport.request(command, {"pw_ok"})
                if response["address"] != run["address"] or response["count"] != run["words"]:
                    raise RuntimeError("firmware returned a mismatched PW response")
                values = self.read(transport, run["address"], run["words"])
                readback.update({run["address"] + offset: word
                                 for offset, word in enumerate(values)})
            with self._lock:
                self._bind(transport)
                self._memory.update(readback)
                for address in readback:
                    self._lost.pop(address, None)
                plan["modified_words"] = self._modified_count()
            plan["readback"] = readback
            return plan
        with self._lock:
            self._bind(transport)
            for write in writes:
                word = int(write["word"])
                self._memory[int(write["address"])] = word - (1 << 32) if word >= (1 << 31) else word
                self._lost.pop(int(write["address"]), None)
            readback = {int(w["address"]): self._memory[int(w["address"])] for w in writes}
            plan["modified_words"] = self._modified_count()
        plan["readback"] = readback
        return plan

    def set(self, transport, key: str, value: float) -> dict[str, Any]:
        self._require(transport, write=True)
        if key not in self.SETTINGS:
            raise ValueError(f"unknown setting {key!r}")
        minimum, maximum, _ = self.SETTINGS[key]
        setting = int(round(float(value)))
        if not minimum <= setting <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
        current = self.settings(transport)
        if str(getattr(transport, "transport_name", "")).startswith("serial:"):
            current.update(transport.snapshot().get("settings") or {})
        if (key == "band_low_hz" and setting >= current.get("band_high_hz", 3500)) or (
                key == "band_high_hz" and setting <= current.get("band_low_hz", 500)):
            raise ValueError("band_low_hz must stay below band_high_hz; change the other corner first")
        if str(getattr(transport, "transport_name", "")).startswith("serial:"):
            response = transport.request(f"SET,{key},{setting}", {"set_ok"})
            if response["key"] != key or response["value"] != setting:
                raise RuntimeError("firmware returned a mismatched SET response")
        with self._lock:
            self._bind(transport)
            self._settings[key] = setting
            settings = dict(self._settings)
        apply_line = getattr(transport, "apply_line", None)
        if apply_line is not None and not str(transport.transport_name).startswith("serial:"):
            # The emulator answers as the firmware would, so the GUI sees the change.
            apply_line(f"SET_OK,{key},{setting}")
        return settings
