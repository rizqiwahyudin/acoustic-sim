"""Cached 48 kHz acoustic level provider for the Heimdall firmware emulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Callable

import numpy as np
import pyroomacoustics as pra
from scipy.fft import next_fast_len
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly

from acoustic_utils import air_absorption_kwargs, spl_to_amplitude


ACOUSTIC_MODEL_VERSION = "heimdall-acoustic-v9"
ROOM_CACHE_MODEL_VERSION = "heimdall-room-v1"
DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parent / "data" / "heimdall_acoustic_contract.json"
DEFAULT_FIRMWARE_HEADER_PATH = Path(__file__).resolve().parents[1] / "MAX78002" / "beam_table_2d.h"
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parent / ".cache" / "heimdall_acoustic"
DEFAULT_ROOM_CACHE_ROOT = DEFAULT_CACHE_ROOT / "rooms"
ProgressCallback = Callable[[dict], None]


@dataclass(frozen=True)
class AcousticScenario:
    scenario: str = "conference_evasive"
    room_dimensions_m: tuple[float, float, float] = (20.0, 40.0, 10.0)
    array_center_m: tuple[float, float, float] = (10.0, 0.5, 1.0)
    rt60_s: float = 1.2
    duration_s: float = 30.0
    level_step_s: float = 0.005
    reflection_order: int = 2
    drone_waypoints: int = 8
    crowd_talkers: int = 12
    speech_azimuth_deg: float = 25.0
    speech_range_m: float = 12.0
    speech_height_m: float = 1.5
    drone_spl_db: float = 78.0
    crowd_spl_db: float = 67.0
    speech_enabled: bool = True
    speech_spl_db: float = 72.0
    microphone_noise_spl_db: float = 30.0
    seed: int = 78002

    def validate(self) -> None:
        if self.scenario not in {
            "conference_evasive",
            "stationary_2m",
            "handheld_2m_drone",
            "handheld_2m_mixed",
        }:
            raise ValueError(f"unknown acoustic scenario: {self.scenario}")
        if tuple(self.room_dimensions_m) != (20.0, 40.0, 10.0):
            raise ValueError("initial acoustic emulator room must remain 20 x 40 x 10 m")
        if tuple(self.array_center_m) != (10.0, 0.5, 1.0):
            raise ValueError("initial acoustic emulator array position must remain [10, 0.5, 1] m")
        if not 0.0 < self.rt60_s <= 3.0:
            raise ValueError("rt60_s must be between 0 and 3 seconds")
        if not 0.05 <= self.duration_s <= 120.0:
            raise ValueError("duration_s must be between 0.05 and 120 seconds")
        if not 0.0025 <= self.level_step_s <= 0.1:
            raise ValueError("level_step_s must be between 2.5 and 100 milliseconds")
        if not 0 <= self.reflection_order <= 6:
            raise ValueError("reflection_order must be between 0 and 6")
        if not 1 <= self.drone_waypoints <= 32:
            raise ValueError("drone_waypoints must be between 1 and 32")
        if not 0 <= self.crowd_talkers <= 64:
            raise ValueError("crowd_talkers must be between 0 and 64")


@dataclass(frozen=True)
class BeamContract:
    path: Path
    source_sha256: str
    sampling_rate_hz: int
    speed_of_sound_m_s: float
    max_delay_samples: int
    rows: int
    columns: int
    sectors: int
    microphones: int
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    microphone_positions_mm: np.ndarray
    delay_addresses: np.ndarray
    delay_fixpt: np.ndarray
    delay_samples: np.ndarray
    fir_stages: tuple[np.ndarray, ...]
    fir_stage_modes: tuple[str, ...]


@dataclass(frozen=True)
class AcousticGridSpec:
    rows: int = 7
    columns: int = 7
    azimuth_min_deg: float = -70.0
    azimuth_max_deg: float = 70.0
    elevation_min_deg: float = -60.0
    elevation_max_deg: float = 60.0

    def validate(self) -> None:
        if not 1 <= self.rows <= 20 or not 1 <= self.columns <= 20:
            raise ValueError("acoustic grid must be between 1x1 and 20x20")
        if not -90.0 <= self.azimuth_min_deg < self.azimuth_max_deg <= 90.0:
            raise ValueError("azimuth limits must satisfy -90 <= minimum < maximum <= 90")
        if not -90.0 <= self.elevation_min_deg < self.elevation_max_deg <= 90.0:
            raise ValueError("elevation limits must satisfy -90 <= minimum < maximum <= 90")


@dataclass(frozen=True)
class AcousticCache:
    cache_id: str
    directory: Path
    manifest: dict
    levels_raw: np.ndarray


@dataclass(frozen=True)
class RoomAcousticCache:
    cache_id: str
    directory: Path
    manifest: dict
    microphone_transfer: np.ndarray


class PreparationCancelled(RuntimeError):
    pass


def acoustic_scenario(name: str) -> AcousticScenario:
    if name == "conference_evasive":
        return AcousticScenario()
    if name == "stationary_2m":
        return replace(
            AcousticScenario(),
            scenario=name,
            drone_waypoints=1,
            crowd_talkers=0,
            speech_enabled=False,
        )
    if name in {"handheld_2m_drone", "handheld_2m_mixed"}:
        drone_only = name == "handheld_2m_drone"
        return replace(
            AcousticScenario(),
            scenario=name,
            drone_waypoints=16,
            crowd_talkers=0 if drone_only else 12,
            speech_enabled=not drone_only,
        )
    raise ValueError(f"unknown acoustic scenario: {name}")


def _signed_8_24(raw: int) -> float:
    signed = raw if raw < 0x80000000 else raw - 0x100000000
    return signed / float(1 << 24)


def load_beam_contract(path=DEFAULT_CONTRACT_PATH,
                       firmware_header_path=DEFAULT_FIRMWARE_HEADER_PATH) -> BeamContract:
    path = Path(path)
    data = json.loads(path.read_text(encoding="ascii"))
    if data.get("schema") != "heimdall-beam-table-v1":
        raise ValueError("unsupported Heimdall beam contract schema")
    if firmware_header_path is not None and Path(firmware_header_path).is_file():
        firmware_hash = sha256(Path(firmware_header_path).read_bytes()).hexdigest()
        if firmware_hash != str(data.get("source_c_header_sha256", "")).lower():
            raise ValueError("acoustic beam contract does not match firmware beam_table_2d.h")
    rows = int(data["rows"])
    columns = int(data["columns"])
    sectors = int(data["sectors"])
    microphones = int(data["microphones"])
    if rows * columns != sectors or microphones != 44:
        raise ValueError("beam contract dimensions are invalid")
    if int(data["sampling_rate_hz"]) != 48000 or int(data["max_delay_samples"]) != 68:
        raise ValueError("acoustic emulator requires the deployed 48 kHz MaxDelay=68 contract")

    azimuth = np.asarray(data["azimuth_deg"], dtype=np.float64)
    elevation = np.asarray(data["elevation_deg"], dtype=np.float64)
    positions = np.asarray(data["microphone_positions_mm"], dtype=np.float64)
    addresses = np.asarray(data["delay_addresses"], dtype=np.uint16)
    delay_fixpt = np.asarray(data["delay_fixpt_8_24"], dtype=np.uint32)
    if azimuth.shape != (columns,) or elevation.shape != (rows,):
        raise ValueError("beam contract angle arrays do not match dimensions")
    if positions.shape != (microphones, 2):
        raise ValueError("beam contract microphone positions must be 44 [right, up] pairs")
    if addresses.shape != (microphones,) or len(np.unique(addresses)) != microphones:
        raise ValueError("beam contract delay addresses are invalid")
    if delay_fixpt.shape != (sectors, microphones):
        raise ValueError("beam contract delay table has the wrong shape")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(azimuth)) or not np.all(np.isfinite(elevation)):
        raise ValueError("beam contract contains non-finite values")

    max_delay = int(data["max_delay_samples"])
    delay_samples = delay_fixpt.astype(np.float64) / float(1 << 24) * max_delay
    if np.any(delay_samples < 0.0) or np.any(delay_samples > max_delay + 1e-6):
        raise ValueError("beam contract contains out-of-range delay values")

    fir_stages = []
    for stage in data.get("fir_stages_fixpt_8_24", []):
        coefficients = np.asarray([_signed_8_24(int(value)) for value in stage], dtype=np.float64)
        if coefficients.size == 0 or not np.all(np.isfinite(coefficients)):
            raise ValueError("beam contract FIR stage is invalid")
        fir_stages.append(coefficients)
    if len(fir_stages) != 2:
        raise ValueError("beam contract must contain two common FIR stages")
    fir_stage_modes = tuple(data.get("fir_stage_modes", ()))
    if fir_stage_modes != ("highpass", "lowpass"):
        raise ValueError("beam contract FIR stage modes must be highpass then lowpass")

    return BeamContract(
        path=path,
        source_sha256=str(data["source_c_header_sha256"]).lower(),
        sampling_rate_hz=int(data["sampling_rate_hz"]),
        speed_of_sound_m_s=float(data["speed_of_sound_m_s"]),
        max_delay_samples=max_delay,
        rows=rows,
        columns=columns,
        sectors=sectors,
        microphones=microphones,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        microphone_positions_mm=positions,
        delay_addresses=addresses,
        delay_fixpt=delay_fixpt,
        delay_samples=delay_samples,
        fir_stages=tuple(fir_stages),
        fir_stage_modes=fir_stage_modes,
    )


def _cell_centers(minimum_deg: float, maximum_deg: float, count: int) -> np.ndarray:
    step = (float(maximum_deg) - float(minimum_deg)) / int(count)
    return float(minimum_deg) + (np.arange(int(count), dtype=np.float64) + 0.5) * step


def build_acoustic_grid(deployment: BeamContract,
                        spec: AcousticGridSpec) -> tuple[BeamContract, str]:
    """Return deployment delays when exact, otherwise a quantized simulation grid."""
    spec.validate()
    azimuth = _cell_centers(spec.azimuth_min_deg, spec.azimuth_max_deg, spec.columns)
    elevation = _cell_centers(spec.elevation_min_deg, spec.elevation_max_deg, spec.rows)
    if (spec.rows == deployment.rows and spec.columns == deployment.columns
            and np.allclose(azimuth, deployment.azimuth_deg, atol=1e-12)
            and np.allclose(elevation, deployment.elevation_deg, atol=1e-12)):
        return deployment, "deployment_contract"

    local = np.column_stack((
        deployment.microphone_positions_mm[:, 0] / 1000.0,
        deployment.microphone_positions_mm[:, 1] / 1000.0,
        np.zeros(deployment.microphones),
    ))
    local -= local.mean(axis=0, keepdims=True)
    directions = []
    for elevation_deg in elevation:
        elevation_rad = math.radians(float(elevation_deg))
        for azimuth_deg in azimuth:
            azimuth_rad = math.radians(float(azimuth_deg))
            directions.append((
                math.cos(elevation_rad) * math.sin(azimuth_rad),
                math.sin(elevation_rad),
                math.cos(elevation_rad) * math.cos(azimuth_rad),
            ))
    projections = np.asarray(directions, dtype=np.float64) @ local.T
    delay_samples = projections / deployment.speed_of_sound_m_s * deployment.sampling_rate_hz
    delay_samples -= delay_samples.min(axis=1, keepdims=True)
    if float(np.max(delay_samples)) > deployment.max_delay_samples + 1e-9:
        raise ValueError("custom acoustic grid exceeds the deployed fractional-delay range")
    delay_fixpt = np.rint(
        delay_samples / deployment.max_delay_samples * (1 << 24)
    ).astype(np.uint32)
    quantized_delays = delay_fixpt.astype(np.float64) / (1 << 24) * deployment.max_delay_samples
    return replace(
        deployment,
        rows=spec.rows,
        columns=spec.columns,
        sectors=spec.rows * spec.columns,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        delay_fixpt=delay_fixpt,
        delay_samples=quantized_delays,
    ), "exploratory_simulation"


def room_microphone_positions(contract: BeamContract, scenario: AcousticScenario) -> np.ndarray:
    """Map local [right, up] coordinates onto a vertical plane facing global +Y."""
    offsets = contract.microphone_positions_mm / 1000.0
    offsets -= offsets.mean(axis=0, keepdims=True)
    center = np.asarray(scenario.array_center_m, dtype=np.float64)
    positions = np.vstack((
        center[0] + offsets[:, 0],
        np.full(contract.microphones, center[1]),
        center[2] + offsets[:, 1],
    ))
    room = np.asarray(scenario.room_dimensions_m, dtype=np.float64)
    if np.any(positions < 0.0) or np.any(positions > room[:, None]):
        raise ValueError("transformed microphone positions are outside the room")
    return positions


def room_evasive_position(elapsed_s: float, scenario: AcousticScenario) -> np.ndarray:
    """Closed 15-second evasive route (the requested 2x stress profile)."""
    phase = 2.0 * math.pi * (float(elapsed_s) / 15.0)
    x = 10.0 + 6.0 * math.sin(phase) + 1.5 * math.sin(3.0 * phase + 0.4)
    y = 20.0 + 12.0 * math.sin(phase + 0.7) + 2.5 * math.sin(4.0 * phase)
    z = 2.25 + 0.65 * math.sin(2.0 * phase + 0.2)
    return np.asarray((x, y, z), dtype=np.float64)


def stationary_speaker_position(scenario: AcousticScenario) -> np.ndarray:
    center = np.asarray(scenario.array_center_m, dtype=np.float64)
    return center + np.asarray((0.0, 2.0, 0.0), dtype=np.float64)


def handheld_speaker_position(elapsed_s: float, scenario: AcousticScenario) -> np.ndarray:
    """Closed slow figure-eight in the vertical array plane at fixed 2 m depth."""
    phase = 2.0 * math.pi * (float(elapsed_s) / 12.0)
    center = np.asarray(scenario.array_center_m, dtype=np.float64)
    return np.asarray((
        center[0] + 0.75 * math.sin(phase),
        center[1] + 2.0,
        1.6 + 0.4 * math.sin(2.0 * phase),
    ), dtype=np.float64)


def speech_speaker_position(scenario: AcousticScenario) -> np.ndarray:
    center = np.asarray(scenario.array_center_m, dtype=np.float64)
    angle = math.radians(scenario.speech_azimuth_deg)
    return np.asarray((
        center[0] + scenario.speech_range_m * math.sin(angle),
        center[1] + scenario.speech_range_m * math.cos(angle),
        scenario.speech_height_m,
    ), dtype=np.float64)


def _trajectory_positions(scenario: AcousticScenario) -> list[np.ndarray]:
    if scenario.scenario == "stationary_2m":
        return [stationary_speaker_position(scenario)]
    route_period_s = _trajectory_period_s(scenario)
    position_at = (handheld_speaker_position
                   if scenario.scenario.startswith("handheld_2m_")
                   else room_evasive_position)
    return [
        position_at(index * route_period_s / scenario.drone_waypoints, scenario)
        for index in range(scenario.drone_waypoints)
    ]


def _trajectory_period_s(scenario: AcousticScenario) -> float:
    return 12.0 if scenario.scenario.startswith("handheld_2m_") else 15.0


def _interpolated_source_position(elapsed_s: float, scenario: AcousticScenario,
                                  source_positions: np.ndarray) -> np.ndarray:
    if scenario.scenario == "stationary_2m":
        return source_positions[0]
    waypoint_position = (elapsed_s / _trajectory_period_s(scenario)) * len(source_positions)
    low = int(math.floor(waypoint_position)) % len(source_positions)
    high = (low + 1) % len(source_positions)
    ratio = waypoint_position - math.floor(waypoint_position)
    return (1.0 - ratio) * source_positions[low] + ratio * source_positions[high]


def load_audio_48k(path: Path, duration_s: float) -> np.ndarray:
    sample_rate, samples = wavfile.read(path)
    samples = np.asarray(samples)
    original_dtype = samples.dtype
    if samples.ndim > 1:
        samples = samples.astype(np.float64).mean(axis=1)
    else:
        samples = samples.astype(np.float64)
    if np.issubdtype(original_dtype, np.integer):
        limits = np.iinfo(original_dtype)
        samples /= max(abs(limits.min), limits.max)
    divisor = math.gcd(int(sample_rate), 48000)
    if sample_rate != 48000:
        samples = resample_poly(samples, 48000 // divisor, int(sample_rate) // divisor)
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    if rms > 1e-12:
        samples /= rms
    required = max(1, int(round(duration_s * 48000)))
    if samples.size < required:
        samples = np.tile(samples, math.ceil(required / max(1, samples.size)))
    samples = samples[:required].astype(np.float64, copy=False)
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    return samples / rms if rms > 1e-12 else samples


def _late_reverb_tail(length: int, fs: int, rt60_s: float, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    elapsed = np.arange(length, dtype=np.float64) / fs
    decay = np.exp(-math.log(1000.0) * elapsed / rt60_s)
    noise = rng.standard_normal(length) * decay
    norm = np.linalg.norm(noise)
    return noise / norm if norm > 1e-12 else noise


def _room_rirs(source: np.ndarray, microphones: np.ndarray, scenario: AcousticScenario,
               seed: int) -> list[np.ndarray]:
    absorption, _ = pra.inverse_sabine(scenario.rt60_s, list(scenario.room_dimensions_m))
    room = pra.ShoeBox(
        list(scenario.room_dimensions_m),
        fs=48000,
        materials=pra.Material(absorption),
        max_order=scenario.reflection_order,
        **air_absorption_kwargs(20.0, 50.0),
    )
    room.add_microphone_array(pra.MicrophoneArray(microphones, fs=48000))
    room.add_source(np.asarray(source, dtype=np.float64))
    room.compute_rir()
    early = [np.asarray(room.rir[mic][0], dtype=np.float64) for mic in range(microphones.shape[1])]
    tail_length = int(round(scenario.rt60_s * 48000))
    output = []
    for mic, impulse in enumerate(early):
        rng = np.random.default_rng(seed + mic)
        tail = _late_reverb_tail(tail_length, 48000, scenario.rt60_s, rng)
        direct_peak = float(np.max(np.abs(impulse))) if impulse.size else 0.0
        tail *= direct_peak * 4.0
        padded = np.pad(impulse, (0, max(0, tail_length - impulse.size)))[:tail_length]
        output.append(padded + tail)
    return output


def _combined_fir(contract: BeamContract) -> np.ndarray:
    result = np.asarray((1.0,), dtype=np.float64)
    for stage, mode in zip(contract.fir_stages, contract.fir_stage_modes):
        effective_stage = stage
        if mode == "highpass":
            effective_stage = -stage.copy()
            effective_stage[effective_stage.size // 2] += 1.0
        result = np.convolve(result, effective_stage)
    return result


def _microphone_transfer_bins(rirs: list[np.ndarray], signal_fft_size: int) -> np.ndarray:
    rir_length = max(len(rir) for rir in rirs)
    transfer_fft_size = next_fast_len(rir_length + 1)
    if transfer_fft_size % signal_fft_size:
        transfer_fft_size = signal_fft_size * math.ceil(transfer_fft_size / signal_fft_size)
    microphone_transfer = np.stack([
        np.fft.rfft(np.pad(rir, (0, transfer_fft_size - len(rir))))
        for rir in rirs
    ])
    bins = np.arange(signal_fft_size // 2 + 1)
    transfer_bins = bins * (transfer_fft_size // signal_fft_size)
    return microphone_transfer[:, transfer_bins]


def _beam_transfer(contract: BeamContract, microphone_transfer: np.ndarray,
                   signal_fft_size: int,
                   detector_fir: np.ndarray | None = None) -> np.ndarray:
    frequencies = np.fft.rfftfreq(signal_fft_size, 1.0 / contract.sampling_rate_hz)
    phase = np.exp(
        -2j * np.pi * frequencies[None, None, :]
        * contract.delay_samples[:, :, None] / contract.sampling_rate_hz
    )
    effective = np.sum(microphone_transfer[None, :, :] * phase, axis=1)
    fir = _combined_fir(contract) if detector_fir is None else detector_fir
    fir_response = np.fft.rfft(fir, signal_fft_size)
    return effective * fir_response[None, :]


def _beam_transfer_powers(contract: BeamContract, microphone_transfer: np.ndarray,
                          signal_fft_size: int,
                          detector_fir: np.ndarray | None = None) -> np.ndarray:
    return np.square(np.abs(
        _beam_transfer(contract, microphone_transfer, signal_fft_size, detector_fir)
    ))


def _effective_transfer_powers(contract: BeamContract, rirs: list[np.ndarray],
                               signal_fft_size: int,
                               detector_fir: np.ndarray | None = None) -> np.ndarray:
    return _beam_transfer_powers(
        contract,
        _microphone_transfer_bins(rirs, signal_fft_size),
        signal_fft_size,
        detector_fir,
    )


def _plane_wave_transfer_powers(contract: BeamContract, directions: np.ndarray,
                                signal_fft_size: int,
                                detector_fir: np.ndarray | None = None) -> np.ndarray:
    local = np.column_stack((
        contract.microphone_positions_mm[:, 0] / 1000.0,
        contract.microphone_positions_mm[:, 1] / 1000.0,
        np.zeros(contract.microphones),
    ))
    projections = directions @ local.T
    propagation_delays = (projections.max(axis=1, keepdims=True) - projections) \
        / contract.speed_of_sound_m_s * contract.sampling_rate_hz
    frequencies = np.fft.rfftfreq(signal_fft_size, 1.0 / contract.sampling_rate_hz)
    acoustic_phase = np.exp(
        -2j * np.pi * frequencies[None, None, :]
        * propagation_delays[:, :, None] / contract.sampling_rate_hz
    )
    steering_phase = np.exp(
        -2j * np.pi * frequencies[None, None, :]
        * contract.delay_samples[:, :, None] / contract.sampling_rate_hz
    )
    response = np.sum(
        acoustic_phase[:, None, :, :] * steering_phase[None, :, :, :], axis=2
    )
    fir = _combined_fir(contract) if detector_fir is None else detector_fir
    fir_response = np.fft.rfft(fir, signal_fft_size)
    return np.mean(np.square(np.abs(response * fir_response[None, None, :])), axis=0)


def _plane_wave_microphone_transfer(contract: BeamContract, directions: np.ndarray,
                                    signal_fft_size: int,
                                    detector_fir: np.ndarray) -> np.ndarray:
    local = np.column_stack((
        contract.microphone_positions_mm[:, 0] / 1000.0,
        contract.microphone_positions_mm[:, 1] / 1000.0,
        np.zeros(contract.microphones),
    ))
    local -= local.mean(axis=0, keepdims=True)
    projections = directions @ local.T
    propagation_delays = (projections.max(axis=1, keepdims=True) - projections) \
        / contract.speed_of_sound_m_s * contract.sampling_rate_hz
    frequencies = np.fft.rfftfreq(signal_fft_size, 1.0 / contract.sampling_rate_hz)
    phase = np.exp(
        -2j * np.pi * frequencies[None, None, :]
        * propagation_delays[:, :, None] / contract.sampling_rate_hz
    )
    fir_response = np.fft.rfft(detector_fir, signal_fft_size)
    return phase * fir_response[None, None, :]


def _window_spectral_power(samples: np.ndarray, frame_samples: int,
                           signal_fft_size: int) -> np.ndarray:
    frame_count = math.ceil(samples.size / frame_samples)
    padded = np.pad(samples, (0, frame_count * frame_samples - samples.size))
    frames = padded.reshape(frame_count, frame_samples)
    window = np.hanning(frame_samples)
    scale = max(float(np.sum(np.square(window))), 1e-12)
    spectra = np.fft.rfft(frames * window[None, :], signal_fft_size)
    return np.square(np.abs(spectra)) / scale


def _levels_from_spectra(spectrum_power: np.ndarray, transfer_power: np.ndarray,
                         signal_fft_size: int) -> np.ndarray:
    weights = np.ones(spectrum_power.shape[1], dtype=np.float64)
    if weights.size > 2:
        weights[1:-1] = 2.0
    return (spectrum_power * weights[None, :]) @ transfer_power.T / signal_fft_size


def apply_detector_envelope(levels: np.ndarray, step_s: float,
                            rise_db_s: float = 10000.0,
                            decay_db_s: float = 10000.0) -> np.ndarray:
    """Apply a dB-domain rise/decay envelope independently to every sector."""
    values = np.asarray(levels, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("detector levels must be [time, sector]")
    output = np.zeros_like(values)
    if values.shape[0] == 0:
        return output
    previous_db = 20.0 * np.log10(np.maximum(values[0], 1e-8))
    output[0] = np.power(10.0, previous_db / 20.0)
    rise_step = float(rise_db_s) * float(step_s)
    decay_step = float(decay_db_s) * float(step_s)
    for index, row in enumerate(values[1:], start=1):
        target_db = 20.0 * np.log10(np.maximum(row, 1e-8))
        delta = target_db - previous_db
        previous_db += np.where(
            delta >= 0.0,
            np.minimum(delta, rise_step),
            np.maximum(delta, -decay_step),
        )
        output[index] = np.power(10.0, previous_db / 20.0)
    return output


def _wrapped_segment(samples: np.ndarray, start: int, length: int) -> np.ndarray:
    indices = (np.arange(length, dtype=np.int64) + int(start)) % samples.size
    return samples[indices]


def _interpolate_transfer(transfers: np.ndarray, elapsed_s: float,
                          scenario: AcousticScenario) -> np.ndarray:
    if scenario.scenario == "stationary_2m":
        return transfers[0]
    position = (elapsed_s / _trajectory_period_s(scenario)) * len(transfers)
    low = int(math.floor(position)) % len(transfers)
    high = (low + 1) % len(transfers)
    ratio = position - math.floor(position)
    return (1.0 - ratio) * transfers[low] + ratio * transfers[high]


def _render_source_to_microphones(signal: np.ndarray,
                                  microphone_transfer: np.ndarray) -> np.ndarray:
    impulse_length = (microphone_transfer.shape[1] - 1) * 2
    impulses = np.fft.irfft(microphone_transfer, n=impulse_length, axis=1)
    output_length = signal.size + impulse_length - 1
    fft_size = next_fast_len(output_length)
    signal_fft = np.fft.rfft(signal, fft_size)
    impulse_fft = np.fft.rfft(impulses, fft_size, axis=1)
    return np.fft.irfft(impulse_fft * signal_fft[None, :], fft_size, axis=1)[:, :signal.size]


def _beamform_microphone_audio(signals: np.ndarray, delay_samples: np.ndarray,
                               sampling_rate_hz: int) -> np.ndarray:
    fft_size = next_fast_len(signals.shape[1])
    frequencies = np.fft.rfftfreq(fft_size, 1.0 / sampling_rate_hz)
    spectra = np.fft.rfft(signals, fft_size, axis=1)
    phase = np.exp(
        -2j * np.pi * frequencies[None, :] * delay_samples[:, None] / sampling_rate_hz
    )
    return np.fft.irfft(np.sum(spectra * phase, axis=0), fft_size)[:signals.shape[1]]


def _direction_delay_samples(contract: BeamContract, azimuth_deg: float,
                             elevation_deg: float) -> np.ndarray:
    local = np.column_stack((
        contract.microphone_positions_mm[:, 0] / 1000.0,
        contract.microphone_positions_mm[:, 1] / 1000.0,
        np.zeros(contract.microphones),
    ))
    local -= local.mean(axis=0, keepdims=True)
    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    direction = np.asarray((
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
        math.cos(elevation) * math.cos(azimuth),
    ))
    delays = local @ direction / contract.speed_of_sound_m_s * contract.sampling_rate_hz
    delays -= delays.min()
    return np.rint(delays / contract.max_delay_samples * (1 << 24)) \
        / (1 << 24) * contract.max_delay_samples


def render_acoustic_audition(cache: AcousticCache, start_s: float = 0.0,
                             duration_s: float = 3.0, selected_sector: int | None = None,
                             room_cache_root=DEFAULT_ROOM_CACHE_ROOT) -> dict:
    """Render short simulator-only audio clips from cached room transfer data."""
    if not 0.25 <= duration_s <= 5.0:
        raise ValueError("audition duration must be between 0.25 and 5 seconds")
    scenario = AcousticScenario(**cache.manifest["scenario"])
    deployment = load_beam_contract()
    grid = AcousticGridSpec(**cache.manifest.get("grid", asdict(AcousticGridSpec())))
    contract, _ = build_acoustic_grid(deployment, grid)
    if selected_sector is None:
        selected_sector = contract.sectors // 2
    if not 0 <= selected_sector < contract.sectors:
        raise ValueError("selected audition sector is outside the acoustic grid")
    room_cache = load_room_acoustic_cache(cache.manifest["room_cache_id"], room_cache_root)
    signal_fft_size = int(room_cache.manifest["signal_fft_size"])
    detector_fir = _combined_fir(contract)
    detector_response = np.fft.rfft(detector_fir, signal_fft_size)
    sample_count = int(round(duration_s * contract.sampling_rate_hz))
    start_sample = int(round(float(start_s) * contract.sampling_rate_hz))
    base = Path(__file__).resolve().parent
    drone = load_audio_48k(base / "audio" / "drone.wav", scenario.duration_s)
    crowd = load_audio_48k(base / "audio" / "crowd.wav", scenario.duration_s)
    drone_signal = _wrapped_segment(drone, start_sample, sample_count) \
        * spl_to_amplitude(scenario.drone_spl_db)
    midpoint_s = float(start_s) + duration_s / 2.0
    drone_transfer = _interpolate_transfer(
        np.asarray(room_cache.microphone_transfer[:scenario.drone_waypoints]),
        midpoint_s,
        scenario,
    ) * detector_response[None, :]
    microphone_audio = _render_source_to_microphones(drone_signal, drone_transfer)
    generated_mix = fftconvolve(drone_signal, detector_fir, mode="same")

    if scenario.speech_enabled:
        speech_signal = _wrapped_segment(
            crowd, start_sample + int(7.3 * contract.sampling_rate_hz), sample_count
        ) * spl_to_amplitude(scenario.speech_spl_db)
        speech_index = int(room_cache.manifest["speech_source_index"])
        speech_transfer = np.asarray(room_cache.microphone_transfer[speech_index]) \
            * detector_response[None, :]
        microphone_audio += _render_source_to_microphones(speech_signal, speech_transfer)
        generated_mix += fftconvolve(speech_signal, detector_fir, mode="same")

    if scenario.crowd_talkers:
        rng = np.random.default_rng(scenario.seed)
        directions = rng.standard_normal((scenario.crowd_talkers, 3))
        directions[:, 2] *= 0.35
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        crowd_transfers = _plane_wave_microphone_transfer(
            deployment, directions, signal_fft_size, detector_fir
        )
        crowd_scale = spl_to_amplitude(scenario.crowd_spl_db) / math.sqrt(scenario.crowd_talkers)
        for index, transfer in enumerate(crowd_transfers):
            crowd_signal = _wrapped_segment(
                crowd,
                start_sample + int((index + 1) * 0.37 * contract.sampling_rate_hz),
                sample_count,
            ) * crowd_scale
            microphone_audio += _render_source_to_microphones(crowd_signal, transfer)
            generated_mix += fftconvolve(crowd_signal, detector_fir, mode="same")

    truth_position = _interpolated_source_position(
        midpoint_s,
        scenario,
        np.asarray(room_cache.manifest["source_positions_m"], dtype=np.float64),
    )
    offset = truth_position - np.asarray(scenario.array_center_m, dtype=np.float64)
    horizontal_range = math.hypot(float(offset[0]), float(offset[1]))
    truth_azimuth = math.degrees(math.atan2(float(offset[0]), float(offset[1])))
    truth_elevation = math.degrees(math.atan2(float(offset[2]), horizontal_range))
    return {
        "sample_rate_hz": contract.sampling_rate_hz,
        "start_s": float(start_s),
        "duration_s": float(duration_s),
        "selected_sector": int(selected_sector),
        "truth_azimuth_deg": truth_azimuth,
        "truth_elevation_deg": truth_elevation,
        "clips": {
            "generated": generated_mix,
            "single_mic": microphone_audio[0],
            "unsteered": np.sum(microphone_audio, axis=0),
            "truth": _beamform_microphone_audio(
                microphone_audio,
                _direction_delay_samples(deployment, truth_azimuth, truth_elevation),
                contract.sampling_rate_hz,
            ),
            "selected": _beamform_microphone_audio(
                microphone_audio,
                contract.delay_samples[selected_sector],
                contract.sampling_rate_hz,
            ),
        },
    }


def _scenario_hash(contract_path: Path, scenario: AcousticScenario,
                   drone_path: Path, crowd_path: Path) -> str:
    digest = sha256()
    digest.update(ACOUSTIC_MODEL_VERSION.encode("ascii"))
    digest.update(contract_path.read_bytes())
    digest.update(drone_path.read_bytes())
    digest.update(crowd_path.read_bytes())
    digest.update(json.dumps(asdict(scenario), sort_keys=True).encode("ascii"))
    return digest.hexdigest()


def _room_cache_hash(contract: BeamContract, scenario: AcousticScenario,
                     drone_path: Path, crowd_path: Path) -> str:
    digest = sha256()
    digest.update(ROOM_CACHE_MODEL_VERSION.encode("ascii"))
    digest.update(contract.microphone_positions_mm.tobytes())
    digest.update(str(contract.sampling_rate_hz).encode("ascii"))
    digest.update(json.dumps(asdict(scenario), sort_keys=True).encode("ascii"))
    digest.update(drone_path.read_bytes())
    digest.update(crowd_path.read_bytes())
    return digest.hexdigest()


def load_room_acoustic_cache(cache_id: str,
                             cache_root=DEFAULT_ROOM_CACHE_ROOT) -> RoomAcousticCache:
    directory = Path(cache_root) / cache_id
    manifest_path = directory / "manifest.json"
    transfer_path = directory / "microphone_transfer.npy"
    if not manifest_path.is_file() or not transfer_path.is_file():
        raise FileNotFoundError(f"room acoustic cache is incomplete: {cache_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ROOM_CACHE_MODEL_VERSION or manifest.get("cache_id") != cache_id:
        raise ValueError("room acoustic cache manifest is incompatible")
    if sha256(transfer_path.read_bytes()).hexdigest() != manifest.get("transfer_sha256"):
        raise ValueError("room acoustic cache transfer checksum mismatch")
    transfer = np.load(transfer_path, mmap_mode="r")
    if tuple(transfer.shape) != tuple(manifest["transfer_shape"]) or transfer.dtype != np.complex128:
        raise ValueError("room acoustic transfer array is incompatible")
    return RoomAcousticCache(cache_id, directory, manifest, transfer)


def prepare_room_acoustic_cache(scenario=AcousticScenario(),
                                contract_path=DEFAULT_CONTRACT_PATH,
                                cache_root=DEFAULT_ROOM_CACHE_ROOT,
                                progress: ProgressCallback | None = None,
                                cancel_event: threading.Event | None = None) -> RoomAcousticCache:
    scenario.validate()
    contract = load_beam_contract(contract_path)
    base = Path(__file__).resolve().parent
    drone_path = base / "audio" / "drone.wav"
    crowd_path = base / "audio" / "crowd.wav"
    cache_id = _room_cache_hash(contract, scenario, drone_path, crowd_path)
    directory = Path(cache_root) / cache_id
    manifest_path = directory / "manifest.json"
    transfer_path = directory / "microphone_transfer.npy"
    if manifest_path.is_file() and transfer_path.is_file():
        try:
            return load_room_acoustic_cache(cache_id, cache_root)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            manifest_path.unlink(missing_ok=True)
            transfer_path.unlink(missing_ok=True)

    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    microphones = room_microphone_positions(contract, scenario)
    frame_samples = int(round(scenario.level_step_s * contract.sampling_rate_hz))
    signal_fft_size = next_fast_len(max(512, frame_samples * 2))
    source_positions = _trajectory_positions(scenario)
    speech_position = speech_speaker_position(scenario) if scenario.speech_enabled else None
    rir_sources = source_positions + ([] if speech_position is None else [speech_position])
    transfers = []
    measured_rt60_values = []
    for index, source in enumerate(rir_sources):
        if cancel_event is not None and cancel_event.is_set():
            raise PreparationCancelled("acoustic preparation canceled")
        _progress(progress, "room-rirs", index, len(rir_sources), started)
        rirs = _room_rirs(source, microphones, scenario, scenario.seed + index * 1000)
        try:
            measured_rt60_values.append(float(
                pra.experimental.measure_rt60(rirs[0], fs=contract.sampling_rate_hz, decay_db=20)
            ))
        except Exception:
            pass
        transfers.append(_microphone_transfer_bins(rirs, signal_fft_size))
    _progress(progress, "room-rirs", len(rir_sources), len(rir_sources), started)

    microphone_transfer = np.stack(transfers).astype(np.complex128, copy=False)
    manifest = {
        "schema": ROOM_CACHE_MODEL_VERSION,
        "cache_id": cache_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scenario": asdict(scenario),
        "signal_fft_size": signal_fft_size,
        "source_positions_m": [position.tolist() for position in source_positions],
        "speech_source_index": len(source_positions) if speech_position is not None else None,
        "transfer_shape": list(microphone_transfer.shape),
        "transfer_dtype": "complex128",
        "measured_rt60_s": (None if not measured_rt60_values
                              else float(np.median(measured_rt60_values))),
        "preparation_seconds": time.perf_counter() - started,
    }
    with tempfile.TemporaryDirectory(dir=directory) as temporary_directory:
        temporary_directory = Path(temporary_directory)
        temporary_transfer = temporary_directory / "microphone_transfer.npy"
        temporary_manifest = temporary_directory / "manifest.json"
        np.save(temporary_transfer, microphone_transfer, allow_pickle=False)
        manifest["transfer_sha256"] = sha256(temporary_transfer.read_bytes()).hexdigest()
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_transfer, transfer_path)
        os.replace(temporary_manifest, manifest_path)
    return load_room_acoustic_cache(cache_id, cache_root)


def load_acoustic_cache(cache_id: str, cache_root=DEFAULT_CACHE_ROOT) -> AcousticCache:
    directory = Path(cache_root) / cache_id
    manifest_path = directory / "manifest.json"
    levels_path = directory / "levels_raw.npy"
    if not manifest_path.is_file() or not levels_path.is_file():
        raise FileNotFoundError(f"acoustic cache is incomplete: {cache_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ACOUSTIC_MODEL_VERSION or manifest.get("cache_id") != cache_id:
        raise ValueError("acoustic cache manifest is incompatible")
    if sha256(levels_path.read_bytes()).hexdigest() != manifest.get("levels_sha256"):
        raise ValueError("acoustic cache level checksum mismatch")
    levels = np.load(levels_path, mmap_mode="r")
    if tuple(levels.shape) != tuple(manifest["levels_shape"]) or levels.dtype != np.uint32:
        raise ValueError("acoustic cache level array is incompatible")
    return AcousticCache(cache_id, directory, manifest, levels)


def _progress(callback: ProgressCallback | None, stage: str, completed: int,
              total: int, started: float) -> None:
    if callback is None:
        return
    stage_ranges = {
        "room-rirs": (0.0, 90.0),
        "spectral-beamforming": (90.0, 98.0),
        "detector": (98.0, 99.0),
        "cache-write": (99.0, 100.0),
    }
    start_percent, end_percent = stage_ranges.get(stage, (0.0, 100.0))
    fraction = completed / max(1, total)
    elapsed = time.perf_counter() - started
    eta_s = None
    if stage == "room-rirs" and completed > 0:
        eta_s = elapsed / completed * max(0, total - completed)
    callback({
        "stage": stage,
        "completed": completed,
        "total": total,
        "percent": 100.0 * fraction,
        "overall_percent": start_percent + (end_percent - start_percent) * fraction,
        "elapsed_s": elapsed,
        "eta_s": eta_s,
    })


def prepare_acoustic_cache(scenario=AcousticScenario(), contract_path=DEFAULT_CONTRACT_PATH,
                           cache_root=DEFAULT_CACHE_ROOT, progress: ProgressCallback | None = None,
                           cancel_event: threading.Event | None = None,
                           grid_spec: AcousticGridSpec | None = None,
                           room_cache_root: Path | None = None) -> AcousticCache:
    scenario.validate()
    deployment_contract = load_beam_contract(contract_path)
    grid_spec = grid_spec or AcousticGridSpec()
    contract, grid_mode = build_acoustic_grid(deployment_contract, grid_spec)
    base = Path(__file__).resolve().parent
    drone_path = base / "audio" / "drone.wav"
    crowd_path = base / "audio" / "crowd.wav"
    room_cache_id = _room_cache_hash(
        deployment_contract, scenario, drone_path, crowd_path
    )
    digest = sha256()
    digest.update(ACOUSTIC_MODEL_VERSION.encode("ascii"))
    digest.update(room_cache_id.encode("ascii"))
    digest.update(json.dumps(asdict(grid_spec), sort_keys=True).encode("ascii"))
    cache_id = digest.hexdigest()
    directory = Path(cache_root) / cache_id
    manifest_path = directory / "manifest.json"
    levels_path = directory / "levels_raw.npy"
    if manifest_path.is_file() and levels_path.is_file():
        try:
            return load_acoustic_cache(cache_id, cache_root)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            manifest_path.unlink(missing_ok=True)
            levels_path.unlink(missing_ok=True)

    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    frame_samples = int(round(scenario.level_step_s * 48000))
    signal_fft_size = next_fast_len(max(512, frame_samples * 2))
    detector_fir = _combined_fir(contract)
    drone = load_audio_48k(drone_path, scenario.duration_s)
    crowd = load_audio_48k(crowd_path, scenario.duration_s)
    room_cache = prepare_room_acoustic_cache(
        scenario,
        contract_path=contract_path,
        cache_root=(Path(room_cache_root) if room_cache_root is not None
                    else Path(cache_root) / "rooms"),
        progress=progress,
        cancel_event=cancel_event,
    )
    if int(room_cache.manifest["signal_fft_size"]) != signal_fft_size:
        raise ValueError("room cache FFT size does not match detector projection")
    source_positions = np.asarray(room_cache.manifest["source_positions_m"], dtype=np.float64)
    beam_transfers = [
        _beam_transfer(contract, microphone_transfer, signal_fft_size, detector_fir)
        for microphone_transfer in room_cache.microphone_transfer
    ]
    transfer_powers = [np.square(np.abs(transfer)) for transfer in beam_transfers]

    _progress(progress, "spectral-beamforming", 0, 3, started)
    drone_spectrum = _window_spectral_power(drone, frame_samples, signal_fft_size)
    crowd_spectrum = _window_spectral_power(crowd, frame_samples, signal_fft_size)
    if scenario.scenario == "stationary_2m":
        drone_spectrum[:] = np.mean(drone_spectrum, axis=0, keepdims=True)
    frame_count = drone_spectrum.shape[0]
    levels_power = np.zeros((frame_count, contract.sectors), dtype=np.float64)

    if scenario.scenario == "stationary_2m":
        drone_transfer = transfer_powers[0]
        levels_power += _levels_from_spectra(
            drone_spectrum * spl_to_amplitude(scenario.drone_spl_db) ** 2,
            drone_transfer,
            signal_fft_size,
        )
    else:
        drone_transfers = np.stack(beam_transfers[:len(source_positions)])
        route_period_s = _trajectory_period_s(scenario)
        for frame in range(frame_count):
            elapsed_s = frame * scenario.level_step_s
            position = (elapsed_s / route_period_s) * len(source_positions)
            low = int(math.floor(position)) % len(source_positions)
            high = (low + 1) % len(source_positions)
            ratio = position - math.floor(position)
            transfer = np.square(np.abs(
                (1.0 - ratio) * drone_transfers[low] + ratio * drone_transfers[high]
            ))
            levels_power[frame] += _levels_from_spectra(
                drone_spectrum[frame:frame + 1]
                * spl_to_amplitude(scenario.drone_spl_db) ** 2,
                transfer,
                signal_fft_size,
            )[0]
        elapsed = np.arange(frame_count, dtype=np.float64) * scenario.level_step_s
        profile_time = np.mod(elapsed * 2.0, 17.0)
        dropout = ((profile_time >= 11.5) & (profile_time < 12.4)
               if scenario.scenario == "conference_evasive"
               else np.zeros(frame_count, dtype=bool))
        if np.any(dropout):
            drone_only = np.zeros_like(levels_power)
            for frame in np.where(dropout)[0]:
                elapsed_s = frame * scenario.level_step_s
                position = (elapsed_s / route_period_s) * len(source_positions)
                low = int(math.floor(position)) % len(source_positions)
                high = (low + 1) % len(source_positions)
                ratio = position - math.floor(position)
                transfer = np.square(np.abs(
                    (1.0 - ratio) * drone_transfers[low] + ratio * drone_transfers[high]
                ))
                drone_only[frame] = _levels_from_spectra(
                    drone_spectrum[frame:frame + 1]
                    * spl_to_amplitude(scenario.drone_spl_db) ** 2,
                    transfer,
                    signal_fft_size,
                )[0]
            levels_power[dropout] -= drone_only[dropout]
    _progress(progress, "spectral-beamforming", 1, 3, started)

    if scenario.speech_enabled:
        speech_signal = np.roll(crowd, int(7.3 * 48000))
        speech_spectrum = _window_spectral_power(speech_signal, frame_samples, signal_fft_size)
        levels_power += _levels_from_spectra(
            speech_spectrum * spl_to_amplitude(scenario.speech_spl_db) ** 2,
            transfer_powers[-1],
            signal_fft_size,
        )
    _progress(progress, "spectral-beamforming", 2, 3, started)

    if scenario.crowd_talkers:
        rng = np.random.default_rng(scenario.seed)
        directions = rng.standard_normal((scenario.crowd_talkers, 3))
        directions[:, 2] *= 0.35
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        diffuse_transfer = _plane_wave_transfer_powers(
            contract, directions, signal_fft_size, detector_fir
        )
        levels_power += _levels_from_spectra(
            crowd_spectrum * spl_to_amplitude(scenario.crowd_spl_db) ** 2,
            diffuse_transfer,
            signal_fft_size,
        )
    _progress(progress, "spectral-beamforming", 3, 3, started)

    _progress(progress, "detector", 0, 1, started)
    noise_pressure = spl_to_amplitude(scenario.microphone_noise_spl_db)
    levels_power += np.square(noise_pressure * math.sqrt(contract.microphones))
    rms = np.sqrt(np.maximum(levels_power, 0.0))
    rms = apply_detector_envelope(rms, scenario.level_step_s)
    levels_raw = np.clip(np.rint(rms * (1 << 24)), 0, 0x7FFFFFFF).astype(np.uint32)
    _progress(progress, "detector", 1, 1, started)

    manifest = {
        "schema": ACOUSTIC_MODEL_VERSION,
        "cache_id": cache_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibrated": False,
        "scenario": asdict(scenario),
        "grid": asdict(grid_spec),
        "grid_mode": grid_mode,
        "azimuth_deg": contract.azimuth_deg.tolist(),
        "elevation_deg": contract.elevation_deg.tolist(),
        "room_cache_id": room_cache.cache_id,
        "room_preparation_seconds": room_cache.manifest["preparation_seconds"],
        "contract_sha256": sha256(Path(contract_path).read_bytes()).hexdigest(),
        "source_c_header_sha256": deployment_contract.source_sha256,
        "levels_shape": list(levels_raw.shape),
        "levels_dtype": "uint32",
        "sample_rate_hz": 48000,
        "level_step_s": scenario.level_step_s,
        "preparation_seconds": time.perf_counter() - started,
        "measured_rt60_s": room_cache.manifest.get("measured_rt60_s"),
        "acoustic_method": "order-2 early reflections plus deterministic diffuse RT60 tail",
        "crowd_method": "12-direction diffuse plane-wave field",
        "detector_filter": "SigmaStudio exported order-10 1 kHz HP + 4 kHz LP FIR cascade",
        "exported_fir_status": "applied with highpass then lowpass stage semantics",
        "parity_scope": (
            "deployed delays, FIR topology, unity-gain sum, envelope settings, "
            "sequential policy; uncalibrated analog sensitivity and empirical late reverb"
        ),
    }

    with tempfile.TemporaryDirectory(dir=directory) as temporary_directory:
        _progress(progress, "cache-write", 0, 1, started)
        temporary_directory = Path(temporary_directory)
        temporary_levels = temporary_directory / "levels_raw.npy"
        temporary_manifest = temporary_directory / "manifest.json"
        np.save(temporary_levels, levels_raw, allow_pickle=False)
        manifest["levels_sha256"] = sha256(temporary_levels.read_bytes()).hexdigest()
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_levels, levels_path)
        os.replace(temporary_manifest, manifest_path)
    _progress(progress, "cache-write", 1, 1, started)
    return load_acoustic_cache(cache_id, cache_root)


class AcousticCacheLevelProvider:
    def __init__(self, cache: AcousticCache, clock=time.monotonic):
        self.cache = cache
        self.clock = clock
        self.started_at = clock()
        self.level_step_s = float(cache.manifest["level_step_s"])
        self.scenario = AcousticScenario(**cache.manifest["scenario"])
        self.source_positions = np.stack(_trajectory_positions(self.scenario))

    def _frame(self) -> int:
        elapsed = self.clock() - self.started_at
        return int(elapsed / self.level_step_s) % self.cache.levels_raw.shape[0]

    def level_for_sector(self, sector: int) -> int:
        return int(self.cache.levels_raw[self._frame(), sector])

    def _truth(self) -> dict:
        frame = self._frame()
        elapsed_s = frame * self.level_step_s
        position = _interpolated_source_position(
            elapsed_s, self.scenario, self.source_positions
        )
        offset = position - np.asarray(self.scenario.array_center_m, dtype=np.float64)
        horizontal_range = math.hypot(float(offset[0]), float(offset[1]))
        profile_time = math.fmod(elapsed_s * 2.0, 17.0)
        dropout = (self.scenario.scenario == "conference_evasive"
                   and 11.5 <= profile_time < 12.4)
        return {
            "frame": frame,
            "elapsed_s": elapsed_s,
            "loop_elapsed_s": math.fmod(elapsed_s, _trajectory_period_s(self.scenario)),
            "azimuth_deg": math.degrees(math.atan2(float(offset[0]), float(offset[1]))),
            "elevation_deg": math.degrees(math.atan2(float(offset[2]), horizontal_range)),
            "range_m": float(np.linalg.norm(offset)),
            "room_position_m": [float(value) for value in position],
            "source_active": not dropout,
        }

    def metadata(self) -> dict:
        scenario = self.cache.manifest["scenario"]
        return {
            "acoustic_cache_id": self.cache.cache_id,
            "acoustic_scenario": scenario["scenario"],
            "acoustic_calibrated": bool(self.cache.manifest["calibrated"]),
            "acoustic_model": self.cache.manifest["schema"],
            "acoustic_measured_rt60_s": self.cache.manifest.get("measured_rt60_s"),
            "acoustic_preparation_seconds": self.cache.manifest.get("preparation_seconds"),
            "acoustic_detector_filter": self.cache.manifest.get("detector_filter"),
            "acoustic_exported_fir_status": self.cache.manifest.get("exported_fir_status"),
            "acoustic_parity_scope": self.cache.manifest.get("parity_scope"),
            "acoustic_grid": self.cache.manifest.get("grid"),
            "acoustic_grid_mode": self.cache.manifest.get("grid_mode", "deployment_contract"),
            "acoustic_room_cache_id": self.cache.manifest.get("room_cache_id"),
            "emulator_truth": self._truth(),
        }
