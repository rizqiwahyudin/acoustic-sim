"""
FastAPI backend for live acoustic beamforming simulation.

Runs a single-trial pyroomacoustics simulation on demand and returns
the SRP-PHAT spatial power spectrum + beamformed audio.

Usage:
    python sim_server.py              # starts on port 8766
"""

import asyncio
import base64
import csv
import datetime
import io
import json
import os
import pathlib
import re
import subprocess
import time
import wave
from contextlib import asynccontextmanager
from typing import Any, Literal

import numpy as np
import pyroomacoustics as pra
import scipy.io.wavfile as wavfile
from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import Headers
from starlette.websockets import WebSocketClose

from array_theory import theory_cut
from bank_fit import BankFit
from heimdall_transport import DeployedProgram, EmulatorHeimdallTransport, SerialHeimdallTransport
from dsp_registry import DspParameterService, plan_writes, setting_order
from study_runner import StudyRunner
from heimdall_acoustic import (
    AcousticCacheLevelProvider,
    AcousticGridSpec,
    PreparationCancelled,
    acoustic_scenario,
    default_contract_path,
    load_beam_contract,
    prepare_acoustic_cache,
    render_acoustic_audition,
)

from acoustic_utils import (
    air_absorption_kwargs,
    apply_codec_quantization,
    apply_crosstalk,
    apply_crosstalk_fir,
    apply_mic_mismatch_v2,
    atmospheric_elevation_bias_deg,
    atmospheric_z_bias,
    build_freq_bin_mask,
    build_materials,
    chunk_signal_with_crossfade,
    compute_top_n_peaks,
    crowd_positions_mixed,
    feature_snr_db,
    MICCANVAS_POSITIONS,
    mips_estimate,
    load_crosstalk_fir,
    log_mel_features,
    make_custom,
    make_trajectory,
    measure_rt60_from_rir,
    ml_path_quantize_audio,
    ml_path_quantize_features,
    ml_path_snr_db,
    scan_directions,
    render_spectrogram_png_b64,
    spl_to_amplitude,
    synthesize_diffuse_crowd_plane_waves,
    wall_adjacent_positions,
    CODEC_BIT_DEPTH_DEFAULT,
    CROSSTALK_COUPLING_DB_DEFAULT,
    CROWD_SPL_DB,
    DEFAULT_FMAX,
    DEFAULT_FMIN,
    DRONE_SPL_DB,
    EXHIBITION_HALL_MATERIALS,
    MATERIAL_CHOICES,
    MIC_NOISE_FLOOR_DB,
    ML_DEFAULT_BIT_DEPTH,
    ML_DEFAULT_FEATURE_BIT_DEPTH,
    ML_DEFAULT_N_MELS,
    PA_SPL_DB,
)

# ── Constants ──────────────────────────────────────────────────────────────────

FS = 16_000
NFFT = 1024
HOP = 512
C = 343.0
MARGIN = 0.3

SRP_AZ = np.linspace(0, 2 * np.pi, 72, endpoint=False)
SRP_COLAT = np.linspace(0, np.pi, 19)

# ── Audio loading ──────────────────────────────────────────────────────────────

AUDIO_DIR = pathlib.Path("audio")

DRONE_AUDIO = None
CROWD_AUDIO = None


def _load_and_normalize(path):
    sr, data = wavfile.read(path)
    data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != FS:
        n_out = int(len(data) * FS / sr)
        data = np.interp(np.linspace(0, len(data) - 1, n_out),
                         np.arange(len(data)), data)
    rms = np.sqrt(np.mean(data ** 2))
    if rms > 1e-12:
        data /= rms
    return data


def load_audio():
    global DRONE_AUDIO, CROWD_AUDIO
    drone_path = AUDIO_DIR / "drone.wav"
    crowd_path = AUDIO_DIR / "crowd.wav"

    if drone_path.exists():
        DRONE_AUDIO = _load_and_normalize(drone_path)
        print(f"Loaded drone audio: {len(DRONE_AUDIO)/FS:.1f}s")
    else:
        print("drone.wav not found -- will use synthetic signal")

    if crowd_path.exists():
        CROWD_AUDIO = _load_and_normalize(crowd_path)
        print(f"Loaded crowd audio: {len(CROWD_AUDIO)/FS:.1f}s")
    else:
        print("crowd.wav not found -- will use synthetic noise")


# ── Geometry builders ──────────────────────────────────────────────────────────

def make_uca(center, mic_count=12, radius=0.15):
    xy = pra.circular_2D_array(center[:2], mic_count, 0.0, radius)
    z_row = np.full((1, mic_count), center[2])
    return np.vstack([xy, z_row])


def make_cross(center, mic_count=12, half_length=0.15):
    mics_per_arm = mic_count // 2
    offsets = np.linspace(-half_length, half_length, mics_per_arm + 1)
    offsets = offsets[np.abs(offsets) > 1e-9]
    pts = []
    for o in offsets:
        pts.append([center[0] + o, center[1], center[2]])
    for o in offsets:
        pts.append([center[0], center[1], center[2] + o])
    return np.array(pts).T


def make_ula(center, mic_count=12, length=0.30):
    offsets = np.linspace(-length / 2, length / 2, mic_count)
    R = np.zeros((3, mic_count))
    R[0, :] = center[0] + offsets
    R[1, :] = center[1]
    R[2, :] = center[2]
    return R


def make_cylinder(center, mic_count=12, radius=0.15, separation=0.12):
    mics_per_ring = mic_count // 2
    half_sep = separation / 2.0
    bot_xy = pra.circular_2D_array(center[:2], mics_per_ring, 0.0, radius)
    bot_z = np.full((1, mics_per_ring), center[2] - half_sep)
    top_xy = pra.circular_2D_array(center[:2], mics_per_ring,
                                    np.pi / mics_per_ring, radius)
    top_z = np.full((1, mics_per_ring), center[2] + half_sep)
    return np.hstack([
        np.vstack([bot_xy, bot_z]),
        np.vstack([top_xy, top_z]),
    ])


def build_array(geometry, center, mic_count, radius, ring_separation,
                room_dim=None, custom_mic_positions=None):
    if geometry == "UCA":
        return make_uca(center, mic_count, radius), []
    elif geometry == "CROSS":
        return make_cross(center, mic_count, radius), []
    elif geometry == "ULA":
        return make_ula(center, mic_count, radius * 2), []
    elif geometry == "CYLINDER":
        return make_cylinder(center, mic_count, radius, ring_separation), []
    elif geometry == "CUSTOM":
        if not custom_mic_positions or len(custom_mic_positions) < 2:
            raise HTTPException(
                status_code=400,
                detail="CUSTOM requires >= 2 mic positions",
            )
        try:
            return make_custom(
                center, custom_mic_positions,
                room_dim=room_dim, margin=MARGIN,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    elif geometry == "MICCANVAS":
        try:
            return make_custom(center, MICCANVAS_POSITIONS, room_dim=room_dim, margin=MARGIN)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise ValueError(f"Unknown geometry: {geometry}")


# ── Signal helpers ─────────────────────────────────────────────────────────────

def drone_signal_synthetic(n_samples, rng):
    t = np.arange(n_samples) / FS
    sig = np.zeros(n_samples)
    for f0 in [150, 300, 600, 900]:
        sig += np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
    sig += 0.3 * rng.standard_normal(n_samples)
    return sig / np.max(np.abs(sig))


def get_drone_signal(n_samples, rng):
    if DRONE_AUDIO is not None:
        if len(DRONE_AUDIO) >= n_samples:
            start = rng.integers(0, len(DRONE_AUDIO) - n_samples + 1)
            return DRONE_AUDIO[start:start + n_samples].copy()
        else:
            repeats = (n_samples // len(DRONE_AUDIO)) + 1
            return np.tile(DRONE_AUDIO, repeats)[:n_samples].copy()
    return drone_signal_synthetic(n_samples, rng)


def get_crowd_segments(n_segments, n_samples, rng):
    """Return list of n_segments independent crowd noise arrays."""
    segments = []
    if CROWD_AUDIO is not None:
        total = len(CROWD_AUDIO)
        max_segments = total // n_samples
        for i in range(n_segments):
            if i < max_segments:
                seg = CROWD_AUDIO[i * n_samples:(i + 1) * n_samples].copy()
            else:
                shift = rng.integers(0, n_samples)
                base_idx = (i % max_segments) * n_samples
                seg = np.roll(CROWD_AUDIO[base_idx:base_idx + n_samples].copy(), shift)
            segments.append(seg)
    else:
        for _ in range(n_segments):
            segments.append(rng.standard_normal(n_samples))
    return segments


def drone_position(center, az_deg, el_deg, distance, room_dim):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    src = np.array([
        center[0] + distance * np.cos(el) * np.cos(az),
        center[1] + distance * np.cos(el) * np.sin(az),
        center[2] + distance * np.sin(el),
    ])
    return np.clip(src, MARGIN, room_dim - MARGIN)


# ── Beamforming ────────────────────────────────────────────────────────────────

def delay_and_sum(signals, mic_positions, az_rad, el_rad, fs):
    """Beamform toward a direction using delay-and-sum."""
    colat = np.pi / 2 - el_rad
    d = np.array([
        np.sin(colat) * np.cos(az_rad),
        np.sin(colat) * np.sin(az_rad),
        np.cos(colat),
    ])
    center = mic_positions.mean(axis=1)
    delays = (mic_positions - center[:, None]).T @ d / C
    delays -= delays.min()

    n_samples = signals.shape[1]
    output = np.zeros(n_samples)
    for m in range(signals.shape[0]):
        shift = int(round(delays[m] * fs))
        if shift >= n_samples:
            continue
        output[shift:] += signals[m, :n_samples - shift]
    output /= signals.shape[0]
    return output


def encode_wav_b64_raw(audio, fs):
    """Encode an already-normalised float audio array (peak ≤ 1.0) as 16-bit WAV/base64."""
    int16 = np.clip(audio, -1.0, 1.0) * 32767.0
    int16 = int16.astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, fs, int16)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# Fixed full-scale reference used when audio normalization is disabled, so
# the downloaded .wav files preserve absolute amplitudes across separate
# trials. 1 Pa ≈ 94 dB SPL, which comfortably brackets drone / crowd /
# PA levels in our sim without clipping at typical integration windows.
AUDIO_FULL_SCALE_PA = 1.0


def encode_three_wavs_joint(clips, fs, normalize=True):
    """Encode three audio clips as base64 WAVs.

    When ``normalize=True`` (default): the three clips (raw, unsteered,
    beamformed) are normalised by a single joint peak value so that their
    relative loudness is preserved on playback. This makes it easy to hear
    that beamforming improves SNR -- normalising each clip independently
    would erase that difference.

    When ``normalize=False``: each clip is encoded against a fixed
    ``AUDIO_FULL_SCALE_PA`` reference (1 Pa) instead. That scale is the
    same for every trial, so the downloaded .wav amplitudes preserve the
    *absolute* Pascal values. Use this mode when you want to compare
    loudness across two separate trials (e.g. a 1/r² free-field falloff
    sanity check).
    """
    if normalize:
        peak = max(float(np.max(np.abs(c))) for c in clips)
        if peak < 1e-12:
            peak = 1.0
        return [encode_wav_b64_raw(c / peak * 0.9, fs) for c in clips]
    return [encode_wav_b64_raw(c / AUDIO_FULL_SCALE_PA, fs) for c in clips]


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(_app):
    load_audio()
    yield
    _acoustic_preparation.cancel()
    _replace_heimdall_transport(None)


app = FastAPI(title="Acoustic Beamformer Simulation", lifespan=_lifespan)

LOCAL_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")


class LocalOriginMiddleware:
    """A web page from another computer must not command the board (CORS alone does not stop the request)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            origin = Headers(scope=scope).get("origin")
            if origin is not None and not LOCAL_ORIGIN.match(origin):
                if scope["type"] == "http":
                    response = JSONResponse({"detail": "This server answers pages on this computer only."},
                                            status_code=403)
                else:
                    response = WebSocketClose(code=1008)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=LOCAL_ORIGIN.pattern,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LocalOriginMiddleware)


class SimRequest(BaseModel):
    geometry: str = "UCA"
    custom_mic_positions: list[list[float]] | None = None
    beam_method: str = "srp_phat"  # "srp_phat" | "steered_das" | "beam_bank_das"
    beam_bank_angles_deg: list[float] | None = None
    frac_delay_max_samples: int = 8
    frac_delay_taps: int = 16
    target_fs_hz: int = 48_000
    mic_count: int = 12
    radius: float = 0.15
    ring_separation: float = 0.12
    room_length: float = 50.0
    room_width: float = 40.0
    room_height: float = 12.0
    rt60: float = 1.5
    source_az_deg: float = 60.0
    source_el_deg: float = 30.0
    source_distance: float = 6.0
    drone_spl_db: float = DRONE_SPL_DB
    crowd_spl_db: float = CROWD_SPL_DB
    pa_spl_db: float = PA_SPL_DB
    mic_noise_floor_db: float = MIC_NOISE_FLOOR_DB
    seed: int = -1
    diffuse: bool = True
    crowd_count: int = 30
    pa_count: int = 8
    mic_mismatch: bool = False
    integration_ms: int = 1000

    absorption_mode: str = "rt60"  # "rt60" | "materials"
    floor_material: str = EXHIBITION_HALL_MATERIALS["floor"]
    ceiling_material: str = EXHIBITION_HALL_MATERIALS["ceiling"]
    east_material: str = EXHIBITION_HALL_MATERIALS["east"]
    west_material: str = EXHIBITION_HALL_MATERIALS["west"]
    south_material: str = EXHIBITION_HALL_MATERIALS["south"]
    north_material: str = EXHIBITION_HALL_MATERIALS["north"]

    crosstalk: bool = False
    crosstalk_db: float = CROSSTALK_COUPLING_DB_DEFAULT
    quantization: bool = False
    bit_depth: int = CODEC_BIT_DEPTH_DEFAULT

    # Phase 2b: atmosphere and moving source.
    temperature_c: float = 20.0
    humidity_pct: float = 50.0
    temp_gradient_c_per_m: float = 0.0
    moving_source: bool = False
    trajectory_type: str = "straight"  # "straight" | "arc"
    speed_mps: float = 5.0
    heading_deg: float = 0.0
    n_trajectory_chunks: int = 8

    # Phase 3A: MAX78002 fixed-point ML-path preview.
    ml_preview: bool = False
    ml_bit_depth: int = ML_DEFAULT_BIT_DEPTH          # 8 or 16
    ml_feature_bit_depth: int = ML_DEFAULT_FEATURE_BIT_DEPTH  # 8 or 16
    ml_n_mels: int = ML_DEFAULT_N_MELS

    # Phase 3B: crowd-noise spatial-correlation model.
    crowd_model: str = "point_source"  # "point_source" | "plane_wave"
    n_plane_waves: int = 64

    # Phase 3C: FIR capacitive-coupling crosstalk.
    crosstalk_model: str = "simple"   # "simple" | "fir_capacitive"
    crosstalk_corner_hz: float = 500.0
    crosstalk_fir_path: str = ""

    # Phase 3+: SRP-PHAT band and optional harmonic-comb weighting.
    fmin_hz: float = DEFAULT_FMIN
    fmax_hz: float = DEFAULT_FMAX
    harmonic_comb: bool = False
    drone_fundamental_hz: float = 200.0

    # Audio-normalization toggle: default keeps the legacy joint-peak
    # normalization so beamforming SNR gain is audible in playback. Turn
    # off to preserve absolute amplitude across trials (use for 1/r^2
    # free-field falloff checks etc.).
    normalize_audio: bool = True


def run_live_trial(req: SimRequest) -> dict:
    """Pure (non-FastAPI) entry point for a single simulation trial.

    Broken out from the ``/simulate`` route so that the parity test and
    future tooling can invoke it in-process without spinning up a server.
    """
    t0 = time.time()
    if req.seed < 0:
        seed_used = int(np.random.SeedSequence().entropy & 0x7FFFFFFF)
    else:
        seed_used = int(req.seed)
    rng = np.random.default_rng(seed_used)

    room_dim = np.array([req.room_length, req.room_width, req.room_height])
    warmup_samples = int(0.1 * FS)
    n_samples_requested = int(FS * req.integration_ms / 1000)
    n_samples_sim = n_samples_requested + warmup_samples

    array_center = room_dim / 2
    array_center[2] = 1.0

    array_R, mic_clip_notes = build_array(
        req.geometry, array_center, req.mic_count,
        req.radius, req.ring_separation,
        room_dim=room_dim,
        custom_mic_positions=req.custom_mic_positions,
    )

    sigma2 = spl_to_amplitude(req.mic_noise_floor_db) ** 2

    materials_mode = (req.absorption_mode == "materials")
    anechoic = (not materials_mode) and req.rt60 <= 0.0

    air_kw = air_absorption_kwargs(req.temperature_c, req.humidity_pct)

    if anechoic:
        room = pra.AnechoicRoom(3, fs=FS, sigma2_awgn=sigma2)
    elif materials_mode:
        mats = build_materials(
            req.floor_material, req.ceiling_material,
            req.east_material, req.west_material,
            req.south_material, req.north_material,
        )
        room = pra.ShoeBox(
            room_dim.tolist(), fs=FS, sigma2_awgn=sigma2,
            materials=mats, max_order=6,
            **air_kw,
        )
    else:
        try:
            e_absorption, max_order = pra.inverse_sabine(req.rt60, room_dim.tolist())
        except ValueError:
            e_absorption = 0.5
            max_order = 3
        max_order = min(max_order, 6)
        room = pra.ShoeBox(
            room_dim.tolist(), fs=FS, sigma2_awgn=sigma2,
            materials=pra.Material(e_absorption),
            max_order=max_order,
            **air_kw,
        )

    src_pos = drone_position(array_center, req.source_az_deg, req.source_el_deg,
                             req.source_distance, room_dim)
    drone_sig = get_drone_signal(n_samples_sim, rng) * spl_to_amplitude(req.drone_spl_db)

    # First-order gradient beam-bending: shift each source position in z
    # before handing it to pyroomacoustics. A static source is identical to
    # today's code path when temp_gradient_c_per_m == 0.
    grad = float(req.temp_gradient_c_per_m)

    # Phase 2b moving source: render as K phantom static sources along the
    # trajectory, each fed a Hann-crossfaded chunk of the drone signal. One
    # simulate() call still renders them all, so Doppler + direction-change
    # come for free from the differing ISM delays.
    duration_s = float(n_samples_sim) / FS
    if req.moving_source:
        n_chunks = max(int(req.n_trajectory_chunks), 1)
        traj = make_trajectory(
            src_pos,
            speed_mps=float(req.speed_mps),
            heading_deg=float(req.heading_deg),
            trajectory_type=str(req.trajectory_type),
            duration_s=duration_s,
            n_chunks=n_chunks,
            array_center=array_center,
        )
        chunks = chunk_signal_with_crossfade(
            drone_sig, n_chunks, FS, crossfade_ms=10.0,
        )
    else:
        traj = [np.asarray(src_pos, dtype=np.float64)]
        chunks = [drone_sig]

    trajectory_out = []
    for pos, chunk in zip(traj, chunks):
        shifted = atmospheric_z_bias(pos, array_center, grad)
        shifted = np.clip(shifted, MARGIN, room_dim - MARGIN)
        room.add_source(shifted.tolist(), signal=chunk)
        trajectory_out.append(shifted.tolist())

    crowd_positions = []
    pa_positions = []
    plane_wave_mode = (req.crowd_model == "plane_wave")
    plane_wave_crowd_source_signals = None  # retained for post-simulate() addition
    if req.diffuse:
        crowd_positions = crowd_positions_mixed(
            room_dim, req.crowd_count, z_height=1.5,
            array_center=array_center, rng=rng,
        )
        pa_positions = wall_adjacent_positions(
            room_dim, req.pa_count, z_height=3.0, rng=rng,
        )

        # Crowd audio segments are needed either way; in plane-wave mode we
        # feed them to the diffuse-field generator rather than to ShoeBox.
        n_plane_src_needed = max(int(req.n_plane_waves), req.crowd_count)
        crowd_segs = get_crowd_segments(
            n_plane_src_needed if plane_wave_mode else req.crowd_count,
            n_samples_requested, rng,
        )
        pa_segs = get_crowd_segments(req.pa_count, n_samples_sim, rng)

        crowd_scale = spl_to_amplitude(req.crowd_spl_db)
        pa_scale = spl_to_amplitude(req.pa_spl_db)

        if not plane_wave_mode:
            for pos, seg in zip(crowd_positions, crowd_segs):
                room.add_source(pos, signal=seg * crowd_scale)
        else:
            plane_wave_crowd_source_signals = [
                seg * crowd_scale for seg in crowd_segs
            ]

        for pos, seg in zip(pa_positions, pa_segs):
            room.add_source(pos, signal=seg * pa_scale)

    room.add_microphone_array(pra.MicrophoneArray(array_R, fs=FS))
    room.simulate()

    signals = room.mic_array.signals[:, warmup_samples:warmup_samples + n_samples_requested]

    if plane_wave_mode and plane_wave_crowd_source_signals is not None:
        diffuse = synthesize_diffuse_crowd_plane_waves(
            array_R,
            duration_s=float(n_samples_requested) / FS,
            fs=FS,
            n_planes=int(req.n_plane_waves),
            source_signals=plane_wave_crowd_source_signals,
            rng=rng,
        )
        # Line up diffuse length with simulate() output before adding.
        n = min(signals.shape[1], diffuse.shape[1])
        signals[:, :n] = signals[:, :n] + diffuse[:, :n]

    # Hardware-impairments chain (order = closest-to-ADC last).
    if req.mic_mismatch:
        signals = apply_mic_mismatch_v2(signals, FS, rng)
    if req.crosstalk:
        if req.crosstalk_model == "fir_capacitive":
            measured = load_crosstalk_fir(req.crosstalk_fir_path)
            signals = apply_crosstalk_fir(
                signals, FS,
                coupling_db=req.crosstalk_db,
                corner_hz=float(req.crosstalk_corner_hz),
                measured_fir=measured,
            )
        else:
            signals = apply_crosstalk(signals, coupling_db=req.crosstalk_db)
    if req.quantization:
        signals = apply_codec_quantization(signals, bit_depth=req.bit_depth)

    freq_bins = build_freq_bin_mask(
        FS, NFFT,
        fmin_hz=req.fmin_hz,
        fmax_hz=req.fmax_hz,
        harmonic_comb=req.harmonic_comb,
        f0_hz=req.drone_fundamental_hz,
    )
    beam_method = str(req.beam_method).lower().strip()
    valid_beam_methods = {"srp_phat", "steered_das", "beam_bank_das"}
    if beam_method not in valid_beam_methods:
        raise HTTPException(
            status_code=400,
            detail=f"beam_method must be one of {sorted(valid_beam_methods)}",
        )

    beam_scan = None
    mips = mips_estimate(
        beam_method,
        n_mics=int(array_R.shape[1]),
        k_beams=1,
        n_taps=int(req.frac_delay_taps),
        target_fs_hz=int(req.target_fs_hz),
    )

    if beam_method == "srp_phat":
        X = np.array([
            pra.transform.stft.analysis(sig, NFFT, HOP).T
            for sig in signals
        ])

        doa = pra.doa.SRP(
            array_R, FS, NFFT, c=C, num_src=1, dim=3,
            azimuth=SRP_AZ, colatitude=SRP_COLAT,
        )
        doa.locate_sources(X, freq_bins=freq_bins)

        est_az_rad = float(np.atleast_1d(doa.azimuth_recon)[0])
        est_colat_rad = float(np.atleast_1d(doa.colatitude_recon)[0])
        est_az_deg = float(np.rad2deg(est_az_rad))
        est_el_deg = float(90.0 - np.rad2deg(est_colat_rad))

        grid_vals = np.array(doa.grid.values, copy=True)
        power_2d = grid_vals.reshape(len(SRP_COLAT), len(SRP_AZ)).tolist()
        top_peaks = compute_top_n_peaks(
            grid_vals, SRP_AZ, SRP_COLAT, n=3, min_angular_sep_deg=15.0,
        )
        bf_audio = delay_and_sum(
            signals, array_R, est_az_rad, np.deg2rad(est_el_deg), FS,
        )
    else:
        if req.beam_bank_angles_deg and len(req.beam_bank_angles_deg) > 0:
            angles_deg = [float(a) % 360.0 for a in req.beam_bank_angles_deg]
        else:
            angles_deg = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))

        beams, powers_db, clamp_notes_per_angle = scan_directions(
            signals, array_R, angles_rad, FS,
            c=C,
            frac_delay_max_samples=int(req.frac_delay_max_samples),
            n_taps=int(req.frac_delay_taps),
        )
        argmax_idx = int(np.argmax(powers_db))
        est_az_deg = float(angles_deg[argmax_idx])
        est_el_deg = 0.0
        bf_audio = beams[argmax_idx]
        power_2d = []
        top_peaks = []

        clamp_notes = []
        for i, notes in enumerate(clamp_notes_per_angle):
            for n in notes:
                clamp_notes.append({
                    "angle_deg": float(angles_deg[i]),
                    "idx": int(n["idx"]),
                    "before_samples": float(n["before_samples"]),
                    "after_samples": float(n["after_samples"]),
                    "max_samples": float(n["max_samples"]),
                })
        beam_scan = {
            "angles_deg": [float(a) for a in angles_deg],
            "powers_db": [float(x) for x in powers_db.tolist()],
            "argmax_idx": argmax_idx,
            "clamp_notes": clamp_notes,
        }
        k_for_mips = len(angles_deg) if beam_method == "beam_bank_das" else 1
        mips = mips_estimate(
            beam_method,
            n_mics=int(array_R.shape[1]),
            k_beams=k_for_mips,
            n_taps=int(req.frac_delay_taps),
            target_fs_hz=int(req.target_fs_hz),
        )
        if beam_method == "steered_das":
            mips["scan_latency_s"] = (
                float(len(angles_deg)) * float(req.integration_ms) / 1000.0
            )

    n_out = signals.shape[1]
    raw_audio = signals[0, :n_out].copy()
    unsteered_audio = signals[:, :n_out].mean(axis=0)

    raw_b64, unsteered_b64, bf_b64 = encode_three_wavs_joint(
        [raw_audio, unsteered_audio, bf_audio], FS,
        normalize=req.normalize_audio,
    )

    # Phase 3A: MAX78002 ML-path preview -- quantize the beamformed audio
    # post-beamforming, run a proxy log-mel feature extractor, and report
    # the SNR hit at each stage. The preview is expensive enough that we
    # only compute it on request.
    ml_audio_b64 = None
    ml_spectrogram_png_b64 = None
    ml_path_snr_db_val = None
    feature_snr_db_val = None
    if req.ml_preview:
        ml_bits = int(req.ml_bit_depth)
        feat_bits = int(req.ml_feature_bit_depth)
        n_mels = int(req.ml_n_mels)

        ml_audio = ml_path_quantize_audio(bf_audio, bit_depth=ml_bits)
        ml_path_snr_db_val = round(ml_path_snr_db(bf_audio, ml_audio), 2)

        # Features computed on the quantized audio (what the CNN sees).
        # The mel filterbank tracks the user-configured DOA band so the
        # spectrogram matches the slice of spectrum SRP-PHAT is using --
        # they share the same analog-front-end bandpass on real hardware.
        mel_features = log_mel_features(
            ml_audio, fs=FS, n_mels=n_mels,
            fmin=req.fmin_hz, fmax=req.fmax_hz,
        )
        mel_features_ref = log_mel_features(
            bf_audio, fs=FS, n_mels=n_mels,
            fmin=req.fmin_hz, fmax=req.fmax_hz,
        )
        mel_features_q = ml_path_quantize_features(
            mel_features, bit_depth=feat_bits,
        )
        feature_snr_db_val = round(
            feature_snr_db(mel_features_ref, mel_features_q), 2,
        )

        # Encode the quantized audio as a 4th downloadable WAV. In
        # normalize=True mode we peak-normalize it like the other three
        # clips; in normalize=False mode we keep its absolute amplitude
        # scaled by AUDIO_FULL_SCALE_PA to match the other clips.
        if req.normalize_audio:
            ml_audio_b64 = encode_wav_b64_raw(
                np.clip(ml_audio / max(float(np.max(np.abs(ml_audio))), 1e-12) * 0.9,
                        -1.0, 1.0),
                FS,
            )
        else:
            ml_audio_b64 = encode_wav_b64_raw(ml_audio / AUDIO_FULL_SCALE_PA, FS)
        ml_spectrogram_png_b64 = render_spectrogram_png_b64(
            mel_features_q, fs=FS, n_mels=n_mels,
            fmin=req.fmin_hz, fmax=req.fmax_hz,
        )

    mic_rel = (array_R - array_center[:, None]).T.tolist()

    atm_bias_deg = atmospheric_elevation_bias_deg(
        src_pos, array_center, grad,
    )

    image_sources = []
    if not anechoic and hasattr(room, 'sources') and len(room.sources) > 0:
        src0 = room.sources[0]
        if hasattr(src0, 'images') and src0.images.shape[1] > 1:
            n_img = min(src0.images.shape[1], 7)
            image_sources = src0.images[:, 1:n_img].T.tolist()

    rir_data = []
    rt60_actual = None
    if hasattr(room, 'rir') and room.rir is not None and len(room.rir) > 0:
        raw_rir = np.asarray(room.rir[0][0])
        rir_data = raw_rir[:min(len(raw_rir), 2000)].tolist()
        if not anechoic and materials_mode:
            rt60_actual = measure_rt60_from_rir(raw_rir, fs=FS, decay_db=20)

    elapsed = time.time() - t0

    return {
        "power": power_2d,
        "top_peaks": top_peaks,
        "beam_method": beam_method,
        "beam_scan": beam_scan,
        "mips": mips,
        "est_az_deg": round(est_az_deg, 1),
        "est_el_deg": round(est_el_deg, 1),
        "true_az_deg": req.source_az_deg,
        "true_el_deg": req.source_el_deg,
        "mic_positions": mic_rel,
        "mic_clip_notes": mic_clip_notes,
        "audio_b64": bf_b64,
        "raw_audio_b64": raw_b64,
        "unsteered_audio_b64": unsteered_b64,
        "ml_audio_b64": ml_audio_b64,
        "ml_spectrogram_png_b64": ml_spectrogram_png_b64,
        "ml_path_snr_db": ml_path_snr_db_val,
        "feature_snr_db": feature_snr_db_val,
        "elapsed_s": round(elapsed, 2),
        "seed_used": seed_used,
        "rt60_actual": (round(rt60_actual, 2) if rt60_actual is not None else None),
        "params": req.model_dump(),
        "room_dim": room_dim.tolist(),
        "source_pos": src_pos.tolist(),
        "trajectory": trajectory_out,
        "atmospheric_bias_deg": round(float(atm_bias_deg), 3),
        "array_center": array_center.tolist(),
        "crowd_positions": crowd_positions,
        "pa_positions": pa_positions,
        "image_sources": image_sources,
        "rir": rir_data,
        "fmin_hz": float(req.fmin_hz),
        "fmax_hz": float(req.fmax_hz),
        "harmonic_comb": bool(req.harmonic_comb),
        "drone_fundamental_hz": float(req.drone_fundamental_hz),
        "n_freq_bins": int(freq_bins.size),
    }


@app.post("/simulate")
def simulate(req: SimRequest):
    return run_live_trial(req)


@app.get("/materials")
def list_materials():
    """Expose the curated material list so the frontend can build dropdowns
    without hard-coding names that might drift from the backend."""
    return {"choices": list(MATERIAL_CHOICES),
            "exhibition_hall": dict(EXHIBITION_HALL_MATERIALS)}


# ── Hardware Serial Reader ─────────────────────────────────────────────────────

import threading

_heimdall_transport = None
_heimdall_transport_lock = threading.Lock()

# The Study screen's test runner. While a test runs, only its own connections may command the board.
_study_runner = StudyRunner()
RUN_HEADER = "x-heimdall-run"


def _unlocked(request: Request) -> None:
    if not _study_runner.allows(request.headers.get(RUN_HEADER)):
        raise HTTPException(status_code=423, detail=_study_runner.lock_text())


# ── Hardware WebSocket endpoint ────────────────────────────────────────────────

_hw_true_az_deg = None   # User-set ground truth for calibration
_hw_true_el_deg = None


class HardwareConnectRequest(BaseModel):
    transport: str = "serial"
    port: str | None = None
    baud: int = 921600
    rows: int = 7
    columns: int = 7
    azimuth_min_deg: float = -70.0
    azimuth_max_deg: float = 70.0
    elevation_min_deg: float = -60.0
    elevation_max_deg: float = 60.0
    flight_profile: str = "crossing"
    flight_speed: float = 1.0
    acoustic_cache_id: str | None = None


class AcousticPrepareRequest(BaseModel):
    scenario: str = "conference_evasive"
    rows: int = 7
    columns: int = 7
    azimuth_min_deg: float = -70.0
    azimuth_max_deg: float = 70.0
    elevation_min_deg: float = -60.0
    elevation_max_deg: float = 60.0


class AcousticAuditionRequest(BaseModel):
    acoustic_cache_id: str
    start_s: float = 0.0
    duration_s: float = 3.0
    selected_sector: int | None = None


class AcousticPreparationManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.worker = None
        self.cache = None
        self.status = {"state": "idle", "stage": "idle", "percent": 0.0}

    def snapshot(self):
        with self.lock:
            return dict(self.status)

    def start(self, scenario_name, grid_spec=None):
        scenario = acoustic_scenario(scenario_name)
        grid_spec = grid_spec or AcousticGridSpec()
        grid_spec.validate()
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                raise ValueError("an acoustic scene is already being prepared")
            self.cancel_event = threading.Event()
            self.cache = None
            self.status = {
                "state": "preparing",
                "stage": "starting",
                "percent": 0.0,
                "scenario": scenario_name,
                "grid": grid_spec.__dict__,
            }
            self.worker = threading.Thread(
                target=self._prepare,
                args=(scenario, grid_spec),
                daemon=True,
            )
            self.worker.start()
        return self.snapshot()

    def _prepare(self, scenario, grid_spec):
        def report(progress):
            with self.lock:
                self.status.update(progress)

        try:
            cache = prepare_acoustic_cache(
                scenario,
                progress=report,
                cancel_event=self.cancel_event,
                grid_spec=grid_spec,
            )
        except PreparationCancelled:
            with self.lock:
                self.status.update({"state": "canceled", "stage": "canceled"})
            return
        except Exception as error:
            with self.lock:
                self.status.update({
                    "state": "failed",
                    "stage": "failed",
                    "error": str(error),
                })
            return
        with self.lock:
            self.cache = cache
            grid = cache.manifest.get("grid", grid_spec.__dict__)
            self.status.update({
                "state": "ready",
                "stage": "ready",
                "percent": 100.0,
                "cache_id": cache.cache_id,
                "preparation_seconds": cache.manifest["preparation_seconds"],
                "measured_rt60_s": cache.manifest.get("measured_rt60_s"),
                "calibrated": cache.manifest["calibrated"],
                "grid": grid,
                "grid_mode": cache.manifest.get("grid_mode", "deployment_contract"),
                "room_cache_id": cache.manifest.get("room_cache_id"),
            })

    def cancel(self):
        self.cancel_event.set()
        return self.snapshot()

    def ready_cache(self, cache_id=None):
        with self.lock:
            cache = self.cache
        if cache is None or (cache_id is not None and cache.cache_id != cache_id):
            raise ValueError("prepare the requested acoustic scene before connecting")
        return cache


_acoustic_preparation = AcousticPreparationManager()


def _replace_heimdall_transport(transport):
    global _heimdall_transport
    with _heimdall_transport_lock:
        previous = _heimdall_transport
        _heimdall_transport = transport
    if previous is not None:
        try:
            previous.send_command("X")
        except (OSError, ValueError):
            pass
        previous.stop()
        if isinstance(previous, threading.Thread):
            previous.join(timeout=1.0)
    if transport is not None:
        transport.start()


@app.post("/hw_connect", dependencies=[Depends(_unlocked)])
def hw_connect(request: HardwareConnectRequest):
    """Connect serial hardware or start the protocol-faithful emulator."""
    try:
        if request.transport == "emulator":
            transport = EmulatorHeimdallTransport(
                rows=request.rows,
                columns=request.columns,
                azimuth_min_deg=request.azimuth_min_deg,
                azimuth_max_deg=request.azimuth_max_deg,
                elevation_min_deg=request.elevation_min_deg,
                elevation_max_deg=request.elevation_max_deg,
                flight_profile=request.flight_profile,
                flight_speed=request.flight_speed,
                program=_deployed_program(),
                parameter_service=_dsp_service,
                bank_fit=_bank_fit(),
            )
        elif request.transport == "acoustic":
            cache = _acoustic_preparation.ready_cache(request.acoustic_cache_id)
            grid = cache.manifest.get("grid", AcousticGridSpec().__dict__)
            rows = int(grid["rows"])
            columns = int(grid["columns"])
            if cache.levels_raw.shape[1] != rows * columns:
                raise ValueError("acoustic cache sector count does not match its simulation grid")
            transport = EmulatorHeimdallTransport(
                rows=rows,
                columns=columns,
                azimuth_min_deg=float(grid["azimuth_min_deg"]),
                azimuth_max_deg=float(grid["azimuth_max_deg"]),
                elevation_min_deg=float(grid["elevation_min_deg"]),
                elevation_max_deg=float(grid["elevation_max_deg"]),
                level_provider=AcousticCacheLevelProvider(cache),
                program=_deployed_program(),
                parameter_service=_dsp_service,
                bank_fit=_bank_fit(),
            )
        elif request.transport == "serial":
            if not request.port:
                raise ValueError("serial port is required")
            transport = SerialHeimdallTransport(request.port, request.baud)
        else:
            raise ValueError("transport must be 'serial' or 'emulator'")
        _replace_heimdall_transport(transport)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": "connecting", "transport": transport.transport_name}


@app.post("/hw_acoustic_prepare")
def hw_acoustic_prepare(request: AcousticPrepareRequest):
    try:
        return _acoustic_preparation.start(
            request.scenario,
            AcousticGridSpec(
                rows=request.rows,
                columns=request.columns,
                azimuth_min_deg=request.azimuth_min_deg,
                azimuth_max_deg=request.azimuth_max_deg,
                elevation_min_deg=request.elevation_min_deg,
                elevation_max_deg=request.elevation_max_deg,
            ),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/hw_acoustic_status")
def hw_acoustic_status():
    return _acoustic_preparation.snapshot()


@app.post("/hw_acoustic_cancel")
def hw_acoustic_cancel():
    return _acoustic_preparation.cancel()


@app.post("/hw_acoustic_audition")
def hw_acoustic_audition(request: AcousticAuditionRequest):
    try:
        cache = _acoustic_preparation.ready_cache(request.acoustic_cache_id)
        rendered = render_acoustic_audition(
            cache,
            start_s=request.start_s,
            duration_s=request.duration_s,
            selected_sector=request.selected_sector,
        )
        names = list(rendered["clips"])
        encoded = encode_three_wavs_joint(
            [rendered["clips"][name] for name in names],
            rendered["sample_rate_hz"],
        )
        return {
            "sample_rate_hz": rendered["sample_rate_hz"],
            "start_s": rendered["start_s"],
            "duration_s": rendered["duration_s"],
            "selected_sector": rendered["selected_sector"],
            "truth_azimuth_deg": rendered["truth_azimuth_deg"],
            "truth_elevation_deg": rendered["truth_elevation_deg"],
            "clips_b64": dict(zip(names, encoded)),
            "normalization": "joint_peak",
        }
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/hw_disconnect", dependencies=[Depends(_unlocked)])
def hw_disconnect():
    _replace_heimdall_transport(None)
    return {"status": "disconnected"}


@app.get("/hw_status")
def hw_status():
    """Return the active serial or emulator transport state."""
    with _heimdall_transport_lock:
        transport = _heimdall_transport
    if transport is None:
        return {"available": False, "connected": False, "transport": "disconnected"}
    snapshot = transport.snapshot()
    return {
        "available": True,
        "connected": snapshot["connected"],
        "transport": snapshot["transport"],
        "configuration": snapshot["configuration"],
        "scan_count": snapshot["scan_count"],
        "protocol_errors": snapshot["protocol_errors"],
        "settings": snapshot.get("settings") or {},
        "map_hash": snapshot.get("map_hash"),
        "tones": snapshot.get("tones") or {},
        "classes": snapshot.get("classes") or {},
        "direction_set": snapshot.get("direction_set"),
        "recording_meta": snapshot.get("recording_meta"),
        "bank": snapshot.get("bank"),
        "outputs": snapshot.get("outputs"),
        "ch1": snapshot.get("ch1"),
        "level_time_constant_ms": _deployed_program().level_time_constant_ms,
        "taper": snapshot.get("taper"),
        "dsp": snapshot.get("dsp"),
        "tables": snapshot.get("tables"),
    }


@app.get("/hw_ports")
def hw_ports():
    """List serial ports the operator can pick from (read-only)."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return {"ports": []}
    ports = []
    for port in sorted(list_ports.comports(), key=lambda item: item.device):
        ports.append({
            "device": port.device,
            "description": port.description or "",
            "hwid": port.hwid or "",
        })
    return {"ports": ports}


_DEPLOYMENT_CACHE = {}


def _deployment():
    """The beam table contract of the selected DSP export, read once; None if unusable."""
    if "contract" not in _DEPLOYMENT_CACHE:
        try:
            contract = json.loads(default_contract_path().read_text(encoding="ascii"))
        except (OSError, ValueError):
            contract = None
        _DEPLOYMENT_CACHE["contract"] = contract
    return _DEPLOYMENT_CACHE["contract"]


def _array_layout():
    """Microphone positions of the deployed array, from the contract.

    The firmware does not report its geometry, so the host supplies it for
    drawing. Returns None when the contract is missing or malformed.
    """
    try:
        data = _deployment()
        return {
            "units": "mm",
            "coordinate_system": data.get("coordinate_system"),
            "positions": [[float(value) for value in point] for point in data["microphone_positions_mm"]],
            "taper_sigma": float(data.get("taper_sigma", 0.0)),
            "mount": data.get("array_mount") or "upright",
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _deployed_program():
    """The program the emulators report, so they print what the firmware would."""
    try:
        return DeployedProgram.from_contract(_deployment())
    except (KeyError, TypeError, ValueError):
        return DeployedProgram()


# ── Study: DSP parameters (register map from the SigmaStudio export) ─────────

_dsp_service = DspParameterService()


class DspWrite(BaseModel):
    address: int
    word: int


class DspWriteRequest(BaseModel):
    writes: list[DspWrite]


class DspReadRequest(BaseModel):
    address: int
    count: int = 1


class DspSetRequest(BaseModel):
    key: str
    value: float


def _current_transport():
    with _heimdall_transport_lock:
        return _heimdall_transport


@app.get("/dsp/registry")
def dsp_registry():
    try:
        return _dsp_service.registry()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/dsp/status")
def dsp_status():
    return _dsp_service.status(_current_transport())


@app.post("/dsp/plan")
def dsp_plan(request: DspWriteRequest):
    try:
        return plan_writes([write.model_dump() for write in request.writes], _dsp_service.registry())
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _dsp_call(action):
    try:
        return action()
    except PermissionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=423, detail=str(error)) from error
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/dsp/memory")
def dsp_memory():
    return _dsp_call(lambda: {"words": {str(k): v for k, v in _dsp_service.memory(_current_transport()).items()}})


@app.post("/dsp/read")
def dsp_read(request: DspReadRequest):
    return _dsp_call(lambda: {"address": request.address,
                              "words": _dsp_service.read(_current_transport(), request.address, request.count)})


@app.post("/dsp/write", dependencies=[Depends(_unlocked)])
def dsp_write(request: DspWriteRequest):
    return _dsp_call(lambda: _dsp_service.write(_current_transport(), [w.model_dump() for w in request.writes]))


@app.post("/dsp/set", dependencies=[Depends(_unlocked)])
def dsp_set(request: DspSetRequest):
    return _dsp_call(lambda: {"settings": _dsp_service.set(_current_transport(), request.key, request.value)})


@app.post("/dsp/lost/forget")
def dsp_lost_forget():
    _dsp_service.forget_lost()
    return {"lost": []}


# ── Study: the characterisation tests, run by study_runner.py ────────────────

class StudyRunRequest(BaseModel):
    entry: str
    options: dict[str, Any] = {}
    configuration: str = "flashed"


class StudyAnswerRequest(BaseModel):
    text: str = ""


class StudyStopRequest(BaseModel):
    force: bool = False


def _idle(transport, seconds=3.0):
    if transport.snapshot().get("mode") == "IDLE":
        return True
    transport.send_command("X")
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if transport.snapshot().get("mode") == "IDLE":
            return True
        time.sleep(0.05)
    return False


def _study_edits(transport):
    """Writable DSP words that the Study changed (the backend's copy of what it wrote), with labels."""
    registry = _dsp_service.registry()
    memory = _dsp_service.memory(transport)
    edits = []
    for item in registry["parameters"]:
        if not item["writable"]:
            continue
        for offset, default in enumerate(item["default_words"]):
            address = item["address"] + offset
            word = memory.get(address, default)
            if word != default:
                label = f"{item['block']}.{item['name']}" + (f"[{offset}]" if item["words"] > 1 else "")
                edits.append({"address": address, "word": word, "default": default, "label": label})
    return edits


def _run_prepare(configuration):
    """Before a test: list the Study's edits and changed settings; a flashed run sets them to flashed."""
    transport = _current_transport()
    if transport is None:
        raise PermissionError("Connect the board on the Hardware screen first.")
    messages = []
    supported, reason = _dsp_service.supports(transport)
    edits = _study_edits(transport) if supported else []
    if not supported:
        messages.append(f"DSP edits not checked: {reason}")
    settings = dict(_dsp_service.status(transport).get("settings") or {})
    defaults = DspParameterService._default_settings()
    changed = {key: value for key, value in settings.items() if key in defaults and value != defaults[key]}
    if configuration == "flashed" and (edits or changed):
        if not _idle(transport):
            raise RuntimeError("The board did not stop. Click Stop on the Hardware screen, then run again.")
        if edits:
            _dsp_service.write(transport, [{"address": e["address"], "word": e["default"]} for e in edits])
            messages.append(f"Flashed run: {len(edits)} edited DSP words are at their flashed values during the test.")
        for key, value in setting_order(settings, {key: defaults[key] for key in changed}):
            _dsp_service.set(transport, key, value)
        if changed:
            messages.append("Flashed run: settings at their defaults during the test (" + ", ".join(
                f"{key} {value} to {defaults[key]}" for key, value in changed.items()) + ").")
    elif configuration == "edits":
        messages.append(f"Run with the current edits: {len(edits)} DSP words differ from the flashed program. "
                        "The test writes them again after each boot check (I).")
    return {"edits": edits, "settings": settings, "changed_settings": changed, "messages": messages}


def _run_finish(context):
    """After a test: put the Study's edits back on the board, and the settings after a flashed run."""
    edits = context.get("edits") or []
    changed = context.get("changed_settings") if context.get("configuration") == "flashed" else {}
    if not edits and not changed:
        return []
    transport = _current_transport()
    if transport is None or not _idle(transport):
        return ["The board is not connected or did not stop: the Study's edits and settings were not put back."]
    messages = []
    if edits:
        _dsp_service.write(transport, [{"address": e["address"], "word": e["word"]} for e in edits])
        messages.append(f"The Study's {len(edits)} edited DSP words are back on the board.")
    current = dict(_dsp_service.status(transport).get("settings") or {})
    for key, value in setting_order(current, changed or {}):
        _dsp_service.set(transport, key, value)
    if changed:
        messages.append("The settings are back: " + ", ".join(f"{key} {value}" for key, value in changed.items()) + ".")
    return messages


def _runner_call(action):
    try:
        return action()
    except (FileNotFoundError, KeyError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (RuntimeError, PermissionError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/study/tests")
def study_tests():
    return _runner_call(_study_runner.catalogue)


@app.get("/study/run")
def study_run_status(since: int = 0):
    return _study_runner.status(since)


@app.post("/study/run")
def study_run_start(request: StudyRunRequest):
    return _runner_call(lambda: _study_runner.start(request.entry, request.options, request.configuration,
                                                    prepare=_run_prepare, finish=_run_finish))


@app.post("/study/run/answer")
def study_run_answer(request: StudyAnswerRequest):
    return _runner_call(lambda: _study_runner.answer(request.text))


@app.post("/study/run/stop")
def study_run_stop(request: StudyStopRequest):
    return _runner_call(lambda: _study_runner.stop(request.force))


@app.get("/study/results")
def study_results():
    return {"results": _study_runner.results()}


@app.get("/study/results/item")
def study_result(path: str):
    return _runner_call(lambda: _study_runner.result(path))


@app.get("/study/results/file")
def study_result_file(path: str):
    return FileResponse(_runner_call(lambda: _study_runner.result_file(path)))


@app.post("/study/speaker_check")
def study_speaker_check():
    return _runner_call(_study_runner.speaker_check)


@app.get("/study/theory/cut")
def study_theory_cut(axis: str = "azimuth", steer_azimuth_deg: float = 0.0,
                     steer_elevation_deg: float = 0.0, frequency_hz: float = 2000.0,
                     minimum_deg: float = -60.0, maximum_deg: float = 60.0,
                     step_deg: float = 1.0):
    try:
        return theory_cut(
            axis=axis,
            steer_azimuth_deg=steer_azimuth_deg,
            steer_elevation_deg=steer_elevation_deg,
            frequency_hz=frequency_hz,
            minimum_deg=minimum_deg,
            maximum_deg=maximum_deg,
            step_deg=step_deg,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


RECORDING_SCHEMA = "heimdall-recording-v2"
DATASET_ENV = "HEIMDALL_DATASET_DIR"
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
INDEX_FIELDS = [
    "file", "started_at", "stopped_at", "duration_s", "channels", "channel_roles",
    "class", "source", "source_url", "licence", "clip_offset_s", "distance_m", "azimuth_deg", "elevation_deg",
    "level", "room", "session", "notes", "steer_azimuth_deg", "steer_elevation_deg", "map_hash", "dsp_edits",
    "dsp_put_back_by_I", "settings_changed", "listen_copy",
]
_dataset_lock = threading.Lock()


class RecordingLabel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    label_class: Literal["drone", "not_drone"] | None = Field(None, alias="class")
    source: str = ""
    source_url: str = ""
    licence: str = ""
    clip_offset_s: float | None = None
    distance_m: float | None = None
    azimuth_deg: float | None = None
    elevation_deg: float | None = None
    level: str = ""
    room: str = ""
    session: str = ""
    notes: str = ""


class ListenCopyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    suffix: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")
    preset: str
    kind: Literal["band", "bank"]
    low_hz: float | None = None
    high_hz: float | None = None
    design: str = ""
    anti_alias_taps: list[float] | None = None
    band_pass_taps: list[float] | None = None
    band_pass_factor: int | None = None


class RecordingSaveRequest(BaseModel):
    label: RecordingLabel = RecordingLabel()
    listen_copy: ListenCopyRecord | None = None


def _active_capture():
    with _heimdall_transport_lock:
        transport = _heimdall_transport
    if transport is None:
        raise HTTPException(status_code=404, detail="no hardware transport is active")
    capture = transport.get_audio_download()
    if capture is None:
        raise HTTPException(status_code=404, detail="no completed audio download is available")
    return transport, capture


def _wav_bytes(capture) -> bytes:
    if capture["bits_per_sample"] != 16 or capture["channels"] not in {1, 2}:
        raise HTTPException(status_code=422, detail="unsupported captured PCM format")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(capture["channels"])
        wav.setsampwidth(capture["bits_per_sample"] // 8)
        wav.setframerate(capture["sample_rate_hz"])
        wav.writeframes(capture["pcm"])
    return output.getvalue()


def _iso_time(unix_s):
    if unix_s is None:
        return None
    return datetime.datetime.fromtimestamp(unix_s).astimezone().isoformat(timespec="milliseconds")


def _git_state(path):
    """Commit and uncommitted tracked changes of one repository, or None."""
    try:
        commit = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=5, check=True).stdout.strip()
        changes = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                                 capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "dirty": bool(changes)}


def _dsp_record(transport):
    """The Study's DSP edits at save time, and the edits that a boot check (I) or a program load put back."""
    try:
        supported, reason = _dsp_service.supports(transport)
        if not supported:
            return {"checked": False, "reason": reason}
        registry = _dsp_service.registry()
        memory = _dsp_service.memory(transport)
        lost = _dsp_service.lost()
    except (FileNotFoundError, PermissionError, RuntimeError) as error:
        return {"checked": False, "reason": str(error)}
    words = {}
    for item in registry["parameters"]:
        if item["writable"]:
            for offset, flashed in enumerate(item["default_words"]):
                words[item["address"] + offset] = (item, offset, flashed)

    def entry(address, word):
        item, offset, flashed = words[address]
        return {"block": item["block"], "parameter": item["name"],
                "index": offset if item["words"] > 1 else None,
                "address": address, "word": word, "flashed": flashed}

    return {
        "checked": True,
        "export_sha256": registry["export"]["sha256"],
        "edits": [entry(address, memory[address]) for address in sorted(words)
                  if memory.get(address, words[address][2]) != words[address][2]],
        # Each put-back word ran as flashed; "word" is the edit that the board lost.
        "put_back_by_I": [entry(item["address"], item["word"]) for item in lost if item["address"] in words],
    }


def _recording_record(transport, capture):
    snapshot = transport.snapshot()
    channels = int(capture["channels"])
    bytes_per_frame = channels * int(capture["bits_per_sample"]) // 8
    frames = len(capture["pcm"]) // max(1, bytes_per_frame)
    recording_meta = snapshot.get("recording_meta") or {}
    roles = ["steered_beam"]
    if channels > 1:
        source = int(recording_meta.get("ch1_source", 0))
        index = int(recording_meta.get("ch1_index", 0))
        roles.append({0: "steered_beam", 1: f"microphone_{index}",
                      2: f"bank_beam_{index}"}.get(source, "unknown"))
    settings = snapshot.get("settings") or {}
    defaults = DspParameterService._default_settings()
    return {
        "schema": RECORDING_SCHEMA,
        "recording": {
            "started_at": _iso_time(capture.get("started_unix_s")),
            "stopped_at": _iso_time(capture.get("stopped_unix_s")),
            "downloaded_at": _iso_time(capture.get("downloaded_unix_s")),
            "clock": "host clock when the board reported the start and the stop",
        },
        "label": None,
        "files": None,
        "audio": {
            "sample_rate_hz": int(capture["sample_rate_hz"]),
            "bits_per_sample": int(capture["bits_per_sample"]),
            "channels": channels,
            "frames": frames,
            "duration_s": frames / int(capture["sample_rate_hz"]),
        },
        "channel_roles": roles,
        "steering": snapshot.get("last_steer"),
        "settings": settings,
        "settings_changed": {key: value for key, value in settings.items()
                             if key in defaults and value != defaults[key]},
        "dsp": _dsp_record(transport),
        "listen_copy": None,
        "transport": snapshot.get("transport"),
        "map_hash": snapshot.get("map_hash"),
        "configuration": snapshot.get("configuration"),
        "recording_meta": recording_meta,
        "direction_set": snapshot.get("direction_set"),
        "host": {"git": {"project": _git_state(PROJECT_DIR),
                         "acoustic_sim": _git_state(pathlib.Path(__file__).resolve().parent)}},
        "generated_at_unix_s": time.time(),
    }


def _dataset_dir() -> pathlib.Path:
    configured = os.environ.get(DATASET_ENV)
    return pathlib.Path(configured) if configured else PROJECT_DIR / "datasets" / "heimdall_recordings"


def _index_row(record) -> dict[str, Any]:
    label = record["label"] or {}
    steering = record["steering"] or {}
    dsp = record["dsp"]
    return {
        **{key: label.get(key) for key in INDEX_FIELDS if key in label},
        "file": record["files"]["wav"],
        "started_at": record["recording"]["started_at"],
        "stopped_at": record["recording"]["stopped_at"],
        "duration_s": round(record["audio"]["duration_s"], 3),
        "channels": record["audio"]["channels"],
        "channel_roles": ";".join(record["channel_roles"]),
        "steer_azimuth_deg": steering.get("azimuth_deg"),
        "steer_elevation_deg": steering.get("elevation_deg"),
        "map_hash": record["map_hash"],
        "dsp_edits": len(dsp["edits"]) if dsp.get("checked") else "not checked",
        "dsp_put_back_by_I": len(dsp["put_back_by_I"]) if dsp.get("checked") else "not checked",
        "settings_changed": json.dumps(record["settings_changed"], sort_keys=True),
        "listen_copy": (record["listen_copy"] or {}).get("file", ""),
    }


def _save_to_dataset(record, wav: bytes, root: pathlib.Path) -> None:
    """Write the WAV and the record; one index row per WAV (a second save of the same capture replaces it)."""
    folder = root / record["files"]["wav"].split("/")[0]
    folder.mkdir(parents=True, exist_ok=True)
    (root / record["files"]["wav"]).write_bytes(wav)
    (root / record["files"]["json"]).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    index = root / "index.csv"
    rows = []
    if index.exists():
        with index.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("file") != record["files"]["wav"]]
    rows.append(_index_row(record))
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@app.get("/hw_recording.wav")
def hw_recording_wav():
    _, capture = _active_capture()
    return Response(
        content=_wav_bytes(capture),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=heimdall-recording.wav"},
    )


@app.get("/hw_recording_metadata.json")
def hw_recording_metadata():
    transport, capture = _active_capture()
    return _recording_record(transport, capture)


@app.post("/hw_recording/save")
def hw_recording_save(request: RecordingSaveRequest):
    """Label the last download and save it with its record into the dataset folder; returns the record."""
    transport, capture = _active_capture()
    wav = _wav_bytes(capture)
    record = _recording_record(transport, capture)
    stamp_s = capture.get("started_unix_s") or capture.get("downloaded_unix_s") or time.time()
    local = datetime.datetime.fromtimestamp(stamp_s).astimezone()
    stem = f"heimdall-{local.strftime('%Y-%m-%dT%H-%M-%S')}"
    day = local.strftime("%Y-%m-%d")
    record["label"] = request.label.model_dump(by_alias=True)
    record["files"] = {"wav": f"{day}/{stem}.wav", "json": f"{day}/{stem}.json"}
    if request.listen_copy is not None:
        copy = request.listen_copy.model_dump(exclude_none=True)
        record["listen_copy"] = {"file": f"{stem}-{copy.pop('suffix')}.wav", "saved_in_dataset": False, **copy}
    root = _dataset_dir()
    try:
        with _dataset_lock:
            _save_to_dataset(record, wav, root)
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"Could not save into the dataset folder {root}: {error}") from error
    return {**record, "dataset_folder": str(root)}


@app.websocket("/realtime_hw")
async def realtime_hw_ws(ws: WebSocket):
    """Stream parsed Heimdall state and forward commands to the active transport."""
    global _hw_true_az_deg, _hw_true_el_deg

    with _heimdall_transport_lock:
        transport = _heimdall_transport
    if transport is None:
        await ws.close(code=4001, reason="No serial or emulator transport configured.")
        return

    await ws.accept()
    await ws.send_json(_hardware_payload("init", transport.snapshot()))
    run_token = ws.query_params.get("run")
    paused = False
    last_state_key = None

    async def push_loop():
        nonlocal last_state_key, paused
        while True:
            if paused:
                await asyncio.sleep(0.05)
                continue
            snapshot = transport.snapshot()
            truth_frame = snapshot.get("emulator_truth", {}).get("frame")
            truth_tick = truth_frame // 10 if isinstance(truth_frame, int) else None
            expected_audio = snapshot["audio_download_expected"]
            audio_progress_percent = (
                0 if expected_audio == 0
                else snapshot["audio_download_received"] * 100 // expected_audio
            )
            state_key = (snapshot["sequence"], truth_tick, audio_progress_percent, _study_runner.revision)
            if state_key == last_state_key:
                await asyncio.sleep(0.010)
                continue
            last_state_key = state_key
            try:
                await ws.send_json(_hardware_payload("frame", snapshot))
            except (WebSocketDisconnect, RuntimeError):
                return
            await asyncio.sleep(0.005)

    loop_task = asyncio.create_task(push_loop())

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            mtype = msg.get("type")
            if mtype == "pause":
                paused = True
            elif mtype == "resume":
                paused = False
            elif mtype == "command":
                if not _study_runner.allows(run_token):
                    await ws.send_json({"type": "command_error", "detail": _study_runner.lock_text()})
                    continue
                try:
                    transport.send_command(str(msg.get("command", "")))
                except (OSError, ValueError) as error:
                    await ws.send_json({"type": "command_error", "detail": str(error)})
            elif mtype == "set_true_dir":
                _hw_true_az_deg = float(msg["az_deg"]) if msg.get("az_deg") is not None else None
                _hw_true_el_deg = float(msg["el_deg"]) if msg.get("el_deg") is not None else None
            elif mtype == "request_init":
                if _study_runner.allows(run_token):
                    transport.send_command("I")
                await ws.send_json(_hardware_payload("init", transport.snapshot()))
    except WebSocketDisconnect:
        pass
    finally:
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass


def _bank_with_directions(bank):
    """The firmware reports bank levels, its target and its pattern fit; the beam directions come from the contract."""
    if bank is None:
        return None
    program = _deployed_program()
    if len(program.bank_azimuth_deg) != bank["rows"] * bank["columns"]:
        return bank
    return {**bank, "azimuth_deg": list(program.bank_azimuth_deg),
            "elevation_deg": list(program.bank_elevation_deg)}


def _bank_fit():
    """The deployed bank's pattern fit, for the emulators (the board runs its own); None without mic delays."""
    if "bank_fit" not in _DEPLOYMENT_CACHE:
        try:
            _DEPLOYMENT_CACHE["bank_fit"] = BankFit.from_contract(_deployment())
        except (KeyError, TypeError, ValueError):
            _DEPLOYMENT_CACHE["bank_fit"] = None
    return _DEPLOYMENT_CACHE["bank_fit"]


def _hardware_payload(payload_type, snapshot):
    configuration = snapshot["configuration"]
    candidates = []
    if configuration is not None:
        for row in range(configuration["rows"]):
            for column in range(configuration["columns"]):
                level_db = snapshot["levels_db"][row][column]
                if level_db is not None:
                    candidates.append((level_db, row, column))
    candidates.sort(reverse=True)

    target = snapshot["target"]
    last_record = snapshot["last_record"]
    last_steer = snapshot["last_steer"]
    if target is not None:
        estimated_azimuth = target["azimuth_deg"]
        estimated_elevation = target["elevation_deg"]
    elif last_steer is not None:
        estimated_azimuth = last_steer["azimuth_deg"]
        estimated_elevation = last_steer["elevation_deg"]
    elif candidates and snapshot["azimuth_deg"] and snapshot["elevation_deg"]:
        _, row, column = candidates[0]
        estimated_azimuth = snapshot["azimuth_deg"][column]
        estimated_elevation = snapshot["elevation_deg"][row]
    else:
        estimated_azimuth = None
        estimated_elevation = None

    margin_db = candidates[0][0] - candidates[1][0] if len(candidates) > 1 else 0.0
    timing = snapshot["last_timing"]
    scan_rate_hz = None
    if (timing is not None and timing.get("wall_us", 0) > 0 and configuration is not None
            and timing.get("sector_count") == configuration["sectors"]):
        scan_rate_hz = 1_000_000.0 / timing["wall_us"]
    transport_name = snapshot["transport"]
    grid_source = (snapshot.get("acoustic_grid_mode", "deployment_contract")
                   if transport_name == "acoustic-emulator"
                   else "fast_emulator" if transport_name == "emulator"
                   else "firmware_reported")
    extra = {"array_layout": _array_layout()} if payload_type == "init" else {}
    return {
        **extra,
        "type": payload_type,
        "sequence": snapshot["sequence"],
        "timestamp_s": time.time(),
        "mode": "hardware",
        "beam_method": "hardware_sector_scan",
        "transport": snapshot["transport"],
        "study_run": _study_runner.brief(),
        "grid_source": grid_source,
        "connected": snapshot["connected"],
        "configuration": configuration,
        "azimuth_deg": snapshot["azimuth_deg"],
        "elevation_deg": snapshot["elevation_deg"],
        "levels_raw": snapshot["levels_raw"],
        "levels_db": snapshot["levels_db"],
        "scan_count": snapshot["scan_count"],
        "last_measurement": snapshot.get("last_measurement"),
        "firmware_mode": snapshot["mode"],
        "target": target,
        "last_steer": last_steer,
        "last_timing": timing,
        "last_error": snapshot["last_error"],
        "last_record": last_record,
        "audio_format": snapshot["audio_format"],
        "recording_state": snapshot["recording_state"],
        "last_recording": snapshot["last_recording"],
        "audio_download_expected": snapshot["audio_download_expected"],
        "audio_download_received": snapshot["audio_download_received"],
        "audio_download_progress": snapshot["audio_download_progress"],
        "audio_download_ready": snapshot["audio_download_ready"],
        "audio_download_error": snapshot["audio_download_error"],
        "protocol_errors": snapshot["protocol_errors"],
        "settings": snapshot.get("settings") or {},
        "map_hash": snapshot.get("map_hash"),
        "tones": snapshot.get("tones") or {},
        "classes": snapshot.get("classes") or {},
        "direction_set": snapshot.get("direction_set"),
        "recording_meta": snapshot.get("recording_meta"),
        "bank": _bank_with_directions(snapshot.get("bank")),
        "outputs": snapshot.get("outputs"),
        "array_mount": snapshot.get("array_mount"),
        "ch1": snapshot.get("ch1"),
        "level_time_constant_ms": _deployed_program().level_time_constant_ms,
        "taper": snapshot.get("taper"),
        "dsp": snapshot.get("dsp"),
        "tables": snapshot.get("tables"),
        "est_az_deg": estimated_azimuth,
        "est_el_deg": estimated_elevation,
        "true_az_deg": _hw_true_az_deg,
        "true_el_deg": _hw_true_el_deg,
        "margin_db": margin_db,
        "argmax_db": candidates[0][0] if candidates else None,
        "scan_rate_hz": scan_rate_hz,
        "target_update_rate_hz": snapshot["target_update_rate_hz"],
        "target_update_age_s": snapshot["target_update_age_s"],
        "emulator_flight_profile": snapshot.get("emulator_flight_profile"),
        "emulator_flight_speed": snapshot.get("emulator_flight_speed"),
        "acoustic_cache_id": snapshot.get("acoustic_cache_id"),
        "acoustic_scenario": snapshot.get("acoustic_scenario"),
        "acoustic_calibrated": snapshot.get("acoustic_calibrated"),
        "acoustic_model": snapshot.get("acoustic_model"),
        "acoustic_measured_rt60_s": snapshot.get("acoustic_measured_rt60_s"),
        "acoustic_preparation_seconds": snapshot.get("acoustic_preparation_seconds"),
        "acoustic_detector_filter": snapshot.get("acoustic_detector_filter"),
        "acoustic_exported_fir_status": snapshot.get("acoustic_exported_fir_status"),
        "acoustic_grid": snapshot.get("acoustic_grid"),
        "acoustic_grid_mode": snapshot.get("acoustic_grid_mode"),
        "acoustic_room_cache_id": snapshot.get("acoustic_room_cache_id"),
        "emulator_truth": snapshot.get("emulator_truth"),
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Acoustic beamformer sim server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--serial-port", type=str, default=None,
                        help="COM port for MAX78002 hardware (e.g. COM3). Enables /realtime_hw.")
    parser.add_argument("--serial-baud", type=int, default=921600)
    parser.add_argument("--emulator", action="store_true",
                        help="Start the Heimdall firmware emulator")
    args = parser.parse_args()

    if args.emulator:
        _replace_heimdall_transport(EmulatorHeimdallTransport())
        print("Heimdall firmware emulator started")
    elif args.serial_port:
        try:
            _replace_heimdall_transport(
                SerialHeimdallTransport(args.serial_port, args.serial_baud)
            )
            print(f"Hardware serial reader started on {args.serial_port} @ {args.serial_baud} baud")
        except Exception as e:
            print(f"Failed to open serial port {args.serial_port}: {e}")
            _replace_heimdall_transport(None)

    uvicorn.run(app, host=args.host, port=args.port, reload=False)
