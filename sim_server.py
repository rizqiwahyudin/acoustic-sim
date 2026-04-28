"""
FastAPI backend for live acoustic beamforming simulation.

Runs a single-trial pyroomacoustics simulation on demand and returns
the SRP-PHAT spatial power spectrum + beamformed audio.

Usage:
    python sim_server.py              # starts on port 8766
"""

import base64
import io
import pathlib
import time

import numpy as np
import pyroomacoustics as pra
import scipy.io.wavfile as wavfile
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    load_crosstalk_fir,
    log_mel_features,
    make_trajectory,
    measure_rt60_from_rir,
    ml_path_quantize_audio,
    ml_path_quantize_features,
    ml_path_snr_db,
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
# FMIN / FMAX are retained as legacy defaults for any caller that imports
# them directly; the live path now reads req.fmin_hz / req.fmax_hz instead.
FMIN = DEFAULT_FMIN
FMAX = DEFAULT_FMAX
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


def build_array(geometry, center, mic_count, radius, ring_separation):
    if geometry == "UCA":
        return make_uca(center, mic_count, radius)
    elif geometry == "CROSS":
        return make_cross(center, mic_count, radius)
    elif geometry == "ULA":
        return make_ula(center, mic_count, radius * 2)
    elif geometry == "CYLINDER":
        return make_cylinder(center, mic_count, radius, ring_separation)
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

app = FastAPI(title="Acoustic Beamformer Simulation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimRequest(BaseModel):
    geometry: str = "UCA"
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

    # Phase 3A: MAX78000 fixed-point ML-path preview.
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

    array_R = build_array(req.geometry, array_center, req.mic_count,
                          req.radius, req.ring_separation)

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

    X = np.array([
        pra.transform.stft.analysis(sig, NFFT, HOP).T
        for sig in signals
    ])

    freq_bins = build_freq_bin_mask(
        FS, NFFT,
        fmin_hz=req.fmin_hz,
        fmax_hz=req.fmax_hz,
        harmonic_comb=req.harmonic_comb,
        f0_hz=req.drone_fundamental_hz,
    )

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

    n_out = signals.shape[1]
    raw_audio = signals[0, :n_out].copy()
    unsteered_audio = signals[:, :n_out].mean(axis=0)
    bf_audio = delay_and_sum(signals, array_R, est_az_rad,
                             np.deg2rad(est_el_deg), FS)

    raw_b64, unsteered_b64, bf_b64 = encode_three_wavs_joint(
        [raw_audio, unsteered_audio, bf_audio], FS,
        normalize=req.normalize_audio,
    )

    # Phase 3A: MAX78000 ML-path preview -- quantize the beamformed audio
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
        "est_az_deg": round(est_az_deg, 1),
        "est_el_deg": round(est_el_deg, 1),
        "true_az_deg": req.source_az_deg,
        "true_el_deg": req.source_el_deg,
        "mic_positions": mic_rel,
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


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    load_audio()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sim_server:app", host="127.0.0.1", port=8766, reload=False)
