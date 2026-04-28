#!/usr/bin/env python
"""
3D DOA Array Geometry Comparison -- Exhibition Hall Simulation

Compares UCA, Standing Cross, ULA, and Cylinder microphone array geometries
for direction-of-arrival estimation (azimuth + elevation) under varying
reverberation, drone SPL, and diffuse noise conditions in a 3D exhibition hall.

Source levels are specified in dB SPL at 1 m (referenced to 20 uPa) via the
shared acoustic_utils module, matching sim_server.py so batch and live
simulators use an identical physical model.

Usage:
    python run_comparison.py              # full sweep (~1816 trials, ~3.5-4 hrs)
    python run_comparison.py --test       # quick validation (~160 trials, ~5-10 min)
    python run_comparison.py --plots-only # regenerate plots from existing CSV
"""

import argparse
import csv
import json
import pathlib
import time

import numpy as np
import pyroomacoustics as pra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from acoustic_utils import (
    air_absorption_kwargs,
    apply_crosstalk,
    apply_crosstalk_fir,
    atmospheric_z_bias,
    build_freq_bin_mask,
    build_materials,
    crowd_positions_mixed,
    feature_snr_db as _feature_snr_db,
    load_crosstalk_fir,
    log_mel_features,
    measure_rt60_from_rir,
    ml_path_quantize_audio,
    ml_path_quantize_features,
    ml_path_snr_db as _ml_path_snr_db,
    spl_to_amplitude,
    synthesize_diffuse_crowd_plane_waves,
    wall_adjacent_positions,
    CROSSTALK_COUPLING_DB_DEFAULT,
    CROWD_SPL_DB,
    DEFAULT_FMAX,
    DEFAULT_FMIN,
    EXHIBITION_HALL_MATERIALS,
    MIC_NOISE_FLOOR_DB,
    ML_DEFAULT_BIT_DEPTH,
    ML_DEFAULT_FEATURE_BIT_DEPTH,
    ML_DEFAULT_N_MELS,
    PA_SPL_DB,
)

FS = 16_000
NFFT = 1024
HOP = 512
# Legacy module-level defaults retained for any caller that imports them
# directly; the batch path now reads FMIN_HZ / FMAX_HZ globals (set via CLI)
# through build_freq_bin_mask.
FMIN = DEFAULT_FMIN
FMAX = DEFAULT_FMAX
C = 343.0

SIGNAL_SECONDS = 1.0
WARMUP_SAMPLES = int(0.1 * FS)  # 100 ms warm-up trimmed after room.simulate()
ROOM_DIM = np.array([20.0, 15.0, 10.0])
ARRAY_CENTER = np.array([10.0, 7.5, 1.0])
SOURCE_DISTANCE = 4.0
MAX_ORDER_CAP = 10
MARGIN = 0.3

GEOMETRIES = ["UCA", "CROSS", "ULA", "CYLINDER"]
GEO_COLORS = {"UCA": "C0", "CROSS": "C1", "ULA": "C2", "CYLINDER": "C3"}
GEO_MARKERS = {"UCA": "o", "CROSS": "^", "ULA": "s", "CYLINDER": "D"}

RT60S_FULL = [0.0, 0.5, 0.8, 1.0, 1.5]
DRONE_SPLS_FULL = [85.0, 80.0, 78.0, 75.0, 70.0]
DEFAULT_EL = 30.0
DEFAULT_AZ = 60.0
DEFAULT_DRONE_SPL = 78.0
DEFAULT_RT60_REV = 1.0
ELEVATIONS_SWEEP = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
AZIMUTHS_FULL = np.arange(0, 360, 10, dtype=float)
SEEDS_FULL = [0, 1]

SRP_AZ = np.linspace(0, 2 * np.pi, 72, endpoint=False)
SRP_COLAT = np.linspace(0, np.pi, 19)

OUT = pathlib.Path("results")
CSV_PATH = OUT / "metrics.csv"
SPECTRA_PATH = OUT / "spectra.json"
CSV_FIELDS = [
    "geometry", "rt60", "drone_spl_db", "diffuse",
    "true_az_deg", "true_el_deg", "est_az_deg", "est_el_deg",
    "az_error_deg", "el_error_deg", "total_error_deg", "seed",
    # Phase 3: populated when --ml-preview is passed; empty otherwise.
    "ml_path_snr_db", "feature_snr_db",
]

# Materials-profile mode -- set by --materials-profile on the CLI.
# When not None, run_single_trial uses per-wall materials instead of
# inverse_sabine, CSV_PATH and SPECTRA_PATH are redirected to
# *_materials.csv / *_materials.json, and plot output filenames are
# suffixed with "_materials".
MATERIALS_PROFILE = None
MATERIALS_PROFILE_RT60 = None  # measured RT60 of the profile (populated in main)
PLOT_SUFFIX = ""

# Atmosphere (Phase 2b) -- set by --temperature / --humidity / --temp-gradient
# CLI flags. Defaults match the pra air-absorption defaults used in earlier
# phases so existing sweeps are unaffected.
#
# NOTE: moving source is deliberately *not* added to the batch sweep. The
# live simulator exposes it for what-if exploration; the batch path ranks
# geometries under static-source conditions so runs stay apples-to-apples.
TEMPERATURE_C = 20.0
HUMIDITY_PCT = 50.0
TEMP_GRADIENT_C_PER_M = 0.0

# Phase 3 batch knobs -- all default to the Phase 2b baseline so unmodified
# sweeps stay apples-to-apples. Overridden by the --ml-preview /
# --crowd-model / --crosstalk-model CLI flags in main().
CROWD_MODEL = "point_source"            # "point_source" | "plane_wave"
N_PLANE_WAVES = 64
CROSSTALK_MODEL = "simple"              # "simple" | "fir_capacitive"
CROSSTALK_CORNER_HZ = 500.0
CROSSTALK_COUPLING_DB = CROSSTALK_COUPLING_DB_DEFAULT
CROSSTALK_FIR_PATH = ""
ML_PREVIEW = False
ML_BIT_DEPTH = ML_DEFAULT_BIT_DEPTH
ML_FEATURE_BIT_DEPTH = ML_DEFAULT_FEATURE_BIT_DEPTH
ML_N_MELS = ML_DEFAULT_N_MELS

# Phase 3+ DOA-band tuning. Defaults match DEFAULT_FMIN / DEFAULT_FMAX so
# unchanged sweeps stay byte-compatible; overridden by --fmin / --fmax /
# --harmonic-comb / --drone-fundamental in main().
FMIN_HZ = DEFAULT_FMIN
FMAX_HZ = DEFAULT_FMAX
HARMONIC_COMB = False
DRONE_FUNDAMENTAL_HZ = 200.0


# ─── Helper functions ─────────────────────────────────────────────────────────

def drone_like_signal(fs, seconds=SIGNAL_SECONDS, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = int(fs * seconds)
    t = np.arange(n) / fs
    sig = np.zeros(n)
    for f0 in [150, 300, 600, 900]:
        sig += np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
    sig += 0.3 * rng.standard_normal(n)
    return sig / np.max(np.abs(sig))


def wrap_angle_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def angular_distance_deg(az1, el1, az2, el2):
    a1, e1 = np.deg2rad(az1), np.deg2rad(el1)
    a2, e2 = np.deg2rad(az2), np.deg2rad(el2)
    v1 = np.array([np.cos(e1) * np.cos(a1), np.cos(e1) * np.sin(a1), np.sin(e1)])
    v2 = np.array([np.cos(e2) * np.cos(a2), np.cos(e2) * np.sin(a2), np.sin(e2)])
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(dot)))


def drone_position(center, az_deg, el_deg, distance, room_dim, margin=MARGIN):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    src = np.array([
        center[0] + distance * np.cos(el) * np.cos(az),
        center[1] + distance * np.cos(el) * np.sin(az),
        center[2] + distance * np.sin(el),
    ])
    return np.clip(src, margin, room_dim - margin)


# ─── Geometry builders ────────────────────────────────────────────────────────

def make_uca(center, mics=12, radius=0.15):
    xy = pra.circular_2D_array(center[:2], mics, 0.0, radius)
    z_row = np.full((1, mics), center[2])
    return np.vstack([xy, z_row])


def make_cross(center, mics_per_arm=6, half_length=0.15):
    offsets = np.linspace(-half_length, half_length, mics_per_arm + 1)
    offsets = offsets[np.abs(offsets) > 1e-9]
    pts = []
    for o in offsets:
        pts.append([center[0] + o, center[1], center[2]])
    for o in offsets:
        pts.append([center[0], center[1], center[2] + o])
    return np.array(pts).T


def make_ula(center, mics=12, length=0.30):
    offsets = np.linspace(-length / 2, length / 2, mics)
    R = np.zeros((3, mics))
    R[0, :] = center[0] + offsets
    R[1, :] = center[1]
    R[2, :] = center[2]
    return R


def make_cylinder(center, mics_per_ring=6, radius=0.15, separation=0.12):
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


def build_geometry(name, center):
    if name == "UCA":
        return make_uca(center)
    elif name == "CROSS":
        return make_cross(center)
    elif name == "ULA":
        return make_ula(center)
    elif name == "CYLINDER":
        return make_cylinder(center)
    raise ValueError(f"Unknown geometry: {name}")


# ─── Diffuse source builder ──────────────────────────────────────────────────

def make_diffuse_sources_3d(room_dim, fs, seconds, rng, array_center):
    """Build (position, signal) pairs for the diffuse noise bed.

    Uses the shared placement helpers so batch and live simulators agree,
    and scales synthetic noise by dB SPL at 1 m so the mic sees physically
    meaningful Pascal-unit pressure.
    """
    sources = []
    n_samples = int(fs * seconds) + WARMUP_SAMPLES
    for pos in crowd_positions_mixed(room_dim, 12, z_height=1.2,
                                      array_center=array_center, rng=rng):
        sig = rng.standard_normal(n_samples) * spl_to_amplitude(CROWD_SPL_DB)
        sources.append((np.array(pos), sig))
    for pos in wall_adjacent_positions(room_dim, 4, z_height=room_dim[2] - 1.0,
                                        rng=rng):
        sig = rng.standard_normal(n_samples) * spl_to_amplitude(PA_SPL_DB)
        sources.append((np.array(pos, dtype=float), sig))
    return sources


# ─── Single trial ─────────────────────────────────────────────────────────────

def run_single_trial(array_R, true_az_deg, true_el_deg, rt60, drone_spl_db,
                     room_dim, diffuse=False, seed=0,
                     materials_profile=None,
                     temperature_c=None, humidity_pct=None,
                     temp_gradient_c_per_m=None,
                     crowd_model=None, n_plane_waves=None,
                     crosstalk_model=None, crosstalk_corner_hz=None,
                     crosstalk_coupling_db=None, crosstalk_fir_path=None,
                     ml_preview=None, ml_bit_depth=None,
                     ml_feature_bit_depth=None, ml_n_mels=None,
                     fmin_hz=None, fmax_hz=None,
                     harmonic_comb=None, drone_fundamental_hz=None):
    """Run one static-source batch trial.

    Atmosphere parameters default to the module-level ``TEMPERATURE_C``,
    ``HUMIDITY_PCT``, ``TEMP_GRADIENT_C_PER_M`` globals (set by the
    ``--temperature`` / ``--humidity`` / ``--temp-gradient`` CLI flags in
    ``main``). Phase 3 parameters (crowd model, crosstalk model, ML preview)
    similarly default to module-level globals set by CLI flags so unchanged
    sweeps produce unchanged CSVs.
    """
    temperature_c = (TEMPERATURE_C if temperature_c is None else float(temperature_c))
    humidity_pct = (HUMIDITY_PCT if humidity_pct is None else float(humidity_pct))
    temp_gradient_c_per_m = (TEMP_GRADIENT_C_PER_M if temp_gradient_c_per_m is None
                             else float(temp_gradient_c_per_m))

    crowd_model = (CROWD_MODEL if crowd_model is None else str(crowd_model))
    n_plane_waves = (N_PLANE_WAVES if n_plane_waves is None else int(n_plane_waves))
    crosstalk_model = (CROSSTALK_MODEL if crosstalk_model is None
                       else str(crosstalk_model))
    crosstalk_corner_hz = (CROSSTALK_CORNER_HZ if crosstalk_corner_hz is None
                           else float(crosstalk_corner_hz))
    crosstalk_coupling_db = (CROSSTALK_COUPLING_DB if crosstalk_coupling_db is None
                             else float(crosstalk_coupling_db))
    crosstalk_fir_path = (CROSSTALK_FIR_PATH if crosstalk_fir_path is None
                          else str(crosstalk_fir_path))
    ml_preview = (ML_PREVIEW if ml_preview is None else bool(ml_preview))
    ml_bit_depth = (ML_BIT_DEPTH if ml_bit_depth is None else int(ml_bit_depth))
    ml_feature_bit_depth = (ML_FEATURE_BIT_DEPTH if ml_feature_bit_depth is None
                            else int(ml_feature_bit_depth))
    ml_n_mels = (ML_N_MELS if ml_n_mels is None else int(ml_n_mels))

    fmin_hz = (FMIN_HZ if fmin_hz is None else float(fmin_hz))
    fmax_hz = (FMAX_HZ if fmax_hz is None else float(fmax_hz))
    harmonic_comb = (HARMONIC_COMB if harmonic_comb is None else bool(harmonic_comb))
    drone_fundamental_hz = (DRONE_FUNDAMENTAL_HZ if drone_fundamental_hz is None
                            else float(drone_fundamental_hz))

    rng = np.random.default_rng(seed)

    sigma2 = spl_to_amplitude(MIC_NOISE_FLOOR_DB) ** 2
    air_kw = air_absorption_kwargs(temperature_c, humidity_pct)

    if materials_profile == "exhibition_hall":
        mats = build_materials(**EXHIBITION_HALL_MATERIALS)
        room = pra.ShoeBox(
            room_dim, fs=FS, sigma2_awgn=sigma2,
            materials=mats, max_order=min(6, MAX_ORDER_CAP),
            **air_kw,
        )
    elif rt60 <= 0.0:
        room = pra.AnechoicRoom(3, fs=FS, sigma2_awgn=sigma2)
    else:
        try:
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
        except ValueError:
            return None, None, None, None, None, None, None
        max_order = min(max_order, MAX_ORDER_CAP)
        room = pra.ShoeBox(
            room_dim, fs=FS, sigma2_awgn=sigma2,
            materials=pra.Material(e_absorption),
            max_order=max_order,
            **air_kw,
        )

    src_pos = drone_position(ARRAY_CENTER, true_az_deg, true_el_deg,
                             SOURCE_DISTANCE, room_dim)
    # Apply first-order gradient beam-bending by shifting the source
    # position in z; pyroomacoustics then produces a RIR that reflects
    # the bent arrival. At dT/dz = 0 this is a no-op.
    src_pos = atmospheric_z_bias(src_pos, ARRAY_CENTER, temp_gradient_c_per_m)
    src_pos = np.clip(src_pos, MARGIN, np.asarray(room_dim) - MARGIN)
    # Include the warm-up pad in the source signal so we can trim 100 ms
    # of RIR onset artefacts from the mic signals without eating into the
    # 1-second integration window.
    n_src = int(FS * SIGNAL_SECONDS) + WARMUP_SAMPLES
    drone_sig = drone_like_signal(FS, n_src / FS, rng) * spl_to_amplitude(drone_spl_db)
    room.add_source(src_pos, signal=drone_sig)

    # Phase 3B: optional plane-wave crowd replaces the point-source crowd
    # component of the diffuse bed (PA sources stay as point sources since
    # they are localised and directional).
    plane_wave_crowd_sources = None
    if diffuse:
        if crowd_model == "plane_wave":
            # PA as point sources; crowd deferred to plane-wave synthesis
            # after simulate(). We reuse make_diffuse_sources_3d's PA-tail
            # placement by only pulling the PA half here.
            n_samples_sim = int(FS * SIGNAL_SECONDS) + WARMUP_SAMPLES
            for pos in wall_adjacent_positions(room_dim, 4,
                                               z_height=room_dim[2] - 1.0,
                                               rng=rng):
                sig = rng.standard_normal(n_samples_sim) * spl_to_amplitude(PA_SPL_DB)
                room.add_source(np.array(pos, dtype=float), signal=sig)
            n_plane_src = max(int(n_plane_waves), 12)
            plane_wave_crowd_sources = [
                rng.standard_normal(int(FS * SIGNAL_SECONDS))
                * spl_to_amplitude(CROWD_SPL_DB)
                for _ in range(n_plane_src)
            ]
        else:
            for pos, sig in make_diffuse_sources_3d(room_dim, FS, SIGNAL_SECONDS,
                                                     rng, ARRAY_CENTER):
                room.add_source(pos, signal=sig)

    room.add_microphone_array(pra.MicrophoneArray(array_R, fs=FS))
    room.simulate()

    mic_signals = room.mic_array.signals[:, WARMUP_SAMPLES:]

    # Phase 3B: add plane-wave crowd field directly to mic signals.
    if plane_wave_crowd_sources is not None:
        diffuse_pw = synthesize_diffuse_crowd_plane_waves(
            array_R,
            duration_s=float(mic_signals.shape[1]) / FS,
            fs=FS,
            n_planes=int(n_plane_waves),
            source_signals=plane_wave_crowd_sources,
            rng=rng,
        )
        n = min(mic_signals.shape[1], diffuse_pw.shape[1])
        mic_signals = mic_signals.copy()
        mic_signals[:, :n] = mic_signals[:, :n] + diffuse_pw[:, :n]

    # Phase 3C: optional FIR crosstalk. Only applied when crosstalk_model
    # is set to "fir_capacitive"; the legacy batch sweep (crosstalk_model
    # == "simple" and CROSSTALK_COUPLING_DB default not forced-on) leaves
    # signals untouched.
    if crosstalk_model == "fir_capacitive":
        measured = load_crosstalk_fir(crosstalk_fir_path)
        mic_signals = apply_crosstalk_fir(
            mic_signals, FS,
            coupling_db=crosstalk_coupling_db,
            corner_hz=crosstalk_corner_hz,
            measured_fir=measured,
        )

    X = np.array([
        pra.transform.stft.analysis(sig, NFFT, HOP).T
        for sig in mic_signals
    ])

    freq_bins = build_freq_bin_mask(
        FS, NFFT,
        fmin_hz=fmin_hz, fmax_hz=fmax_hz,
        harmonic_comb=harmonic_comb,
        f0_hz=drone_fundamental_hz,
    )

    doa = pra.doa.SRP(
        array_R, FS, NFFT, c=C, num_src=1, dim=3,
        azimuth=SRP_AZ, colatitude=SRP_COLAT,
    )
    doa.locate_sources(X, freq_bins=freq_bins)

    est_az_rad = float(np.atleast_1d(doa.azimuth_recon)[0])
    est_colat_rad = float(np.atleast_1d(doa.colatitude_recon)[0])
    est_az_deg = np.rad2deg(est_az_rad)
    est_el_deg = 90.0 - np.rad2deg(est_colat_rad)

    grid_az = np.array(doa.grid.azimuth, copy=True)
    grid_colat = np.array(doa.grid.colatitude, copy=True)
    grid_vals = np.array(doa.grid.values, copy=True)

    # Phase 3A: optional MAX78000 ML-path preview metrics. We beamform
    # toward the estimated DOA, quantize to int8/int16, and report the
    # SNR hit on audio + log-mel features. Cheap enough to enable for a
    # single-environment sweep (<5% overhead per trial).
    ml_snr = None
    feat_snr = None
    if ml_preview:
        el_rad = np.deg2rad(est_el_deg)
        colat = np.pi / 2 - el_rad
        d = np.array([
            np.sin(colat) * np.cos(est_az_rad),
            np.sin(colat) * np.sin(est_az_rad),
            np.cos(colat),
        ])
        center = array_R.mean(axis=1)
        delays = (array_R - center[:, None]).T @ d / C
        delays -= delays.min()
        n_out = mic_signals.shape[1]
        bf_audio = np.zeros(n_out, dtype=np.float64)
        for m in range(mic_signals.shape[0]):
            shift = int(round(delays[m] * FS))
            if shift >= n_out:
                continue
            bf_audio[shift:] += mic_signals[m, :n_out - shift]
        bf_audio /= mic_signals.shape[0]

        q_audio = ml_path_quantize_audio(bf_audio, bit_depth=ml_bit_depth)
        ml_snr = float(round(_ml_path_snr_db(bf_audio, q_audio), 2))

        mel_ref = log_mel_features(bf_audio, fs=FS, n_mels=ml_n_mels)
        mel_q = log_mel_features(q_audio, fs=FS, n_mels=ml_n_mels)
        mel_q = ml_path_quantize_features(mel_q, bit_depth=ml_feature_bit_depth)
        feat_snr = float(round(_feature_snr_db(mel_ref, mel_q), 2))

    return est_az_deg, est_el_deg, grid_az, grid_colat, grid_vals, ml_snr, feat_snr


# ─── Trial list builder ──────────────────────────────────────────────────────

def build_trial_list(test_mode=False, materials_profile=None,
                     materials_rt60=None):
    if test_mode:
        azimuths = np.array([0, 60, 90, 180, 270], dtype=float)
        rt60s = [0.0, 1.0]
        drone_spls = [80.0, 70.0]
        elevations_sweep = [20.0, 50.0]
        seeds = [0]
    else:
        azimuths = AZIMUTHS_FULL
        rt60s = RT60S_FULL
        drone_spls = DRONE_SPLS_FULL
        elevations_sweep = ELEVATIONS_SWEEP
        seeds = SEEDS_FULL

    if materials_profile is not None:
        # Single-environment sweep: one measured RT60 bucket for the
        # selected materials profile. Faster, and the plots collapse the
        # rt60 axis to a single bin.
        rt60s = [float(materials_rt60 if materials_rt60 is not None else 0.0)]

    seen = set()
    trials = []

    def _add(rt60, drone_spl_db, diffuse, az, el, seed):
        key = (rt60, drone_spl_db, diffuse, az, el, seed)
        if key not in seen:
            seen.add(key)
            trials.append(dict(rt60=rt60, drone_spl_db=drone_spl_db,
                               diffuse=diffuse,
                               true_az_deg=az, true_el_deg=el, seed=seed))

    for rt60 in rt60s:
        for az in azimuths:
            for seed in seeds:
                _add(rt60, DEFAULT_DRONE_SPL, False, float(az), DEFAULT_EL, seed)

    for el in elevations_sweep:
        for seed in seeds:
            _add(DEFAULT_RT60_REV, DEFAULT_DRONE_SPL, False, DEFAULT_AZ, el, seed)

    for spl in drone_spls:
        for seed in seeds:
            _add(DEFAULT_RT60_REV, spl, False, DEFAULT_AZ, DEFAULT_EL, seed)

    for az in azimuths:
        for seed in seeds:
            _add(DEFAULT_RT60_REV, 70.0, True, float(az), DEFAULT_EL, seed)

    return trials


# ─── CSV I/O ──────────────────────────────────────────────────────────────────

def load_completed(csv_path):
    done = set()
    p = pathlib.Path(csv_path)
    if not p.exists():
        return done
    with open(p, "r", newline="") as f:
        for r in csv.DictReader(f):
            done.add((
                r["geometry"], float(r["rt60"]), float(r["drone_spl_db"]),
                r["diffuse"] == "True", float(r["true_az_deg"]),
                float(r["true_el_deg"]), int(r["seed"]),
            ))
    return done


def append_row(csv_path, row):
    p = pathlib.Path(csv_path)
    write_header = not p.exists() or p.stat().st_size == 0
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_csv(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                geometry=r["geometry"],
                rt60=float(r["rt60"]),
                drone_spl_db=float(r["drone_spl_db"]),
                diffuse=r["diffuse"] == "True",
                true_az_deg=float(r["true_az_deg"]),
                true_el_deg=float(r["true_el_deg"]),
                est_az_deg=float(r["est_az_deg"]),
                est_el_deg=float(r["est_el_deg"]),
                az_error_deg=float(r["az_error_deg"]),
                el_error_deg=float(r["el_error_deg"]),
                total_error_deg=float(r["total_error_deg"]),
                seed=int(r["seed"]),
            ))
    return rows


# ─── Main simulation loop ────────────────────────────────────────────────────

def run_all_trials(test_mode=False, materials_profile=None,
                   materials_rt60=None):
    OUT.mkdir(exist_ok=True)
    done = load_completed(CSV_PATH)
    trial_list = build_trial_list(test_mode, materials_profile=materials_profile,
                                  materials_rt60=materials_rt60)
    total = len(trial_list) * len(GEOMETRIES)
    spectra = []

    print(f"Conditions: {len(trial_list)}  x  {len(GEOMETRIES)} geometries  "
          f"=  {total} trials")
    print(f"Room: {ROOM_DIM[0]:.0f}x{ROOM_DIM[1]:.0f}x{ROOM_DIM[2]:.0f} m  |  "
          f"max_order cap: {MAX_ORDER_CAP}  |  "
          f"SRP grid: {len(SRP_AZ)}x{len(SRP_COLAT)}={len(SRP_AZ)*len(SRP_COLAT)}\n")

    t0 = time.time()
    idx = 0
    new_count = 0

    for geo in GEOMETRIES:
        R = build_geometry(geo, ARRAY_CENTER)
        assert R.shape == (3, 12), f"{geo} shape {R.shape} != (3, 12)"

        for trial in trial_list:
            idx += 1
            key = (geo, trial["rt60"], trial["drone_spl_db"], trial["diffuse"],
                   trial["true_az_deg"], trial["true_el_deg"], trial["seed"])
            if key in done:
                continue

            result = run_single_trial(
                R, trial["true_az_deg"], trial["true_el_deg"],
                trial["rt60"], trial["drone_spl_db"], ROOM_DIM,
                trial["diffuse"], trial["seed"],
                materials_profile=materials_profile,
            )
            if result is None or result[0] is None:
                done.add(key)
                continue
            est_az, est_el, g_az, g_colat, g_vals, ml_snr, feat_snr = result

            az_err = abs(wrap_angle_deg(est_az - trial["true_az_deg"]))
            el_err = abs(est_el - trial["true_el_deg"])
            total_err = angular_distance_deg(
                trial["true_az_deg"], trial["true_el_deg"], est_az, est_el)

            row = dict(
                geometry=geo,
                rt60=trial["rt60"],
                drone_spl_db=trial["drone_spl_db"],
                diffuse=trial["diffuse"],
                true_az_deg=round(trial["true_az_deg"], 4),
                true_el_deg=round(trial["true_el_deg"], 4),
                est_az_deg=round(est_az, 4),
                est_el_deg=round(est_el, 4),
                az_error_deg=round(az_err, 4),
                el_error_deg=round(el_err, 4),
                total_error_deg=round(total_err, 4),
                seed=trial["seed"],
                ml_path_snr_db=("" if ml_snr is None else ml_snr),
                feature_snr_db=("" if feat_snr is None else feat_snr),
            )
            append_row(CSV_PATH, row)
            done.add(key)
            new_count += 1

            is_cherry = (
                abs(trial["true_az_deg"] - DEFAULT_AZ) < 1e-9
                and abs(trial["true_el_deg"] - DEFAULT_EL) < 1e-9
                and trial["seed"] == 0
                and (
                    (abs(trial["rt60"] - 1.0) < 1e-9
                     and not trial["diffuse"]
                     and abs(trial["drone_spl_db"] - DEFAULT_DRONE_SPL) < 1e-9)
                    or trial["diffuse"]
                )
            )
            if is_cherry:
                cond = "diffuse" if trial["diffuse"] else "rt60_1.0"
                spectra.append(dict(
                    geometry=geo, condition=cond,
                    grid_az_rad=g_az.tolist(),
                    grid_colat_rad=g_colat.tolist(),
                    grid_vals=g_vals.tolist(),
                    est_az_deg=est_az, est_el_deg=est_el,
                ))

            if new_count % 20 == 0 or idx == total:
                elapsed = time.time() - t0
                rate = new_count / max(elapsed, 0.001)
                remaining = (total - idx) / max(rate, 0.001) if rate > 0 else 0
                diff_tag = "DIFF " if trial["diffuse"] else "     "
                print(
                    f"  [{idx}/{total}]  {rate:.2f} trials/s  "
                    f"~{remaining/60:.0f}min left  |  "
                    f"{geo:8s} rt60={trial['rt60']:.1f} "
                    f"drone={trial['drone_spl_db']:3.0f}dB {diff_tag}"
                    f"az={trial['true_az_deg']:5.0f} el={trial['true_el_deg']:4.0f}  "
                    f"err_az={az_err:.1f} err_el={el_err:.1f} err_tot={total_err:.1f}"
                )

    if spectra:
        with open(SPECTRA_PATH, "w") as f:
            json.dump(spectra, f)

    elapsed = time.time() - t0
    print(f"\nSimulation complete: {new_count} new trials in {elapsed:.1f}s")


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _filter(rows, **kwargs):
    out = rows
    for k, v in kwargs.items():
        if isinstance(v, float):
            out = [r for r in out if abs(r[k] - v) < 1e-9]
        elif isinstance(v, bool):
            out = [r for r in out if r[k] == v]
        else:
            out = [r for r in out if r[k] == v]
    return out


def _conditions_list():
    conds = []
    for rt60 in RT60S_FULL:
        conds.append(dict(rt60=rt60, drone_spl_db=DEFAULT_DRONE_SPL, diffuse=False,
                          label="Anechoic" if rt60 == 0 else f"RT60={rt60:.1f}s"))
    conds.append(dict(rt60=DEFAULT_RT60_REV, drone_spl_db=70.0, diffuse=True,
                      label="Diffuse"))
    return conds


# ─── Plot 1: Azimuth error polar (2x3) ───────────────────────────────────────

def plot_az_error_polar(rows):
    conds = _conditions_list()
    fig, axes = plt.subplots(2, 3, subplot_kw=dict(projection="polar"),
                             figsize=(18, 12))
    axes_flat = axes.flatten()

    for ci, cond in enumerate(conds):
        ax = axes_flat[ci]
        for geo in GEOMETRIES:
            subset = _filter(rows, geometry=geo, rt60=cond["rt60"],
                             drone_spl_db=cond["drone_spl_db"],
                             diffuse=cond["diffuse"])
            if not subset:
                continue
            az_map = {}
            for r in subset:
                az_map.setdefault(r["true_az_deg"], []).append(r["az_error_deg"])
            azs = sorted(az_map)
            means = [np.mean(az_map[a]) for a in azs]
            az_rad = np.deg2rad(np.append(azs, azs[0]))
            vals = np.append(means, means[0])
            ax.plot(az_rad, vals, label=geo, color=GEO_COLORS[geo],
                    marker=GEO_MARKERS[geo], markersize=3, linewidth=1.5)
        ax.set_title(cond["label"], pad=15, fontsize=11)
        ax.set_rlabel_position(135)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Azimuth Error vs True Azimuth (deg)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / f"az_error_polar{PLOT_SUFFIX}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    az_error_polar.png")


# ─── Plot 2: Azimuth error heatmap (2x3) ─────────────────────────────────────

def _error_heatmap_panel(ax, rows, cond, error_key):
    for gi, geo in enumerate(GEOMETRIES):
        subset = _filter(rows, geometry=geo, rt60=cond["rt60"],
                         drone_spl_db=cond["drone_spl_db"],
                         diffuse=cond["diffuse"])
        if not subset:
            continue
        az_map = {}
        for r in subset:
            az_map.setdefault(r["true_az_deg"], []).append(r[error_key])
        azs = sorted(az_map)
        for j, a in enumerate(azs):
            val = np.mean(az_map[a])
            ax.fill_between([j - 0.5, j + 0.5], gi - 0.5, gi + 0.5,
                            color=plt.cm.viridis_r(min(val / 90.0, 1.0)))
    ax.set_yticks(range(len(GEOMETRIES)))
    ax.set_yticklabels(GEOMETRIES, fontsize=8)
    azs_all = sorted({r["true_az_deg"] for r in rows if not r["diffuse"]})
    if not azs_all:
        azs_all = sorted({r["true_az_deg"] for r in rows})
    tick_idx = list(range(0, len(azs_all), max(1, len(azs_all) // 6)))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{azs_all[i]:.0f}" for i in tick_idx], fontsize=7)
    ax.set_title(cond["label"], fontsize=10)


def plot_az_error_heatmap(rows):
    conds = _conditions_list()
    fig, axes = plt.subplots(2, 3, figsize=(20, 8))
    axes_flat = axes.flatten()
    for ci, cond in enumerate(conds):
        _error_heatmap_panel(axes_flat[ci], rows, cond, "az_error_deg")
    fig.suptitle("Azimuth Error Heatmap (geometry x azimuth)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / f"az_error_heatmap{PLOT_SUFFIX}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    az_error_heatmap.png")


# ─── Plot 3: Elevation error polar (2x3) ─────────────────────────────────────

def plot_el_error_polar(rows):
    conds = _conditions_list()
    fig, axes = plt.subplots(2, 3, subplot_kw=dict(projection="polar"),
                             figsize=(18, 12))
    axes_flat = axes.flatten()

    for ci, cond in enumerate(conds):
        ax = axes_flat[ci]
        for geo in GEOMETRIES:
            subset = _filter(rows, geometry=geo, rt60=cond["rt60"],
                             drone_spl_db=cond["drone_spl_db"],
                             diffuse=cond["diffuse"])
            if not subset:
                continue
            az_map = {}
            for r in subset:
                az_map.setdefault(r["true_az_deg"], []).append(r["el_error_deg"])
            azs = sorted(az_map)
            means = [np.mean(az_map[a]) for a in azs]
            az_rad = np.deg2rad(np.append(azs, azs[0]))
            vals = np.append(means, means[0])
            ax.plot(az_rad, vals, label=geo, color=GEO_COLORS[geo],
                    marker=GEO_MARKERS[geo], markersize=3, linewidth=1.5)
        ax.set_title(cond["label"], pad=15, fontsize=11)
        ax.set_rlabel_position(135)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Elevation Error vs True Azimuth (deg)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / f"el_error_polar{PLOT_SUFFIX}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    el_error_polar.png")


# ─── Plot 4: Elevation error heatmap (2x3) ───────────────────────────────────

def plot_el_error_heatmap(rows):
    conds = _conditions_list()
    fig, axes = plt.subplots(2, 3, figsize=(20, 8))
    axes_flat = axes.flatten()
    for ci, cond in enumerate(conds):
        _error_heatmap_panel(axes_flat[ci], rows, cond, "el_error_deg")
    fig.suptitle("Elevation Error Heatmap (geometry x azimuth)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / f"el_error_heatmap{PLOT_SUFFIX}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    el_error_heatmap.png")


# ─── Plot 5: Performance summary (2x2) ───────────────────────────────────────

def plot_performance_summary(rows):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    non_diff = _filter(rows, diffuse=False, drone_spl_db=DEFAULT_DRONE_SPL)
    for geo in GEOMETRIES:
        geo_rows = _filter(non_diff, geometry=geo)
        rt60_map = {}
        for r in geo_rows:
            rt60_map.setdefault(r["rt60"], []).append(r["total_error_deg"])
        rt60s = sorted(rt60_map)
        if not rt60s:
            continue
        means = [np.mean(rt60_map[v]) for v in rt60s]
        stds = [np.std(rt60_map[v]) for v in rt60s]
        ax1.plot(rt60s, means, "o-", label=geo, color=GEO_COLORS[geo], linewidth=2)
        ax1.fill_between(rt60s, np.maximum(np.array(means) - stds, 0),
                         np.array(means) + stds, color=GEO_COLORS[geo], alpha=0.12)
    ax1.set_xlabel("RT60 (s)")
    ax1.set_ylabel("Mean total error (deg)")
    ax1.set_title("Total Angular Error vs RT60")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    n_geo = len(GEOMETRIES)
    width = 0.8 / n_geo
    rt60_vals = sorted({r["rt60"] for r in non_diff})
    x = np.arange(len(rt60_vals))
    for i, geo in enumerate(GEOMETRIES):
        rates = []
        for rt in rt60_vals:
            group = _filter(non_diff, geometry=geo, rt60=rt)
            if group:
                ok = sum(1 for r in group if r["total_error_deg"] < 5.0)
                rates.append(100.0 * ok / len(group))
            else:
                rates.append(0.0)
        offset = (i - (n_geo - 1) / 2) * width
        ax2.bar(x + offset, rates, width, label=geo, color=GEO_COLORS[geo])
    ax2.set_xlabel("RT60 (s)")
    ax2.set_ylabel("Success rate (%)")
    ax2.set_title("Success Rate (total error < 5 deg) vs RT60")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{v:.1f}" for v in rt60_vals])
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis="y", alpha=0.3)

    spl_rows = _filter(rows, rt60=DEFAULT_RT60_REV, diffuse=False)
    for geo in GEOMETRIES:
        geo_rows = _filter(spl_rows, geometry=geo)
        spl_map = {}
        for r in geo_rows:
            spl_map.setdefault(r["drone_spl_db"], []).append(r["total_error_deg"])
        spls = sorted(spl_map)
        if not spls:
            continue
        means = [np.mean(spl_map[s]) for s in spls]
        ax3.plot(spls, means, "o-", label=geo, color=GEO_COLORS[geo], linewidth=2)
    ax3.set_xlabel("Drone SPL at 1 m (dB)")
    ax3.set_ylabel("Mean total error (deg)")
    ax3.set_title("Total Angular Error vs Drone SPL")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    el_rows = _filter(rows, rt60=DEFAULT_RT60_REV,
                      drone_spl_db=DEFAULT_DRONE_SPL, diffuse=False)
    for geo in GEOMETRIES:
        geo_rows = _filter(el_rows, geometry=geo)
        el_map = {}
        for r in geo_rows:
            if abs(r["true_az_deg"] - DEFAULT_AZ) < 1e-9:
                el_map.setdefault(r["true_el_deg"], []).append(r["el_error_deg"])
        els = sorted(el_map)
        if not els:
            continue
        means = [np.mean(el_map[e]) for e in els]
        stds = [np.std(el_map[e]) for e in els]
        ax4.plot(els, means, "o-", label=geo, color=GEO_COLORS[geo], linewidth=2)
        ax4.fill_between(els, np.maximum(np.array(means) - stds, 0),
                         np.array(means) + stds, color=GEO_COLORS[geo], alpha=0.12)
    ax4.set_xlabel("True elevation (deg)")
    ax4.set_ylabel("Mean elevation error (deg)")
    ax4.set_title("Elevation Error vs Elevation Angle")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(OUT / f"performance_summary{PLOT_SUFFIX}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    performance_summary.png")


# ─── Plot 6: Summary heatmap ─────────────────────────────────────────────────

def plot_summary_heatmap(rows):
    conditions = []
    labels = []

    for rt60 in RT60S_FULL:
        conditions.append(dict(rt60=rt60, drone_spl_db=DEFAULT_DRONE_SPL,
                               diffuse=False))
        labels.append("Anechoic" if rt60 == 0 else f"RT60={rt60:.1f}")

    for spl in [85.0, 75.0, 70.0]:
        conditions.append(dict(rt60=DEFAULT_RT60_REV, drone_spl_db=spl,
                               diffuse=False))
        labels.append(f"Drone={int(spl)}dB")

    conditions.append(dict(rt60=DEFAULT_RT60_REV, drone_spl_db=70.0, diffuse=True))
    labels.append("Diffuse")

    mat = np.full((len(GEOMETRIES), len(conditions)), np.nan)
    for j, cond in enumerate(conditions):
        for i, geo in enumerate(GEOMETRIES):
            group = _filter(rows, geometry=geo, rt60=cond["rt60"],
                            drone_spl_db=cond["drone_spl_db"],
                            diffuse=cond["diffuse"])
            if group:
                mat[i, j] = np.mean([r["total_error_deg"] for r in group])

    fig, ax = plt.subplots(figsize=(14, 4))
    vmax = max(np.nanmax(mat), 10) if not np.all(np.isnan(mat)) else 90
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=vmax)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, fontsize=9, rotation=40, ha="right")
    ax.set_yticks(range(len(GEOMETRIES)))
    ax.set_yticklabels(GEOMETRIES, fontsize=11)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                color = "white" if mat[i, j] > vmax * 0.55 else "black"
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    ax.set_title("Mean Total Angular Error (deg) -- Geometry x Condition",
                 fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, label="Mean error (deg)", shrink=0.9, pad=0.02)
    fig.savefig(OUT / f"summary_heatmap{PLOT_SUFFIX}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    summary_heatmap.png")


# ─── Plot 7: SRP spectra (2x4) ───────────────────────────────────────────────

def plot_srp_spectra(spectra_list):
    if not spectra_list:
        print("    (no spectra to plot)")
        return

    rt60_specs = [s for s in spectra_list if s["condition"] == "rt60_1.0"]
    diff_specs = [s for s in spectra_list if s["condition"] == "diffuse"]

    fig, axes = plt.subplots(2, 4, subplot_kw=dict(projection="polar"),
                             figsize=(20, 10))

    for row_specs, row_axes, row_label in [
        (rt60_specs, axes[0], "RT60 = 1.0 s"),
        (diff_specs, axes[1], "Diffuse"),
    ]:
        geo_map = {s["geometry"]: s for s in row_specs}
        for gi, geo in enumerate(GEOMETRIES):
            ax = row_axes[gi]
            if geo not in geo_map:
                ax.set_title(f"{geo}\n({row_label})", fontsize=10)
                continue
            s = geo_map[geo]
            g_az = np.array(s["grid_az_rad"])
            g_colat = np.array(s["grid_colat_rad"])
            g_vals = np.array(s["grid_vals"])

            est_colat = np.deg2rad(90.0 - s["est_el_deg"])
            colat_unique = np.unique(g_colat)
            nearest_colat = colat_unique[np.argmin(np.abs(colat_unique - est_colat))]
            mask = np.abs(g_colat - nearest_colat) < 1e-6
            slice_az = g_az[mask]
            slice_vals = g_vals[mask]

            if len(slice_vals) > 0:
                slice_vals = slice_vals / (slice_vals.max() + 1e-12)
                order = np.argsort(slice_az)
                slice_az = slice_az[order]
                slice_vals = slice_vals[order]
                az_plot = np.append(slice_az, slice_az[0])
                v_plot = np.append(slice_vals, slice_vals[0])
                ax.fill(az_plot, v_plot, alpha=0.25, color=GEO_COLORS[geo])
                ax.plot(az_plot, v_plot, color=GEO_COLORS[geo], linewidth=1.5)

            true_az_rad = np.deg2rad(DEFAULT_AZ)
            ax.plot([true_az_rad, true_az_rad], [0, 1], "--", color="red",
                    linewidth=2, label="True")
            est_az_rad = np.deg2rad(s["est_az_deg"])
            ax.plot([est_az_rad, est_az_rad], [0, 1], "-", color="green",
                    linewidth=2, label="Est")
            ax.set_title(f"{geo}\n({row_label})", fontsize=10)
            ax.set_rlim(0, 1.05)
            if gi == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("SRP-PHAT Spatial Spectrum (azimuth slice at estimated elevation)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / f"srp_spectra{PLOT_SUFFIX}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    srp_spectra.png")


# ─── Plot 8: Room scene ──────────────────────────────────────────────────────

def plot_room_scene():
    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(16, 7))

    rect_top = plt.Rectangle((0, 0), ROOM_DIM[0], ROOM_DIM[1],
                               fill=False, edgecolor="black", linewidth=2)
    ax_top.add_patch(rect_top)

    # Illustrative crowd/PA layout: fixed seed so the diagram is stable.
    viz_rng = np.random.default_rng(7)
    crowd_pos = crowd_positions_mixed(ROOM_DIM, 12, z_height=1.2,
                                       array_center=ARRAY_CENTER, rng=viz_rng)
    pa_pos = wall_adjacent_positions(ROOM_DIM, 4, z_height=ROOM_DIM[2] - 1.0,
                                      rng=viz_rng)
    cx_list = [p[0] for p in crowd_pos]
    cy_list = [p[1] for p in crowd_pos]
    ax_top.scatter(cx_list, cy_list, c="gray", s=30, marker="x", alpha=0.6,
                   label="Crowd noise")

    drone = drone_position(ARRAY_CENTER, DEFAULT_AZ, DEFAULT_EL,
                           SOURCE_DISTANCE, ROOM_DIM)
    ax_top.scatter(drone[0], drone[1], c="red", s=150, marker="*", zorder=5,
                   label=f"Drone (az={DEFAULT_AZ}, el={DEFAULT_EL})")

    for geo in GEOMETRIES:
        R = build_geometry(geo, ARRAY_CENTER)
        ax_top.scatter(R[0], R[1], c=GEO_COLORS[geo], s=20,
                       marker=GEO_MARKERS[geo], label=f"{geo}", zorder=4)

    ax_top.set_xlim(-0.5, ROOM_DIM[0] + 0.5)
    ax_top.set_ylim(-0.5, ROOM_DIM[1] + 0.5)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.set_title("Top-down view (x-y plane)", fontsize=12)
    ax_top.legend(fontsize=7, loc="upper left")
    ax_top.grid(True, alpha=0.15)
    ax_top.text(ROOM_DIM[0] / 2, -0.3, f"{ROOM_DIM[0]:.0f} m", ha="center",
                fontsize=9)
    ax_top.text(-0.3, ROOM_DIM[1] / 2, f"{ROOM_DIM[1]:.0f} m", ha="center",
                fontsize=9, rotation=90)

    rect_side = plt.Rectangle((0, 0), ROOM_DIM[0], ROOM_DIM[2],
                                fill=False, edgecolor="black", linewidth=2)
    ax_side.add_patch(rect_side)

    ax_side.axhline(y=ARRAY_CENTER[2], color="blue", linestyle="--", alpha=0.5,
                     label=f"Array (z={ARRAY_CENTER[2]:.1f} m)")
    ax_side.scatter(drone[0], drone[2], c="red", s=150, marker="*", zorder=5,
                    label=f"Drone (z={drone[2]:.1f} m)")
    ax_side.plot([ARRAY_CENTER[0], drone[0]], [ARRAY_CENTER[2], drone[2]],
                 "r--", alpha=0.4)

    ax_side.scatter(cx_list, [1.2] * len(cx_list), c="gray", s=30, marker="x",
                    alpha=0.6, label="Crowd (z=1.2 m)")

    pa_x = [p[0] for p in pa_pos]
    pa_z = [p[2] for p in pa_pos]
    ax_side.scatter(pa_x, pa_z, c="orange", s=60, marker="v", zorder=4,
                    label=f"PA (z={ROOM_DIM[2] - 1.0:.1f} m)")

    ax_side.set_xlim(-0.5, ROOM_DIM[0] + 0.5)
    ax_side.set_ylim(-0.5, ROOM_DIM[2] + 0.5)
    ax_side.set_aspect("equal")
    ax_side.set_xlabel("x (m)")
    ax_side.set_ylabel("z (m)")
    ax_side.set_title("Side view (x-z plane)", fontsize=12)
    ax_side.legend(fontsize=7, loc="upper left")
    ax_side.grid(True, alpha=0.15)
    ax_side.text(ROOM_DIM[0] / 2, ROOM_DIM[2] + 0.3,
                 f"{ROOM_DIM[2]:.0f} m ceiling", ha="center", fontsize=9)

    fig.suptitle("Exhibition Hall -- 3D Room Layout", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / f"room_scene{PLOT_SUFFIX}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    room_scene.png")


# ─── Plot orchestrator ────────────────────────────────────────────────────────

def generate_all_plots():
    if not CSV_PATH.exists():
        print("No results CSV found -- run simulation first.")
        return

    print("\nGenerating plots...")
    rows = load_csv(CSV_PATH)

    plot_az_error_polar(rows)
    plot_az_error_heatmap(rows)
    plot_el_error_polar(rows)
    plot_el_error_heatmap(rows)
    plot_performance_summary(rows)
    plot_summary_heatmap(rows)

    if SPECTRA_PATH.exists():
        with open(SPECTRA_PATH, "r") as f:
            spectra = json.load(f)
        plot_srp_spectra(spectra)

    plot_room_scene()
    print(f"\nAll plots saved to {OUT}/")


# ─── Main ─────────────────────────────────────────────────────────────────────

def _measure_profile_rt60(materials_profile):
    """Run one throwaway RIR computation to get the measured RT60 of the
    requested materials profile in the batch room geometry."""
    if materials_profile != "exhibition_hall":
        return None
    mats = build_materials(**EXHIBITION_HALL_MATERIALS)
    room = pra.ShoeBox(ROOM_DIM.tolist(), fs=FS,
                       materials=mats, max_order=6,
                       **air_absorption_kwargs(TEMPERATURE_C, HUMIDITY_PCT))
    src = drone_position(ARRAY_CENTER, DEFAULT_AZ, DEFAULT_EL,
                         SOURCE_DISTANCE, ROOM_DIM)
    room.add_source(src, signal=np.zeros(100))
    room.add_microphone_array(pra.MicrophoneArray(
        ARRAY_CENTER.reshape(3, 1), fs=FS))
    room.compute_rir()
    rir = np.asarray(room.rir[0][0])
    return measure_rt60_from_rir(rir, fs=FS, decay_db=20)


def main():
    parser = argparse.ArgumentParser(
        description="3D DOA Geometry Comparison -- Exhibition Hall Simulation")
    parser.add_argument("--test", action="store_true",
                        help="Quick run with reduced parameter set (~160 trials)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerate plots from existing CSV")
    parser.add_argument("--materials-profile", choices=["exhibition_hall"],
                        default=None,
                        help="Use per-wall materials instead of inverse_sabine. "
                             "Writes metrics_materials.csv and *_materials.png plots.")
    parser.add_argument("--temperature", type=float, default=None,
                        metavar="C",
                        help="Air temperature in Celsius for air absorption "
                             "(Phase 2b). Default 20 C.")
    parser.add_argument("--humidity", type=float, default=None,
                        metavar="PCT",
                        help="Relative humidity in %% for air absorption "
                             "(Phase 2b). Default 50%%.")
    parser.add_argument("--temp-gradient", type=float, default=None,
                        metavar="CPERM",
                        help="Vertical temperature gradient dT/dz in C/m for "
                             "first-order beam-bending (Phase 2b). Default 0. "
                             "Typical convention hall with ceiling lights: ~1.5.")
    parser.add_argument("--crowd-model", choices=["point_source", "plane_wave"],
                        default=None,
                        help="Crowd-noise spatial correlation model (Phase 3). "
                             "'point_source' keeps the legacy discrete crowd "
                             "placement; 'plane_wave' synthesizes an isotropic "
                             "diffuse field. Default point_source.")
    parser.add_argument("--n-plane-waves", type=int, default=None, metavar="N",
                        help="Number of plane waves for --crowd-model plane_wave "
                             "(Phase 3). Default 64.")
    parser.add_argument("--crosstalk-model", choices=["simple", "fir_capacitive"],
                        default=None,
                        help="Preamp/ADC crosstalk model (Phase 3). "
                             "'simple' = flat neighbour leakage; 'fir_capacitive' "
                             "= 1-pole HPF leakage path. Default simple.")
    parser.add_argument("--crosstalk-corner-hz", type=float, default=None,
                        metavar="HZ",
                        help="Corner frequency for --crosstalk-model fir_capacitive "
                             "(Phase 3). Default 500 Hz.")
    parser.add_argument("--crosstalk-coupling-db", type=float, default=None,
                        metavar="DB",
                        help="Neighbour-channel voltage coupling ratio in dB when "
                             "a FIR crosstalk model is active (Phase 3). Default -40 dB.")
    parser.add_argument("--crosstalk-fir-path", type=str, default=None,
                        metavar="PATH",
                        help="Optional path to a measured crosstalk FIR "
                             "(.json or .npz). When set, overrides the analytic "
                             "HPF model.")
    parser.add_argument("--ml-preview", action="store_true",
                        help="Enable MAX78000 ML-path preview per trial (Phase 3). "
                             "Adds ml_path_snr_db and feature_snr_db columns to "
                             "the output CSV. Adds a few % overhead per trial.")
    parser.add_argument("--ml-bit-depth", type=int, default=None,
                        choices=[8, 16], metavar="BITS",
                        help="Bit-depth for ML-path audio quantization "
                             "(Phase 3). Default 8.")
    parser.add_argument("--ml-feature-bit-depth", type=int, default=None,
                        choices=[8, 16], metavar="BITS",
                        help="Bit-depth for ML-path log-mel feature quantization "
                             "(Phase 3). Default 8.")
    parser.add_argument("--ml-n-mels", type=int, default=None, metavar="N",
                        help="Number of mel bands for ML-path feature extractor "
                             "(Phase 3). Default 64.")
    parser.add_argument("--fmin", type=float, default=None, metavar="HZ",
                        help="SRP-PHAT band lower edge in Hz (Phase 3+). "
                             "Default 200.")
    parser.add_argument("--fmax", type=float, default=None, metavar="HZ",
                        help="SRP-PHAT band upper edge in Hz (Phase 3+). "
                             "Default 2000.")
    parser.add_argument("--harmonic-comb", action="store_true",
                        help="Restrict SRP-PHAT bins to a +/-10 Hz window "
                             "around each harmonic n*f0 of the drone "
                             "fundamental (Phase 3+).")
    parser.add_argument("--drone-fundamental", type=float, default=None,
                        metavar="HZ",
                        help="Drone blade-pass fundamental f0 in Hz used by "
                             "--harmonic-comb (Phase 3+). Default 200.")
    args = parser.parse_args()

    global MATERIALS_PROFILE, MATERIALS_PROFILE_RT60, PLOT_SUFFIX
    global CSV_PATH, SPECTRA_PATH, RT60S_FULL, DEFAULT_RT60_REV
    global TEMPERATURE_C, HUMIDITY_PCT, TEMP_GRADIENT_C_PER_M
    global CROWD_MODEL, N_PLANE_WAVES
    global CROSSTALK_MODEL, CROSSTALK_CORNER_HZ, CROSSTALK_COUPLING_DB
    global CROSSTALK_FIR_PATH
    global ML_PREVIEW, ML_BIT_DEPTH, ML_FEATURE_BIT_DEPTH, ML_N_MELS
    global FMIN_HZ, FMAX_HZ, HARMONIC_COMB, DRONE_FUNDAMENTAL_HZ

    if args.temperature is not None:
        TEMPERATURE_C = float(args.temperature)
    if args.humidity is not None:
        HUMIDITY_PCT = float(args.humidity)
    if args.temp_gradient is not None:
        TEMP_GRADIENT_C_PER_M = float(args.temp_gradient)
    if (args.temperature is not None or args.humidity is not None
            or args.temp_gradient is not None):
        print(f"[atmosphere] T={TEMPERATURE_C:.1f} C  RH={HUMIDITY_PCT:.0f}%  "
              f"dT/dz={TEMP_GRADIENT_C_PER_M:+.2f} C/m")

    if args.crowd_model is not None:
        CROWD_MODEL = str(args.crowd_model)
    if args.n_plane_waves is not None:
        N_PLANE_WAVES = int(args.n_plane_waves)
    if args.crosstalk_model is not None:
        CROSSTALK_MODEL = str(args.crosstalk_model)
    if args.crosstalk_corner_hz is not None:
        CROSSTALK_CORNER_HZ = float(args.crosstalk_corner_hz)
    if args.crosstalk_coupling_db is not None:
        CROSSTALK_COUPLING_DB = float(args.crosstalk_coupling_db)
    if args.crosstalk_fir_path is not None:
        CROSSTALK_FIR_PATH = str(args.crosstalk_fir_path)
    if args.ml_preview:
        ML_PREVIEW = True
    if args.ml_bit_depth is not None:
        ML_BIT_DEPTH = int(args.ml_bit_depth)
    if args.ml_feature_bit_depth is not None:
        ML_FEATURE_BIT_DEPTH = int(args.ml_feature_bit_depth)
    if args.ml_n_mels is not None:
        ML_N_MELS = int(args.ml_n_mels)

    if CROWD_MODEL != "point_source" or CROSSTALK_MODEL != "simple" or ML_PREVIEW:
        print(f"[phase3] crowd_model={CROWD_MODEL}  "
              f"crosstalk_model={CROSSTALK_MODEL}  "
              f"ml_preview={'ON' if ML_PREVIEW else 'off'}")

    if args.fmin is not None:
        FMIN_HZ = float(args.fmin)
    if args.fmax is not None:
        FMAX_HZ = float(args.fmax)
    if args.harmonic_comb:
        HARMONIC_COMB = True
    if args.drone_fundamental is not None:
        DRONE_FUNDAMENTAL_HZ = float(args.drone_fundamental)

    if (FMIN_HZ != DEFAULT_FMIN or FMAX_HZ != DEFAULT_FMAX
            or HARMONIC_COMB):
        comb_tag = (f"comb ON @ f0={DRONE_FUNDAMENTAL_HZ:.0f} Hz"
                    if HARMONIC_COMB else "comb off")
        print(f"[doa-band] band={FMIN_HZ:.0f}-{FMAX_HZ:.0f} Hz  {comb_tag}")

    if args.materials_profile:
        MATERIALS_PROFILE = args.materials_profile
        PLOT_SUFFIX = f"_{args.materials_profile}"
        CSV_PATH = OUT / f"metrics{PLOT_SUFFIX}.csv"
        SPECTRA_PATH = OUT / f"spectra{PLOT_SUFFIX}.json"
        MATERIALS_PROFILE_RT60 = _measure_profile_rt60(args.materials_profile)
        if MATERIALS_PROFILE_RT60 is None:
            raise RuntimeError(f"Could not measure RT60 for profile {args.materials_profile}")
        # Keep the full-precision value so _filter(rt60=...) matches the
        # CSV rows exactly. Labels still render via "{rt60:.1f}" formatting.
        RT60S_FULL = [float(MATERIALS_PROFILE_RT60)]
        DEFAULT_RT60_REV = RT60S_FULL[0]
        OUT.mkdir(exist_ok=True)
        print(f"[materials-profile] {MATERIALS_PROFILE} "
              f"measured RT60 = {MATERIALS_PROFILE_RT60:.2f} s")
        print(f"[materials-profile] CSV -> {CSV_PATH}  plots -> *{PLOT_SUFFIX}.png")

    if args.plots_only:
        generate_all_plots()
        return

    run_all_trials(test_mode=args.test,
                   materials_profile=MATERIALS_PROFILE,
                   materials_rt60=MATERIALS_PROFILE_RT60)
    generate_all_plots()


if __name__ == "__main__":
    main()
