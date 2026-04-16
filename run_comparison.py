#!/usr/bin/env python
"""
3D DOA Array Geometry Comparison -- Exhibition Hall Simulation

Compares UCA, Standing Cross, ULA, and Cylinder microphone array geometries
for direction-of-arrival estimation (azimuth + elevation) under varying
reverberation, SNR, and diffuse noise conditions in a 3D exhibition hall.

Usage:
    python run_comparison.py              # full sweep (~1816 trials, ~3.5-4 hrs)
    python run_comparison.py --test       # quick validation (~160 trials, ~5-10 min)
    python run_comparison.py --plots-only # regenerate plots from existing CSV
"""

import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np
import pyroomacoustics as pra

SIGNAL_SECONDS = 1.0
ROOM_DIM = np.array([20.0, 15.0, 10.0])
ARRAY_CENTER = np.array([10.0, 7.5, 1.0])
SOURCE_DISTANCE = 4.0
MAX_ORDER_CAP = 10
MARGIN = 0.3

GEOMETRIES = ["UCA", "CROSS", "ULA", "CYLINDER"]
GEO_COLORS = {"UCA": "C0", "CROSS": "C1", "ULA": "C2", "CYLINDER": "C3"}
GEO_MARKERS = {"UCA": "o", "CROSS": "^", "ULA": "s", "CYLINDER": "D"}

RT60S_FULL = [0.0, 0.5, 0.8, 1.0, 1.5]
SNRS_FULL = [20.0, 10.0, 5.0, 0.0, -5.0]
DEFAULT_EL = 30.0
DEFAULT_AZ = 60.0
DEFAULT_SNR = 10.0
DEFAULT_RT60_REV = 1.0
ELEVATIONS_SWEEP = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
AZIMUTHS_FULL = np.arange(0, 360, 10, dtype=float)
SEEDS_FULL = [0, 1]

SRP_AZ = np.linspace(0, 2 * np.pi, 72, endpoint=False)
SRP_COLAT = np.linspace(0, np.pi, 19)

PA_POSITIONS = [[2.0, 2.0, 9.0], [18.0, 2.0, 9.0],
                [2.0, 13.0, 9.0], [18.0, 13.0, 9.0]]
OUT = pathlib.Path("results")
CSV_PATH = OUT / "metrics.csv"
SPECTRA_PATH = OUT / "spectra.json"
CSV_FIELDS = [
    "geometry", "rt60", "snr_db", "diffuse",
    "true_az_deg", "true_el_deg", "est_az_deg", "est_el_deg",
    "az_error_deg", "el_error_deg", "total_error_deg", "seed",
]


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


def perimeter_positions_3d(room_dim, n, z_height, margin=0.5):
    positions = []
    lx = room_dim[0] - 2 * margin
    ly = room_dim[1] - 2 * margin
    perimeter = 2 * (lx + ly)
    spacing = perimeter / n
    for i in range(n):
        d = i * spacing
        if d < lx:
            positions.append([margin + d, margin, z_height])
        elif d < lx + ly:
            positions.append([room_dim[0] - margin, margin + (d - lx), z_height])
        elif d < 2 * lx + ly:
            positions.append([room_dim[0] - margin - (d - lx - ly),
                              room_dim[1] - margin, z_height])
        else:
            positions.append([margin,
                              room_dim[1] - margin - (d - 2 * lx - ly), z_height])
    return positions


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

def make_diffuse_sources_3d(room_dim, fs, seconds, rng):
    sources = []
    n_samples = int(fs * seconds)
    for pos in perimeter_positions_3d(room_dim, 12, z_height=1.2, margin=0.5):
        sig = rng.standard_normal(n_samples) * 0.3
        sources.append((np.array(pos), sig))
    for pos in PA_POSITIONS:
        sig = rng.standard_normal(n_samples) * 0.2
        sources.append((np.array(pos, dtype=float), sig))
    return sources


# ─── Single trial ─────────────────────────────────────────────────────────────

def run_single_trial(array_R, true_az_deg, true_el_deg, rt60, snr_db,
                     room_dim, diffuse=False, seed=0):
    rng = np.random.default_rng(seed)

    sigma2 = 10.0 ** (-snr_db / 10.0) / (4.0 * np.pi * SOURCE_DISTANCE) ** 2

    if rt60 <= 0.0:
        room = pra.AnechoicRoom(3, fs=FS, sigma2_awgn=sigma2)
    else:
        try:
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
        except ValueError:
            return None, None, None, None, None
        max_order = min(max_order, MAX_ORDER_CAP)
        room = pra.ShoeBox(
            room_dim, fs=FS, sigma2_awgn=sigma2,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )

    src_pos = drone_position(ARRAY_CENTER, true_az_deg, true_el_deg,
                             SOURCE_DISTANCE, room_dim)
    room.add_source(src_pos, signal=drone_like_signal(FS, SIGNAL_SECONDS, rng))

    if diffuse:
        for pos, sig in make_diffuse_sources_3d(room_dim, FS, SIGNAL_SECONDS, rng):
            room.add_source(pos, signal=sig)

    room.add_microphone_array(pra.MicrophoneArray(array_R, fs=FS))
    room.simulate()

    X = np.array([
        pra.transform.stft.analysis(sig, NFFT, HOP).T
        for sig in room.mic_array.signals
    ])

    df = FS / NFFT
    freq_bins = np.arange(int(FMIN / df), int(FMAX / df))

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

    return est_az_deg, est_el_deg, grid_az, grid_colat, grid_vals


# ─── Trial list builder ──────────────────────────────────────────────────────

def build_trial_list(test_mode=False):
    if test_mode:
        azimuths = np.array([0, 60, 90, 180, 270], dtype=float)
        rt60s = [0.0, 1.0]
        snrs = [10.0, 0.0]
        elevations_sweep = [20.0, 50.0]
        seeds = [0]
    else:
        azimuths = AZIMUTHS_FULL
        rt60s = RT60S_FULL
        snrs = SNRS_FULL
        elevations_sweep = ELEVATIONS_SWEEP
        seeds = SEEDS_FULL

    seen = set()
    trials = []

    def _add(rt60, snr_db, diffuse, az, el, seed):
        key = (rt60, snr_db, diffuse, az, el, seed)
        if key not in seen:
            seen.add(key)
            trials.append(dict(rt60=rt60, snr_db=snr_db, diffuse=diffuse,
                               true_az_deg=az, true_el_deg=el, seed=seed))

    for rt60 in rt60s:
        for az in azimuths:
            for seed in seeds:
                _add(rt60, DEFAULT_SNR, False, float(az), DEFAULT_EL, seed)

    for el in elevations_sweep:
        for seed in seeds:
            _add(DEFAULT_RT60_REV, DEFAULT_SNR, False, DEFAULT_AZ, el, seed)

    for snr in snrs:
        for seed in seeds:
            _add(DEFAULT_RT60_REV, snr, False, DEFAULT_AZ, DEFAULT_EL, seed)

    for az in azimuths:
        for seed in seeds:
            _add(DEFAULT_RT60_REV, 5.0, True, float(az), DEFAULT_EL, seed)

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
                r["geometry"], float(r["rt60"]), float(r["snr_db"]),
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
                snr_db=float(r["snr_db"]),
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

def run_all_trials(test_mode=False):
    OUT.mkdir(exist_ok=True)
    done = load_completed(CSV_PATH)
    trial_list = build_trial_list(test_mode)
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
            key = (geo, trial["rt60"], trial["snr_db"], trial["diffuse"],
                   trial["true_az_deg"], trial["true_el_deg"], trial["seed"])
            if key in done:
                continue

            est_az, est_el, g_az, g_colat, g_vals = run_single_trial(
                R, trial["true_az_deg"], trial["true_el_deg"],
                trial["rt60"], trial["snr_db"], ROOM_DIM,
                trial["diffuse"], trial["seed"],
            )
            if est_az is None:
                done.add(key)
                continue

            az_err = abs(wrap_angle_deg(est_az - trial["true_az_deg"]))
            el_err = abs(est_el - trial["true_el_deg"])
            total_err = angular_distance_deg(
                trial["true_az_deg"], trial["true_el_deg"], est_az, est_el)

            row = dict(
                geometry=geo,
                rt60=trial["rt60"],
                snr_db=trial["snr_db"],
                diffuse=trial["diffuse"],
                true_az_deg=round(trial["true_az_deg"], 4),
                true_el_deg=round(trial["true_el_deg"], 4),
                est_az_deg=round(est_az, 4),
                est_el_deg=round(est_el, 4),
                az_error_deg=round(az_err, 4),
                el_error_deg=round(el_err, 4),
                total_error_deg=round(total_err, 4),
                seed=trial["seed"],
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
                     and abs(trial["snr_db"] - DEFAULT_SNR) < 1e-9)
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
                    f"snr={trial['snr_db']:3.0f}dB {diff_tag}"
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
        conds.append(dict(rt60=rt60, snr_db=DEFAULT_SNR, diffuse=False,
                          label="Anechoic" if rt60 == 0 else f"RT60={rt60:.1f}s"))
    conds.append(dict(rt60=DEFAULT_RT60_REV, snr_db=5.0, diffuse=True,
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
                             snr_db=cond["snr_db"], diffuse=cond["diffuse"])
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
    fig.savefig(OUT / "az_error_polar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    az_error_polar.png")


# ─── Plot 2: Azimuth error heatmap (2x3) ─────────────────────────────────────

def _error_heatmap_panel(ax, rows, cond, error_key):
    for gi, geo in enumerate(GEOMETRIES):
        subset = _filter(rows, geometry=geo, rt60=cond["rt60"],
                         snr_db=cond["snr_db"], diffuse=cond["diffuse"])
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
    fig.savefig(OUT / "az_error_heatmap.png", dpi=150, bbox_inches="tight")
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
                             snr_db=cond["snr_db"], diffuse=cond["diffuse"])
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
    fig.savefig(OUT / "el_error_polar.png", dpi=150, bbox_inches="tight")
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
    fig.savefig(OUT / "el_error_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    el_error_heatmap.png")


# ─── Plot 5: Performance summary (2x2) ───────────────────────────────────────

def plot_performance_summary(rows):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    non_diff = _filter(rows, diffuse=False, snr_db=DEFAULT_SNR)
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

    snr_rows = _filter(rows, rt60=DEFAULT_RT60_REV, diffuse=False)
    for geo in GEOMETRIES:
        geo_rows = _filter(snr_rows, geometry=geo)
        snr_map = {}
        for r in geo_rows:
            snr_map.setdefault(r["snr_db"], []).append(r["total_error_deg"])
        snrs = sorted(snr_map)
        if not snrs:
            continue
        means = [np.mean(snr_map[s]) for s in snrs]
        ax3.plot(snrs, means, "o-", label=geo, color=GEO_COLORS[geo], linewidth=2)
    ax3.set_xlabel("SNR (dB)")
    ax3.set_ylabel("Mean total error (deg)")
    ax3.set_title("Total Angular Error vs SNR")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    el_rows = _filter(rows, rt60=DEFAULT_RT60_REV, snr_db=DEFAULT_SNR,
                      diffuse=False)
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
    fig.savefig(OUT / "performance_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    performance_summary.png")


# ─── Plot 6: Summary heatmap ─────────────────────────────────────────────────

def plot_summary_heatmap(rows):
    conditions = []
    labels = []

    for rt60 in RT60S_FULL:
        conditions.append(dict(rt60=rt60, snr_db=DEFAULT_SNR, diffuse=False))
        labels.append("Anechoic" if rt60 == 0 else f"RT60={rt60:.1f}")

    for snr in [20.0, 5.0, 0.0, -5.0]:
        conditions.append(dict(rt60=DEFAULT_RT60_REV, snr_db=snr, diffuse=False))
        labels.append(f"SNR={int(snr)}")

    conditions.append(dict(rt60=DEFAULT_RT60_REV, snr_db=5.0, diffuse=True))
    labels.append("Diffuse")

    mat = np.full((len(GEOMETRIES), len(conditions)), np.nan)
    for j, cond in enumerate(conditions):
        for i, geo in enumerate(GEOMETRIES):
            group = _filter(rows, geometry=geo, rt60=cond["rt60"],
                            snr_db=cond["snr_db"], diffuse=cond["diffuse"])
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
    fig.savefig(OUT / "summary_heatmap.png", dpi=150, bbox_inches="tight")
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
    fig.savefig(OUT / "srp_spectra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    srp_spectra.png")


# ─── Plot 8: Room scene ──────────────────────────────────────────────────────

def plot_room_scene():
    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(16, 7))

    rect_top = plt.Rectangle((0, 0), ROOM_DIM[0], ROOM_DIM[1],
                               fill=False, edgecolor="black", linewidth=2)
    ax_top.add_patch(rect_top)

    crowd_pos = perimeter_positions_3d(ROOM_DIM, 12, 1.2, 0.5)
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

    pa_x = [p[0] for p in PA_POSITIONS]
    pa_z = [p[2] for p in PA_POSITIONS]
    ax_side.scatter(pa_x, pa_z, c="orange", s=60, marker="v", zorder=4,
                    label="PA (z=9.0 m)")

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
    fig.savefig(OUT / "room_scene.png", dpi=150, bbox_inches="tight")
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

def main():
    parser = argparse.ArgumentParser(
        description="3D DOA Geometry Comparison -- Exhibition Hall Simulation")
    parser.add_argument("--test", action="store_true",
                        help="Quick run with reduced parameter set (~160 trials)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerate plots from existing CSV")
    args = parser.parse_args()

    if args.plots_only:
        generate_all_plots()
        return

    run_all_trials(test_mode=args.test)
    generate_all_plots()


if __name__ == "__main__":
    main()
