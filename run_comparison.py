#!/usr/bin/env python
"""
DOA Array Geometry Comparison

Compares UCA, Cross, and ULA microphone array geometries for direction-of-arrival
estimation under varying reverberation, SNR, and diffuse noise conditions.

Usage:
    python run_comparison.py              # full sweep
    python run_comparison.py --test       # quick validation (~1 min)
    python run_comparison.py --plot-only  # regenerate plots from existing CSV
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyroomacoustics as pra

# ─── Configuration ────────────────────────────────────────────────────────────

FS = 16000
C = 343.0
NFFT = 256
HOP = NFFT // 2
FMIN = 500
FMAX = 2000  # below UCA aliasing threshold (~2.2 kHz) for fair comparison
SIGNAL_SECONDS = 2.0
SOURCE_DISTANCE = 3.0
ROOM_DIM = np.array([20.0, 15.0])
MAX_ORDER_CAP = 17  # bounds reverberant sim time; actual RT60 may be shorter
N_DIFFUSE = 24
RESULTS_DIR = "results"

GEOMETRIES = ["UCA", "CROSS", "ULA"]
AZIMUTHS_FULL = np.arange(0, 360, 5, dtype=float)
RT60S_FULL = [0.0, 0.4, 0.6, 1.0, 1.5]
SNRS_FULL = [20, 10, 5, 0, -5]
SEEDS_FULL = [0, 1, 2]

RT60_SWEEP_SNR = 10      # fixed SNR for the RT60 sweep
SNR_SWEEP_RT60 = 1.0     # fixed RT60 for the SNR sweep
DIFFUSE_RT60 = 1.0
DIFFUSE_SNR = 5

GEO_COLORS = {"UCA": "#1f77b4", "CROSS": "#ff7f0e", "ULA": "#2ca02c"}
GEO_MARKERS = {"UCA": "o", "CROSS": "s", "ULA": "^"}

CSV_FIELDS = [
    "geometry", "rt60", "snr_db", "diffuse",
    "true_az_deg", "est_az_deg", "abs_error_deg", "seed",
]


# ─── Signal generation ────────────────────────────────────────────────────────

def drone_like_signal(fs, seconds=SIGNAL_SECONDS, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(int(fs * seconds)) / fs
    freqs = [220, 440, 880, 1760]
    sig = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    sig *= 0.5
    sig += 0.05 * rng.standard_normal(len(t))
    return sig.astype(np.float32)


# ─── Array geometry builders ──────────────────────────────────────────────────

def make_uca(center_xy, mics=12, radius=0.15):
    return pra.circular_2D_array(center_xy, mics, 0.0, radius)


def make_cross(center_xy, mics_per_axis=6, half_length=0.15):
    offsets = np.linspace(-half_length, half_length, mics_per_axis + 1)
    offsets = offsets[np.abs(offsets) > 1e-9]
    pts = []
    for o in offsets:
        pts.append([center_xy[0] + o, center_xy[1]])
    for o in offsets:
        pts.append([center_xy[0], center_xy[1] + o])
    return np.array(pts).T


def make_ula(center_xy, mics=12, length=0.30):
    offsets = np.linspace(-length / 2, length / 2, mics)
    R = np.zeros((2, mics))
    R[0, :] = center_xy[0] + offsets
    R[1, :] = center_xy[1]
    return R


def build_geometry(name, center_xy):
    if name == "UCA":
        return make_uca(center_xy)
    elif name == "CROSS":
        return make_cross(center_xy)
    elif name == "ULA":
        return make_ula(center_xy)
    raise ValueError(f"Unknown geometry: {name}")


# ─── Diffuse noise sources ───────────────────────────────────────────────────

def perimeter_positions(room_dim, n, margin=0.3):
    W, H = room_dim
    w, h = W - 2 * margin, H - 2 * margin
    perim = 2 * (w + h)
    positions = []
    for i in range(n):
        d = i / n * perim
        if d < w:
            x, y = margin + d, margin
        elif d < w + h:
            x, y = W - margin, margin + (d - w)
        elif d < 2 * w + h:
            x, y = W - margin - (d - w - h), H - margin
        else:
            x, y = margin, H - margin - (d - 2 * w - h)
        positions.append(np.array([x, y]))
    return positions


def make_diffuse_sources(room_dim, n_sources=N_DIFFUSE, fs=FS,
                         seconds=SIGNAL_SECONDS, rng=None):
    if rng is None:
        rng = np.random.default_rng(99)
    positions = perimeter_positions(room_dim, n_sources)
    n_samples = int(fs * seconds)
    sources = []
    for pos in positions:
        noise = rng.standard_normal(n_samples).astype(np.float32) * 0.3
        sources.append((pos, noise))
    return sources


# ─── Angle utilities ──────────────────────────────────────────────────────────

def wrap_angle_deg(a):
    return (a + 180) % 360 - 180


def abs_angular_error(est, true):
    return abs(wrap_angle_deg(est - true))


# ─── Single trial ─────────────────────────────────────────────────────────────

def run_single_trial(array_R, true_az_deg, rt60, snr_db, room_dim,
                     diffuse=False, seed=0):
    rng = np.random.default_rng(seed)
    center = room_dim / 2.0

    az_rad = np.deg2rad(true_az_deg)
    src_pos = center + SOURCE_DISTANCE * np.array([np.cos(az_rad), np.sin(az_rad)])

    sigma2 = 10.0 ** (-snr_db / 10.0) / (4.0 * np.pi * SOURCE_DISTANCE) ** 2

    if rt60 <= 0.0:
        room = pra.AnechoicRoom(2, fs=FS, sigma2_awgn=sigma2)
    else:
        try:
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
        except ValueError:
            return None, None, None
        max_order = min(max_order, MAX_ORDER_CAP)
        room = pra.ShoeBox(
            room_dim, fs=FS, sigma2_awgn=sigma2,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )

    room.add_source(src_pos, signal=drone_like_signal(FS, SIGNAL_SECONDS, rng))

    if diffuse:
        for pos, sig in make_diffuse_sources(room_dim, N_DIFFUSE, FS,
                                             SIGNAL_SECONDS, rng):
            room.add_source(pos, signal=sig)

    room.add_microphone_array(pra.MicrophoneArray(array_R, fs=FS))
    room.simulate()

    X = np.array([
        pra.transform.stft.analysis(sig, NFFT, HOP).T
        for sig in room.mic_array.signals
    ])

    df = FS / NFFT
    freq_bins = np.arange(int(FMIN / df), int(FMAX / df))

    doa = pra.doa.SRP(array_R, FS, NFFT, c=C, num_src=1)
    doa.locate_sources(X, freq_bins=freq_bins)

    est_az_deg = float(np.atleast_1d(doa.azimuth_recon)[0] * 180.0 / np.pi)
    grid_az = np.array(doa.grid.azimuth, copy=True)
    grid_vals = np.array(doa.grid.values, copy=True)

    return est_az_deg, grid_az, grid_vals


# ─── Trial list construction ─────────────────────────────────────────────────

def build_trial_list(azimuths, rt60s, snrs, seeds):
    seen = set()
    trials = []

    def _add(rt60, snr_db, diffuse, az, seed):
        key = (rt60, snr_db, diffuse, az, seed)
        if key not in seen:
            seen.add(key)
            trials.append(dict(
                rt60=rt60, snr_db=snr_db, diffuse=diffuse,
                true_az_deg=az, seed=seed,
            ))

    for rt60 in rt60s:
        for az in azimuths:
            for seed in seeds:
                _add(rt60, RT60_SWEEP_SNR, False, float(az), seed)

    for snr in snrs:
        for az in azimuths:
            for seed in seeds:
                _add(SNR_SWEEP_RT60, snr, False, float(az), seed)

    for az in azimuths:
        for seed in seeds:
            _add(DIFFUSE_RT60, DIFFUSE_SNR, True, float(az), seed)

    return trials


# ─── CSV I/O ──────────────────────────────────────────────────────────────────

def load_completed(csv_path):
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, "r", newline="") as f:
        for r in csv.DictReader(f):
            done.add((
                r["geometry"], float(r["rt60"]), float(r["snr_db"]),
                r["diffuse"] == "True", float(r["true_az_deg"]), int(r["seed"]),
            ))
    return done


def flush_rows(csv_path, rows):
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


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
                est_az_deg=float(r["est_az_deg"]),
                abs_error_deg=float(r["abs_error_deg"]),
                seed=int(r["seed"]),
            ))
    return rows


# ─── Simulation loop ─────────────────────────────────────────────────────────

def run_all_trials(trial_list, out_dir):
    csv_path = os.path.join(out_dir, "metrics.csv")
    spectra_path = os.path.join(out_dir, "spectra.json")

    done = load_completed(csv_path)
    center = ROOM_DIM / 2.0
    geometries = {name: build_geometry(name, center) for name in GEOMETRIES}

    total = len(trial_list) * len(GEOMETRIES)
    pending = []
    spectra = {}
    t0 = time.time()
    completed = 0
    skipped = 0

    for trial in trial_list:
        for geo_name in GEOMETRIES:
            completed += 1
            key = (geo_name, trial["rt60"], trial["snr_db"],
                   trial["diffuse"], trial["true_az_deg"], trial["seed"])
            if key in done:
                skipped += 1
                continue

            est, grid_az, grid_vals = run_single_trial(
                geometries[geo_name], trial["true_az_deg"], trial["rt60"],
                trial["snr_db"], ROOM_DIM, trial["diffuse"], trial["seed"],
            )
            if est is None:
                skipped += 1
                done.add(key)
                continue
            err = abs_angular_error(est, trial["true_az_deg"])

            pending.append(dict(
                geometry=geo_name,
                rt60=trial["rt60"],
                snr_db=trial["snr_db"],
                diffuse=trial["diffuse"],
                true_az_deg=trial["true_az_deg"],
                est_az_deg=round(est, 4),
                abs_error_deg=round(err, 4),
                seed=trial["seed"],
            ))
            done.add(key)

            is_cherry = (
                trial["true_az_deg"] == 60.0
                and trial["seed"] == 0
                and (
                    (abs(trial["rt60"] - 1.0) < 1e-9
                     and abs(trial["snr_db"] - RT60_SWEEP_SNR) < 1e-9
                     and not trial["diffuse"])
                    or trial["diffuse"]
                )
            )
            if is_cherry:
                tag = "diffuse" if trial["diffuse"] else f"rt60={trial['rt60']:.1f}"
                spectra[f"{geo_name}_{tag}"] = dict(
                    grid_az=grid_az.tolist(),
                    grid_vals=grid_vals.tolist(),
                    true_az_deg=trial["true_az_deg"],
                    est_az_deg=est,
                    geometry=geo_name,
                    rt60=trial["rt60"],
                    diffuse=trial["diffuse"],
                )

            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - t0
                new_done = completed - skipped
                rate = new_done / max(elapsed, 0.001)
                remaining = (total - completed) / max(rate, 0.001) if rate > 0 else 0
                print(
                    f"  [{completed}/{total}]  {rate:.1f} trials/s  "
                    f"~{remaining:.0f}s left  |  "
                    f"{geo_name:5s} rt60={trial['rt60']:.1f} "
                    f"snr={trial['snr_db']:3.0f}dB "
                    f"{'DIFF ' if trial['diffuse'] else '     '}"
                    f"az={trial['true_az_deg']:5.0f}°  err={err:.1f}°"
                )
                if pending:
                    flush_rows(csv_path, pending)
                    pending.clear()

    if pending:
        flush_rows(csv_path, pending)

    if spectra:
        with open(spectra_path, "w") as f:
            json.dump(spectra, f)

    elapsed = time.time() - t0
    new_count = completed - skipped
    print(f"\nSimulation complete: {new_count} new trials in {elapsed:.1f}s "
          f"({skipped} skipped from previous run)")


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _filter(rows, **kwargs):
    out = rows
    for k, v in kwargs.items():
        if isinstance(v, float):
            out = [r for r in out if abs(r[k] - v) < 1e-9]
        else:
            out = [r for r in out if r[k] == v]
    return out


def _group_by_azimuth(subset):
    az_map = {}
    for r in subset:
        az_map.setdefault(r["true_az_deg"], []).append(r["abs_error_deg"])
    azimuths = sorted(az_map)
    means = np.array([np.mean(az_map[a]) for a in azimuths])
    return np.array(azimuths), means


# ─── Plot: DOA error vs azimuth (polar) ──────────────────────────────────────

def plot_doa_error_vs_azimuth(rows, rt60, snr_db, diffuse, out_dir, suffix=""):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

    max_err = 0
    for geo in GEOMETRIES:
        subset = _filter(rows, geometry=geo, rt60=float(rt60),
                         snr_db=float(snr_db), diffuse=diffuse)
        if not subset:
            continue
        azimuths, means = _group_by_azimuth(subset)
        az_rad = np.deg2rad(np.append(azimuths, azimuths[0]))
        vals = np.append(means, means[0])
        ax.plot(az_rad, vals, label=geo, color=GEO_COLORS[geo], linewidth=2)
        max_err = max(max_err, vals.max())

    cond = "Anechoic" if rt60 == 0 else f"RT60={rt60:.1f}s"
    cond += f", SNR={snr_db}dB"
    if diffuse:
        cond += " + diffuse noise"
    ax.set_title(f"Absolute DOA Error vs Azimuth\n{cond}", pad=20, fontsize=12)
    ax.set_rlabel_position(135)
    if max_err > 0:
        ax.set_ylim(0, max(max_err * 1.15, 5))
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))

    fname = f"error_vs_azimuth{suffix}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    {fname}")


# ─── Plot: mean error vs RT60 ────────────────────────────────────────────────

def plot_mean_error_vs_rt60(rows, out_dir):
    subset = _filter(rows, snr_db=float(RT60_SWEEP_SNR), diffuse=False)
    if not subset:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for geo in GEOMETRIES:
        geo_rows = _filter(subset, geometry=geo)
        if not geo_rows:
            continue
        rt60_map = {}
        for r in geo_rows:
            rt60_map.setdefault(r["rt60"], []).append(r["abs_error_deg"])
        rt60s = sorted(rt60_map)
        means = np.array([np.mean(rt60_map[v]) for v in rt60s])
        stds = np.array([np.std(rt60_map[v]) for v in rt60s])
        ax.plot(rt60s, means, "o-", label=geo, color=GEO_COLORS[geo], linewidth=2)
        ax.fill_between(rt60s, np.maximum(means - stds, 0), means + stds,
                        color=GEO_COLORS[geo], alpha=0.12)

    ax.set_xlabel("RT60 (s)")
    ax.set_ylabel("Mean absolute DOA error (°)")
    ax.set_title(f"DOA Error vs Reverberation  (SNR = {RT60_SWEEP_SNR} dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fname = "mean_error_vs_rt60.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    {fname}")


# ─── Plot: success rate vs RT60 ──────────────────────────────────────────────

def plot_success_rate_vs_rt60(rows, out_dir, threshold=5.0):
    subset = _filter(rows, snr_db=float(RT60_SWEEP_SNR), diffuse=False)
    if not subset:
        return

    rt60_vals = sorted({r["rt60"] for r in subset})
    fig, ax = plt.subplots(figsize=(8, 5))
    n_geo = len(GEOMETRIES)
    width = 0.8 / n_geo
    x = np.arange(len(rt60_vals))

    for i, geo in enumerate(GEOMETRIES):
        rates = []
        for rt in rt60_vals:
            group = _filter(subset, geometry=geo, rt60=rt)
            if group:
                ok = sum(1 for r in group if r["abs_error_deg"] < threshold)
                rates.append(100.0 * ok / len(group))
            else:
                rates.append(0.0)
        offset = (i - (n_geo - 1) / 2) * width
        ax.bar(x + offset, rates, width, label=geo, color=GEO_COLORS[geo])

    ax.set_xlabel("RT60 (s)")
    ax.set_ylabel(f"Success rate  (error < {threshold}°)  %")
    ax.set_title(f"DOA Success Rate vs Reverberation  (SNR = {RT60_SWEEP_SNR} dB)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in rt60_vals])
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)

    fname = "success_rate_vs_rt60.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    {fname}")


# ─── Plot: mean error vs SNR ─────────────────────────────────────────────────

def plot_mean_error_vs_snr(rows, out_dir):
    subset = _filter(rows, rt60=float(SNR_SWEEP_RT60), diffuse=False)
    if not subset:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for geo in GEOMETRIES:
        geo_rows = _filter(subset, geometry=geo)
        if not geo_rows:
            continue
        snr_map = {}
        for r in geo_rows:
            snr_map.setdefault(r["snr_db"], []).append(r["abs_error_deg"])
        snrs = sorted(snr_map)
        means = np.array([np.mean(snr_map[s]) for s in snrs])
        stds = np.array([np.std(snr_map[s]) for s in snrs])
        ax.plot(snrs, means, "o-", label=geo, color=GEO_COLORS[geo], linewidth=2)
        ax.fill_between(snrs, np.maximum(means - stds, 0), means + stds,
                        color=GEO_COLORS[geo], alpha=0.12)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mean absolute DOA error (°)")
    ax.set_title(f"DOA Error vs SNR  (RT60 = {SNR_SWEEP_RT60} s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fname = "mean_error_vs_snr.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    {fname}")


# ─── Plot: summary heatmap ───────────────────────────────────────────────────

def plot_summary_heatmap(rows, out_dir):
    conditions = []
    labels = []

    rt60_in_data = sorted({r["rt60"] for r in rows
                           if not r["diffuse"]
                           and abs(r["snr_db"] - RT60_SWEEP_SNR) < 1e-9})
    for rt in rt60_in_data:
        conditions.append(dict(rt60=rt, snr_db=RT60_SWEEP_SNR, diffuse=False))
        labels.append("Anechoic" if rt == 0 else f"RT60={rt:.1f}s")

    snr_in_data = sorted({r["snr_db"] for r in rows
                          if not r["diffuse"]
                          and abs(r["rt60"] - SNR_SWEEP_RT60) < 1e-9}, reverse=True)
    for snr in snr_in_data:
        if abs(snr - RT60_SWEEP_SNR) < 1e-9:
            continue
        conditions.append(dict(rt60=SNR_SWEEP_RT60, snr_db=snr, diffuse=False))
        labels.append(f"SNR={int(snr)}dB")

    if any(r["diffuse"] for r in rows):
        conditions.append(dict(rt60=DIFFUSE_RT60, snr_db=DIFFUSE_SNR, diffuse=True))
        labels.append("Diffuse")

    if not conditions:
        return

    mat = np.full((len(GEOMETRIES), len(conditions)), np.nan)
    for j, cond in enumerate(conditions):
        for i, geo in enumerate(GEOMETRIES):
            group = _filter(rows, geometry=geo, rt60=float(cond["rt60"]),
                            snr_db=float(cond["snr_db"]), diffuse=cond["diffuse"])
            if group:
                mat[i, j] = np.mean([r["abs_error_deg"] for r in group])

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.1), 3.5))
    vmax = np.nanmax(mat) if np.nanmax(mat) > 0 else 10
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=vmax)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, fontsize=9, rotation=40, ha="right")
    ax.set_yticks(range(len(GEOMETRIES)))
    ax.set_yticklabels(GEOMETRIES, fontsize=11)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                color = "white" if mat[i, j] > vmax * 0.55 else "black"
                ax.text(j, i, f"{mat[i, j]:.1f}°", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    ax.set_title("Mean DOA Error (°) — Geometry × Condition", fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, label="Mean error (°)", shrink=0.9, pad=0.02)

    fname = "summary_heatmap.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    {fname}")


# ─── Plot: SRP spatial spectrum ───────────────────────────────────────────────

def plot_srp_spectrum(grid_az, grid_vals, true_az_deg, est_az_deg,
                      geo_name, condition_str, out_dir, fname):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))

    vals = grid_vals / (grid_vals.max() + 1e-12)
    az_plot = np.append(grid_az, grid_az[0])
    vals_plot = np.append(vals, vals[0])

    ax.fill(az_plot, vals_plot, alpha=0.25, color=GEO_COLORS.get(geo_name, "#1f77b4"))
    ax.plot(az_plot, vals_plot, color=GEO_COLORS.get(geo_name, "#1f77b4"), linewidth=1.5)

    true_rad = np.deg2rad(true_az_deg)
    ax.plot([true_rad, true_rad], [0, 1], "--", color="green", linewidth=2.5,
            label=f"True ({true_az_deg:.0f}°)")
    est_rad = np.deg2rad(est_az_deg)
    ax.plot([est_rad, est_rad], [0, 1], "-", color="red", linewidth=2.5,
            label=f"Est ({est_az_deg:.1f}°)")

    ax.set_rlim(0, 1.05)
    ax.set_title(f"SRP-PHAT Spatial Spectrum\n{geo_name} — {condition_str}",
                 pad=20, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=9)

    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    {fname}")


# ─── Plot: room scene ────────────────────────────────────────────────────────

def plot_room_scene(room_dim, out_dir):
    center = room_dim / 2.0
    geometries = {n: build_geometry(n, center) for n in GEOMETRIES}

    true_az = 60.0
    az_rad = np.deg2rad(true_az)
    src = center + SOURCE_DISTANCE * np.array([np.cos(az_rad), np.sin(az_rad)])
    diff_pos = perimeter_positions(room_dim, N_DIFFUSE)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [2, 1]})

    # ── left: full room ──
    ax = axes[0]
    rect = plt.Rectangle((0, 0), room_dim[0], room_dim[1],
                          fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(rect)

    dx = [p[0] for p in diff_pos]
    dy = [p[1] for p in diff_pos]
    ax.scatter(dx, dy, c="gray", s=40, marker="x", alpha=0.6, label="Noise sources")
    ax.scatter(*src, c="red", s=150, marker="*", zorder=5, label="Drone")

    for name, R in geometries.items():
        ax.scatter(R[0], R[1], c=GEO_COLORS[name], s=35,
                   marker=GEO_MARKERS[name], label=f"{name} mics", zorder=4)

    arrow_len = 2.5
    ax.annotate("", xy=(center[0] + arrow_len * np.cos(az_rad),
                        center[1] + arrow_len * np.sin(az_rad)),
                xytext=center,
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
    ax.text(center[0] + (arrow_len + 0.4) * np.cos(az_rad),
            center[1] + (arrow_len + 0.4) * np.sin(az_rad),
            f"true {true_az:.0f}°", fontsize=9, color="red", ha="center")

    ax.set_xlim(-0.5, room_dim[0] + 0.5)
    ax.set_ylim(-0.5, room_dim[1] + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Room Scene  (20 m × 15 m exhibition hall)", fontsize=12)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.15)

    # ── right: zoomed array center ──
    ax2 = axes[1]
    zoom = 0.25
    for name, R in geometries.items():
        ax2.scatter(R[0] - center[0], R[1] - center[1], c=GEO_COLORS[name],
                    s=60, marker=GEO_MARKERS[name], label=name, zorder=4)
    ax2.set_xlim(-zoom, zoom)
    ax2.set_ylim(-zoom, zoom)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Δx (m)")
    ax2.set_ylabel("Δy (m)")
    ax2.set_title("Array Geometries (zoomed)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fname = "room_scene.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    {fname}")


# ─── Plot orchestrator ────────────────────────────────────────────────────────

def generate_all_plots(out_dir):
    csv_path = os.path.join(out_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        print("No results CSV found — run simulation first.")
        return

    print("\nGenerating plots...")
    rows = load_csv(csv_path)

    rt60_vals = sorted({r["rt60"] for r in rows
                        if not r["diffuse"]
                        and abs(r["snr_db"] - RT60_SWEEP_SNR) < 1e-9})
    for rt in rt60_vals:
        plot_doa_error_vs_azimuth(rows, rt, RT60_SWEEP_SNR, False, out_dir,
                                  suffix=f"_rt60_{rt:.1f}")

    if any(r["diffuse"] for r in rows):
        plot_doa_error_vs_azimuth(rows, DIFFUSE_RT60, DIFFUSE_SNR, True, out_dir,
                                  suffix="_diffuse")

    plot_mean_error_vs_rt60(rows, out_dir)
    plot_success_rate_vs_rt60(rows, out_dir)
    plot_mean_error_vs_snr(rows, out_dir)
    plot_summary_heatmap(rows, out_dir)

    spectra_path = os.path.join(out_dir, "spectra.json")
    if os.path.exists(spectra_path):
        with open(spectra_path, "r") as f:
            spectra = json.load(f)
        for sdata in spectra.values():
            cond = "diffuse" if sdata["diffuse"] else f"RT60={sdata['rt60']:.1f}s"
            safe = cond.replace("=", "").replace(".", "p")
            fname = f"srp_spectrum_{sdata['geometry']}_{safe}.png"
            plot_srp_spectrum(
                np.array(sdata["grid_az"]), np.array(sdata["grid_vals"]),
                sdata["true_az_deg"], sdata["est_az_deg"],
                sdata["geometry"], cond, out_dir, fname,
            )

    plot_room_scene(ROOM_DIM, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DOA Array Geometry Comparison — UCA vs Cross vs ULA")
    parser.add_argument("--test", action="store_true",
                        help="Quick run with a small parameter subset")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulation, regenerate plots from existing CSV")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.test:
        azimuths = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
        rt60s = [0.0, 1.0]
        snrs = [10, 0]
        seeds = [0]
    else:
        azimuths = AZIMUTHS_FULL
        rt60s = RT60S_FULL
        snrs = SNRS_FULL
        seeds = SEEDS_FULL

    if not args.plot_only:
        trials = build_trial_list(azimuths, rt60s, snrs, seeds)
        n_total = len(trials) * len(GEOMETRIES)
        print(f"Conditions: {len(trials)}  ×  {len(GEOMETRIES)} geometries  "
              f"=  {n_total} trials")
        print(f"Freq band: {FMIN}–{FMAX} Hz  |  Room: "
              f"{ROOM_DIM[0]:.0f}×{ROOM_DIM[1]:.0f} m  |  "
              f"max_order cap: {MAX_ORDER_CAP}\n")
        run_all_trials(trials, RESULTS_DIR)

    generate_all_plots(RESULTS_DIR)


if __name__ == "__main__":
    main()
