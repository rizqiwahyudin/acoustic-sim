"""Heimdall characterisation tests (MAX78002/docs/CHARACTERISATION_PLAN.md), run through the GUI backend.

    python scripts/characterise.py n1     noise floor vs number of microphones (speaker off)
    python scripts/characterise.py n2     noise floor vs steering direction (speaker off)
    python scripts/characterise.py a3     beam-pattern cuts through the speaker, fitted with theory
    python scripts/characterise.py a4     pointing across speaker positions: slope, offset, residuals
    python scripts/characterise.py a8     front-to-back ratio
    python scripts/characterise.py l1     Live (bank) direction vs speaker position, against the firmware's centroid
    python scripts/characterise.py r1     room reverberation time per octave (interrupted noise)
    python scripts/characterise.py s1     system sensitivity and equivalent noise with an SPL meter
    python scripts/characterise.py a9     path gain, directivity index (diffuse field), signal and noise gain
    python scripts/characterise.py a10    spatial aliasing: grating lobes at 5 and 8 kHz (beam FIR swapped for a high-pass)
    python scripts/characterise.py p2     audio-path latency after a steer, and the guard_us to use
    python scripts/characterise.py s3     linearity, and whether the DSP detector word follows amplitude or power
    python scripts/characterise.py l5     detection margin: Live against continuous scanning, sound A in 6 dB steps
    python scripts/characterise.py l6     two sources: drone clip on the speaker, speech or music on a phone
    python scripts/characterise.py t1     tonal detector on silence, a 200 Hz series, the drone clip, crowd (--real-drone)
    python scripts/characterise.py d1     time constant of the scan level detector (Study check of a detector preset)

The speaker tests say where to put the speaker. a3 and l1 then find it with the steered beam (--truth array,
the default: rough placement is enough, no tape); with --truth measured they ask for the measured position.
a4 always needs measured positions (tape or a phone measuring app). Needs sim_server.py
connected to the board, the speaker as the Windows default output and a quiet room. Writes every capture
(captures.csv), summary.json and figures to --out, and restores gains, settings, Ch1 and the grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from array_theory import (band_directivity_index_db, band_frequencies, diffuse_power,  # noqa: E402
                          first_null_half_width, fit_cut, steered_response, steering_cut_db)
from board_regression import WS_URL, Board, apply_run_edits, connect, http, play, record  # noqa: E402
from gui_run import ask_operator, run_edits  # noqa: E402
from heimdall_acoustic import default_contract_path, load_beam_contract  # noqa: E402
from live_log import analyse as live_log_analyse, save_log  # noqa: E402
from study_session import (FIELDS, MCU_BEAM, WINDOWS, Results, Session, make_signals, power,  # noqa: E402
                           power_mean_db, refine_peak_2d, signal_db, snr_db)

BAND_HZ = (500.0, 3500.0)
N1_COUNTS = (0, 1, 2, 4, 8, 11, 16, 22, 32, 44)
N2_ANGLES = (-60.0, -30.0, 0.0, 30.0, 60.0)


def l90(levels):
    """Level exceeded 90% of the time (dB): the usual measure of background noise under intermittent sounds."""
    return float(np.percentile(levels, 10))


def summarise(values):
    """Mean and standard error of per-cycle results."""
    values = np.asarray(values, float)
    se = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return float(values.mean()), se


def find(data, key):
    """First value stored under `key` anywhere in a nested contract."""
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for value in data.values():
            found = find(value, key)
            if found is not None:
                return found
    return None


class Array:
    """Deployed geometry and the three noise models, averaged over the 0.5-3.5 kHz measurement band."""

    def __init__(self):
        self.contract = load_beam_contract()
        self.freqs = np.linspace(*BAND_HZ, 61)

    def diffuse_sum(self, members):
        """Band mean of sum_ij sinc(k d_ij): beam power from diffuse noise, one microphone = 1."""
        return float(np.mean([diffuse_power(self.contract, f, members=members) for f in self.freqs]))

    def steered(self, azimuth_deg, elevation_deg):
        """(common-mode, diffuse) beam power steered to a direction, relative to steering at 0,0, in dB."""
        common = steered_response(self.contract, 0.0, 0.0, azimuth_deg, elevation_deg, self.freqs)[0]
        diffuse = np.mean([diffuse_power(self.contract, f, azimuth_deg, elevation_deg) for f in self.freqs])
        diffuse0 = np.mean([diffuse_power(self.contract, f) for f in self.freqs])
        return 10 * np.log10(common), 10 * np.log10(diffuse / diffuse0)


def write_words(words):
    """Write {address: word} through the backend (allow-listed, read back by the firmware)."""
    if not words:
        return
    status, body = http("POST", "/dsp/write", {"writes": [{"address": a, "word": w} for a, w in words.items()]})
    if status != 200:
        raise RuntimeError(f"DSP write refused ({status}): {body}")


def reload_dsp(board):
    """I reloads the taper and tables and puts the deployment grid back."""
    board.stop()
    board.command("I")
    board.pump(4.0, lambda s: (s.get("tables") or {}).get("verified") == (s.get("tables") or {}).get("words"))
    apply_run_edits(board)


def n1(session, args, out, array):
    """Noise floor (L90) vs number of unmuted microphones, beam at 0,0, speaker off, in interleaved cycles."""
    contract = json.loads(Path(default_contract_path()).read_text(encoding="utf-8"))
    addresses = [int(a) for a in find(contract, "gain_addresses")]
    unity = [int(w) for w in find(contract, "gain_fixpt_8_24")]
    order = [int(m) for m in np.random.default_rng(0).permutation(len(addresses))]
    sector = session.speaker_sector(0, 0)
    current = dict(zip(addresses, unity))
    cycles = {count: [] for count in N1_COUNTS}
    readings = {count: [] for count in N1_COUNTS}
    try:
        with session.settings(**MCU_BEAM):
            for cycle in range(args.cycles):
                for count in (N1_COUNTS if cycle % 2 == 0 else N1_COUNTS[::-1]):
                    on = set(order[:count])
                    wanted = {a: (unity[m] if m in on else 0) for m, a in enumerate(addresses)}
                    session.board.stop()
                    write_words({a: w for a, w in wanted.items() if current.get(a) != w})
                    current = wanted
                    capture = session.capture("n1", f"{count} microphones, cycle {cycle + 1}", sector,
                                              args.seconds or 4.0, value=count)
                    cycles[count].append(l90(capture["levels"]))
                    readings[count] += capture["levels"]
                    print(f"  cycle {cycle + 1}, {count:2d} microphones: L90 {cycles[count][-1]:7.2f} dBFS", flush=True)
    finally:
        session.board.stop()
        write_words({a: w for a, w in zip(addresses, unity) if current.get(a) != w})
        reload_dsp(session.board)

    rows = []
    for count in N1_COUNTS:
        level, se = summarise(cycles[count])
        rows.append({"count": count, "level_db": level, "se_db": se, "cycles_db": cycles[count],
                     "median_db": float(np.median(readings[count])), "mean_db": power_mean_db(readings[count]),
                     "diffuse_sum": array.diffuse_sum(order[:count]) if count else 0.0})
    floor = power(rows[0]["level_db"])
    full = rows[-1]
    full_excess = float(power(full["level_db"]) - floor)
    for row in rows[1:]:
        excess = float(power(row["level_db"]) - floor)
        row["excess_db"] = 10 * np.log10(excess) if excess > 0 else None
        row["usable"] = bool(excess > floor)
        n, n_all = row["count"], full["count"]
        row["independent_db"] = 10 * np.log10(full_excess * n / n_all)
        row["common_db"] = 10 * np.log10(full_excess * (n / n_all) ** 2)
        row["diffuse_db"] = 10 * np.log10(full_excess * row["diffuse_sum"] / full["diffuse_sum"])
    usable = [r for r in rows[1:] if r["usable"] and r["excess_db"] is not None]
    slope = None
    if len(usable) >= 3:
        x = 10 * np.log10([r["count"] for r in usable])
        slope = float(np.polyfit(x, [r["excess_db"] for r in usable], 1)[0])
    session.values["n1"] = {"statistic": "L90 per cycle, mean over cycles", "link_floor_dbfs": rows[0]["level_db"],
                            "rows": rows, "slope": slope, "order": order}
    session.report.info("N1 link floor (all muted)", f"{rows[0]['level_db']:.1f} dBFS")
    session.report.info("N1 slope of excess vs 10 log10 N", "not enough points above the link floor"
                        if slope is None else f"{slope:.2f} (1 = independent per microphone, 2 = common to all)")
    # No model lets a subset read above the full array; when one does, a source in the room has a direction.
    loud = [r["count"] for r in rows[1:-1] if r["level_db"] > full["level_db"] + 3.0]
    session.values["n1"]["valid"] = not loud
    if loud:
        session.report.info("N1 NOT VALID", f"{loud} microphones read more than 3 dB above all {full['count']}: "
                                            "something in the room is on; repeat when it is quiet")
    figure_n1(rows, out)


def n2(session, args, out, array):
    """Noise floor (L90), all microphones on, steered along the azimuth and elevation axes, speaker off."""
    cycles = {}
    with session.settings(**MCU_BEAM):
        for cycle in range(args.cycles):
            for axis in ("azimuth", "elevation"):
                if axis == "azimuth":
                    spec = (N2_ANGLES[0], N2_ANGLES[-1], len(N2_ANGLES), 0.0, 0.0, 1)
                else:
                    spec = (0.0, 0.0, 1, N2_ANGLES[0], N2_ANGLES[-1], len(N2_ANGLES))
                azimuths, elevations = session.grid(*spec)
                sectors = range(len(N2_ANGLES)) if cycle % 2 == 0 else range(len(N2_ANGLES) - 1, -1, -1)
                for sector in sectors:
                    az = azimuths[sector] if axis == "azimuth" else azimuths[0]
                    el = elevations[0] if axis == "azimuth" else elevations[sector]
                    capture = session.capture("n2", f"{az:g}, {el:g}, cycle {cycle + 1}", sector, args.seconds or 4.0,
                                              value=az if axis == "azimuth" else el, azimuth_deg=az, elevation_deg=el)
                    cycles.setdefault((axis, az, el), []).append(l90(capture["levels"]))
                    print(f"  cycle {cycle + 1}, {axis} {az:6.1f}, {el:6.1f}: L90 {cycles[(axis, az, el)][-1]:7.2f} dBFS",
                          flush=True)
    rows = []
    for (axis, az, el), values in cycles.items():
        level, se = summarise(values)
        common, diffuse = array.steered(az, el)
        rows.append({"axis": axis, "azimuth_deg": az, "elevation_deg": el, "level_db": level, "se_db": se,
                     "cycles_db": values, "common_db": common, "diffuse_db": diffuse})
    rows.sort(key=lambda r: (r["axis"], r["azimuth_deg"] + r["elevation_deg"]))
    centre = float(np.mean([r["level_db"] for r in rows if r["azimuth_deg"] == 0 and r["elevation_deg"] == 0]))
    for row in rows:
        row["relative_db"] = row["level_db"] - centre
    away = [r for r in rows if abs(r["azimuth_deg"]) >= 30 or abs(r["elevation_deg"]) >= 30]
    measured, common = np.mean([r["relative_db"] for r in away]), np.mean([r["common_db"] for r in away])
    session.values["n2"] = {"statistic": "L90 per cycle, mean over cycles", "rows": rows, "centre_dbfs": centre,
                            "mean_off_axis_db": float(measured)}
    session.report.info("N2 floor at 0,0", f"{centre:.1f} dBFS")
    session.report.info("N2 mean change at 30-60 deg", f"{measured:+.1f} dB (independent or diffuse about 0, "
                                                        f"common to all channels {common:+.1f})")
    loudest = max(rows, key=lambda r: r["relative_db"])
    session.values["n2"]["valid"] = loudest["relative_db"] - loudest["diffuse_db"] <= 3.0
    if not session.values["n2"]["valid"]:
        session.report.info("N2 NOT VALID", f"{loudest['azimuth_deg']:g}, {loudest['elevation_deg']:g} reads "
                                            f"{loudest['relative_db']:+.1f} dB: a source in that direction is on")
    figure_n2(rows, out)


def figure_n1(rows, out):
    import matplotlib.pyplot as plt
    data = [r for r in rows[1:] if r["excess_db"] is not None]
    counts = [r["count"] for r in data]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(counts, [r["excess_db"] for r in data], yerr=[r["se_db"] for r in data], fmt="o", color="k",
                label="measured (link floor subtracted)")
    for key, label, style in (("independent_db", "independent per microphone: 10 log N", "--"),
                              ("common_db", "common to all channels: 20 log N", ":"),
                              ("diffuse_db", "diffuse room noise (this geometry)", "-")):
        ax.plot(counts, [r[key] for r in data], style, label=label)
    ax.axhline(rows[0]["level_db"], color="gray", lw=0.8)
    ax.text(1, rows[0]["level_db"] + 0.5, f"link floor {rows[0]['level_db']:.1f} dBFS", color="gray", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(counts, [str(c) for c in counts])
    ax.set_xlabel("microphones unmuted in the beam (beam at 0,0, speaker off)")
    ax.set_ylabel("beam noise above the link floor, L90 (dBFS, 0.5-3.5 kHz)")
    ax.set_title("N1: how the beam's noise floor grows with the number of microphones\n"
                 "models anchored at 44 microphones", fontsize=11)
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(out, "n1_noise_vs_microphones.png"), dpi=150, bbox_inches="tight")


def figure_n2(rows, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, axis in zip(axes, ("azimuth", "elevation")):
        data = [r for r in rows if r["axis"] == axis]
        x = [r["azimuth_deg"] if axis == "azimuth" else r["elevation_deg"] for r in data]
        ax.errorbar(x, [r["relative_db"] for r in data], yerr=[r["se_db"] for r in data], fmt="o-", color="k",
                    label="measured")
        ax.plot(x, [0] * len(x), "--", label="independent per microphone")
        ax.plot(x, [r["diffuse_db"] for r in data], "-", label="diffuse room noise")
        ax.plot(x, [r["common_db"] for r in data], ":", label="common to all channels")
        ax.set_xlabel(f"{axis} the beam is steered to (deg)")
        ax.set_title(f"steered in {axis}")
    axes[0].set_ylabel("beam noise floor (L90) relative to 0,0 (dB)")
    axes[0].legend(fontsize=9)
    fig.suptitle("N2: beam noise floor vs steering, speaker off (0.5-3.5 kHz)")
    fig.savefig(os.path.join(out, "n2_noise_vs_steering.png"), dpi=150, bbox_inches="tight")


# Speaker tests ------------------------------------------------------------------------------------

BANDS = {"B1k": (1000.0, 2.0, 48.0), "B2k": (2000.0, 1.0, 24.0), "B3k": (3150.0, 1.0, 24.0)}
A_BAND = np.linspace(500.0, 3500.0, 31)
OCTAVES = {"500 Hz": 500.0, "1 kHz": 1000.0, "2 kHz": 2000.0, "4 kHz": 4000.0}
L1_POSITIONS = "-30,0;-22.5,0;-15,0;-7.5,0;0,0;7.5,0;15,0;22.5,0;30,0;0,-15;0,-7.5;0,7.5;0,15"


def describe(nominal_az, nominal_el, distance_m):
    """Where a source at (az, el, distance) goes, in words for the operator."""
    a, e = np.radians(nominal_az), np.radians(nominal_el)
    right, up = distance_m * np.cos(e) * np.sin(a), distance_m * np.sin(e)
    side = (f"{abs(100 * right):.0f} cm {'right' if right >= 0 else 'left'} of the centre line (seen from behind "
            "the array)" if abs(right) >= 0.05 else "straight ahead")
    height = (f", {abs(100 * up):.0f} cm {'above' if up >= 0 else 'below'} the array centre" if abs(up) >= 0.05
              else ", at the height of the array centre")
    return f"{side}{height}, about {distance_m:g} m away, facing the array"


def place(nominal_az, nominal_el, distance_m):
    """Rough placement; the array then measures where the speaker is. Returns the nominal distance."""
    ask_operator(f"\nSpeaker roughly {describe(nominal_az, nominal_el, distance_m)}. Rest it on something rather than "
                 "holding it; close is fine, the array measures where it is. Enter when ready.")
    return distance_m


def parse_numbers(text, count):
    """Numbers as typed at a prompt; a decimal comma works ("72,5", "152,5 200 10")."""
    text = text.strip()
    parts = [text] if count == 1 else text.split() if " " in text else text.replace(";", ",").split(",")
    values = [float(part.strip(",;").replace(",", ".")) for part in parts if part.strip(",;")]
    if len(values) != count:
        raise ValueError(f"expected {count} numbers, got {len(values)}")
    return values


def ask_numbers(prompt, count=1, optional=False):
    """Ask until `count` numbers are typed; None for an empty answer when optional."""
    while True:
        text = ask_operator(prompt, numbers=count, optional=optional).strip()
        if optional and not text:
            return None
        try:
            return parse_numbers(text, count)
        except ValueError:
            print(f"  type {count} number{'s' if count > 1 else ''}" + (" or Enter" if optional else ""), flush=True)


def ask_position(nominal_az, nominal_el, distance_m):
    """Tell the operator where the speaker goes; return the measured (azimuth, elevation, distance)."""
    a, e = np.radians(nominal_az), np.radians(nominal_el)
    x, z, h = distance_m * np.cos(e) * np.sin(a), distance_m * np.cos(e) * np.cos(a), distance_m * np.sin(e)
    print(f"\nSpeaker to az {nominal_az:g}, el {nominal_el:g}: driver {100 * x:.1f} cm right, {100 * z:.1f} cm "
          f"forward, {100 * h:.1f} cm up from the array centre, facing the array.", flush=True)
    measured = ask_numbers("Measured right, forward, up in cm (Enter = as above): ", 3, optional=True)
    if measured:
        x, z, h = (v / 100.0 for v in measured)
    return position_angles(x, z, h)


def position_angles(right_m, forward_m, up_m):
    """(azimuth, elevation, distance) of a point, in the firmware's convention."""
    r = float(np.sqrt(right_m ** 2 + forward_m ** 2 + up_m ** 2))
    return float(np.degrees(np.arctan2(right_m, forward_m))), float(np.degrees(np.arcsin(up_m / r))), r


def parse_positions(text):
    return [tuple(float(v) for v in item.split(",")) for item in text.split(";") if item.strip()]


def axis_spec(axis, centre_az, centre_el, step, half):
    """DG arguments for a one-axis direction set centred on (centre_az, centre_el)."""
    count = int(round(2 * half / step)) + 1
    if axis == "azimuth":
        return centre_az - half, centre_az + half, count, centre_el, centre_el, 1
    return centre_az, centre_az, 1, centre_el - half, centre_el + half, count


def real_drift(values_db, errors_db, limit_db=0.5, z=2.326):
    """True when reference readings change by more than limit_db and by more than their scatter explains.

    Chi-square of the readings about their mean against the 99% point (Wilson-Hilferty, z = 2.326)."""
    values, errors = np.asarray(values_db, float), np.maximum(np.asarray(errors_db, float), 1e-3)
    if len(values) < 2 or np.ptp(values) <= limit_db:
        return False
    dof = len(values) - 1
    chi2 = float(np.sum(((values - np.average(values, weights=errors ** -2)) / errors) ** 2))
    return chi2 > dof * (1 - 2 / (9 * dof) + z * np.sqrt(2 / (9 * dof))) ** 3


def measure_cut(session, axis, centre, step, half, signal, repeats, seconds, rng, label):
    """One silent pass, then shuffled repeats with the speaker on and the centre re-measured every 10
    points (its drift is corrected and reported). Signal = on minus off, in power."""
    azimuths, elevations = session.grid(*axis_spec(axis, centre[0], centre[1], step, half))
    angles = np.array(azimuths if axis == "azimuth" else elevations, float)
    n, middle = len(angles), len(angles) // 2

    def capture(s, text, duration):
        return session.capture(label, f"{axis} {angles[s]:g} {text}", s, duration, value=float(angles[s]),
                               azimuth_deg=azimuths[s] if axis == "azimuth" else azimuths[0],
                               elevation_deg=elevations[0] if axis == "azimuth" else elevations[s])

    session.sound(None)
    off = np.array([capture(s, "silent", 0.5)["mean_db"] for s in range(n)])
    session.sound(signal)
    levels, errors, drifts, real = [], [], [], []
    for repeat in range(repeats):
        order = [int(s) for s in rng.permutation(n)]
        on, refs = {}, [(0, capture(middle, "reference", seconds))]
        for i, s in enumerate(order):
            on[s] = capture(s, f"repeat {repeat + 1}", seconds)
            if (i + 1) % 10 == 0 or i + 1 == n:
                refs.append((i + 1, capture(middle, "reference", seconds)))
        steps = [i for i, _ in refs]
        values = [c["mean_db"] for _, c in refs]
        drifts.append(float(max(values) - min(values)))
        real.append(real_drift(values, [c["std_db"] / np.sqrt(c["n"]) for _, c in refs]))
        place = {s: i + 0.5 for i, s in enumerate(order)}
        levels.append([on[s]["mean_db"] - (np.interp(place[s], steps, values) - np.mean(values)) for s in range(n)])
        errors.append([on[s]["std_db"] / np.sqrt(on[s]["n"]) for s in range(n)])
    session.sound(None)
    runs = np.array([[signal_db(level, off[s]) for s, level in enumerate(run)] for run in levels])
    runs[~np.isfinite(runs)] = np.nan
    within = np.sqrt(np.mean(np.square(errors), axis=0) / repeats)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        between = np.nanstd(runs, axis=0, ddof=1) / np.sqrt(repeats) if repeats > 1 else np.zeros(n)
        mean = np.nanmean(runs, axis=0)
    snr = np.array([snr_db(np.mean(np.array(levels)[:, s]), off[s]) for s in range(n)])
    return {"angles": angles, "signal_db": mean, "se_db": np.fmax(within, np.nan_to_num(between)),
            "snr_db": snr, "repeats_db": runs, "drift_db": drifts, "drift_real": real, "silent_db": off,
            "on_db": np.mean(np.array(levels), axis=0)}


def fit_axis(cut, centre_value, rel_model, db_model, shifts=None):
    """Fit the theory to the averaged cut; widen the errors to the spread of per-repeat fits if larger."""
    rel = cut["angles"] - centre_value
    ok = np.isfinite(cut["signal_db"]) & (cut["snr_db"] >= 10.0)
    if ok.sum() < 4:
        return None
    result = fit_cut(rel[ok], cut["signal_db"][ok], cut["se_db"][ok], rel_model, db_model, shifts=shifts)
    if result is None:
        return None
    shifts_found, widths = [], []
    for run in cut["repeats_db"]:
        good = ok & np.isfinite(run)
        single = fit_cut(rel[good], run[good], cut["se_db"][good] * np.sqrt(len(cut["repeats_db"])),
                         rel_model, db_model, shifts=shifts)
        if single:
            shifts_found.append(single["shift_deg"])
            widths.append(single["width_3db_deg"])
    if len(shifts_found) > 1:
        result["shift_sigma_deg"] = max(result["shift_sigma_deg"] or 0.0,
                                        float(np.std(shifts_found, ddof=1) / np.sqrt(len(shifts_found))))
        result["width_sigma_deg"] = max(result["width_sigma_deg"] or 0.0,
                                        float(np.std(widths, ddof=1) / np.sqrt(len(widths))))
    result["repeat_shifts_deg"], result["repeat_widths_deg"] = shifts_found, widths
    return result


def sidelobe_bound(cut, centre_value, rel_model, db_model):
    """Highest measured point outside the theoretical first nulls, dB re the peak (an upper bound indoors)."""
    minima = [i for i in range(1, len(db_model) - 1)
              if db_model[i] < db_model[i - 1] and db_model[i] <= db_model[i + 1] and db_model[i] < -6]
    left = max((rel_model[i] for i in minima if rel_model[i] < 0), default=None)
    right = min((rel_model[i] for i in minima if rel_model[i] > 0), default=None)
    rel = cut["angles"] - centre_value
    if left is None or right is None:
        return None
    outside = ((rel < left) | (rel > right)) & np.isfinite(cut["signal_db"])
    if not outside.any():
        return None
    return float(np.nanmax(cut["signal_db"][outside]) - np.nanmax(cut["signal_db"]))


def locate(session, contract, centre, distance_m, repeats, seconds, rng, label):
    """Find the speaker with the steered beam: 7x7 search at 4 deg around `centre`, then 1 deg cuts through its
    peak fitted with theory. Returns ((az, el), (sigma_az, sigma_el), fits, coarse peak)."""
    session.sound("B2k")
    azimuths, elevations = session.grid(centre[0] - 12, centre[0] + 12, 7, centre[1] - 12, centre[1] + 12, 7)
    grid = np.zeros((7, 7))
    for s in range(49):
        r, c = divmod(s, 7)
        grid[r, c] = session.capture(label, f"search {azimuths[c]:g}, {elevations[r]:g}", s, 0.5,
                                     azimuth_deg=azimuths[c], elevation_deg=elevations[r])["mean_db"]
    coarse = refine_peak_2d(grid, azimuths, elevations)
    found, sigma, fits = [], [], {}
    for i, axis in enumerate(("azimuth", "elevation")):
        cut = measure_cut(session, axis, coarse, 1.0, 12.0, "B2k", repeats, seconds, rng, label)
        rel_model, db_model = steering_cut_db(contract, coarse, axis, band_frequencies(2000.0), distance_m)
        fits[axis] = fit_axis(cut, coarse[i], rel_model, db_model)
        found.append(coarse[i] + fits[axis]["shift_deg"] if fits[axis] else float("nan"))
        sigma.append(fits[axis]["shift_sigma_deg"] if fits[axis] else float("nan"))
    session.restore_grid()
    return tuple(found), tuple(sigma), fits, coarse


def a3(session, args, out, array):
    """Azimuth and elevation cuts through the speaker per band, fitted with the theory."""
    contract = load_beam_contract()
    rng = np.random.default_rng(args.seed)
    if args.truth == "array":
        distance = place(0.0, 0.0, args.distance)
        with session.settings(**MCU_BEAM):
            (az, el), _, _, _ = locate(session, contract, (0.0, 0.0), distance, 1, 1.0, rng, "a3 locate")
        if not (np.isfinite(az) and np.isfinite(el)):
            raise RuntimeError("the steered beam did not find the speaker; is it on and roughly in front?")
        print(f"  speaker found at {az:+.2f}, {el:+.2f} deg", flush=True)
    else:
        az, el, distance = ask_position(0.0, 0.0, args.distance)
    results, plots = {}, []
    with session.settings(**MCU_BEAM):
        for band in args.bands.split(","):
            centre_hz, step, half = BANDS[band]
            for axis in ("azimuth", "elevation"):
                cut = measure_cut(session, axis, (az, el), step, half, band, args.repeats, args.seconds or 1.0, rng,
                                  "a3")
                centre_value = az if axis == "azimuth" else el
                rel_model, db_model = steering_cut_db(contract, (az, el), axis, band_frequencies(centre_hz), distance)
                fit = fit_axis(cut, centre_value, rel_model, db_model, shifts=np.arange(-10.0, 10.001, 0.02))
                psl = sidelobe_bound(cut, centre_value, rel_model, db_model)
                name = f"{band} {axis}"
                results[name] = {"fit": fit, "sidelobe_bound_db": psl, "drift_db": cut["drift_db"],
                                 "drift_real": cut["drift_real"],
                                 "angles": cut["angles"], "signal_db": cut["signal_db"], "se_db": cut["se_db"],
                                 "snr_db": cut["snr_db"]}
                plots.append((name, cut, centre_value, rel_model, db_model, fit))
                if fit is None:
                    session.report.add(f"A3 {name} width", False, "not enough points 10 dB above the silent floor")
                    continue
                ok = abs(fit["width_3db_deg"] - fit["theory_width_3db_deg"]) <= 3 * (fit["width_sigma_deg"] or 0.0)
                session.report.add(f"A3 {name} width", ok and not fit["edge"],
                                   f"{fit['width_3db_deg']:.1f} +- {fit['width_sigma_deg']:.1f} deg, theory "
                                   f"{fit['theory_width_3db_deg']:.1f}; reduced chi2 {fit['reduced_chi2']:.1f}")
                session.report.info(f"A3 {name} peak", f"{fit['shift_deg']:+.2f} +- {fit['shift_sigma_deg']:.2f} deg "
                                    f"from the speaker; drift {max(cut['drift_db']):.2f} dB"
                                    + ("" if psl is None else f"; sidelobes <= {psl:.1f} dB (room-limited)"))
                if any(cut["drift_real"]):
                    session.report.info(f"A3 {name} drift", "over 0.5 dB and more than the reading scatter: "
                                                            "repeat this cut")
    session.values["a3"] = {"speaker": {"azimuth_deg": az, "elevation_deg": el, "distance_m": distance},
                            "cuts": results}
    figure_cuts(plots, out)


def a4(session, args, out, array):
    """Pointing at several speaker positions: 7x7 search, then 1 deg cuts through its peak, fitted."""
    contract = load_beam_contract()
    rng = np.random.default_rng(args.seed)
    rows = []
    print("A4 needs each speaker position measured (tape, or a phone measuring app): here the array cannot be "
          "its own reference.", flush=True)
    with session.settings(**MCU_BEAM):
        for nominal in parse_positions(args.positions):
            az, el, distance = ask_position(nominal[0], nominal[1], args.distance)
            found, sigma, fits, coarse = locate(session, contract, (az, el), distance, args.repeats,
                                                args.seconds or 1.0, rng, "a4")
            rows.append({"true_az": az, "true_el": el, "distance_m": distance, "measured_az": found[0],
                         "measured_el": found[1], "sigma_az": sigma[0], "sigma_el": sigma[1],
                         "coarse": list(coarse), "fits": fits})
            print(f"  true {az:+.2f}, {el:+.2f}: measured {found[0]:+.2f} +- {sigma[0]:.2f}, "
                  f"{found[1]:+.2f} +- {sigma[1]:.2f}", flush=True)
    lines = {}
    for axis, true, key, err, other in (("azimuth", "true_az", "measured_az", "sigma_az", "true_el"),
                                        ("elevation", "true_el", "measured_el", "sigma_el", "true_az")):
        pts = [r for r in rows if np.isfinite(r[key]) and abs(r[other]) < 5.0]
        if len(pts) < 3:
            continue
        x = np.array([r[true] for r in pts])
        y = np.array([r[key] for r in pts])
        s = np.sqrt(np.array([r[err] for r in pts]) ** 2 + args.alignment_sigma ** 2)
        (slope, offset), cov = np.polyfit(x, y, 1, w=1 / s, cov="unscaled")
        residual = y - (offset + slope * x)
        lines[axis] = {"slope": float(slope), "slope_sigma": float(np.sqrt(cov[0, 0])), "offset_deg": float(offset),
                       "offset_sigma_deg": float(np.sqrt(cov[1, 1])), "residuals_deg": residual.tolist(),
                       "sigma_deg": s.tolist()}
        session.report.add(f"A4 {axis} slope", abs(slope - 1) <= 3 * np.sqrt(cov[0, 0]),
                           f"{slope:.4f} +- {np.sqrt(cov[0, 0]):.4f} (1 = right scale and sign)")
        session.report.add(f"A4 {axis} offset", abs(offset) <= 0.5,
                           f"{offset:+.2f} +- {np.sqrt(cov[1, 1]):.2f} deg (alignment)")
        session.report.add(f"A4 {axis} residuals", bool(np.all(np.abs(residual) <= 3 * s)),
                           "largest " + f"{np.max(np.abs(residual)):.2f} deg, 3 sigma {3 * s[np.argmax(np.abs(residual))]:.2f}")
    session.values["a4"] = {"positions": rows, "lines": lines, "alignment_sigma_deg": args.alignment_sigma}
    figure_pointing(rows, lines, out)


def a8(session, args, out, array):
    """Beam at 0,0 with the speaker on the axis in front, then behind at the same distance."""
    sector = session.speaker_sector(0, 0)
    seconds = args.seconds or 3.0
    levels = {}
    with session.settings(**MCU_BEAM):
        for side in ("front", "behind"):
            ask_operator(f"\nA8: speaker on the array axis {side} the array, {args.distance:g} m from its centre, "
                         "facing it. Enter when ready.")
            session.sound(None)
            silent = session.capture("a8", f"{side} silent", sector, seconds)["mean_db"]
            for signal in ("B1k", "B2k", "A"):
                session.sound(signal)
                values = [signal_db(session.capture("a8", f"{side} {signal}", sector, seconds)["mean_db"], silent)
                          for _ in range(3)]
                levels[(side, signal)] = values
            session.sound(None)
    result = {}
    for signal in ("B1k", "B2k", "A"):
        front, behind = np.array(levels[("front", signal)]), np.array(levels[("behind", signal)])
        ratio = float(front.mean() - behind.mean())
        se = float(np.sqrt(front.var(ddof=1) / 3 + behind.var(ddof=1) / 3))
        result[signal] = {"front_to_back_db": ratio, "se_db": se}
        text = f"{ratio:+.1f} +- {se:.1f} dB (theory 0 for a flat array)"
        if signal == "B1k":
            session.report.add("A8 front to back, 1 kHz", abs(ratio) <= 3.0, text)
        else:
            session.report.info(f"A8 front to back, {signal}", text)
    session.values["a8"] = result


def bank_prediction(bank, contract, az, el, distance_m):
    """Where Live should report a source: the firmware's 3x3 centroid (beam_bank.c interpolate) applied to
    theory bank levels (RMS amplitudes, quiet background), 0.5-3.5 kHz."""
    baz, bel = np.asarray(bank["azimuth_deg"], float), np.asarray(bank["elevation_deg"], float)
    rows, columns = int(bank["rows"]), int(bank["columns"])
    amplitude = np.sqrt(steered_response(contract, az, el, baz, bel, A_BAND, distance_m))
    r0, c0 = divmod(int(np.argmax(amplitude)), columns)
    sx = sy = total = 0.0
    for r in range(max(0, r0 - 1), min(rows, r0 + 2)):
        for c in range(max(0, c0 - 1), min(columns, c0 + 2)):
            b = r * columns + c
            e = np.radians(bel[b])
            sx += amplitude[b] * np.cos(e) * np.sin(np.radians(baz[b]))
            sy += amplitude[b] * np.sin(e)
            total += amplitude[b]
    e = np.arcsin(sy / total)
    return float(np.degrees(np.arcsin(np.clip(sx / total / np.cos(e), -1, 1)))), float(np.degrees(e))


def live_trial(session, sound, listen):
    """Stare starts in silence (its background), the sound plays; median over the last 4 s of the peak beam
    (BANK_TARGET) and of the pattern fit (BANK_FIT, while there is a target): {"peak": (az, el) or None, "fit": ...}."""
    board, found, fitted = session.board, [], []

    def collect(state):
        bank = state.get("bank") or {}
        target, fit = bank.get("target"), bank.get("fit")
        if state.get("firmware_mode") == "B" and target:
            now = time.monotonic()
            found.append((now, float(target["azimuth_deg"]), float(target["elevation_deg"])))
            if fit:
                fitted.append((now, float(fit["azimuth_deg"]), float(fit["elevation_deg"])))
        return False

    board.stop()
    board.command("B")
    board.pump(2.0)
    session.sound(sound)
    end = time.monotonic() + listen
    board.pump(listen, collect)
    board.stop()
    session.sound(None)

    def median(points):
        late = [(a, e) for t, a, e in points if t >= end - 4.0]
        return (float(np.median([a for a, _ in late])), float(np.median([e for _, e in late]))) if late else None

    return {"peak": median(found), "fit": median(fitted)}


def live_summary(hits, truth_sigma):
    """Median, spread and uncertainty of the trial directions (an array of (az, el)) at one position."""
    out = {"median_az": float(np.median(hits[:, 0])), "median_el": float(np.median(hits[:, 1])),
           "spread_az": float(np.subtract(*np.percentile(hits[:, 0], [75, 25]))),
           "spread_el": float(np.subtract(*np.percentile(hits[:, 1], [75, 25])))}
    for axis, key in (("az", 0), ("el", 1)):
        se = 1.253 * np.std(hits[:, key], ddof=1) / np.sqrt(len(hits)) if len(hits) > 1 else 0.0
        out[f"sigma_{axis}"] = float(np.hypot(se, truth_sigma[key]))
    return out


def l1(session, args, out, array):
    """Live direction vs speaker position, against the firmware centroid applied to theory bank levels."""
    contract = load_beam_contract()
    bank = find(json.loads(Path(default_contract_path()).read_text(encoding="utf-8")), "bank")
    rng = np.random.default_rng(args.seed)
    rows = []
    for nominal in parse_positions(args.positions or L1_POSITIONS):
        if args.truth == "array":
            distance = place(nominal[0], nominal[1], args.distance)
        else:
            az, el, distance = ask_position(nominal[0], nominal[1], args.distance)
        results = [live_trial(session, args.sound, args.listen) for _ in range(args.trials)]
        trials = [r["peak"] for r in results]
        fit_trials = [r["fit"] for r in results]
        hits = np.array([t for t in trials if t])
        fit_hits = np.array([t for t in fit_trials if t])
        if args.truth == "array":
            centre = (float(np.median(hits[:, 0])), float(np.median(hits[:, 1]))) if len(hits) else nominal
            with session.settings(**MCU_BEAM):
                (az, el), truth_sigma, _, _ = locate(session, contract, centre, distance, 1, 1.0, rng, "l1 locate")
        else:
            truth_sigma = (args.alignment_sigma, args.alignment_sigma)
        predicted = bank_prediction(bank, contract, az, el, distance) if np.isfinite([az, el]).all() else (
            float("nan"), float("nan"))
        row = {"true_az": az, "true_el": el, "distance_m": distance, "truth": args.truth,
               "truth_sigma_deg": list(truth_sigma), "predicted_az": predicted[0],
               "predicted_el": predicted[1], "trials": trials, "detected": len(hits) / len(trials),
               "fit_trials": fit_trials}
        if len(fit_hits):
            row.update({f"fit_{key}": value for key, value in live_summary(fit_hits, truth_sigma).items()})
        if len(hits):
            row.update(live_summary(hits, truth_sigma))
            fit_text = (f"; fit {row['fit_median_az']:+.1f}, {row['fit_median_el']:+.1f}" if len(fit_hits) else "")
            print(f"  true {az:+.1f}, {el:+.1f}: Live {row['median_az']:+.1f}, {row['median_el']:+.1f}{fit_text}; "
                  f"predicted {predicted[0]:+.1f}, {predicted[1]:+.1f}; detected {row['detected']:.0%}", flush=True)
        rows.append(row)
    detected = [r for r in rows if "median_az" in r and np.isfinite([r["true_az"], r["true_el"]]).all()]
    lost = [r for r in rows if not np.isfinite([r["true_az"], r["true_el"]]).all()]
    if lost:
        session.report.add("L1 speaker found by the steered beam", False,
                           f"not found at {len(lost)} position(s): speaker off, too quiet or outside the search")
    near = all(abs(r["median_az"] - r["predicted_az"]) <= 2 * r["sigma_az"] and
               abs(r["median_el"] - r["predicted_el"]) <= 2 * r["sigma_el"] for r in detected)
    session.report.add("L1 every position within 2 sigma of the prediction", bool(detected) and near,
                       f"{len(detected)} of {len(rows)} positions detected; speaker direction from "
                       + ("the steered beam's fit (Live vs the array's own fine beam)" if args.truth == "array"
                          else "measured positions"))
    session.report.add("L1 detection", all(r["detected"] >= 0.95 for r in rows),
                       f"lowest {min(r['detected'] for r in rows):.0%} of trials")
    errors = [np.hypot(r["median_az"] - r["true_az"], r["median_el"] - r["true_el"]) for r in detected]
    if errors:
        session.report.info("L1 error vs true position", f"median {np.median(errors):.1f} deg, largest "
                                                          f"{np.max(errors):.1f} deg")
    fit_errors = [np.hypot(r["fit_median_az"] - r["true_az"], r["fit_median_el"] - r["true_el"])
                  for r in rows if "fit_median_az" in r and np.isfinite([r["true_az"], r["true_el"]]).all()]
    if fit_errors:
        session.report.info("L1 pattern fit error vs true position", f"median {np.median(fit_errors):.1f} deg, "
                                                                      f"largest {np.max(fit_errors):.1f} deg")
    session.values["l1"] = {"sound": args.sound, "positions": rows}
    figure_live(rows, out)


def decay_time(x, centre_hz, rate=48000, stop_s=1.5):
    """Reverberation time of an octave band from an interrupted-noise recording (ISO 3382-2 style)."""
    spectrum = np.fft.rfft(x - np.mean(x))
    f = np.fft.rfftfreq(len(x), 1 / rate)
    y = np.fft.irfft(spectrum * ((f >= centre_hz / np.sqrt(2)) & (f <= centre_hz * np.sqrt(2))), len(x))
    window = int(0.01 * rate)
    envelope = 10 * np.log10(np.convolve(y ** 2, np.ones(window) / window, mode="same") + 1e-20)
    t = np.arange(len(x)) / rate
    steady = float(np.median(envelope[(t > 0.2) & (t < stop_s - 0.1)]))
    tail = float(np.median(envelope[t > t[-1] - 0.5]))
    start = np.nonzero((t > stop_s - 0.2) & (envelope < steady - 5))[0]
    result = {"rt60_s": None, "method": None, "range_db": steady - tail}
    if not start.size:
        return result
    for depth, method in ((25, "T20"), (20, "T15"), (15, "T10")):
        if steady - tail < depth + 10:
            continue
        below = np.nonzero((t > t[start[0]]) & (envelope < steady - depth))[0]
        if below.size:
            span = slice(start[0], below[0])
            slope = np.polyfit(t[span], envelope[span], 1)[0]
            result.update({"rt60_s": float(-60 / slope) if slope < 0 else None, "method": method})
            break
    return result


def r1(session, args, out, array):
    """Interrupted broadband noise recorded on one microphone; T20 (or T15/T10) per octave."""
    session.route(1, args.microphone)
    found = {name: [] for name in OCTAVES}
    for repeat in range(args.repeats):
        session.sound("W")
        session.board.pump(2.0)
        stats, pcm, reason = record(session.board, args.record, at=1.5, action=lambda: play(None))
        session.sound(None)
        if pcm is None:
            session.report.info(f"R1 recording {repeat + 1}", f"failed: {reason}")
            continue
        for name, centre in OCTAVES.items():
            found[name].append(decay_time(pcm[:, 1], centre))
    answer = ask_numbers("\nRoom volume in m^3, length x width x height (for example 5 x 4 x 2.5 = 50; Enter to skip): ",
                         optional=True)
    volume = answer[0] if answer else None
    result = {}
    for name, values in found.items():
        times = [v["rt60_s"] for v in values if v["rt60_s"]]
        rt = float(np.mean(times)) if times else None
        result[name] = {"rt60_s": rt, "sd_s": float(np.std(times, ddof=1)) if len(times) > 1 else None,
                        "methods": [v["method"] for v in values], "range_db": [v["range_db"] for v in values],
                        "critical_distance_m": 0.057 * np.sqrt(volume / rt) if rt and volume else None}
        session.report.info(f"R1 RT60 {name}", "not measurable (decay range too small)" if rt is None else
                            f"{rt:.2f} s ({', '.join(m for m in result[name]['methods'] if m)})"
                            + (f", critical distance {result[name]['critical_distance_m']:.2f} m (Q = 1)"
                               if result[name]["critical_distance_m"] else ""))
    session.values["r1"] = {"microphone": args.microphone, "volume_m3": volume, "octaves": result}


def s1(session, args, out, array):
    """Sensitivity of microphone channels and the beam from an SPL meter with 1/3-octave noise at 1 kHz (with a
    tone, room modes make the meter and each microphone read different levels), then their silent levels in dB SPL
    and the room background. The scale is at the MCU (through the DSP and the analog link), not at the PDM output."""
    microphones = [int(m) for m in args.microphones.split(",")]
    sector = session.speaker_sector(0, 0)
    seconds = args.seconds or 3.0
    ask_operator("\nS1: speaker on the array axis; SPL meter beside the array, its microphone level with the array "
                 "face and pointing at the speaker, A weighting, slow. Enter to start the noise.")
    session.sound("B1k")
    spl = ask_numbers("SPL meter reading with the noise, the middle of what it shows over 10 s (dB): ")[0]
    on, off = {}, {}
    with session.settings(**WINDOWS, channel=1):
        for m in microphones:
            session.route(1, m)
            on[m] = session.capture("s1", f"microphone {m}, tone", sector, seconds, value=m)["mean_db"]
    with session.settings(**MCU_BEAM):
        on["beam"] = session.capture("s1", "beam, tone", sector, seconds)["mean_db"]
    session.sound(None)
    with session.settings(**WINDOWS, channel=1):
        for m in microphones:
            session.route(1, m)
            off[m] = session.capture("s1", f"microphone {m}, silent", sector, seconds, value=m)["mean_db"]
    with session.settings(**MCU_BEAM):
        off["beam"] = session.capture("s1", "beam, silent", sector, seconds)["mean_db"]
    answer = ask_numbers("Meter reading now, in silence: the room background (dB(A), Enter to skip): ", optional=True)
    background = answer[0] if answer else None
    result = {}
    for key in [*microphones, "beam"]:
        sensitivity = signal_db(on[key], off[key]) - (spl - 94.0)
        result[str(key)] = {"sensitivity_dbfs_at_94": sensitivity, "silent_dbfs": off[key],
                            "silent_db_spl": off[key] - sensitivity + 94.0}
        name = "beam" if key == "beam" else f"microphone {key}"
        session.report.info(f"S1 {name}", f"{sensitivity:.1f} dBFS at 94 dB SPL, 1 kHz band; silent level "
                                          f"{off[key] - sensitivity + 94.0:.1f} dB SPL, so SNR re 94 dB SPL "
                                          f"{sensitivity - off[key]:.1f} dB (0.5-3.5 kHz, unweighted, includes the room)")
    if background is not None:
        session.report.info("S1 room background", f"{background:.1f} dB(A); the noise read {spl - background:.1f} dB above "
                            "it" + ("" if spl - background >= 10 else ", under 10 dB: turn the speaker up and repeat"))
    session.values["s1"] = {"meter_db_spl": spl, "signal": "B1k", "background_dba": background, "channels": result}


def figure_cuts(plots, out):
    import matplotlib.pyplot as plt
    if not plots:
        return
    fig, axes = plt.subplots(len(plots) // 2, 2, figsize=(12, 4.2 * (len(plots) // 2)), squeeze=False)
    for ax, (name, cut, centre_value, rel_model, db_model, fit) in zip(axes.flat, plots):
        rel = cut["angles"] - centre_value
        peak = np.nanmax(cut["signal_db"])
        ax.errorbar(rel, cut["signal_db"] - peak, yerr=cut["se_db"], fmt="o", ms=3, color="k", label="measured")
        if fit:
            shown = np.interp((rel_model - fit["shift_deg"]) * fit["scale"], rel_model, db_model)
            ax.plot(rel_model, shown + fit["level_db"] - peak, "-", label=f"fit: width {fit['width_3db_deg']:.1f} "
                    f"+- {fit['width_sigma_deg']:.1f}, peak {fit['shift_deg']:+.1f} deg")
            ax.plot(rel_model, db_model + fit["level_db"] - peak, "--", lw=1,
                    label=f"theory: width {fit['theory_width_3db_deg']:.1f} deg")
        ax.set_xlim(rel.min() - 1, rel.max() + 1)
        ax.set_ylim(-30, 3)
        ax.set_xlabel("steering angle from the speaker (deg)")
        ax.set_ylabel("dB re peak")
        ax.set_title(name)
        ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "a3_cuts.png"), dpi=150)


def figure_pointing(rows, lines, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (axis, true, key, err) in zip(axes, (("azimuth", "true_az", "measured_az", "sigma_az"),
                                                 ("elevation", "true_el", "measured_el", "sigma_el"))):
        x = np.array([r[true] for r in rows])
        ax.errorbar(x, np.array([r[key] for r in rows]) - x, yerr=[r[err] for r in rows], fmt="o", color="k")
        if axis in lines:
            line = lines[axis]
            grid = np.linspace(x.min(), x.max(), 50)
            ax.plot(grid, line["offset_deg"] + (line["slope"] - 1) * grid, "-",
                    label=f"slope {line['slope']:.3f}, offset {line['offset_deg']:+.2f} deg")
            ax.legend(fontsize=9)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel(f"true {axis} (deg)")
        ax.set_ylabel("measured minus true (deg)")
        ax.set_title(f"Pointing error, {axis}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "a4_pointing.png"), dpi=150)


def figure_live(rows, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (axis, true, other) in zip(axes, (("az", "true_az", "true_el"), ("el", "true_el", "true_az"))):
        data = [r for r in rows if abs(r[other]) < 5 and f"median_{axis}" in r]
        if not data:
            continue
        x = np.array([r[true] for r in data])
        ax.errorbar(x, [r[f"median_{axis}"] - r[true] for r in data], yerr=[r[f"sigma_{axis}"] for r in data],
                    fmt="o", color="k", label="Live peak beam, median of trials")
        fitted = [r for r in data if f"fit_median_{axis}" in r]
        if fitted:
            ax.errorbar([r[true] + 0.6 for r in fitted], [r[f"fit_median_{axis}"] - r[true] for r in fitted],
                        yerr=[r[f"fit_sigma_{axis}"] for r in fitted], fmt="^", color="tab:blue",
                        label="Live pattern fit, median of trials")
        ax.plot(x, [r[f"predicted_{axis}"] - r[true] for r in data], "s--", mfc="none",
                label="predicted: firmware centroid on theory levels")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel(f"true {'azimuth' if axis == 'az' else 'elevation'} (deg)")
        ax.set_ylabel("reported minus true (deg)")
        ax.legend(fontsize=9)
    fig.suptitle("Live (bank) direction error")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "l1_live.png"), dpi=150)


A9_BANDS = ("B1k", "B2k", "B3k")
A10_BANDS = {"B5k": 5000.0, "B8k": 8000.0}


def registry_parameter(group):
    """(address, words) of a DSP parameter group from the backend's register map."""
    status, registry = http("GET", "/dsp/registry")
    if status != 200:
        raise RuntimeError(f"no DSP register map ({status})")
    for parameter in registry["parameters"]:
        if parameter.get("group") == group:
            return int(parameter["address"]), int(parameter["words"])
    raise RuntimeError(f"no {group} in the DSP register map")


def read_words(address, count):
    status, body = http("POST", "/dsp/read", {"address": address, "count": count})
    if status != 200:
        raise RuntimeError(f"DSP read refused ({status}): {body}")
    return [int(word) for word in body["words"]]


def highpass_taps(cutoff_hz, taps=11, rate=48000.0):
    """Linear-phase FIR high-pass: centre tap minus a Hamming-windowed sinc low-pass with unity DC gain."""
    n = np.arange(taps) - (taps - 1) / 2
    low = 2 * cutoff_hz / rate * np.sinc(2 * cutoff_hz / rate * n) * np.hamming(taps)
    high = -low / low.sum()
    high[(taps - 1) // 2] += 1.0
    return high


def to_word(value):
    """8.24 fixed point as the 32-bit two's-complement word the DSP stores."""
    return int(round(value * (1 << 24))) & 0xFFFFFFFF


@contextmanager
def fir_replaced(board, highpass_hz=None):
    """Beam FIR stage 1 as a centre tap of 1 (Study's Bypass), stage 2 the same or a high-pass at highpass_hz;
    the flashed taps are put back and checked."""
    board.stop()
    stages = [registry_parameter(group) for group in ("fir1", "fir2")]
    originals = [read_words(address, words) for address, words in stages]
    try:
        for index, (address, words) in enumerate(stages):
            taps = np.zeros(words)
            taps[words // 2] = 1.0
            if index == 1 and highpass_hz:
                taps = highpass_taps(highpass_hz, words)
            write_words({address + k: to_word(t) for k, t in enumerate(taps)})
        yield
    finally:
        board.stop()
        for (address, words), taps in zip(stages, originals):
            write_words({address + k: tap for k, tap in enumerate(taps)})
        if [read_words(address, words) for address, words in stages] != originals:
            raise RuntimeError("FIR taps not restored: reset the MCU (its boot reloads the DSP program)")


def excess(on_db, off_db):
    """Signal power: on minus off, in power; nan when not above the floor."""
    value = float(power(on_db) - power(off_db))
    return value if value > 0 else float("nan")


def directivity_from_levels(microphones, path_gain, microphone_power, beam_power):
    """DF = N^2 * G * P_mic / P_beam: the diffuse-field beam/microphone ratio corrected by the path gain G."""
    return float(microphones ** 2 * path_gain * microphone_power / beam_power)


def grating_lobes(rel_deg, db, main_half_deg, threshold_db=-10.0):
    """Local maxima at or above threshold_db (re the main peak) outside +-main_half_deg: [(angle, level)]."""
    return [(float(rel_deg[i]), float(db[i])) for i in range(1, len(db) - 1)
            if abs(rel_deg[i]) > main_half_deg and db[i] >= threshold_db and db[i] >= db[i - 1] and db[i] >= db[i + 1]]


def a9(session, args, out, array):
    """(1) Path gain G: one microphone through the beam (others muted) vs the same microphone on Ch1.
    (2) Directivity index by the diffuse-field method: speaker facing away, beam/microphone power ratio / G.
    (3) Speaker close and in front: signal gain and noise gain of the full beam (/ G); their difference is the
    array gain here. WNG proper (16.4 dB) needs a free field and a self-noise-limited floor."""
    contract = load_beam_contract()
    data = json.loads(Path(default_contract_path()).read_text(encoding="utf-8"))
    addresses = [int(a) for a in find(data, "gain_addresses")]
    unity = [int(w) for w in find(data, "gain_fixpt_8_24")]
    count = len(addresses)
    references = [int(m) for m in np.random.default_rng(1).permutation(count)[:args.references]]
    seconds = args.seconds or 3.0
    centre = session.speaker_sector(0, 0)
    rng = np.random.default_rng(args.seed)

    def beam(label, sector=centre):
        return session.capture("a9", label, sector, seconds)["mean_db"]

    def mics(label, which, source=1):
        values = {}
        with session.settings(**WINDOWS, channel=1):
            for m in which:
                session.route(source, m)
                values[m] = session.capture("a9", f"{label}, Ch1 {source},{m}", centre, seconds, value=m)["mean_db"]
        return values

    def gains_only(on):
        session.board.stop()
        write_words({a: (unity[k] if k in on else 0) for k, a in enumerate(addresses)})

    place(0.0, 0.0, args.distance)
    path, links = {}, {}
    try:
        with session.settings(**MCU_BEAM):
            gains_only(set())
            links["ch0"] = beam("link floor, all muted")
            links["ch1"] = mics("link floor, all muted", [0], source=0)[0]
            for m in references[:2]:
                gains_only({m})
                for band in A9_BANDS:
                    session.sound(None)
                    off_b, off_m = beam(f"path {band} silent, microphone {m} alone"), mics(f"path {band} silent", [m])[m]
                    session.sound(band)
                    on_b, on_m = beam(f"path {band}, microphone {m} alone"), mics(f"path {band}", [m])[m]
                    path.setdefault(band, []).append(excess(on_b, off_b) / excess(on_m, off_m))
                session.sound(None)
    finally:
        gains_only(set(range(count)))
        reload_dsp(session.board)
    g = {band: float(np.nanmean(values)) for band, values in path.items()}
    result = {"references": references, "link_floor_dbfs": links, "path_gain_db": {}, "bands": {}}
    for band in A9_BANDS:
        result["path_gain_db"][band] = [10 * np.log10(v) if np.isfinite(v) else None for v in path[band]]
        session.report.info(f"A9 path gain {band}", f"{10 * np.log10(g[band]):+.1f} dB, beam path vs microphone path "
                            f"(microphones {references[:2]}: " + ", ".join(
                                f"{v:+.1f}" for v in result["path_gain_db"][band] if v is not None) + ")")

    ask_operator("\nA9 diffuse field: turn the speaker to face a far wall or corner, as far from the array as the "
                 "room allows, ideally with no direct line to the array. Enter when ready.")
    directions = [(0.0, 0.0), (34.29, 0.0), (-34.29, 0.0), (0.0, 34.29), (0.0, -34.29)]
    sectors = [session.speaker_sector(az, el) for az, el in directions]
    with session.settings(**MCU_BEAM):
        for band in A9_BANDS:
            session.sound(None)
            off_b, off_m = [beam(f"diffuse {band} silent", s) for s in sectors], mics(f"diffuse {band} silent", references)
            session.sound(band)
            on_b, on_m = [beam(f"diffuse {band}", s) for s in sectors], mics(f"diffuse {band}", references)
            session.sound(None)
            beam_power = np.array([excess(a, b) for a, b in zip(on_b, off_b)])
            mic_power = np.array([excess(on_m[m], off_m[m]) for m in references])
            df = directivity_from_levels(count, g[band], np.nanmean(mic_power), np.nanmean(beam_power))
            theory = band_directivity_index_db(contract, band_frequencies(BANDS[band][0]))
            spread = float(10 * np.log10(np.nanmax(beam_power) / np.nanmin(beam_power)))
            measured = 10 * np.log10(df) if df > 0 else float("nan")
            result["bands"].setdefault(band, {}).update({"di_db": measured, "di_theory_db": theory,
                                                         "beam_spread_db": spread})
            session.report.info(f"A9 directivity index {band}", f"{measured:.1f} dB, theory {theory:.1f} "
                                f"(diffuse-field method, approximate in a room); beam spread over 5 directions "
                                f"{spread:.1f} dB (large = direct sound still present)")

    near = place(0.0, 0.0, args.near_distance)
    with session.settings(**MCU_BEAM):
        (az, el), _, _, _ = locate(session, contract, (0.0, 0.0), near, 1, 1.0, rng, "a9 locate")
        if not np.isfinite([az, el]).all():
            az, el = 0.0, 0.0
        session.grid(az, az, 1, el, el, 1)
        for band in A9_BANDS:
            session.sound(None)
            off_b, off_m = beam(f"near {band} silent", 0), mics(f"near {band} silent", references)
            session.sound(band)
            on_b, on_m = beam(f"near {band}", 0), mics(f"near {band}", references)
            session.sound(None)
            signal = excess(on_b, off_b) / np.nanmean([excess(on_m[m], off_m[m]) for m in references]) / g[band]
            noise = (excess(off_b, links["ch0"]) /
                     np.nanmean([excess(off_m[m], links["ch1"]) for m in references]) / g[band])
            direct = count ** 2 * float(steered_response(contract, az, el, az, el,
                                                         band_frequencies(BANDS[band][0]), near)[0])
            values = {"signal_gain_db": 10 * np.log10(signal), "signal_gain_direct_theory_db": 10 * np.log10(direct),
                      "noise_gain_db": 10 * np.log10(noise), "noise_gain_independent_db": 10 * np.log10(count)}
            values["array_gain_db"] = values["signal_gain_db"] - values["noise_gain_db"]
            result["bands"].setdefault(band, {}).update(values)
            session.report.info(f"A9 gains {band}", f"signal {values['signal_gain_db']:.1f} dB (direct sound only "
                                f"{values['signal_gain_direct_theory_db']:.1f}), noise {values['noise_gain_db']:.1f} dB "
                                f"(independent noise {10 * np.log10(count):.1f}); array gain here "
                                f"{values['array_gain_db']:+.1f} dB, WNG theory {10 * np.log10(count):.1f}")
    result["speaker_near"] = {"azimuth_deg": az, "elevation_deg": el, "distance_m": near}
    session.values["a9"] = result


def a10(session, args, out, array):
    """Spatial aliasing: speaker straight ahead, the beam steered across +-60 deg with the FIR pair replaced by
    a 3.5 kHz high-pass (removes the low-frequency floor) and the MCU window unfiltered (band 0), at 5 kHz
    (theory: no grating lobe in view) and 8 kHz (lobes near +-50 deg). Measured false peaks against theory."""
    contract = load_beam_contract()
    distance = place(0.0, 0.0, args.distance)
    seconds = args.seconds or 1.0
    rng = np.random.default_rng(args.seed)
    results, plots = {}, []
    with fir_replaced(session.board, highpass_hz=3500.0), session.settings(**{**MCU_BEAM, "band": 0}):
        for band, centre_hz in A10_BANDS.items():
            for axis in ("azimuth", "elevation"):
                cut = measure_cut(session, axis, (0.0, 0.0), 2.5, 60.0, band, 1, seconds, rng, "a10")
                db = cut["signal_db"] - np.nanmax(cut["signal_db"])
                peak = float(cut["angles"][int(np.nanargmax(cut["signal_db"]))])
                source = (peak, 0.0) if axis == "azimuth" else (0.0, peak)
                rel_model, db_model = steering_cut_db(contract, source, axis, band_frequencies(centre_hz), distance)
                half = first_null_half_width(rel_model, db_model) or 10.0
                limit = 60.0 - abs(peak)
                expected = [lobe for lobe in grating_lobes(rel_model, db_model, half, -6.0) if abs(lobe[0]) <= limit]
                found = grating_lobes(cut["angles"] - peak, np.nan_to_num(db, nan=-99.0), half, -10.0)
                matched = all(any(abs(f[0] - e[0]) <= 5.0 and abs(f[1] - e[1]) <= 6.0 for f in found) for e in expected)
                clean = expected or not any(level >= -6.0 for _, level in found)
                name = f"{band} {axis}"
                results[name] = {"peak_deg": peak, "expected": expected, "found": found,
                                 "max_on_dbfs": float(np.nanmax(cut["on_db"])), "angles": cut["angles"],
                                 "signal_db": cut["signal_db"]}
                plots.append((name, cut["angles"] - peak, db, rel_model, db_model))
                session.report.add(f"A10 {name}", matched and bool(clean),
                                   "theory lobes " + (", ".join(f"{a:+.0f} deg {l:.0f} dB" for a, l in expected) or "none")
                                   + "; measured " + (", ".join(f"{a:+.0f} deg {l:.0f} dB" for a, l in found
                                                               if l >= -6.0) or "none above -6 dB"))
                if results[name]["max_on_dbfs"] > -12.0:
                    session.report.add(f"A10 {name} level", False, f"{results[name]['max_on_dbfs']:.1f} dBFS RMS: band "
                                       "noise peaks about 10 dB higher and may clip; turn the speaker down, repeat")
    session.values["a10"] = {"distance_m": distance, "cuts": results}
    figure_a10(plots, out)


def figure_a10(plots, out):
    import matplotlib.pyplot as plt
    if not plots:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for ax, (name, rel, db, rel_model, db_model) in zip(axes.flat, plots):
        ax.plot(rel, db, "o-", ms=3, color="k", label="measured")
        ax.plot(rel_model, db_model, "--", lw=1, label="theory")
        ax.set_xlim(-62, 62)
        ax.set_ylim(-30, 3)
        ax.set_xlabel("steering angle from the speaker (deg)")
        ax.set_ylabel("dB re main lobe")
        ax.set_title(name)
        ax.legend(fontsize=8, loc="lower center")
    fig.suptitle("A10: grating lobes (false beams) above the spatial aliasing frequency")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "a10_aliasing.png"), dpi=150)


# Latency, detector scale, detection margin, two sources, tones -----------------------------------

AUDIO = Path(__file__).resolve().parent.parent / "audio"
P2_GUARDS_US = (0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 4000, 5000)
P2_WINDOW_US = 2000
LEVEL_STEPS_DB = (0, 6, 12, 18, 24, 30, 36, 42, 48)
ACQUIRE_DB = float(20 * np.log10(1.5))  # the firmware's 1.5x acquisition ratio, on raw amplitude words
TOLERANCE_DEG = 10.0


def level_signal(step_db):
    return "A" if step_db == 0 else f"A-{step_db}"


def angle_off(az, el, reference):
    return float(np.hypot(az - reference[0], el - reference[1]))


def arrival_fraction(steered_db, on_db, off_db):
    """Share of a window read right after a steer that already holds the new direction: 0 all old, 1 all new."""
    on, off = power(on_db), power(off_db)
    return (power(steered_db) - off) / (on - off)


def arrival_fit(guards_us, fractions):
    """Least-squares fit of f = clip((guard - start) / ramp, 0, 1): (start_us, ramp_us). From start + ramp on, a
    window read after a steer holds only the new direction."""
    g, f = np.asarray(guards_us, float), np.asarray(fractions, float)
    keep = np.isfinite(f)
    g, f = g[keep], f[keep]
    if g.size < 3:
        return float("nan"), float("nan")
    starts = np.arange(-5000.0, 10000.0 + 1.0, 25.0)
    best, best_error = (float("nan"), float("nan")), np.inf
    for ramp in np.arange(250.0, 8000.0 + 1.0, 25.0):
        error = np.sum((np.clip((g[None, :] - starts[:, None]) / ramp, 0.0, 1.0) - f[None, :]) ** 2, axis=1)
        i = int(np.argmin(error))
        if error[i] < best_error:
            best_error, best = float(error[i]), (float(starts[i]), float(ramp))
    return best


def threshold_db(levels_db, rates, target=0.5):
    """Quietest level whose detection rate still reaches `target`, interpolated in dB between steps. Returns
    (level, censored); censored means it was still detected at the quietest level played; None if never."""
    order = np.argsort(np.asarray(levels_db, float))[::-1]
    x, r = np.asarray(levels_db, float)[order], np.asarray(rates, float)[order]
    if not np.isfinite(r[0]) or r[0] < target:
        return None, False
    for i in range(1, len(x)):
        if r[i] < target:
            return float(x[i - 1] + (r[i - 1] - target) / (r[i - 1] - r[i]) * (x[i] - x[i - 1])), False
    return float(x[-1]), True


def scan_detects(levels_db, reference, columns, ratio_db=ACQUIRE_DB):
    """One scan pass finds the source: its strongest sector is within one cell of `reference` and at least the
    tracker's acquisition ratio above the pass median."""
    flat = np.asarray(levels_db, float).ravel()
    if not np.isfinite(flat).any():
        return False
    best = int(np.nanargmax(flat))
    (r, c), (r0, c0) = divmod(best, columns), divmod(int(reference), columns)
    return bool(abs(r - r0) <= 1 and abs(c - c0) <= 1 and flat[best] - np.nanmedian(flat) >= ratio_db)


def scan_alarm(levels_db, ratio_db=ACQUIRE_DB):
    """Any sector of the pass at least the acquisition ratio above the median (a target, wherever it is)."""
    flat = np.asarray(levels_db, float).ravel()
    return bool(np.isfinite(flat).any() and np.nanmax(flat) - np.nanmedian(flat) >= ratio_db)


def level_slope(played_db, measured_db, errors_db):
    """Weighted straight line of measured against played level: (slope, sigma, offset, residuals)."""
    x, y = np.asarray(played_db, float), np.asarray(measured_db, float)
    s = np.maximum(np.asarray(errors_db, float), 0.05)
    (slope, offset), cov = np.polyfit(x, y, 1, w=1 / s, cov="unscaled")
    return float(slope), float(np.sqrt(cov[0, 0])), float(offset), (y - (offset + slope * x)).tolist()


def detector_models(played_db, measured_db, errors_db, floor_db):
    """Fit level readings with a word that follows amplitude (signal and floor add in word^2) or power (they add
    in the word itself). played_db is re the loudest step, measured_db and floor_db in dBFS (20 log10 word).
    Returns {model: {"scale_db", "chi2"}}, the scale being the loudest step's signal word without the floor."""
    x, y = np.asarray(played_db, float), np.asarray(measured_db, float)
    s = np.maximum(np.asarray(errors_db, float), 0.05)
    floor = 10 ** (floor_db / 20)
    scales = np.arange(-120.0, 20.0, 0.01)
    a = 10 ** (scales / 20)[:, None]
    words = {"amplitude": np.sqrt((a * 10 ** (x / 20)) ** 2 + floor ** 2), "power": a * 10 ** (x / 10) + floor}
    result = {}
    for model, word in words.items():
        chi2 = np.sum(((y - 20 * np.log10(word)) / s) ** 2, axis=1)
        i = int(np.argmin(chi2))
        result[model] = {"scale_db": float(scales[i]), "chi2": float(chi2[i])}
    return result


def tone_stats(reports):
    """Summary of TONE reports [(f0, score, harmonics, level)]."""
    if not reports:
        return {"reports": 0}
    f0, score, harmonics, level = (np.array(values, float) for values in zip(*reports))
    return {"reports": len(reports), "f0_median_hz": float(np.median(f0)),
            "f0_iqr_hz": float(np.subtract(*np.percentile(f0, [75, 25]))), "score_median_db": float(np.median(score)),
            "score_p10_db": float(np.percentile(score, 10)), "score_p90_db": float(np.percentile(score, 90)),
            "harmonics_median": float(np.median(harmonics)), "level_median_db": float(np.median(level))}


def player(session, name):
    return lambda: session.sound(name)


def live_maps(session, listen, start=None, stop=None, quiet=2.0, log=None, during=None):
    """Stare starts in silence (its background), then start() turns the source on. Returns the number of bank maps
    in the next `listen` s and the BANK_TARGET records they brought: [(map, az, el, beam, second beam)].
    With a `log` dict, it also keeps the levels of every map from the start (for live_log.py); `during(state)`
    runs on every frame of the listening time."""
    board, targets, seen = session.board, [], [None]

    def keep(state):
        bank = state.get("bank") or {}
        maps = int(bank.get("maps") or 0)
        if log is not None and state.get("firmware_mode") == "B" and bank.get("levels_raw") and \
                maps > log["start_maps"] and (not log["maps"] or maps != log["maps"][-1]):
            log["maps"].append(maps)
            log["t"].append(time.monotonic())
            log["levels"].append(list(bank["levels_raw"]))
        return False

    def collect(state):
        keep(state)
        bank = state.get("bank") or {}
        target = bank.get("target")
        if state.get("firmware_mode") == "B" and target:
            key = json.dumps(target, sort_keys=True)
            if key != seen[0]:
                seen[0] = key
                targets.append((int(bank.get("maps") or 0), float(target["azimuth_deg"]),
                                float(target["elevation_deg"]), int(target["beam"]), int(target["second_beam"])))
        return False

    board.stop()
    if log is not None:
        log.update(start_maps=int((board.state.get("bank") or {}).get("maps") or 0), maps=[], t=[], levels=[],
                   targets=targets, events=log.get("events", {}))
    board.command("B")
    board.pump(quiet, collect)
    if start:
        start()
        board.pump(0.3, collect)  # applies frames queued while start() waited
    first = int((board.state.get("bank") or {}).get("maps") or 0)
    if log is not None:
        log["events"]["sound_maps"] = first
    board.pump(listen, (lambda s: collect(s) or during(s)) if during else collect)
    count = int((board.state.get("bank") or {}).get("maps") or 0) - first
    board.stop()
    if stop:
        stop()
    return count, [t for t in targets if t[0] > first]


def scan_passes(session, listen, start=None, stop=None):
    """Continuous scan (C) for `listen` s after start(); the dB map of every completed pass."""
    board, maps = session.board, []
    board.stop()
    if start:
        start()
    last = [int(board.state.get("scan_count") or 0)]

    def collect(state):
        count = int(state.get("scan_count") or 0)
        if count != last[0]:
            last[0] = count
            maps.append(np.array([[np.nan if v is None else v for v in row] for row in state["levels_db"]], float))
        return False

    board.command("C")
    board.pump(listen, collect)
    board.stop()
    if stop:
        stop()
    return maps


def map_hits(targets, reference):
    """Maps with a target within TOLERANCE_DEG of `reference`."""
    return {t[0] for t in targets if angle_off(t[1], t[2], reference) <= TOLERANCE_DEG}


def p2(session, args, out, array):
    """Audio-path latency: from a quiet direction, steer onto the speaker and read a 2 ms window after a guard swept
    from 0 to 5 ms. The guard (guard_us) has to cover the time from a steer to the new beam's sound at the MCU."""
    place(0.0, 0.0, args.distance)
    rng = np.random.default_rng(args.seed)
    rows, columns = session.home[0]["rows"], session.home[0]["columns"]
    on = session.speaker_sector(0, 0)
    corners = sorted({0, columns - 1, (rows - 1) * columns, rows * columns - 1})
    session.sound("A")
    with session.settings(**MCU_BEAM, band=1):
        steady = {s: session.capture("p2", f"steady, sector {s}", s, 1.0, value=s)["mean_db"] for s in [on, *corners]}
    off = min(corners, key=lambda s: steady[s])
    contrast = steady[on] - steady[off]
    if contrast < 6.0:
        session.sound(None)
        session.report.add("P2 contrast between the speaker and a far direction", False,
                           f"{contrast:.1f} dB, needs 6 dB: move the speaker closer (1 m) and repeat")
        return
    guards, fractions, errors = [], [], []
    with session.settings(measure=1, band=1, channel=0, window_us=P2_WINDOW_US, guard_us=500):
        for guard in rng.permutation(P2_GUARDS_US):
            guard = int(guard)
            session.set("guard_us", guard)
            steered = []
            for _ in range(args.steers):
                session.read(f"S,{off}")
                measurement = session.read(f"S,{on}")
                if measurement:
                    steered.append(measurement["level_db"])
            reference = {}
            for s in (on, off):
                session.read(f"S,{s}")
                reference[s] = power_mean_db([m["level_db"] for m in (session.read("M") for _ in range(20)) if m])
            p = power(steered)
            span = float(power(reference[on]) - power(reference[off]))
            guards.append(guard)
            if span > 0 and len(p) > 1:
                fractions.append(float(arrival_fraction(power_mean_db(steered), reference[on], reference[off])))
                errors.append(float(np.std(p, ddof=1) / np.sqrt(len(p)) / span))
            else:
                fractions.append(float("nan"))
                errors.append(float("nan"))
            print(f"  guard {guard:5d} us: {fractions[-1]:.2f} of the window is the new direction "
                  f"(+- {errors[-1]:.2f}, {len(steered)} steers)", flush=True)
    session.sound(None)
    order = np.argsort(guards)
    guards, fractions, errors = (list(np.asarray(v)[order]) for v in (guards, fractions, errors))
    start, ramp = arrival_fit(guards, fractions)
    arrival = start + ramp
    session.report.info("P2 contrast", f"speaker sector {on} {steady[on]:.1f} dBFS, quietest corner {off} "
                                       f"{steady[off]:.1f} dBFS ({contrast:.1f} dB)")
    if not np.isfinite(arrival):
        session.report.add("P2 arrival after a steer", False, "no usable guard step: the speaker sector was not "
                                                              "louder than the far one; check the sound and repeat")
        return
    recommended = int(np.ceil(max(arrival, 0.0) / 250.0) * 250)
    current = int(session.settings_before.get("guard_us", 2000))
    usable = np.isfinite(fractions)
    at_current = float(np.interp(current, np.asarray(guards)[usable], np.asarray(fractions)[usable])) \
        if usable.any() else float("nan")
    session.report.info("P2 arrival after a steer", f"the new direction starts to arrive at {start:.0f} us and fills "
                                                    f"the window from {arrival:.0f} us on (fit)")
    session.report.add(f"P2 the board's {current} us guard covers the audio path", current >= arrival,
                       f"at {current} us the window is {at_current:.0%} new direction" if np.isfinite(at_current) else
                       f"no usable step near {current} us")
    session.report.info("P2 guard to use", f"SET,guard_us,{recommended}")
    session.values["p2"] = {"speaker_sector": on, "far_sector": off, "steady_db": steady, "window_us": P2_WINDOW_US,
                            "guards_us": guards, "fractions": fractions, "fraction_se": errors,
                            "arrival_start_us": start, "arrival_end_us": arrival, "recommended_guard_us": recommended,
                            "board_guard_us": current}
    figure_p2(guards, fractions, errors, (start, ramp), recommended, current, out)


def s3(session, args, out, array):
    """Linearity and the detector's scale: sound A in 6 dB steps, read on the beam with MCU windows and with the DSP
    detector. The MCU window reports RMS; the DSP detector's fit tells whether its word follows amplitude or power,
    which sets what the 1.5x and 1.25x thresholds mean in dB."""
    place(0.0, 0.0, args.distance)
    sector = session.speaker_sector(0, 0)
    seconds = args.seconds or 2.0
    rng = np.random.default_rng(args.seed)
    modes = {"MCU window": {**MCU_BEAM, "band": 1}, "DSP detector": {"measure": 0, "channel": 0}}
    readings = {mode: {step: [] for step in LEVEL_STEPS_DB} for mode in modes}
    floors = {mode: [] for mode in modes}
    for cycle in range(args.repeats):
        for step in [None, *(int(s) for s in rng.permutation(LEVEL_STEPS_DB))]:
            session.sound(None if step is None else level_signal(step))
            for mode, settings in modes.items():
                with session.settings(**settings):
                    label = f"{mode}, {'silent' if step is None else level_signal(step)}, cycle {cycle + 1}"
                    level = session.capture("s3", label, sector, seconds, value=None if step is None else -step)
                (floors[mode] if step is None else readings[mode][step]).append(level["mean_db"])
        session.sound(None)
    result = {}
    for mode in modes:
        floor = power_mean_db(floors[mode])
        played, measured, errors = [], [], []
        for step in LEVEL_STEPS_DB:
            values = readings[mode][step]
            played.append(-step)
            measured.append(power_mean_db(values))
            errors.append(float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.3)
        entry = {"floor_db": floor, "played_db": played, "measured_db": measured, "se_db": errors,
                 "models": detector_models(played, measured, errors, floor)}
        high = [i for i, m in enumerate(measured) if m - floor >= 15.0]
        if len(high) >= 3:
            slope, sigma, offset, residuals = level_slope([played[i] for i in high],
                                                          [signal_db(measured[i], floor) for i in high],
                                                          [errors[i] for i in high])
            entry.update(slope=slope, slope_sigma=sigma, residuals_db=residuals, steps_used=len(high))
        result[mode] = entry
    window, detector = result["MCU window"], result["DSP detector"]
    if "slope" in window:
        ok = abs(window["slope"] - 1) <= max(0.05, 3 * window["slope_sigma"]) and max(map(abs, window["residuals_db"])) <= 1.0
        session.report.add("S3 MCU window linearity", ok,
                           f"slope {window['slope']:.3f} +- {window['slope_sigma']:.3f} dB per dB over "
                           f"{window['steps_used']} steps, largest residual {max(map(abs, window['residuals_db'])):.2f} "
                           f"dB; loudest step {window['measured_db'][0] - window['floor_db']:.1f} dB above the floor")
    else:
        session.report.add("S3 MCU window linearity", False, "fewer than 3 steps 15 dB above the floor: louder")
    better = min(detector["models"], key=lambda m: detector["models"][m]["chi2"])
    other = "power" if better == "amplitude" else "amplitude"
    meaning = ("1.5x = 3.5 dB, 1.25x = 1.9 dB" if better == "amplitude" else "1.5x = 1.8 dB, 1.25x = 1.0 dB")
    session.report.info("S3 DSP detector word follows", f"{better} (chi2 {detector['models'][better]['chi2']:.1f} "
                                                       f"against {detector['models'][other]['chi2']:.1f} for {other}"
                                                       + (f"; slope {detector['slope']:.2f} +- "
                                                          f"{detector['slope_sigma']:.2f}" if "slope" in detector else "")
                                                       + f"), so the thresholds mean {meaning}")
    window_model = min(window["models"], key=lambda m: window["models"][m]["chi2"])
    session.report.add("S3 MCU window follows amplitude (sanity check of the method)", window_model == "amplitude",
                       f"best model: {window_model}")
    session.values["s3"] = result
    figure_s3(result, out)


def l5(session, args, out, array):
    """Detection margin, the design's main claim: per-map detection rate of Live (bank) and of continuous scanning
    (MCU windows) against the level of sound A; the margin is the difference of their 50% levels."""
    place(0.0, 0.0, args.distance)
    rng = np.random.default_rng(args.seed)
    columns = int(session.home[0]["columns"])
    base = {"measure": 0, "window_us": 5000, "band": 1, "channel": 0, "map_record": 1}
    if args.guard is not None:
        base["guard_us"] = args.guard
    steps = level_steps(args)
    live = {step: [] for step in steps}
    scanned = {step: [] for step in steps}
    logs = []

    def logged(label):
        logs.append({"label": label, "events": {}})
        return logs[-1]

    with session.settings(**base):
        count, targets = live_maps(session, args.listen, player(session, "A"), player(session, None),
                                   log=logged("level 0 dB, reference"))
        if not targets:
            raise RuntimeError("Live reported nothing at the loudest level: raise the volume or move the speaker closer")
        reference = (float(np.median([t[1] for t in targets])), float(np.median([t[2] for t in targets])))
        session.set("measure", 1)
        passes = scan_passes(session, args.listen, player(session, "A"), player(session, None))
        strongest = [int(np.nanargmax(m)) for m in passes if np.isfinite(m).any()]
        if not strongest:
            raise RuntimeError("the scan completed no pass")
        sector = int(np.bincount(strongest).argmax())
        print(f"  reference: Live {reference[0]:+.1f}, {reference[1]:+.1f} deg; scan sector {sector}", flush=True)
        for trial in range(args.trials):
            for step in (int(s) for s in rng.permutation(steps)):
                for mode in (("live", "scan") if trial % 2 == 0 else ("scan", "live")):
                    sound = player(session, level_signal(step))
                    if mode == "live":
                        session.set("measure", 0)
                        count, targets = live_maps(session, args.listen, sound, player(session, None),
                                                   log=logged(f"level {-step} dB, trial {trial + 1}"))
                        live[step].append(len(map_hits(targets, reference)) / count if count else float("nan"))
                    else:
                        session.set("measure", 1)
                        passes = scan_passes(session, args.listen, sound, player(session, None))
                        scanned[step].append(float(np.mean([scan_detects(m, sector, columns) for m in passes]))
                                             if passes else float("nan"))
                print(f"  trial {trial + 1}, {level_signal(step)}: Live {live[step][-1]:.0%}, "
                      f"scan {scanned[step][-1]:.0%}", flush=True)
        session.set("measure", 0)
        count, targets = live_maps(session, args.listen, log=logged("silent"))
        live_false = len({t[0] for t in targets}) / count if count else float("nan")
        session.set("measure", 1)
        passes = scan_passes(session, args.listen)
        scan_false = float(np.mean([scan_alarm(m) for m in passes])) if passes else float("nan")
    save_log(os.path.join(out, "l5_maps.npz"), logs)
    levels = [-s for s in steps]
    live_rate = [float(np.nanmean(live[s])) for s in steps]
    scan_rate = [float(np.nanmean(scanned[s])) for s in steps]
    (live_level, live_cut), (scan_level, scan_cut) = threshold_db(levels, live_rate), threshold_db(levels, scan_rate)

    def text(level, cut):
        return "never 50%" if level is None else (f"<= {level:.1f}" if cut else f"{level:.1f}") + " dB"

    session.report.info("L5 50% detection level", f"Live {text(live_level, live_cut)}, scan "
                                                  f"{text(scan_level, scan_cut)} (re sound A at the loudest step)")
    if live_level is None or scan_level is None or (live_cut and scan_cut):
        session.report.add("L5 Live detects at least 3 dB quieter than scanning", False,
                           "no margin: " + ("both still detect at the quietest step, so lower the speaker volume "
                                            "and repeat" if live_cut and scan_cut else "a mode never reached 50%"))
        margin = None
    else:
        margin = scan_level - live_level
        bound = "at least " if live_cut else ("at most " if scan_cut else "")
        session.report.add("L5 Live detects at least 3 dB quieter than scanning", margin >= 3.0,
                           f"margin {bound}{margin:.1f} dB (the design claims 6-7 dB)")
    session.report.info("L5 false alarms in silence", f"Live {live_false:.0%} of maps, scan {scan_false:.0%} of passes")
    session.values["l5"] = {"reference_live": reference, "reference_sector": sector, "levels_db": levels,
                            "live_rates": {s: live[s] for s in steps},
                            "scan_rates": {s: scanned[s] for s in steps}, "live_rate": live_rate,
                            "scan_rate": scan_rate, "live_level_db": live_level, "live_censored": live_cut,
                            "scan_level_db": scan_level, "scan_censored": scan_cut, "margin_db": margin,
                            "false_alarms": {"live": live_false, "scan": scan_false}, "guard_us": args.guard}
    figure_l5(levels, live_rate, scan_rate, live_level, scan_level, out)


def level_steps(args):
    """l5 level steps in dB below sound A: --steps, or the 6 dB default."""
    if not args.steps:
        return LEVEL_STEPS_DB
    return tuple(sorted({int(s) for s in args.steps.split(",")}))


def l7(session, args, out, array):
    """Live log: --trials new Live starts (B) of --listen s each, with the levels of every bank map. With --burst,
    sound A plays at the loudest step for that many seconds, --burst-at s into each start. live_log.py replays
    the firmware's detector on the log: target rate after a start and after the loud sound, their directions, and
    the same for other detector settings."""
    logs = []
    for trial in range(args.trials):
        log = {"label": f"start {trial + 1}", "events": {}}
        burst = {"on": None, "off": None}

        def sound_control(state, log=log, burst=burst):
            elapsed = time.monotonic() - log["t"][0] if log.get("t") else 0.0
            if args.burst > 0 and burst["on"] is None and elapsed >= args.burst_at:
                play(session.signals["A"], loop=True)
                burst["on"] = log["events"]["burst_on"] = time.monotonic()
            elif burst["on"] is not None and burst["off"] is None and time.monotonic() - burst["on"] >= args.burst:
                play(None)
                burst["off"] = log["events"]["burst_off"] = time.monotonic()
            return False

        live_maps(session, args.listen, quiet=0.0, log=log, during=sound_control)
        play(None)
        if burst["on"] is not None and burst["off"] is None:
            log["events"]["burst_off"] = time.monotonic()
        logs.append(log)
        print(f"  start {trial + 1}: {len(log['maps'])} maps logged, {len(log['targets'])} target reports", flush=True)
    save_log(os.path.join(out, "l7_maps.npz"), logs)
    session.values["l7"] = live_log_analyse(out).get("l7", {})
    for run in session.values["l7"].get("runs", []):
        session.report.info(f"L7 {run['label']}", f"{run['maps']}/{run['expected']} maps; the replay agrees with "
                                                  f"{run['replay_agrees']:.0%} of {run['board_targets']} targets")
    for label, values in session.values["l7"].get("settings", {}).items():
        session.report.info(f"L7 {label}", ", ".join(f"{k} {v:.2g}" for k, v in values.items()))


def l6(session, args, out, array):
    """Two sources: the drone clip on the speaker, speech or music on a phone. Which one Live reports, and how often
    it reports the phone with no drone (Live ranks by level; it has no classifier yet)."""
    distance = place(args.drone_az, 0.0, args.distance)
    ask_operator(f"Phone (speech or music, for example a podcast) {describe(args.phone_az, 0.0, distance)}, "
                 "paused. Enter when ready.")
    drone, quiet = player(session, args.sound), player(session, None)

    def ask(text):
        return lambda: ask_operator(f"  {text} Enter.")

    def trials(start=None, stop=None):
        return [live_maps(session, args.listen, start, stop) for _ in range(args.trials)]

    phases = {"drone only": trials(drone, quiet),
              "phone starts after Live": trials(ask("Start the phone now, then"), ask("Pause the phone, then"))}
    drone_hits = [t for _, targets in phases["drone only"] for t in targets]
    phone_hits = [t for _, targets in phases["phone starts after Live"] for t in targets]
    if not drone_hits:
        raise RuntimeError("Live did not report the drone clip on its own: raise the volume or move the speaker closer")
    drone_ref = (float(np.median([t[1] for t in drone_hits])), float(np.median([t[2] for t in drone_hits])))
    phone_ref = ((float(np.median([t[1] for t in phone_hits])), float(np.median([t[2] for t in phone_hits])))
                 if phone_hits else (args.phone_az, 0.0))
    drone_sector, phone_sector = session.speaker_sector(*drone_ref), session.speaker_sector(*phone_ref)
    with session.settings(**MCU_BEAM):
        floors = {s: session.capture("l6", f"silent, sector {s}", s, 2.0)["mean_db"] for s in (drone_sector, phone_sector)}
        drone()
        drone_level = signal_db(session.capture("l6", "drone, its beam", drone_sector, 3.0)["mean_db"], floors[drone_sector])
        quiet()
    ask_operator("\nStart the phone and leave it playing through the next trials. Enter when it plays.")
    with session.settings(**MCU_BEAM):
        phone_level = signal_db(session.capture("l6", "phone, its beam", phone_sector, 3.0)["mean_db"], floors[phone_sector])
    phases["phone on, drone starts"] = trials(drone, quiet)
    phases["phone on, no drone"] = trials()
    ask_operator("Pause the phone. Enter.")
    result = {}
    for name, runs in phases.items():
        counts = {"drone": 0, "phone": 0, "other": 0, "none": 0}
        for count, targets in runs:
            by_map = {}
            for t in targets:
                by_map.setdefault(t[0], t)
            for t in by_map.values():
                counts["drone" if angle_off(t[1], t[2], drone_ref) <= TOLERANCE_DEG else
                       "phone" if angle_off(t[1], t[2], phone_ref) <= TOLERANCE_DEG else "other"] += 1
            counts["none"] += max(count - len(by_map), 0)
        total = max(sum(counts.values()), 1)
        result[name] = {key: value / total for key, value in counts.items()}
        session.report.info(f"L6 {name}", ", ".join(f"{key} {value:.0%}" for key, value in result[name].items())
                            + " of maps")
    relative = phone_level - drone_level
    session.report.info("L6 levels at the array", f"phone {relative:+.1f} dB re the drone clip, each in its own beam")
    both = result["phone on, drone starts"]
    session.report.add("L6 with the phone playing, Live reports the drone more often than the phone",
                       both["drone"] > both["phone"], f"drone {both['drone']:.0%}, phone {both['phone']:.0%} of maps "
                                                      f"(phone {relative:+.1f} dB re the drone)")
    session.values["l6"] = {"drone_reference": drone_ref, "phone_reference": phone_ref, "drone_level_db": drone_level,
                            "phone_level_db": phone_level, "phone_re_drone_db": relative, "phases": result,
                            "sound": args.sound}
    figure_l6(result, relative, out)


def t1(session, args, out, array):
    """Tonal detector: TONE reports from the beam on the speaker for silence, a synthetic 200 Hz harmonic series, the
    drone clip and crowd noise, and a real drone with --real-drone. The DSP detector is used, so no MCU window loads
    the core while the tonal detector runs."""
    place(0.0, 0.0, args.distance)
    sector = session.speaker_sector(0, 0)

    def listen():
        reports, last = [], [None]

        def collect(state):
            tone = (state.get("tones") or {}).get("0")
            key = json.dumps(tone, sort_keys=True) if tone else None
            if tone and key != last[0]:
                last[0] = key
                reports.append((tone["f0_hz"], tone["score_db"], tone["harmonics"], tone["level_db"]))
            return False

        session.board.pump(args.listen, collect)
        return reports

    reports = {}
    with session.settings(measure=0, channel=0, tone=args.tone_ms):
        if session.read(f"S,{sector}") is None:
            raise RuntimeError(f"S,{sector} gave no reading ({session.board.state.get('last_error')})")
        for name, sound in (("silence", None), ("harmonic 200 Hz", "D200"), ("drone clip", args.sound),
                            ("crowd", "crowd")):
            session.sound(sound)
            reports[name] = listen()
        session.sound(None)
        if args.real_drone:
            ask_operator("\nT1: hold the drone 1-2 m in front of the array with its motors running (propellers off "
                         "indoors). Enter to listen.")
            reports["real drone"] = listen()
            ask_operator("Stop the motors. Enter.")
    stats = {name: tone_stats(values) for name, values in reports.items()}
    for name, item in stats.items():
        session.report.info(f"T1 {name}", "no TONE reports" if not item["reports"] else
                            f"f0 {item['f0_median_hz']:.1f} Hz (IQR {item['f0_iqr_hz']:.1f}), score "
                            f"{item['score_median_db']:.1f} dB (10-90% {item['score_p10_db']:.1f} to "
                            f"{item['score_p90_db']:.1f}), {item['harmonics_median']:.0f} harmonics, "
                            f"{item['reports']} reports")
    synthetic = stats["harmonic 200 Hz"]
    session.report.add("T1 synthetic 200 Hz series found", bool(synthetic["reports"]) and
                       abs(synthetic["f0_median_hz"] - 200.0) <= 2.0,
                       f"median {synthetic.get('f0_median_hz', float('nan')):.1f} Hz")
    crowd = stats["crowd"]
    for name in ("drone clip", "real drone"):
        item = stats.get(name)
        if item and item["reports"] and crowd["reports"]:
            session.report.add(f"T1 {name} scores above the crowd", item["score_median_db"] > crowd["score_p90_db"],
                               f"median {item['score_median_db']:.1f} dB against the crowd's 90th percentile "
                               f"{crowd['score_p90_db']:.1f} dB")
    session.values["t1"] = {"sound": args.sound, "tone_ms": args.tone_ms, "stats": stats,
                            "reports": {name: values for name, values in reports.items()}}
    figure_t1(reports, out)


def figure_p2(guards, fractions, errors, fit, recommended, current, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    g, f, e = (np.asarray(v, float) for v in (guards, fractions, errors))
    ok = np.isfinite(f) & np.isfinite(e)
    ax.errorbar(g[ok] / 1000, f[ok], yerr=e[ok], fmt="o", color="k", label="measured")
    if np.isfinite(fit).all():
        x = np.linspace(0, g.max(), 400)
        ax.plot(x / 1000, np.clip((x - fit[0]) / fit[1], 0, 1), "--", label="fit")
    ax.axvline(current / 1000, color="tab:red", lw=0.8, label=f"guard on the board {current / 1000:g} ms")
    ax.axvline(recommended / 1000, color="tab:green", lw=0.8, label=f"guard to use {recommended / 1000:g} ms")
    ax.set_xlabel("guard after the steer (ms)")
    ax.set_ylabel("share of the new direction in the window")
    ax.set_title(f"P2: audio-path latency ({P2_WINDOW_US / 1000:g} ms windows)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "p2_latency.png"), dpi=150)


def figure_s3(result, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (mode, entry) in zip(axes, result.items()):
        x = np.asarray(entry["played_db"], float)
        ax.errorbar(x, entry["measured_db"], yerr=entry["se_db"], fmt="o", color="k", label="measured")
        fine = np.linspace(x.min(), x.max(), 200)
        floor = 10 ** (entry["floor_db"] / 20)
        for model, item in entry["models"].items():
            a = 10 ** (item["scale_db"] / 20)
            word = np.sqrt((a * 10 ** (fine / 20)) ** 2 + floor ** 2) if model == "amplitude" else \
                a * 10 ** (fine / 10) + floor
            ax.plot(fine, 20 * np.log10(word), "--", label=f"follows {model} (chi2 {item['chi2']:.0f})")
        ax.axhline(entry["floor_db"], color="gray", lw=0.8, label="silent floor")
        ax.set_xlabel("played level re the loudest step (dB)")
        ax.set_ylabel("reading (dBFS)")
        ax.set_title(mode)
        ax.legend(fontsize=8)
    fig.suptitle("S3: linearity and the detector's scale")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "s3_levels.png"), dpi=150)


def figure_l5(levels, live_rate, scan_rate, live_level, scan_level, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(levels, live_rate, "o-", label="Live (bank), per map")
    ax.plot(levels, scan_rate, "s-", label="continuous scan, per pass")
    ax.axhline(0.5, color="gray", lw=0.8)
    for level, color in ((live_level, "tab:blue"), (scan_level, "tab:orange")):
        if level is not None:
            ax.axvline(level, color=color, lw=0.8, ls=":")
    ax.set_xlabel("level of sound A re the loudest step (dB)")
    ax.set_ylabel("detection rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("L5: detection margin, Live against scanning")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "l5_detection.png"), dpi=150)


def figure_l6(result, relative, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names, bottom = list(result), np.zeros(len(result))
    for key, color in (("drone", "tab:green"), ("phone", "tab:red"), ("other", "tab:gray"), ("none", "lightgray")):
        values = np.array([result[name][key] for name in names])
        ax.bar(names, values, bottom=bottom, color=color, label=key)
        bottom += values
    ax.set_ylabel("share of bank maps")
    ax.set_title(f"L6: what Live reports with two sources (phone {relative:+.1f} dB re the drone)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "l6_two_sources.png"), dpi=150)


def figure_t1(reports, out):
    import matplotlib.pyplot as plt
    names = [name for name, values in reports.items() if values]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([[r[1] for r in reports[name]] for name in names])
    ax.set_xticks(range(1, len(names) + 1), names)
    ax.set_ylabel("tonal score (dB)")
    ax.set_title("T1: tonal detector score by source")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "t1_tones.png"), dpi=150)


# Detector time constant ---------------------------------------------------------------------------

D1_SETTLES_US = (250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 20000)
# After a 20 ms settle: host waits before an M read (timed from the replies, so to about one frame).
D1_TAIL_MS = (50, 100, 200, 400, 800)
D1_RISE_S = 0.3
D1_RATE_HZ = 48000
# The open scale question: the exponent that turns the word into the smoother's mean square.
D1_SCALES = {"power": 1, "amplitude": 2}


def fit_decay(settles_ms, words, loud, floor, exponent):
    """Fit word**e = floor**e + (loud**e - floor**e) * exp(-(t - delay) / tau) on the points between 3% and
    90% of the step. {tau_ms, delay_ms, residual (rms of the log fit)}, or None."""
    t = np.asarray(settles_ms, float)
    w, lo, hi = (np.asarray(v, float) ** exponent for v in (words, floor, loud))
    y = (w - lo) / (hi - lo)
    use = (y > 0.03) & (y < 0.9)
    if use.sum() < 3:
        return None
    slope, intercept = np.polyfit(t[use], np.log(y[use]), 1)
    if slope >= 0:
        return None
    residual = np.log(y[use]) - (slope * t[use] + intercept)
    return {"tau_ms": float(-1.0 / slope), "delay_ms": float(intercept / -slope),
            "residual": float(np.sqrt(np.mean(residual ** 2)))}


def settle_within(settles_ms, excess_db, within_db):
    """The first settle time with the reading within `within_db` of the quiet level (linear between steps)."""
    for i, (t, e) in enumerate(zip(settles_ms, excess_db)):
        if e <= within_db:
            if i == 0:
                return float(t)
            t0, e0 = settles_ms[i - 1], excess_db[i - 1]
            return float(t0 + (t - t0) * (e0 - within_db) / (e0 - e))
    return None


def d1(session, args, out, array):
    """Time constant of the scan level detector (the DSP detector of the steered beam). The beam holds the
    speaker, then steers to a quiet corner; the firmware reads the detector after settle_us (up to 20 ms, timed
    by the firmware). Later points are M reads timed by the host (coarse). The readings trace the detector's
    decay, with its hold and decay words. Fitted as a power word and as an amplitude word (the scale is an open
    question), and compared with the TCONST word on the device."""
    place(0.0, 0.0, args.distance)
    rng = np.random.default_rng(args.seed)
    rows, columns = session.home[0]["rows"], session.home[0]["columns"]
    on = session.speaker_sector(0, 0)
    corners = sorted({0, columns - 1, (rows - 1) * columns, rows * columns - 1})
    status, registry = http("GET", "/dsp/registry")
    params = {p.get("name"): p for p in (registry.get("parameters", []) if status == 200 else [])
              if p.get("group") == "detector"}
    if "TCONST" not in params:
        raise RuntimeError("the register map has no scan detector TCONST")
    words = {name: read_words(int(p["address"]), 1)[0] for name, p in params.items()}
    alpha = (words["TCONST"] if words["TCONST"] < 2 ** 31 else words["TCONST"] - 2 ** 32) / 2 ** 24
    expected = float(-1000.0 / (D1_RATE_HZ * np.log(1.0 - alpha))) if 0 < alpha < 1 else float("nan")
    hold_ms = words["hold"] / (D1_RATE_HZ / 1000) if "hold" in words else None
    longest = max(D1_SETTLES_US) / 1000
    readings = []

    def held(sector):
        """The detector word (linear) after the beam has held `sector` for 1 s."""
        session.read(f"S,{sector}")
        session.board.pump(1.0)
        levels = [m["raw_level"] / 2 ** 24 for m in (session.read("M") for _ in range(5)) if m]
        return float(np.mean(levels)) if levels else float("nan")

    def keep(t_ms, measurement, timing, nominal):
        level = measurement["raw_level"] / 2 ** 24
        readings.append({"t_ms": t_ms, "word": level, "timing": timing, "nominal": nominal})
        session.writer.writerow({"step": "d1", "label": f"{timing} {nominal}", "sector": off, "value": round(t_ms, 3),
                                 "n": 1, "mean_db": float(20 * np.log10(max(level, 1e-9))), "levels": [round(level, 9)]})

    session.sound("A")
    try:
        with session.settings(measure=0, settle_us=max(D1_SETTLES_US)):
            steady = {s: held(s) for s in [on, *corners]}
            off = min(corners, key=lambda s: steady[s])
            loud, floor = steady[on], steady[off]
            contrast = float(20 * np.log10(loud / floor)) if loud > 0 and floor > 0 else 0.0
            if contrast < 20.0:
                session.report.add("D1 contrast between the speaker and a quiet corner", False,
                                   f"{contrast:.1f} dB, needs 20 dB: turn the speaker up or move it closer, and repeat")
                return
            for settle in rng.permutation([s for s in D1_SETTLES_US for _ in range(args.repeats)]):
                settle = int(settle)
                session.set("settle_us", settle)
                session.read(f"S,{on}")
                session.board.pump(D1_RISE_S)
                measurement = session.read(f"S,{off}")
                if measurement:
                    keep(settle / 1000, measurement, "firmware", f"settle {settle} us")
            session.set("settle_us", max(D1_SETTLES_US))
            for wait in rng.permutation([w for w in D1_TAIL_MS for _ in range(args.repeats)]):
                wait = int(wait)
                session.read(f"S,{on}")
                session.board.pump(D1_RISE_S)
                if session.read(f"S,{off}") is None:
                    continue
                start = time.monotonic()
                session.board.pump(wait / 1000)
                measurement = session.read("M")
                if measurement:
                    keep(longest + (time.monotonic() - start) * 1000, measurement, "host", f"wait {wait} ms")
    finally:
        session.sound(None)
    groups = sorted({(r["timing"], r["nominal"]) for r in readings},
                    key=lambda g: np.mean([r["t_ms"] for r in readings if (r["timing"], r["nominal"]) == g]))
    t = [float(np.mean([r["t_ms"] for r in readings if (r["timing"], r["nominal"]) == g])) for g in groups]
    mean = [float(np.mean([r["word"] for r in readings if (r["timing"], r["nominal"]) == g])) for g in groups]
    excess = [float(20 * np.log10(max(w, 1e-12) / floor)) for w in mean]
    fits = {scale: fit_decay(t, mean, loud, floor, e) for scale, e in D1_SCALES.items()}
    within = {db: settle_within(t, excess, db) for db in (3.0, 1.0)}
    session.values["d1"] = {"words": words, "expected_tau_ms": expected, "hold_ms": hold_ms, "fits": fits,
                            "loud": loud, "floor": floor, "contrast_db": contrast, "sectors": {"on": on, "off": off},
                            "readings": readings, "t_ms": t, "mean_word": mean, "excess_db": excess,
                            "settle_within_ms": {str(db): v for db, v in within.items()}}
    session.report.info("D1 detector words on the device", f"TCONST {expected:.2f} ms" + (
        f", hold {words['hold']} samples ({hold_ms:.1f} ms)" if hold_ms is not None else "") + (
        f", decay {words['decay'] / 2 ** 24:.6f} per sample" if "decay" in words else ""))
    for db, value in within.items():
        session.report.info(f"D1 time to within {db:g} dB of quiet", f"more than {max(t):.0f} ms" if value is None
                            else f"{value:.1f} ms ({'firmware' if value <= longest else 'host'} timed; dB as the "
                                 f"screens show them; the firmware waits 20 ms at most)")
    if not any(fits.values()):
        session.report.add("D1 time constant", False, "the readings do not decay between 3% and 90%: no fit")
        figure_d1(t, excess, readings, loud, floor, fits, expected, out)
        return
    detail = ", ".join(f"{scale} {fit['tau_ms']:.2f} ms" for scale, fit in fits.items() if fit)
    matches = [scale for scale, fit in fits.items()
               if fit and np.isfinite(expected) and abs(fit["tau_ms"] - expected) <= 0.15 * expected]
    session.report.add("D1 time constant against the TCONST word", bool(matches),
                       f"{detail} measured; {expected:.2f} ms from the word (limit +-15%)"
                       + (f"; matches the word read as {' and '.join(matches)}" if matches else ""))
    best = min((fit for fit in fits.values() if fit), key=lambda fit: fit["residual"])
    session.report.info("D1 delay before the decay", f"{best['delay_ms']:.2f} ms" + (
        f" (the hold word is {hold_ms:.1f} ms)" if hold_ms is not None else ""))
    figure_d1(t, excess, readings, loud, floor, fits, expected, out)


def figure_d1(t, excess, readings, loud, floor, fits, expected, out):
    import matplotlib.pyplot as plt
    grid = np.geomspace(min(t), max(t), 300)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for timing, colour in (("firmware", "0.55"), ("host", "tab:orange")):
        points = [r for r in readings if r["timing"] == timing]
        ax.plot([r["t_ms"] for r in points], [20 * np.log10(max(r["word"], 1e-12) / floor) for r in points], ".",
                color=colour, label=f"readings ({timing} timed)")
    ax.plot(t, excess, "ko", label="mean of the repeats")
    for (scale, fit), style in zip(fits.items(), ("-", "-.")):
        if not fit:
            continue
        e = D1_SCALES[scale]
        for tau, line, label in ((fit["tau_ms"], style, f"fit as {scale}: tau {fit['tau_ms']:.2f} ms"),
                                 (expected, ":", f"TCONST word as {scale}: {expected:.2f} ms")):
            if not np.isfinite(tau):
                continue
            mean_square = floor ** e + (loud ** e - floor ** e) * np.exp(-np.maximum(grid - fit["delay_ms"], 0) / tau)
            ax.plot(grid, 20 * np.log10(mean_square ** (1 / e) / floor), line, label=label)
    for db in (3.0, 1.0):
        ax.axhline(db, color="0.8", linewidth=0.8)
    ax.axvline(max(D1_SETTLES_US) / 1000, color="0.8", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("time from the steer to the read (ms); right of the dashed line: host timed")
    ax.set_ylabel("reading above the quiet level (dB, as the screens show)")
    ax.set_title("D1: scan detector after a steer from the speaker to a quiet corner")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out, "d1_detector_decay.png"), dpi=150, bbox_inches="tight")


TESTS = {"n1": n1, "n2": n2, "a3": a3, "a4": a4, "a8": a8, "a9": a9, "a10": a10, "l1": l1, "l5": l5, "l6": l6,
         "l7": l7, "p2": p2, "r1": r1, "s1": s1, "s3": s3, "t1": t1, "d1": d1}
SOUND_TESTS = {"a3", "a4", "a8", "a9", "a10", "l1", "l5", "l6", "l7", "p2", "r1", "s1", "s3", "t1", "d1"}
TEST_DEFAULTS = {"l1": {"trials": 5, "listen": 8.0}, "l5": {"trials": 3, "listen": 6.0},
                 "l6": {"trials": 3, "listen": 8.0}, "l7": {"trials": 5, "listen": 60.0}, "t1": {"listen": 12.0}}


def defaults_for(test):
    """Option values of one test that differ from the parser defaults."""
    values = dict(TEST_DEFAULTS.get(test, {}))
    if test == "l1":
        values["positions"] = L1_POSITIONS
    if test in ("l6", "t1"):
        values["sound"] = "drone"
    return values


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("test", choices=sorted(TESTS))
    parser.add_argument("--seconds", type=float, help="capture per point (default per test)")
    parser.add_argument("--cycles", type=int, default=3, help="n1/n2: interleaved repeats, alternating order")
    parser.add_argument("--repeats", type=int, default=3, help="a3/a4/r1/d1: repeats per set")
    parser.add_argument("--distance", type=float, default=2.0, help="nominal speaker distance (m)")
    parser.add_argument("--bands", default="B2k", help="a3: comma list of B1k, B2k, B3k")
    parser.add_argument("--positions", default="-30,0;-15,0;0,0;15,0;30,0",
                        help="a4/l1: 'az,el;az,el;...' nominal positions (l1 default: 13 positions)")
    parser.add_argument("--alignment-sigma", type=float, default=0.4, help="speaker placement 1-sigma (deg)")
    parser.add_argument("--truth", choices=("array", "measured"), default="array",
                        help="a3/l1: find the speaker with the steered beam, or type in measured positions")
    parser.add_argument("--trials", type=int, help="l1/l5/l6/l7: trials per position, level or phase, or Live "
                                                   "starts (default 5/3/3/5)")
    parser.add_argument("--listen", type=float, help="l1/l5/l6/l7/t1: seconds of sound per trial (default "
                                                     "8/6/8/60/12)")
    parser.add_argument("--sound", default="A", help="l1/l6/t1: signal name, 'drone', 'crowd' or a WAV path "
                                                     "(l6 and t1 default to the drone clip)")
    parser.add_argument("--steers", type=int, default=40, help="p2: steers per guard value")
    parser.add_argument("--guard", type=int, help="l5: guard_us for the scan's MCU windows (from p2)")
    parser.add_argument("--steps", help="l5: level steps in dB below sound A, comma list (default 0,6,...,48)")
    parser.add_argument("--burst", type=float, default=0.0, help="l7: seconds of loud sound A in each start (0: none)")
    parser.add_argument("--burst-at", type=float, default=20.0, help="l7: seconds into each start for the burst")
    parser.add_argument("--drone-az", type=float, default=-15.0, help="l6: speaker azimuth (deg)")
    parser.add_argument("--phone-az", type=float, default=15.0, help="l6: phone azimuth (deg)")
    parser.add_argument("--tone-ms", type=int, default=100, help="t1: tonal detector report period (ms)")
    parser.add_argument("--real-drone", action="store_true", help="t1: also listen to a real drone")
    parser.add_argument("--microphone", type=int, default=4, help="r1: microphone on Ch1")
    parser.add_argument("--microphones", default="0,4,20,24", help="s1: microphones to calibrate")
    parser.add_argument("--references", type=int, default=8, help="a9: reference microphones (seeded, spread)")
    parser.add_argument("--near-distance", type=float, default=1.0, help="a9: speaker distance for the gains (m)")
    parser.add_argument("--record", type=float, default=5.0, help="r1: recording length (s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="output folder (default: temp)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    for key, value in defaults_for(args.test).items():
        if getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)
    out = args.out or os.path.join(tempfile.gettempdir(), time.strftime(f"heimdall_{args.test}_%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    signals = make_signals(out, level_steps(args) if args.test == "l5" else ()) if args.test in SOUND_TESTS else {}
    if signals:
        signals.update({name: str(AUDIO / f"{name}.wav") for name in ("drone", "crowd")
                        if (AUDIO / f"{name}.wav").exists()})
    if args.test in ("l1", "l6", "t1") and args.sound not in signals:
        if not os.path.exists(args.sound):
            sys.exit(f"no signal or file {args.sound}")
        signals[args.sound] = args.sound
    report = Results()
    array = Array()
    with connect(WS_URL, open_timeout=5, max_size=None) as socket, \
            open(os.path.join(out, "captures.csv"), "w", newline="") as handle:
        board = Board(socket)
        board.pump(1.0)
        if not board.state.get("connected"):
            sys.exit("the backend is not connected to the board")
        reload_dsp(board)
        writer = csv.DictWriter(handle, FIELDS, extrasaction="ignore")
        writer.writeheader()
        session = Session(board, signals, writer, report)
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            TESTS[args.test](session, args, out, array)
        finally:
            session.restore()
        summary = {"test": args.test, "started": started, "arguments": vars(args),
                   "map_hash": board.state.get("map_hash"), "settings": session.settings_before,
                   "dsp_edits": {str(a): w for a, w in run_edits().items()}, "values": session.values,
                   "checks": [{"name": name, "result": "info" if passed is None else ("pass" if passed else "fail"),
                               "detail": detail} for name, passed, detail in report.rows]}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(f"results: {out}")


if __name__ == "__main__":
    main()
