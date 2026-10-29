"""Read a telemetry log (heimdall_telemetry.py): a summary of the session, and its Live runs in the format of
live_log.py, so that the Live replay and its analyses work on any logged session.

    python scripts/telemetry_log.py <log> [--npz l7_maps.npz] [--analyse]
        prints the summary; --npz saves the Live runs for live_log.py; --analyse replays them now.

A Live run is the maps from SCAN_STARTED,B to the next SCAN_STOPPED, SCAN_STARTED or READY. HEALTH records
carry the board's own map count, so the summary also says how many LEVELS,B lines the host did not get.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from heimdall_protocol import ProtocolError, parse_line  # noqa: E402


def entries(path):
    """(unix time, direction, text) for each well-formed line of the log."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3 or parts[1] not in {"<", ">", "#"}:
                continue
            try:
                yield float(parts[0]), parts[1], parts[2]
            except ValueError:
                continue


def read_session(path):
    """Live runs (live_log.py format, plus their BANK_FIT records), HEALTH records, notes and counts."""
    runs, health, notes, sent, received, errors = [], [], [], Counter(), Counter(), Counter()
    run = None
    for t, direction, text in entries(path):
        if direction == "#":
            notes.append((t, text))
            if run is not None and text.startswith("mark "):
                run["events"][text[5:]] = t
            continue
        if direction == ">":
            sent[text.split(",", 1)[0]] += 1
            continue
        try:
            record = parse_line(text)
        except ProtocolError:
            errors["unparsed"] += 1
            continue
        kind = record["type"]
        received[kind] += 1
        if kind == "error":
            errors[record.get("reason", "error")] += 1
        if kind == "health":
            health.append((t, record))
        elif kind == "scan_started" and record.get("mode") == "B":
            stamp = datetime.fromtimestamp(t).strftime("%H:%M:%S")
            run = {"label": f"start {len(runs) + 1} ({stamp})", "start_maps": 0, "maps": [], "t": [],
                   "levels": [], "targets": [], "fits": [], "events": {}}
            runs.append(run)
        elif kind in {"scan_started", "scan_stopped", "ready"}:
            run = None
        elif run is not None and kind == "levels" and record.get("source") == "B":
            run["maps"].append(len(run["maps"]) + 1)
            run["t"].append(t)
            run["levels"].append(list(record["raw_levels"]))
        elif run is not None and run["maps"] and kind == "bank_target":
            run["targets"].append((run["maps"][-1], record["azimuth_deg"], record["elevation_deg"],
                                   record["beam"], record["second_beam"]))
        elif run is not None and run["maps"] and kind == "bank_fit":
            run["fits"].append((run["maps"][-1], record["azimuth_deg"], record["elevation_deg"], record["quality"]))
    return {"runs": runs, "health": health, "notes": notes, "sent": sent, "received": received, "errors": errors}


def missed_maps(session):
    """LEVELS,B lines that the board sent between the first and last HEALTH record but the host did not get."""
    health = session["health"]
    if len(health) < 2:
        return None
    first, last = health[0], health[-1]
    board = last[1]["bank_maps"] - first[1]["bank_maps"]
    host = sum(1 for run in session["runs"] for t in run["t"] if first[0] < t <= last[0])
    return max(0, board - host)


def summary(session):
    lines = []
    times = [t for run in session["runs"] for t in run["t"]] + [t for t, _ in session["notes"]]
    if times:
        lines.append(f"session: {datetime.fromtimestamp(min(times)):%Y-%m-%d %H:%M:%S}, "
                     f"{(max(times) - min(times)) / 60:.1f} min")
    lines.append("commands sent: " + ", ".join(f"{k} {v}" for k, v in session["sent"].most_common()))
    for run in session["runs"]:
        span = run["t"][-1] - run["t"][0] if len(run["t"]) > 1 else 0.0
        rate = (len(run["t"]) - 1) / span if span > 0 else 0.0
        lines.append(f"Live {run['label']}: {len(run['maps'])} maps in {span:.1f} s ({rate:.1f}/s), "
                     f"{len(run['targets'])} targets, {len(run['fits'])} fits, marks {list(run['events'])}")
    health = session["health"]
    if health:
        records = [r for _, r in health]
        lines.append(f"HEALTH: {len(records)} records; longest main-loop pass {max(r['max_pass_us'] for r in records)} us; "
                     f"audio overruns +{records[-1]['audio_overruns'] - records[0]['audio_overruns']}; lost console "
                     f"bytes +{records[-1]['console_lost'] - records[0]['console_lost']}; LEVELS,B lines missed "
                     f"{missed_maps(session)}")
    if session["errors"]:
        lines.append("errors: " + ", ".join(f"{k} {v}" for k, v in session["errors"].most_common()))
    for t, text in session["notes"]:
        lines.append(f"  {datetime.fromtimestamp(t):%H:%M:%S} {text}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log")
    parser.add_argument("--npz", help="save the Live runs here (l7_maps.npz format of live_log.py)")
    parser.add_argument("--analyse", action="store_true", help="replay the Live runs with live_log.py")
    args = parser.parse_args()
    session = read_session(args.log)
    print(summary(session))
    runs = [run for run in session["runs"] if run["maps"]]
    if args.npz and runs:
        from live_log import save_log
        save_log(args.npz, runs)
        print(f"saved {len(runs)} Live runs to {args.npz}")
    if args.analyse and runs:
        from live_log import analyse_runs
        analyse_runs("telemetry", runs, {})
