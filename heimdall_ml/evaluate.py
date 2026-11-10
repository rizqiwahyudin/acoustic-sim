"""The same scoreboard for every classifier: detection rate, false alarms per hour, AUC, and both per group.

A classifier reports once per window (every hop), so a negative window at or above the threshold is one false
alarm, and the negative windows cover `hop_s` each.
"""
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Scored:
    score: float
    target: bool
    groups: dict = field(default_factory=dict)


def auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """Probability that a target window scores above a non-target window (ties count half)."""
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if positive.size == 0 or negative.size == 0:
        return None
    values = np.concatenate([positive, negative])
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = ranks[:positive.size].sum()
    return float((rank_sum - positive.size * (positive.size + 1) / 2.0) / (positive.size * negative.size))


def false_alarms_per_hour(negative: np.ndarray, threshold: float, hop_s: float) -> float | None:
    negative = np.asarray(negative, float)
    if negative.size == 0:
        return None
    return float(np.sum(negative >= threshold) / (negative.size * hop_s / 3600.0))


def threshold_for_false_alarms(negative: np.ndarray, hop_s: float, per_hour: float) -> float:
    """The lowest threshold that keeps the false alarms at or below `per_hour`."""
    ordered = np.sort(np.asarray(negative, float))[::-1]
    if ordered.size == 0:
        raise ValueError("no non-target windows")
    allowed = int(np.floor(per_hour * ordered.size * hop_s / 3600.0 + 1e-9))
    if allowed >= ordered.size:
        return float(ordered[-1])
    return float(np.nextafter(ordered[allowed], np.inf))


def _summary(items: list[Scored], threshold: float, hop_s: float) -> dict:
    positive = np.array([item.score for item in items if item.target])
    negative = np.array([item.score for item in items if not item.target])
    return {
        "positives": int(positive.size),
        "negatives": int(negative.size),
        "negative_hours": round(negative.size * hop_s / 3600.0, 4),
        "detection_rate": float(np.mean(positive >= threshold)) if positive.size else None,
        "false_alarms_per_hour": false_alarms_per_hour(negative, threshold, hop_s),
        "auc": auc(positive, negative),
    }


def evaluate(items: list[Scored], hop_s: float, threshold: float | None = None,
             false_alarms_per_hour_target: float | None = None, by: tuple[str, ...] = ()) -> dict:
    """Overall and per-group results at a threshold, or at the threshold for a false-alarm rate."""
    if (threshold is None) == (false_alarms_per_hour_target is None):
        raise ValueError("give a threshold or a false-alarm rate, not both")
    if threshold is None:
        threshold = threshold_for_false_alarms([item.score for item in items if not item.target], hop_s,
                                               false_alarms_per_hour_target)
    result = {"threshold": threshold, "hop_s": hop_s, **_summary(items, threshold, hop_s), "groups": {}}
    for key in by:
        buckets = defaultdict(list)
        for item in items:
            buckets[str(item.groups.get(key, ""))].append(item)
        result["groups"][key] = [{"value": value, **_summary(bucket, threshold, hop_s)}
                                 for value, bucket in sorted(buckets.items())]
    return result


def _cell(value, digits=1, percent=False):
    if value is None:
        return "-"
    return f"{100.0 * value:.{digits}f}%" if percent else f"{value:.{digits}f}"


def report(result: dict) -> str:
    lines = [f"threshold {result['threshold']:.4g}: detection {_cell(result['detection_rate'], percent=True)} of "
             f"{result['positives']} target windows; false alarms {_cell(result['false_alarms_per_hour'])} per hour "
             f"over {result['negative_hours']:.3f} h of other sound; AUC {_cell(result['auc'], 3)}"]
    for key, rows in result["groups"].items():
        lines.append(f"by {key}:")
        for row in rows:
            lines.append(f"  {row['value'] or '(none)':>16s}  detection {_cell(row['detection_rate'], percent=True):>7s}"
                         f" of {row['positives']:5d}   false alarms/h {_cell(row['false_alarms_per_hour']):>8s}"
                         f" over {row['negative_hours']:.3f} h")
    return "\n".join(lines)
