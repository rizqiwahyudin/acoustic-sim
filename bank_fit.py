"""bank_fit.py: the direction whose expected bank map best matches a measured Live map (the pattern fit).

Every direction has a fingerprint: the level that each bank beam gives a plane wave from that direction, through
the bank as the DSP runs it (each microphone's total delay from the contract, then the bank band-pass), summed over
the band. A Live map is fitted as gain x fingerprint + floor by least squares, and the best-fitting direction wins.
The sidelobes are inside the fingerprints, so they count for the right direction instead of making false targets;
the floor takes the even spread of a room. It is a one-source fit: a strong reflection pulls the answer toward it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitResult:
    azimuth_deg: float
    elevation_deg: float
    quality: float  # share of the map's variation that the fitted pattern explains (R squared), 0 to 1
    gain: float
    floor: float

    def as_dict(self) -> dict[str, float]:
        return {"azimuth_deg": round(self.azimuth_deg, 2), "elevation_deg": round(self.elevation_deg, 2),
                "quality": round(self.quality, 4), "gain": round(self.gain, 3), "floor": round(self.floor, 3)}


class BankFit:
    """Fingerprints of a direction grid for one bank, and the fit of a map against them.

    positions_mm are the microphones as mounted (x right, y up, seen from behind), as the contract records them;
    a turned board only renumbers which beam points where, and mic_delays is already in physical beam order.
    """

    def __init__(self, positions_mm, mic_delays, band_pass, rate_hz, speed_of_sound_m_s, *,
                 azimuth_deg=(-40.0, 40.0), elevation_deg=(-40.0, 40.0), step_deg=2.0, scale="amplitude",
                 spectrum="pink", band_hz=(300.0, 5000.0), frequencies=48):
        if scale not in ("amplitude", "power") or spectrum not in ("pink", "flat"):
            raise ValueError("scale must be amplitude or power, spectrum pink or flat")
        positions = np.asarray(positions_mm, float) / 1000.0
        self._x, self._y = (positions - positions.mean(axis=0)).T
        self._delays_s = np.asarray(mic_delays, float) / float(rate_hz)
        if self._delays_s.ndim != 2 or self._delays_s.shape[1] != len(self._x):
            raise ValueError("mic_delays must hold one delay per microphone for every beam")
        self.scale, self.speed = scale, float(speed_of_sound_m_s)
        self._frequencies = np.linspace(band_hz[0], band_hz[1], frequencies)
        taps = np.asarray(band_pass, float)
        response = np.exp(-2j * np.pi * np.outer(self._frequencies, np.arange(len(taps))) / rate_hz) @ taps
        self._weights = np.abs(response) ** 2 / (self._frequencies if spectrum == "pink" else 1.0)
        self.azimuths = np.arange(azimuth_deg[0], azimuth_deg[1] + step_deg / 2, step_deg)
        self.elevations = np.arange(elevation_deg[0], elevation_deg[1] + step_deg / 2, step_deg)
        az, el = np.meshgrid(self.azimuths, self.elevations)
        templates = self._patterns(az.ravel(), el.ravel())
        self._scale = float(templates.max())
        self.templates = templates / self._scale
        self._centred = self.templates - self.templates.mean(axis=1, keepdims=True)
        self._norms = (self._centred ** 2).sum(axis=1)

    @classmethod
    def from_contract(cls, contract, **options) -> "BankFit | None":
        """The fit for a beam table contract, or None when it has no bank or no per-microphone bank delays."""
        bank = (contract or {}).get("bank") or {}
        if not bank.get("mic_delays"):
            return None
        view = contract.get("field_of_view_deg") or {}
        return cls(contract["microphone_positions_mm"], bank["mic_delays"], bank["band_pass"], bank["sample_rate_hz"],
                   contract["speed_of_sound_m_s"], azimuth_deg=tuple(view.get("azimuth", (-40.0, 40.0))),
                   elevation_deg=tuple(view.get("elevation", (-40.0, 40.0))), **options)

    @property
    def beams(self) -> int:
        return self._delays_s.shape[0]

    def unit_patterns(self) -> np.ndarray:
        """What the fit correlates a map with: each template minus its mean, scaled to length 1 (directions x
        beams, elevation-major as the grid). The MAX78002 runs the fit with these (bank_fit_table.h)."""
        return self._centred / np.sqrt(self._norms)[:, None]

    def _patterns(self, azimuth_deg, elevation_deg) -> np.ndarray:
        """Band level of every beam for plane waves from these directions: directions x beams."""
        az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
        u, v = np.cos(el) * np.sin(az), np.sin(el)
        arrival_s = -(np.outer(u, self._x) + np.outer(v, self._y)) / self.speed
        power = np.zeros((len(az), self.beams))
        for frequency, weight in zip(self._frequencies, self._weights):
            omega = 2j * np.pi * frequency
            power += weight * np.abs(np.exp(-omega * arrival_s) @ np.exp(-omega * self._delays_s).T) ** 2
        return np.sqrt(power) if self.scale == "amplitude" else power

    def fingerprint(self, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
        """The expected map (one value per beam) of a source in this direction, on the templates' scale."""
        return self._patterns(np.atleast_1d(azimuth_deg), np.atleast_1d(elevation_deg))[0] / self._scale

    def fit(self, levels) -> FitResult | None:
        """The best direction for one map (levels in the bank's beam order), or None for a flat or invalid map."""
        levels = np.asarray(levels, float)
        if levels.shape != (self.beams,) or not np.all(np.isfinite(levels)):
            return None
        centred = levels - levels.mean()
        total = float(centred @ centred)
        if total <= 0.0:
            return None
        projection = self._centred @ centred
        explained = np.where(projection > 0, projection ** 2 / self._norms, 0.0)
        best = int(np.argmax(explained))
        if explained[best] <= 0.0:
            return None
        row, column = divmod(best, len(self.azimuths))
        grid = explained.reshape(len(self.elevations), len(self.azimuths))
        azimuth = self.azimuths[column] + self._offset(grid[row, :], column) * self._step(self.azimuths)
        elevation = self.elevations[row] + self._offset(grid[:, column], row) * self._step(self.elevations)
        gain = float(projection[best] / self._norms[best])
        return FitResult(float(azimuth), float(elevation), float(explained[best] / total), gain,
                         float(levels.mean() - gain * self.templates[best].mean()))

    @staticmethod
    def _step(values) -> float:
        return float(values[1] - values[0]) if len(values) > 1 else 0.0

    @staticmethod
    def _offset(scores, index) -> float:
        """Parabolic peak offset (grid steps, within half a step) from the scores on both sides."""
        if index == 0 or index == len(scores) - 1:
            return 0.0
        left, centre, right = scores[index - 1], scores[index], scores[index + 1]
        curvature = left - 2 * centre + right
        if curvature >= 0:
            return 0.0
        return float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))


class BankFitStream:
    """Fits the mean of the last few bank maps (4 maps = 200 ms at 20 maps/s).

    On the logs of 2026-10-26 the mean of 4 maps cut the far jumps of the fitted direction by 3 to 5 times against
    single maps, and followed a moving speaker as fast as single maps did. A counter that goes back (a new Live
    start) clears it.
    """

    def __init__(self, fit: BankFit, maps: int = 4):
        self.fit, self.maps = fit, max(1, int(maps))
        self._recent: list[np.ndarray] = []
        self._last_count = None
        self.result: FitResult | None = None

    def push(self, count, levels) -> FitResult | None:
        """Add one map (its counter and levels); returns the fit of the recent mean. A repeated counter is ignored."""
        if count is not None and self._last_count is not None:
            if count == self._last_count:
                return self.result
            if count < self._last_count:
                self._recent = []
        self._last_count = count
        levels = np.asarray(levels, float)
        if levels.shape != (self.fit.beams,) or not np.all(np.isfinite(levels)):
            return self.result
        self._recent = (self._recent + [levels])[-self.maps:]
        self.result = self.fit.fit(np.mean(self._recent, axis=0))
        return self.result
