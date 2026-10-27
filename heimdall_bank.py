"""Host copy of the firmware's bank tracker (MAX78002/beam_bank.c), for the emulator.

Each map, a beam is a candidate when its level stands out by ACQUIRE_RATIO from both
its own slow background and the map's median, and no 4-neighbour is louder. The best
candidate's direction is the excess-weighted centroid of its 3 x 3 neighbourhood in
sine space, and the steered beam goes to the nearest sector. Backgrounds carry
FRACTION_BITS fraction bits; levels at or below LEVEL_FLOOR are never candidates.
A beam above its background keeps it for freeze_maps maps (the freeze_s setting),
then adapts with HELD_SHIFT.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

ACQUIRE_RATIO_NUM, ACQUIRE_RATIO_DEN = 3, 2
BACKGROUND_SHIFT = 5
HELD_SHIFT = 9
FRACTION_BITS = 8
LEVEL_FLOOR = 4
LOST_MAPS = 10
PERIOD_S = 0.05
FREEZE_S = 60  # default of the freeze_s setting
FREEZE_MAPS = round(FREEZE_S / PERIOD_S)


@dataclass
class BankResult:
    beams: list[int]
    levels: list[int]
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    sector: int = 0


def _above(level: int, reference: int) -> bool:
    """level >= ACQUIRE_RATIO x reference, with the reference scaled by 2^FRACTION_BITS."""
    return level > LEVEL_FLOOR and (level << FRACTION_BITS) * ACQUIRE_RATIO_DEN >= reference * ACQUIRE_RATIO_NUM


class BankTracker:
    def __init__(self, rows: int, columns: int, azimuth_deg: list[float], elevation_deg: list[float],
                 sector_azimuth_deg: list[float], sector_elevation_deg: list[float],
                 freeze_maps: int = FREEZE_MAPS) -> None:
        self.rows, self.columns = rows, columns
        self.azimuth_deg, self.elevation_deg = list(azimuth_deg), list(elevation_deg)
        self.sector_azimuth_deg, self.sector_elevation_deg = list(sector_azimuth_deg), list(sector_elevation_deg)
        self.freeze_maps = freeze_maps
        self.background: list[int] | None = None
        self.held_maps: list[int] = []

    def update(self, levels: list[int]) -> BankResult:
        if self.background is None:
            self.background = [level << FRACTION_BITS for level in levels]
            self.held_maps = [0] * len(levels)
            return BankResult([], [])
        field = sorted(levels)[len(levels) // 2] << FRACTION_BITS
        above = [_above(level, bg) and _above(level, field) for level, bg in zip(levels, self.background)]
        candidates = [beam for beam in range(len(levels)) if above[beam] and self._local_peak(levels, beam)]
        candidates.sort(key=lambda beam: -Fraction(levels[beam], max(1, self.background[beam] >> FRACTION_BITS)))
        result = BankResult(candidates[:2], [levels[beam] for beam in candidates[:2]])
        if result.beams:
            result.azimuth_deg, result.elevation_deg = self._interpolate(levels, result.beams[0])
            result.sector = self.nearest_sector(result.azimuth_deg, result.elevation_deg)
        for beam, level in enumerate(levels):
            shift = BACKGROUND_SHIFT
            if above[beam]:
                if self.held_maps[beam] < self.freeze_maps:
                    self.held_maps[beam] += 1
                    continue
                shift = HELD_SHIFT
            else:
                self.held_maps[beam] = 0
            difference = (level << FRACTION_BITS) - self.background[beam]
            step = difference >> shift if difference >= 0 else -((-difference) >> shift)
            self.background[beam] += step
        return result

    def _local_peak(self, levels: list[int], beam: int) -> bool:
        row, column = divmod(beam, self.columns)
        for r, c in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            if 0 <= r < self.rows and 0 <= c < self.columns and levels[r * self.columns + c] > levels[beam]:
                return False
        return True

    def _interpolate(self, levels: list[int], beam: int) -> tuple[float, float]:
        row, column = divmod(beam, self.columns)
        weight = sx = sy = 0.0
        for r in range(row - 1, row + 2):
            for c in range(column - 1, column + 2):
                if not (0 <= r < self.rows and 0 <= c < self.columns):
                    continue
                b = r * self.columns + c
                excess = (levels[b] << FRACTION_BITS) - self.background[b]
                if excess <= 0:
                    continue
                el = math.radians(self.elevation_deg[b])
                sx += excess * math.cos(el) * math.sin(math.radians(self.azimuth_deg[b]))
                sy += excess * math.sin(el)
                weight += excess
        if weight <= 0:
            return self.azimuth_deg[beam], self.elevation_deg[beam]
        el = math.asin(max(-1.0, min(1.0, sy / weight)))
        s = max(-1.0, min(1.0, sx / weight / math.cos(el)))
        return math.degrees(math.asin(s)), math.degrees(el)

    def nearest_sector(self, azimuth_deg: float, elevation_deg: float) -> int:
        columns = len(self.sector_azimuth_deg)
        return min(
            range(columns * len(self.sector_elevation_deg)),
            key=lambda sector: (self.sector_azimuth_deg[sector % columns] - azimuth_deg) ** 2
            + (self.sector_elevation_deg[sector // columns] - elevation_deg) ** 2,
        )
