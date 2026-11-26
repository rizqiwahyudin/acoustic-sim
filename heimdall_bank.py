"""Host copy of the firmware's bank tracker (MAX78002/beam_bank.c), for the emulator.

Each map, a beam is a candidate when its level stands out by ACQUIRE_RATIO from both
its own slow background and the map's median, and no 4-neighbour is louder. The best
candidate's direction is the excess-weighted centroid of its 3 x 3 neighbourhood in
sine space, and the steered beam goes to the nearest sector. Backgrounds carry
FRACTION_BITS fraction bits; levels at or below LEVEL_FLOOR are never candidates.
A beam above its background keeps it for freeze_maps maps (the freeze_s setting),
then adapts with HELD_SHIFT.

With the cfar setting above 0, maps without such a candidate go to BankCfar: each beam's level
over the map median, averaged over about 1.6 s, against the beam's own slow average, in units of
the spread of all beams. It finds steady sources far below the 1.5x rule.
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

# CFAR (HEIMDALL_CFAR_* in heimdall_config.h). Ratios are Q24, deviations Q16.
CFAR_SHORT_SHIFT = 4
CFAR_LONG_SHIFT = 9
CFAR_VAR_SHIFT = 8
CFAR_WARMUP_MAPS = 300
CFAR_MEDIAN_FLOOR = 8
CFAR_DEVIATION_LIMIT = 1 << 20


@dataclass
class BankResult:
    beams: list[int]
    levels: list[int]
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    sector: int = 0
    cfar: bool = False


def _above(level: int, reference: int) -> bool:
    """level >= ACQUIRE_RATIO x reference, with the reference scaled by 2^FRACTION_BITS."""
    return level > LEVEL_FLOOR and (level << FRACTION_BITS) * ACQUIRE_RATIO_DEN >= reference * ACQUIRE_RATIO_NUM


def _shift(value: int, shift: int) -> int:
    """C: value >= 0 ? value >> shift : -((-value) >> shift)."""
    return value >> shift if value >= 0 else -((-value) >> shift)


def _divide(numerator: int, denominator: int) -> int:
    """C integer division, which truncates toward zero."""
    quotient = abs(numerator) // denominator
    return quotient if numerator >= 0 else -quotient


def _growing_shift(maps: int, limit: int) -> int:
    """floor(log2(maps)), at most limit: the averages start as running means."""
    shift = 0
    while shift < limit and (2 << shift) <= maps:
        shift += 1
    return shift


class BankCfar:
    """The bank-map CFAR of MAX78002/beam_bank.c, bit for bit. z10 is the threshold in tenths of a sigma."""

    def __init__(self, rows: int, columns: int, z10: int, freeze_maps: int = FREEZE_MAPS) -> None:
        self.rows, self.columns, self.z10, self.freeze_maps = rows, columns, z10, freeze_maps
        beams = rows * columns
        self.short = [0] * beams
        self.long = [0] * beams
        self.held = [0] * beams
        self.deviation = [0] * beams
        self.var = 0
        self.maps = 0

    def prime(self, levels: list[int], median: int) -> None:
        ratio = self._ratios(levels, median)
        self.short, self.long = list(ratio), list(ratio)
        self.held = [0] * len(levels)
        self.var = 0
        self.maps = 1

    def _ratios(self, levels: list[int], median: int) -> list[int]:
        divisor = max(median, CFAR_MEDIAN_FLOOR)
        return [(level << 24) // divisor for level in levels]

    def update(self, levels: list[int], median: int) -> list[int]:
        """One map after the priming one. Returns up to 2 CFAR candidates, the highest deviation first."""
        self.maps = min(self.maps + 1, 0xFFFFFFFF)
        short_shift = _growing_shift(self.maps, CFAR_SHORT_SHIFT)
        long_shift = _growing_shift(self.maps, CFAR_LONG_SHIFT)
        var_shift = _growing_shift(self.maps, CFAR_VAR_SHIFT)
        warm = self.maps >= CFAR_WARMUP_MAPS
        threshold = self.z10 * self.z10 * self.var
        ratio = self._ratios(levels, median)
        hot = []
        for beam in range(len(levels)):
            self.short[beam] += _shift(ratio[beam] - self.short[beam], short_shift)
            deviation = _divide(self.short[beam] - self.long[beam], max(1, self.long[beam] >> 16))
            deviation = max(-CFAR_DEVIATION_LIMIT, min(CFAR_DEVIATION_LIMIT, deviation))
            self.deviation[beam] = deviation
            hot.append(warm and deviation > 0 and 100 * deviation * deviation >= threshold)

        candidates: list[int] = []
        for beam in range(len(levels)):
            if not hot[beam] or levels[beam] <= LEVEL_FLOOR or not self._peak(beam):
                continue
            if len(candidates) < 2:
                candidates.append(beam)
            elif self.deviation[beam] > self.deviation[candidates[1]]:
                candidates[1] = beam
            if len(candidates) == 2 and self.deviation[candidates[1]] > self.deviation[candidates[0]]:
                candidates.reverse()

        total = count = 0
        rising = []
        for beam, deviation in enumerate(self.deviation):
            # Half the threshold or more: maybe a source, so its own average holds.
            rising.append(warm and deviation > 0 and 400 * deviation * deviation >= threshold)
            total += deviation * deviation
            count += 1
        # A source lifts the map median and so pushes every other beam down: hold the spread while one rises.
        if not any(rising):
            self.var += _shift(total // count - self.var, var_shift)

        for beam in range(len(levels)):
            if rising[beam]:
                if self.held[beam] < self.freeze_maps:
                    self.held[beam] += 1
                    continue
            else:
                self.held[beam] = 0
            self.long[beam] += _shift(ratio[beam] - self.long[beam], long_shift)
        return candidates

    def _peak(self, beam: int) -> bool:
        row, column = divmod(beam, self.columns)
        for r, c in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            if 0 <= r < self.rows and 0 <= c < self.columns and \
                    self.deviation[r * self.columns + c] > self.deviation[beam]:
                return False
        return True


class BankTracker:
    def __init__(self, rows: int, columns: int, azimuth_deg: list[float], elevation_deg: list[float],
                 sector_azimuth_deg: list[float], sector_elevation_deg: list[float],
                 freeze_maps: int = FREEZE_MAPS, cfar_z10: int = 0) -> None:
        self.rows, self.columns = rows, columns
        self.azimuth_deg, self.elevation_deg = list(azimuth_deg), list(elevation_deg)
        self.sector_azimuth_deg, self.sector_elevation_deg = list(sector_azimuth_deg), list(sector_elevation_deg)
        self.freeze_maps = freeze_maps
        self.cfar = BankCfar(rows, columns, cfar_z10, freeze_maps) if cfar_z10 > 0 else None
        self.background: list[int] | None = None
        self.held_maps: list[int] = []

    def update(self, levels: list[int]) -> BankResult:
        median = sorted(levels)[len(levels) // 2]
        if self.background is None:
            self.background = [level << FRACTION_BITS for level in levels]
            self.held_maps = [0] * len(levels)
            if self.cfar is not None:
                self.cfar.prime(levels, median)
            return BankResult([], [])
        field = median << FRACTION_BITS
        above = [_above(level, bg) and _above(level, field) for level, bg in zip(levels, self.background)]
        candidates = [beam for beam in range(len(levels)) if above[beam] and self._local_peak(levels, beam)]
        candidates.sort(key=lambda beam: -Fraction(levels[beam], max(1, self.background[beam] >> FRACTION_BITS)))
        cfar = self.cfar.update(levels, median) if self.cfar is not None else []
        result = BankResult(candidates[:2], [levels[beam] for beam in candidates[:2]])
        if not result.beams and cfar:
            result = BankResult(cfar, [levels[beam] for beam in cfar], cfar=True)
        if result.cfar:
            # A steady weak source has no excess over the 1.5x background: use the beam centre (the fit refines it).
            result.azimuth_deg, result.elevation_deg = self.azimuth_deg[cfar[0]], self.elevation_deg[cfar[0]]
            result.sector = self.nearest_sector(result.azimuth_deg, result.elevation_deg)
        elif result.beams:
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
