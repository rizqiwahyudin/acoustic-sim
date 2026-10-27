/**
 * field.js — what the Field view shows: the beam bank in Live, the sector
 * grid during diagnostic scans. Pure data (no DOM, no Three.js) so the map,
 * the 3D view and the tests share it.
 *
 * Directions are [right, up, forward] unit vectors, seen from behind the array.
 * The bank is an even grid in sines (direction cosines), as the generator's
 * beam_bank.grid_sines builds it; the sector grid is even in degrees.
 */

import { angularEdges, direction } from './geometry.js';

const RAD = Math.PI / 180;
const DEG = 180 / Math.PI;
const SECTOR_MODES = new Set(['F', 'C', 'G']);
/** Which Live direction to show, both from the firmware: its peak beam (BANK_TARGET), its pattern fit (BANK_FIT), or both. */
export const DIRECTIONS = [['firmware', 'Peak'], ['fit', 'Fit'], ['both', 'Both']];

const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;

export function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Column and row centres and edges of the bank in sine space, or null without directions. */
export function bankGrid(bank) {
  const rows = bank?.rows;
  const columns = bank?.columns;
  const az = bank?.azimuth_deg;
  const el = bank?.elevation_deg;
  if (!(rows >= 2 && columns >= 2) || az?.length !== rows * columns || el?.length !== rows * columns) return null;
  const u = (beam) => Math.cos(el[beam] * RAD) * Math.sin(az[beam] * RAD);
  const v = (beam) => Math.sin(el[beam] * RAD);
  const uCentres = Array.from({length: columns}, (_, c) => mean(Array.from({length: rows}, (_, r) => u(r * columns + c))));
  const vCentres = Array.from({length: rows}, (_, r) => mean(Array.from({length: columns}, (_, c) => v(r * columns + c))));
  return {uCentres, vCentres, u: angularEdges(uCentres), v: angularEdges(vCentres)};
}

/** Unit direction at (s, t) in [0, 1]² across a cell, s left to right and t bottom to top. */
export function cellDirection(bounds, s, t) {
  if (bounds.space === 'sine') {
    const u = bounds.u0 + (bounds.u1 - bounds.u0) * s;
    const v = bounds.v0 + (bounds.v1 - bounds.v0) * t;
    return [u, v, Math.sqrt(Math.max(0, 1 - u * u - v * v))];
  }
  return direction(bounds.a0 + (bounds.a1 - bounds.a0) * s, bounds.e0 + (bounds.e1 - bounds.e0) * t);
}

/** Which grid to show: the bank while Live runs, the sectors during scans, else the last one. */
export function fieldKind(frame, {previous = null, monitoring = false} = {}) {
  const bankReady = Boolean(bankGrid(frame?.bank) && frame.bank.levels_db);
  if (frame?.firmware_mode === 'B' && bankReady) return 'bank';
  if (SECTOR_MODES.has(frame?.firmware_mode) || monitoring) return 'sectors';
  if (previous === 'bank' && bankReady) return 'bank';
  if (previous === 'sectors') return 'sectors';
  return bankReady ? 'bank' : 'sectors';
}

function strongestCell(cells) {
  let best = null;
  for (const cell of cells) if (Number.isFinite(cell.level) && (!best || cell.level > best.level)) best = cell;
  return best;
}

function bankField(frame, direction) {
  const bank = frame?.bank;
  const grid = bankGrid(bank);
  if (!grid || !bank.levels_db) return null;
  const {rows, columns} = bank;
  const cells = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      const level = bank.levels_db[row]?.[column];
      cells.push({
        index, row, column,
        azimuth: bank.azimuth_deg[index],
        elevation: bank.elevation_deg[index],
        level: Number.isFinite(level) ? level : null,
        fresh: true,
        bounds: {space: 'sine', u0: grid.u[column], u1: grid.u[column + 1], v0: grid.v[row], v1: grid.v[row + 1]},
      });
    }
  }
  const target = bank.target;
  const fit = bank.fit;
  // Detection stays the firmware's: the fit only gives the direction of a target that the firmware reports.
  const own = target ? {kind: 'target', source: 'firmware', azimuth: target.azimuth_deg, elevation: target.elevation_deg,
    cell: cells[target.beam] ?? null} : null;
  const fitted = target && Number.isFinite(fit?.azimuth_deg) && Number.isFinite(fit?.elevation_deg)
    ? {kind: 'target', source: 'fit', azimuth: fit.azimuth_deg, elevation: fit.elevation_deg,
      cell: cells[cellIndex(grid, columns, rows, fit.azimuth_deg, fit.elevation_deg)] ?? null}
    : null;
  return {
    kind: 'bank',
    space: 'sine',
    rows,
    columns,
    cells,
    key: `bank|${rows}x${columns}|${bank.azimuth_deg.join(',')}|${bank.elevation_deg.join(',')}`,
    axes: {x: grid.u, y: grid.v},
    columnLabels: grid.uCentres.map((u) => Math.asin(u) * DEG),
    rowLabels: grid.vCentres.map((v) => Math.asin(v) * DEG),
    strongest: strongestCell(cells),
    marker: direction === 'fit' ? (fitted ?? own) : own,
    alt: direction === 'both' ? fitted : null,
    background: median(cells.map((cell) => cell.level)),
    truth: frame.emulator_truth || null,
  };
}

/** The bank cell that contains a direction (clamped to the grid). */
function cellIndex(grid, columns, rows, azimuth, elevation) {
  const column = fraction(grid.u, Math.cos(elevation * RAD) * Math.sin(azimuth * RAD));
  const row = fraction(grid.v, Math.sin(elevation * RAD));
  const clamp = (value, count) => Math.min(count - 1, Math.max(0, Math.floor(value)));
  return clamp(row, rows) * columns + clamp(column, columns);
}

function sectorField(frame, cellAge, staleMs) {
  const cfg = frame?.configuration;
  if (!cfg || !frame.levels_db || frame.azimuth_deg?.length !== cfg.columns || frame.elevation_deg?.length !== cfg.rows) return null;
  const az = angularEdges(frame.azimuth_deg);
  const el = angularEdges(frame.elevation_deg).map((value) => Math.max(-89, Math.min(89, value)));
  const cells = [];
  for (let row = 0; row < cfg.rows; row++) {
    for (let column = 0; column < cfg.columns; column++) {
      const level = frame.levels_db[row]?.[column];
      const measured = Number.isFinite(level);
      const age = cellAge(row, column);
      cells.push({
        index: row * cfg.columns + column, row, column,
        azimuth: frame.azimuth_deg[column],
        elevation: frame.elevation_deg[row],
        level: measured ? level : null,
        fresh: measured && Number.isFinite(age) && age <= staleMs,
        bounds: {space: 'angle', a0: az[column], a1: az[column + 1], e0: el[row], e1: el[row + 1]},
      });
    }
  }
  const at = (record) => cells[record.row * cfg.columns + record.column] ?? null;
  const angles = (record, cell) => ({
    azimuth: Number.isFinite(record.azimuth_deg) ? record.azimuth_deg : cell?.azimuth,
    elevation: Number.isFinite(record.elevation_deg) ? record.elevation_deg : cell?.elevation,
  });
  const target = frame.target;
  const steer = !target ? frame.last_steer : null;
  let marker = null;
  if (target) marker = {kind: 'target', ...angles(target, at(target)), cell: at(target)};
  else if (steer) marker = {kind: 'beam', ...angles(steer, at(steer)), cell: at(steer)};
  return {
    kind: 'sectors',
    space: 'angle',
    rows: cfg.rows,
    columns: cfg.columns,
    cells,
    key: `sectors|${cfg.rows}x${cfg.columns}|${frame.azimuth_deg.join(',')}|${frame.elevation_deg.join(',')}`,
    axes: {x: az, y: el},
    columnLabels: [...frame.azimuth_deg],
    rowLabels: [...frame.elevation_deg],
    strongest: strongestCell(cells),
    marker,
    background: median(cells.map((cell) => cell.level)),
    truth: frame.emulator_truth || null,
  };
}

export function buildField(frame, kind, {cellAge = () => null, staleMs = 5000, direction = 'firmware'} = {}) {
  return kind === 'bank' ? bankField(frame, direction) : sectorField(frame, cellAge, staleMs);
}

/** Keep a displayed direction until its measured level is at least 5 dB above the field background. */
export function gateMarker(field, previous = null, candidate = field?.marker) {
  const held = previous?.key === field?.key ? previous : null;
  const level = candidate?.cell?.level;
  const qualifies = Number.isFinite(level) && Number.isFinite(field?.background) && level - field.background >= 5;
  const next = qualifies
    ? {key: field.key, marker: {...candidate, cellIndex: candidate.cell?.index ?? null}}
    : held;
  if (!next) return {marker: null, previous: null, suppressed: Boolean(candidate)};

  const {cellIndex, ...marker} = next.marker;
  return {
    marker: {...marker, cell: Number.isInteger(cellIndex) ? field.cells[cellIndex] ?? null : null},
    previous: next,
    suppressed: Boolean(candidate && !qualifies && !held),
  };
}

function fraction(edges, value) {
  const last = edges.length - 2;
  for (let i = 0; i <= last; i++) {
    if (value <= edges[i + 1] || i === last) return i + (value - edges[i]) / (edges[i + 1] - edges[i]);
  }
  return null;
}

/** Fractional column and row (0 = left / bottom edge) of a direction on the field's grid. */
export function gridPosition(field, azimuth, elevation) {
  if (!field || !Number.isFinite(azimuth) || !Number.isFinite(elevation)) return null;
  const [x, y] = field.space === 'sine'
    ? [Math.cos(elevation * RAD) * Math.sin(azimuth * RAD), Math.sin(elevation * RAD)]
    : [azimuth, elevation];
  return {column: fraction(field.axes.x, x), row: fraction(field.axes.y, y)};
}

function edgeValue(edges, f) {
  const i = Math.max(0, Math.min(edges.length - 2, Math.floor(f)));
  return edges[i] + (f - i) * (edges[i + 1] - edges[i]);
}

/** Unit direction at a fractional grid position (0 = left / bottom edge). */
export function gridDirection(field, column, row) {
  const x = edgeValue(field.axes.x, column);
  const y = edgeValue(field.axes.y, row);
  return field.space === 'sine' ? [x, y, Math.sqrt(Math.max(0, 1 - x * x - y * y))] : direction(x, y);
}

function cubic(p0, p1, p2, p3, t) {
  return p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
}

/**
 * Smooth value between cell centres (Catmull-Rom in both directions) at a
 * fractional grid position; exact at every centre, clamped to [0, 1].
 * values: one number per cell, row-major from the bottom-left.
 */
export function surfaceUnit(field, values, column, row) {
  const {rows, columns} = field;
  const x = column - 0.5;
  const y = row - 0.5;
  const c = Math.floor(x);
  const r = Math.floor(y);
  const at = (rr, cc) => values[Math.min(rows - 1, Math.max(0, rr)) * columns + Math.min(columns - 1, Math.max(0, cc))];
  const across = (rr) => cubic(at(rr, c - 1), at(rr, c), at(rr, c + 1), at(rr, c + 2), x - c);
  const value = cubic(across(r - 1), across(r), across(r + 1), across(r + 2), y - r);
  return Math.max(0, Math.min(1, value));
}
