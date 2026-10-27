import { test } from 'node:test';
import assert from 'node:assert/strict';

import { bankGrid, buildField, cellDirection, fieldKind, gateMarker, gridDirection, gridPosition, median, surfaceUnit } from '../js/hardware/field.js';
import { direction } from '../js/hardware/geometry.js';

const STEP = 2 * Math.sin(40 * Math.PI / 180) / 5;
const SINES = [-2, -1, 0, 1, 2].map((k) => k * STEP);

function bank(levels = null, target = null) {
  const azimuth_deg = [];
  const elevation_deg = [];
  for (let row = 0; row < 5; row++) {
    const el = Math.asin(SINES[row]);
    for (let column = 0; column < 5; column++) {
      elevation_deg.push(el * 180 / Math.PI);
      azimuth_deg.push(Math.asin(SINES[column] / Math.cos(el)) * 180 / Math.PI);
    }
  }
  const levels_db = levels || Array.from({length: 5}, () => Array(5).fill(-35));
  return {rows: 5, columns: 5, azimuth_deg, elevation_deg, levels_db, target};
}

const close = (actual, expected, tolerance = 1e-6) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} is not ${expected}`);

test('the bank grid is even in sines even though its azimuths are not', () => {
  const grid = bankGrid(bank());
  [-0.6428, -0.3857, -0.1286, 0.1286, 0.3857, 0.6428].forEach((edge, i) => {
    close(grid.u[i], edge, 1e-3);
    close(grid.v[i], edge, 1e-3);
  });
  assert.equal(bankGrid({rows: 5, columns: 5}), null);
});

test('every bank cell is centred on its own beam direction', () => {
  const frame = {firmware_mode: 'B', bank: bank()};
  const field = buildField(frame, 'bank');
  for (const cell of field.cells) {
    const centre = cellDirection(cell.bounds, 0.5, 0.5);
    direction(cell.azimuth, cell.elevation).forEach((value, axis) => close(centre[axis], value, 1e-9));
  }
  const position = gridPosition(field, field.cells[13].azimuth, field.cells[13].elevation);
  close(position.column, 3.5);
  close(position.row, 2.5);
});

test('the field follows the mode and keeps the last grid when idle', () => {
  const withBank = {bank: bank(), levels_db: [[-30]], configuration: {rows: 1, columns: 1}};
  assert.equal(fieldKind({...withBank, firmware_mode: 'B'}), 'bank');
  assert.equal(fieldKind({...withBank, firmware_mode: 'C'}), 'sectors');
  assert.equal(fieldKind({...withBank, firmware_mode: 'IDLE'}, {monitoring: true}), 'sectors');
  assert.equal(fieldKind({...withBank, firmware_mode: 'IDLE'}, {previous: 'bank'}), 'bank');
  assert.equal(fieldKind({...withBank, firmware_mode: 'IDLE'}, {previous: 'sectors'}), 'sectors');
  assert.equal(fieldKind({firmware_mode: 'B', bank: {rows: 5, columns: 5}}), 'sectors');
});

test('the bank marks a lock only, and measures the background as the median', () => {
  const levels = Array.from({length: 5}, () => Array(5).fill(-36));
  levels[3][4] = -8;
  const searching = buildField({bank: bank(levels)}, 'bank');
  assert.equal(searching.marker, null);
  assert.equal(searching.strongest.index, 19);
  assert.equal(searching.background, -36);
  const locked = buildField({bank: bank(levels, {beam: 19, azimuth_deg: 24, elevation_deg: 13})}, 'bank');
  assert.deepEqual([locked.marker.kind, locked.marker.azimuth, locked.marker.cell.index], ['target', 24, 19]);
  assert.equal(median([3, null, 1, 2, NaN]), 2);
});

test('the displayed direction moves only when its cell is at least 5 dB above background', () => {
  const levels = Array.from({length: 5}, () => Array(5).fill(-36));
  levels[3][4] = -32;
  const quietInitial = buildField({bank: bank(levels, {beam: 19, azimuth_deg: 24, elevation_deg: 13})}, 'bank');
  const rejected = gateMarker(quietInitial);
  assert.equal(rejected.marker, null);
  assert.equal(rejected.suppressed, true);

  levels[3][4] = -31;
  const first = buildField({bank: bank(levels, {beam: 19, azimuth_deg: 24, elevation_deg: 13})}, 'bank');
  const accepted = gateMarker(first);
  assert.equal(accepted.marker.cell.index, 19);

  levels[3][4] = -32;
  levels[3][2] = -8;
  const quietMove = buildField({bank: bank(levels, {beam: 18, azimuth_deg: 15, elevation_deg: 10})}, 'bank');
  const held = gateMarker(quietMove, accepted.previous);
  assert.equal(held.marker.cell.index, 19);
  assert.equal(held.marker.azimuth, 24);
  assert.equal(held.suppressed, false);

  levels[3][3] = -31;
  const loudMove = buildField({bank: bank(levels, {beam: 18, azimuth_deg: 15, elevation_deg: 10})}, 'bank');
  assert.equal(gateMarker(loudMove, held.previous).marker.cell.index, 18);
});

test('the Live direction can come from the pattern fit, but only for a firmware target', () => {
  const target = {beam: 19, azimuth_deg: 24, elevation_deg: 13};
  const fit = {azimuth_deg: -20, elevation_deg: -2, quality: 0.8};
  const frame = (withTarget) => ({bank: {...bank(null, withTarget ? target : null), fit}});
  const own = buildField(frame(true), 'bank');
  assert.deepEqual([own.marker.source, own.marker.azimuth, own.alt], ['firmware', 24, null]);
  const fitted = buildField(frame(true), 'bank', {direction: 'fit'});
  assert.deepEqual([fitted.marker.source, fitted.marker.azimuth, fitted.marker.cell.index], ['fit', -20, 11]);
  const both = buildField(frame(true), 'bank', {direction: 'both'});
  assert.deepEqual([both.marker.source, both.alt.source, both.alt.elevation], ['firmware', 'fit', -2]);
  assert.equal(buildField(frame(false), 'bank', {direction: 'fit'}).marker, null);
  const noFit = buildField({bank: bank(null, target)}, 'bank', {direction: 'fit'});
  assert.equal(noFit.marker.source, 'firmware');
});

test('the sector field marks the target, else the held beam, and ages its cells', () => {
  const frame = {
    configuration: {rows: 2, columns: 3},
    azimuth_deg: [-20, 0, 20],
    elevation_deg: [-10, 10],
    levels_db: [[-30, -20, null], [-25, -10, -40]],
    last_steer: {row: 0, column: 1, azimuth_deg: 0, elevation_deg: -10},
  };
  const ages = {'0,0': 1000, '1,1': 9000};
  const field = buildField(frame, 'sectors', {cellAge: (row, column) => ages[`${row},${column}`] ?? null});
  assert.equal(field.marker.kind, 'beam');
  assert.equal(field.marker.cell.index, 1);
  assert.deepEqual(field.cells.map((cell) => cell.fresh), [true, false, false, false, false, false]);
  assert.equal(field.strongest.index, 4);
  close(field.axes.x[0], -30);
  const targeted = buildField({...frame, target: {row: 1, column: 2}}, 'sectors');
  assert.deepEqual([targeted.marker.kind, targeted.marker.azimuth, targeted.marker.elevation], ['target', 20, 10]);
});

test('the smooth surface passes through every cell and stays in range', () => {
  const field = buildField({bank: bank()}, 'bank');
  const values = Array(25).fill(0.2);
  values[18] = 1;
  for (const cell of field.cells) close(surfaceUnit(field, values, cell.column + 0.5, cell.row + 0.5), values[cell.index]);
  const between = surfaceUnit(field, values, 3.5, 4);
  assert.ok(between > 0.2 && between < 1, `${between} is not between its neighbours`);
  for (let column = 0; column <= 5; column += 0.25) {
    for (let row = 0; row <= 5; row += 0.25) {
      const value = surfaceUnit(field, values, column, row);
      assert.ok(value >= 0 && value <= 1);
    }
  }
  close(surfaceUnit(field, Array(25).fill(0.4), 0.1, 4.9), 0.4);
  const centre = gridDirection(field, 3.5, 2.5);
  cellDirection(field.cells[13].bounds, 0.5, 0.5).forEach((value, axis) => close(centre[axis], value, 1e-9));
});
