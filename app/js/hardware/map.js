/**
 * map.js — the sector map: one square cell per firmware sector, elevation on
 * the vertical axis (highest at the top) and azimuth across. Works for any
 * grid from 1 × 1 to 20 × 20; values are printed only when cells are large
 * enough to read them.
 */

import { h, setText } from '../shared/dom.js';
import { HATCH, inkOn, levelToUnit, rampColor, rampGradient, rgbCss } from '../shared/colormap.js';
import { deg, num } from '../shared/format.js';
import { STALE_MS } from './session.js';
import { angularEdges } from './geometry.js';

const LABEL_MIN_CELL = 44;

export class SectorMap {
  constructor({onPick, onHover} = {}) {
    this.onPick = onPick;
    this.onHover = onHover;
    this.key = '';
    this.cells = [];
    this.hover = null;
    this.cellPx = 72;

    this.rowLabels = h('div', {class: 'map__rows'});
    this.grid = h('div', {class: 'map__cells'});
    this.truth = h('span', {class: 'map__truth', 'aria-hidden': 'true', hidden: true});
    this.grid.append(this.truth);
    this.colLabels = h('div', {class: 'map__cols'});
    this.ticks = h('div', {class: 'map__ticks'});
    this.legendBar = h('div', {class: 'map__legend-bar', 'aria-hidden': 'true', style: {background: rampGradient('to top')}});

    this.el = h('div', {class: 'map'},
      h('div', {class: 'map__axis-title map__axis-title--el'}, 'Elevation'),
      h('div', {class: 'map__body'},
        this.rowLabels,
        this.grid,
        h('div', {class: 'map__legend'},
          h('span', {class: 'map__legend-unit'}, 'dBFS'),
          h('div', {class: 'map__legend-scale'}, this.legendBar, this.ticks),
        ),
      ),
      this.colLabels,
      h('div', {class: 'map__axis-title map__axis-title--az'}, 'Azimuth'),
    );
    this.grid.addEventListener('mouseleave', () => this.setHover(null));
  }

  setHover(sector) {
    if (this.hover === sector) return;
    this.hover = sector;
    this.onHover?.(sector);
  }

  build(frame, cellPx) {
    const cfg = frame.configuration;
    const key = `${cfg.rows}x${cfg.columns}|${cellPx}|${frame.azimuth_deg?.join(',')}|${frame.elevation_deg?.join(',')}`;
    if (key === this.key) return;
    this.key = key;
    this.cellPx = cellPx;
    const gap = cellPx >= 40 ? 4 : 2;
    this.el.style.setProperty('--cell', `${cellPx}px`);
    this.el.style.setProperty('--gap', `${gap}px`);
    this.el.style.setProperty('--grid-w', `${cfg.columns * cellPx + (cfg.columns - 1) * gap}px`);
    this.el.style.setProperty('--grid-h', `${cfg.rows * cellPx + (cfg.rows - 1) * gap}px`);
    this.grid.style.gridTemplateColumns = `repeat(${cfg.columns}, var(--cell))`;
    this.rowLabels.style.gridTemplateRows = `repeat(${cfg.rows}, var(--cell))`;
    this.colLabels.style.gridTemplateColumns = `repeat(${cfg.columns}, var(--cell))`;

    this.cells = [];
    this.grid.replaceChildren(this.truth);
    const everyRow = Math.max(1, Math.ceil(14 / cellPx));
    const rowNodes = [];
    for (let displayRow = 0; displayRow < cfg.rows; displayRow++) {
      const row = cfg.rows - 1 - displayRow;
      rowNodes.push(h('span', {}, displayRow % everyRow === 0 ? deg(frame.elevation_deg?.[row]) : ''));
      for (let column = 0; column < cfg.columns; column++) {
        const sector = row * cfg.columns + column;
        const text = h('span', {class: 'map__value'});
        const ring = h('span', {class: 'map__ring', hidden: true});
        const cell = h('button', {
          type: 'button',
          class: 'map__cell',
          onClick: () => this.onPick?.(sector),
          onMouseenter: () => this.setHover(sector),
          onFocus: () => this.setHover(sector),
        }, text, ring);
        this.grid.append(cell);
        this.cells[sector] = {cell, text, ring, row, column};
      }
    }
    this.rowLabels.replaceChildren(...rowNodes);
    const everyCol = Math.max(1, Math.ceil(48 / cellPx));
    this.colLabels.replaceChildren(...Array.from({length: cfg.columns}, (_, column) =>
      h('span', {}, column % everyCol === 0 ? deg(frame.azimuth_deg?.[column]) : '')));
  }

  /**
   * state: {floor, showValues, session}
   */
  render(frame, {floor, showValues, session, cellPx}) {
    const cfg = frame?.configuration;
    if (!cfg || !frame.levels_db) return;
    this.build(frame, cellPx);
    this.ticks.replaceChildren(...[0, 0.25, 0.5, 0.75, 1].map((f) => h('span', {}, num(floor * f, 0))));

    const strongest = session.strongest(frame);
    const scanning = !frame.target && !frame.last_steer;
    const target = frame.target;
    const steer = !target ? frame.last_steer : null;
    const showText = this.cellPx >= LABEL_MIN_CELL;

    for (const entry of this.cells) {
      if (!entry) continue;
      const {cell, text, ring, row, column} = entry;
      const sector = row * cfg.columns + column;
      const level = frame.levels_db?.[row]?.[column];
      const measured = Number.isFinite(level);
      const age = session.cellAge(row, column);
      const fresh = measured && Number.isFinite(age) && age <= STALE_MS;
      const rgb = rampColor(levelToUnit(level, floor) ?? 0);
      const fill = rgbCss(rgb);
      const isStrongest = scanning && strongest && strongest.sector === sector;
      const isSteer = steer && steer.row === row && steer.column === column;
      const isTarget = target && target.row === row && target.column === column;

      let background = 'var(--raised)';
      if (measured) background = fresh ? fill : `${HATCH}, ${fill}`;
      cell.style.background = background;
      cell.style.opacity = measured && !fresh ? '0.55' : '1';
      cell.classList.toggle('is-steer', Boolean(isSteer));
      cell.classList.toggle('is-strongest', Boolean(isStrongest));
      cell.classList.toggle('is-hover', this.hover === sector);
      ring.hidden = !isTarget;

      const labelled = showText && measured && (showValues || isStrongest || isTarget || isSteer);
      setText(text, labelled ? num(level, 1) : '');
      text.style.color = inkOn(rgb);

      const place = `Sector ${sector}, azimuth ${deg(frame.azimuth_deg?.[column])}, elevation ${deg(frame.elevation_deg?.[row])}`;
      cell.setAttribute('aria-label', measured
        ? `${place}, ${num(level, 1)} dBFS${fresh ? '' : ', older than 5 s'}. Hold the beam here.`
        : `${place}, not measured yet. Hold the beam here.`);
    }

    const truth = frame.emulator_truth;
    if (truth && frame.azimuth_deg?.length && frame.elevation_deg?.length) {
      const az = angularEdges(frame.azimuth_deg);
      const el = angularEdges(frame.elevation_deg);
      const x = (truth.azimuth_deg - az[0]) / (az.at(-1) - az[0]);
      const y = (truth.elevation_deg - el[0]) / (el.at(-1) - el[0]);
      const inside = x >= 0 && x <= 1 && y >= 0 && y <= 1;
      this.truth.hidden = !inside;
      if (inside) {
        this.truth.style.left = `${(x * 100).toFixed(2)}%`;
        this.truth.style.top = `${((1 - y) * 100).toFixed(2)}%`;
        this.truth.classList.toggle('is-dropout', truth.source_active === false);
      }
    } else {
      this.truth.hidden = true;
    }
  }
}
