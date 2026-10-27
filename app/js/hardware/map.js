/**
 * map.js — the Field as a flat map: one square cell per beam (bank) or
 * sector, elevation up the side (highest at the top) and azimuth across.
 * A crosshair marks the source between cells, with its angles on the axes.
 * Values are printed only when the cells are large enough to read them.
 */

import { h, setText } from '../shared/dom.js';
import { HATCH, inkOn, levelToUnit, rampColor, rampGradient, rgbCss } from '../shared/colormap.js';
import { deg, num } from '../shared/format.js';
import { look } from '../shared/look.js';
import { gridPosition } from './field.js';
import { ditherStep, meterFill } from './lookparts.js';

const LABEL_MIN_CELL = 40;

function axisLabel(value) {
  if (!Number.isFinite(value)) return '';
  const rounded = Math.round(value);
  return rounded === 0 ? '0°' : `${rounded > 0 ? '+' : '−'}${Math.abs(rounded)}°`;
}

export class FieldMap {
  constructor({onPick, onHover} = {}) {
    this.onPick = onPick;
    this.onHover = onHover;
    this.key = '';
    this.cells = [];
    this.hover = null;
    this.cellPx = 72;

    this.rowLabels = h('div', {class: 'fmap__ylab'});
    this.colLabels = h('div', {class: 'fmap__xlab'});
    this.grid = h('div', {class: 'fmap__cells'});
    this.cross = h('span', {class: 'fmap__cross', hidden: true}, h('b'));
    this.pointerX = h('span', {class: 'fmap__pointer fmap__pointer--x', hidden: true});
    this.pointerY = h('span', {class: 'fmap__pointer fmap__pointer--y', hidden: true});
    this.truth = h('span', {class: 'fmap__truth', hidden: true});
    this.alt = h('span', {class: 'fmap__alt', hidden: true, title: 'Pattern fit'});
    this.overlays = [this.cross, this.pointerX, this.pointerY, this.truth, this.alt];
    this.ticks = h('ul', {class: 'fmap__ticks'});
    // The hidden looks draw tiles in six dither steps (lookparts.js), so their scale shows the steps.
    const ramp = look
      ? h('div', {class: 'fmap__ramp lk-steps'}, [0, 1, 2, 3, 4, 5].map((step) => h('i', {dataset: {lkStep: String(step)}})))
      : h('div', {class: 'fmap__ramp', style: {background: rampGradient('to top')}});

    this.el = h('div', {class: 'fmap'},
      h('div', {class: 'fmap__ytitle'}, h('span', {class: 'tri'}, '▲'), h('span', {class: 'fmap__ytext'}, 'ELEVATION'), h('span', {class: 'tri'}, '▼')),
      this.rowLabels,
      this.grid,
      h('div', {class: 'fmap__bar'}, ramp, this.ticks),
      this.colLabels,
      h('div', {class: 'fmap__xtitle'}, h('span', {class: 'tri'}, '◀'), 'AZIMUTH', h('span', {class: 'tri'}, '▶')),
    );
    this.grid.addEventListener('mouseleave', () => this.setHover(null));
  }

  setHover(index) {
    if (this.hover === index) return;
    this.hover = index;
    this.onHover?.(index);
  }

  build(field, cellPx) {
    const key = `${field.key}|${cellPx}`;
    if (key === this.key) return;
    this.key = key;
    this.cellPx = cellPx;
    this.el.style.setProperty('--cell', `${cellPx}px`);
    this.el.style.setProperty('--gap', `${cellPx >= 40 ? 3 : 2}px`);
    if (look) this.el.classList.toggle('fmap--small', cellPx < LABEL_MIN_CELL);
    this.grid.style.gridTemplateColumns = `repeat(${field.columns}, var(--cell))`;
    this.rowLabels.style.gridTemplateRows = `repeat(${field.rows}, var(--cell))`;
    this.colLabels.style.gridTemplateColumns = `repeat(${field.columns}, var(--cell))`;

    this.cells = [];
    const nodes = [];
    for (let displayRow = 0; displayRow < field.rows; displayRow++) {
      const row = field.rows - 1 - displayRow;
      for (let column = 0; column < field.columns; column++) {
        const index = row * field.columns + column;
        const text = h('span', {class: 'fmap__value'});
        const meter = look ? h('span', {class: 'lk-meter', 'aria-hidden': 'true'}, h('i'), h('i'), h('i'), h('i'), h('i')) : null;
        const cell = h('button', {
          type: 'button',
          class: 'fmap__cell',
          onClick: () => this.onPick?.(index),
          onMouseenter: () => this.setHover(index),
          onFocus: () => this.setHover(index),
        }, text, meter);
        this.cells[index] = {cell, text, meter};
        nodes.push(cell);
      }
    }
    this.grid.replaceChildren(...nodes, ...this.overlays);
    const everyRow = Math.max(1, Math.ceil(22 / cellPx));
    const rows = field.rowLabels.map((value, row) => h('span', {}, (field.rows - 1 - row) % everyRow === 0 ? axisLabel(value) : ''));
    this.rowLabels.replaceChildren(...rows.reverse());
    const everyColumn = Math.max(1, Math.ceil(52 / cellPx));
    this.colLabels.replaceChildren(...field.columnLabels.map((value, column) =>
      h('span', {}, column % everyColumn === 0 ? axisLabel(value) : '')));
  }

  /** Put an overlay at a fractional grid position; hidden outside the grid. */
  place(node, position, field) {
    const inside = Boolean(position) && position.column >= 0 && position.column <= field.columns
      && position.row >= 0 && position.row <= field.rows;
    node.hidden = !inside;
    if (inside) {
      node.style.left = `${(position.column / field.columns * 100).toFixed(2)}%`;
      node.style.top = `${((1 - position.row / field.rows) * 100).toFixed(2)}%`;
    }
    return inside;
  }

  /** state: {floor, values: 'strongest' | 'all', cellPx} */
  render(field, {floor, values, cellPx}) {
    if (!field) return;
    this.build(field, cellPx);
    this.ticks.replaceChildren(...[0, 0.25, 0.5, 0.75, 1].map((f) => h('li', {}, f === 0 ? '0 dBFS' : num(floor * f, 0))));

    const marker = field.marker;
    const highlight = marker?.cell ?? (field.markerSuppressed ? null : field.strongest);
    const showText = this.cellPx >= LABEL_MIN_CELL;
    for (const cell of field.cells) {
      const entry = this.cells[cell.index];
      if (!entry) continue;
      const measured = Number.isFinite(cell.level);
      const rgb = rampColor(levelToUnit(cell.level, floor) ?? 0);
      if (look) {
        // A hidden look's tile: a dither step and a five-cell meter, faded when stale.
        const step = measured ? String(ditherStep(cell.level, floor)) : '';
        const fill = String(meterFill(cell.level, floor));
        if (entry.cell.dataset.lkStep !== step) entry.cell.dataset.lkStep = step;
        if (entry.meter.dataset.fill !== fill) entry.meter.dataset.fill = fill;
      } else {
        const fill = rgbCss(rgb);
        entry.cell.style.background = !measured ? 'var(--raised)' : cell.fresh ? fill : `${HATCH}, ${fill}`;
      }
      entry.cell.style.opacity = measured && !cell.fresh ? '0.55' : '1';
      entry.cell.classList.toggle('is-strongest', highlight === cell && marker?.kind !== 'beam');
      entry.cell.classList.toggle('is-held', marker?.kind === 'beam' && marker.cell === cell);
      entry.cell.classList.toggle('is-hover', this.hover === cell.index);
      const labelled = showText && measured && (values === 'all' || highlight === cell);
      setText(entry.text, labelled ? num(cell.level, 0) : '');
      if (!look) entry.text.style.color = inkOn(rgb);
      const place = `${field.kind === 'bank' ? 'Beam' : 'Sector'} ${cell.index}, azimuth ${deg(cell.azimuth)}, elevation ${deg(cell.elevation)}`;
      entry.cell.setAttribute('aria-label', measured ? `${place}, ${num(cell.level, 1)} dBFS` : `${place}, not measured yet`);
    }

    const shown = this.place(this.cross, marker ? gridPosition(field, marker.azimuth, marker.elevation) : null, field);
    this.pointerX.hidden = !shown;
    this.pointerY.hidden = !shown;
    if (shown) {
      this.pointerX.style.left = this.cross.style.left;
      this.pointerY.style.top = this.cross.style.top;
      setText(this.pointerX, `${Math.round(Math.abs(marker.azimuth))}°`);
      setText(this.pointerY, `${Math.round(Math.abs(marker.elevation))}°`);
    }
    const truth = field.truth;
    if (this.place(this.truth, truth ? gridPosition(field, truth.azimuth_deg, truth.elevation_deg) : null, field)) {
      this.truth.classList.toggle('is-dropout', truth.source_active === false);
    }
    this.place(this.alt, field.alt ? gridPosition(field, field.alt.azimuth, field.alt.elevation) : null, field);
  }
}
