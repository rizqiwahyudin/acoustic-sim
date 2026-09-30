/**
 * history.js — the last 30 seconds: direction of the answer and its level.
 */

import { h } from '../shared/dom.js';
import { LineChart } from '../shared/linechart.js';
import { deg, num } from '../shared/format.js';
import { angularEdges } from './geometry.js';
import { HISTORY_WINDOW_S } from './session.js';

const X_TICKS = [
  {value: -30, label: '−30 s'},
  {value: -20, label: '−20 s'},
  {value: -10, label: '−10 s'},
  {value: 0, label: 'now'},
];

export class HistoryCharts {
  constructor() {
    this.direction = new LineChart({label: 'Azimuth and elevation of the answer over the last 30 seconds'});
    this.level = new LineChart({label: 'Level of the answer over the last 30 seconds'});
    this.directionCaption = h('figcaption', {class: 'figure-caption'}, 'Direction of the answer (degrees)');
    this.levelCaption = h('figcaption', {class: 'figure-caption'}, 'Level of the answer (dBFS)');
    this.el = h('section', {class: 'hw-history', 'aria-labelledby': 'hw-history-title'},
      h('h2', {class: 'hw-history__title', id: 'hw-history-title'}, 'Last 30 seconds'),
      h('div', {class: 'hw-history__charts'},
        h('figure', {class: 'hw-figure'}, this.directionCaption, this.direction.el),
        h('figure', {class: 'hw-figure'}, this.levelCaption, this.level.el),
      ),
    );
    this.lastRender = 0;
  }

  render(session, {floor, force = false} = {}) {
    const now = performance.now();
    if (!force && now - this.lastRender < 200) return;
    this.lastRender = now;
    const frame = session.frame;
    const history = session.history;
    const latest = Number.isFinite(frame?.timestamp_s) ? frame.timestamp_s : Date.now() / 1000;
    const recent = history.filter((item) => latest - item.t <= HISTORY_WINDOW_S);

    const az = angularEdges(frame?.azimuth_deg || []);
    const el = angularEdges(frame?.elevation_deg || []);
    const top = Math.ceil(Math.max(az.at(-1) ?? 40, el.at(-1) ?? 40));
    const bottom = Math.floor(Math.min(az[0] ?? -40, el[0] ?? -40));
    this.direction.setAxes({
      x: {min: -HISTORY_WINDOW_S, max: 0, ticks: X_TICKS},
      y: {min: bottom, max: top, ticks: [top, 0, bottom].filter((v, i, all) => all.indexOf(v) === i)
        .map((value) => ({value, label: deg(value, 0)}))},
    });
    const azPoints = recent.map((item) => [item.t - latest, item.azimuth]);
    const elPoints = recent.map((item) => [item.t - latest, item.elevation]);
    const lastAz = [...recent].reverse().find((item) => Number.isFinite(item.azimuth))?.azimuth;
    const lastEl = [...recent].reverse().find((item) => Number.isFinite(item.elevation))?.elevation;
    this.direction.render([
      {points: elPoints, stroke: 'var(--muted)', width: 1.5, gap: 2, end: Number.isFinite(lastEl) ? `el ${deg(lastEl)}` : ''},
      {points: azPoints, stroke: 'var(--accent)', width: 2, gap: 2, end: Number.isFinite(lastAz) ? `az ${deg(lastAz)}` : '', endBold: true},
    ]);

    const f = Number.isFinite(floor) ? floor : -40;
    this.level.setAxes({
      x: {min: -HISTORY_WINDOW_S, max: 0, ticks: X_TICKS},
      y: {min: f, max: 0, ticks: [0, f / 2, f].map((value) => ({value, label: num(value, 0)}))},
    });
    const levelPoints = recent.map((item) => [item.t - latest, item.level]);
    const lastLevel = [...recent].reverse().find((item) => Number.isFinite(item.level))?.level;
    this.level.render([
      {points: levelPoints, stroke: 'var(--accent)', width: session.monitoring ? 2.5 : 2, gap: 2,
        end: Number.isFinite(lastLevel) ? num(lastLevel, 1) : '', endBold: true},
    ]);
    this.levelCaption.textContent = session.monitoring
      ? 'Level of the held beam (dBFS)' : 'Level of the answer (dBFS)';
  }
}
