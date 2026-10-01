/**
 * linechart.js — small SVG line chart with fixed axes and end labels,
 * in the style-A look (thin grid, muted ticks, one accent series).
 */

import { s } from './dom.js';

export class LineChart {
  constructor({width = 460, height = 136, pad = {l: 44, r: 84, t: 12, b: 24}, label = ''} = {}) {
    this.width = width;
    this.height = height;
    this.pad = pad;
    this.el = s('svg', {
      class: 'chart',
      viewBox: `0 0 ${width} ${height}`,
      role: 'img',
      'aria-label': label,
    });
    this.x = {min: 0, max: 1, ticks: []};
    this.y = {min: 0, max: 1, ticks: []};
  }

  /** Redraw at a new size in design pixels; text keeps its size. */
  setSize(width, height) {
    if (!(width > 0 && height > 0)) return false;
    if (Math.abs(width - this.width) < 1 && Math.abs(height - this.height) < 1) return false;
    this.width = width;
    this.height = height;
    this.el.setAttribute('viewBox', `0 0 ${width} ${height}`);
    return true;
  }

  setAxes({x, y}) {
    if (x) this.x = x;
    if (y) this.y = y;
  }

  px(value) {
    const {l, r} = this.pad;
    return l + (value - this.x.min) / (this.x.max - this.x.min || 1) * (this.width - l - r);
  }

  py(value) {
    const {t, b} = this.pad;
    const clamped = Math.max(Math.min(value, Math.max(this.y.min, this.y.max)), Math.min(this.y.min, this.y.max));
    return t + (this.y.max - clamped) / (this.y.max - this.y.min || 1) * (this.height - t - b);
  }

  /**
   * series: [{points: [[x, y], ...], stroke, width, dash, end: 'label', endBold}]
   * guides: [{y, label}] dashed horizontal guides.
   */
  render(series = [], guides = []) {
    const {l, r, t, b} = this.pad;
    const right = this.width - r;
    const bottom = this.height - b;
    const nodes = [];
    for (const tick of this.y.ticks) {
      const y = this.py(tick.value);
      nodes.push(s('line', {class: 'grid', x1: l, x2: right, y1: y, y2: y}));
      nodes.push(s('text', {class: 'tick', x: l - 8, y: y + 4, 'text-anchor': 'end'}, tick.label));
    }
    for (const tick of this.x.ticks) {
      nodes.push(s('text', {class: 'tick', x: this.px(tick.value), y: bottom + 18, 'text-anchor': 'middle'}, tick.label));
    }
    for (const guide of guides) {
      const y = this.py(guide.value);
      nodes.push(s('line', {class: 'guide', x1: l, x2: right, y1: y, y2: y}));
      if (guide.label) nodes.push(s('text', {class: 'tick', x: right + 4, y: y + 4}, guide.label));
    }
    const endLabels = [];
    for (const line of series) {
      const pts = line.points.filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
      if (pts.length === 0) continue;
      let d = '';
      let pen = false;
      let previousX = null;
      for (const [x, y] of line.points) {
        if (!Number.isFinite(x) || !Number.isFinite(y) || (line.gap && previousX !== null && x - previousX > line.gap)) {
          pen = false;
          if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        }
        d += `${pen ? 'L' : 'M'}${this.px(x).toFixed(1)} ${this.py(y).toFixed(1)} `;
        pen = true;
        previousX = x;
      }
      nodes.push(s('path', {
        d,
        fill: 'none',
        stroke: line.stroke || 'var(--accent)',
        'stroke-width': line.width || 2,
        'stroke-dasharray': line.dash || null,
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round',
      }));
      if (line.end) {
        const [, lastY] = pts[pts.length - 1];
        endLabels.push({y: this.py(lastY) + 4, line});
      }
    }
    // Keep end labels at least 13 px apart so equal values stay readable.
    endLabels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < endLabels.length; i++) {
      if (endLabels[i].y - endLabels[i - 1].y < 13) endLabels[i].y = endLabels[i - 1].y + 13;
    }
    for (const {y, line} of endLabels) {
      nodes.push(s('text', {
        x: right + 8,
        y,
        fill: line.stroke || 'var(--accent)',
        'font-size': 12,
        'font-weight': line.endBold ? 600 : 400,
      }, line.end));
    }
    this.el.replaceChildren(...nodes);
  }
}
