/**
 * view.js — Beam pattern (style A): array settings on the left, the 3D
 * pattern in the middle, polar cuts, array figures and the layout on the right.
 */

import { h, options, s, setText } from '../shared/dom.js';
import { rampGradient } from '../shared/colormap.js';
import { MINUS, deg, num } from '../shared/format.js';
import { buildGeometry } from '../geometry.js';
import { heimdallToolPositions } from '../shared/heimdall-array.js';
import { customArray, customArrayEditor } from '../shared/custom-array.js';
import { cuts, figures, makeGain, patternGrid, width3 } from './pattern.js';

const GEOMETRIES = [
  ['UCA', 'Circular ring'],
  ['CROSS', 'Standing cross'],
  ['ULA', 'Linear'],
  ['CYLINDER', 'Stacked rings'],
  ['HEIMDALL', 'Heimdall, 44 microphones'],
  ['CUSTOM', 'Custom coordinates'],
];

export function slider({id, label, min, max, step, value, format, onInput}) {
  const output = h('span', {class: 'slider__value'});
  const input = h('input', {id, type: 'range', class: 'slider__input', min, max, step, value});
  const update = () => setText(output, format(Number(input.value)));
  input.addEventListener('input', () => { update(); onInput(Number(input.value)); });
  update();
  return {
    el: h('div', {class: 'slider'}, h('div', {class: 'slider__head'}, h('label', {for: id}, label), output), input),
    input,
    set(v) { input.value = v; update(); },
  };
}

export function createView(container) {
  const state = {geo: 'UCA', count: 12, radius: 0.15, sep: 0.12, az: 60, el: 30, freq: 2000};
  let scene = null;
  let visible = false;

  // ── Left: array and steering ──────────────────────────────────────────
  const geoSelect = h('select', {id: 'bp-geo', class: 'select', onChange: () => { state.geo = geoSelect.value; syncControls(); update(); }},
    options(GEOMETRIES, state.geo));
  const count = slider({id: 'bp-mics', label: 'Microphones', min: 4, max: 32, step: 1, value: state.count,
    format: (v) => String(v), onInput: (v) => { state.count = v; update(); }});
  const radius = slider({id: 'bp-radius', label: 'Ring radius', min: 0.03, max: 1, step: 0.01, value: state.radius,
    format: (v) => `${v.toFixed(2)} m`, onInput: (v) => { state.radius = v; update(); }});
  const sep = slider({id: 'bp-sep', label: 'Ring separation', min: 0.02, max: 0.5, step: 0.01, value: state.sep,
    format: (v) => `${v.toFixed(2)} m`, onInput: (v) => { state.sep = v; update(); }});
  const editor = customArrayEditor({getSeed: () => ({count: state.count, radius: state.radius, separation: state.sep})});
  const heimdallNote = h('p', {class: 'note'}, 'The deployed array, seen facing it: it looks along 0° azimuth. Azimuth here runs counter-clockwise seen from above, so Heimdall’s right is negative azimuth.');

  const az = slider({id: 'bp-az', label: 'Azimuth', min: 0, max: 355, step: 5, value: state.az,
    format: (v) => `${v}°`, onInput: (v) => { state.az = v; update(); }});
  const el = slider({id: 'bp-el', label: 'Elevation', min: -90, max: 90, step: 5, value: state.el,
    format: (v) => `${num(v, 0)}°`, onInput: (v) => { state.el = v; update(); }});
  const freq = slider({id: 'bp-freq', label: 'Frequency', min: 50, max: 20000, step: 50, value: state.freq,
    format: (v) => `${v} Hz`, onInput: (v) => { state.freq = v; update(); }});

  const left = h('aside', {class: 'bp-settings', 'aria-label': 'Array settings'},
    h('section', {class: 'section', 'aria-labelledby': 'bp-array-h'},
      h('h2', {class: 'section__title', id: 'bp-array-h'}, 'Array'),
      h('div', {class: 'field'}, h('label', {for: 'bp-geo'}, 'Geometry'), geoSelect),
      count.el, radius.el, sep.el, heimdallNote, editor,
    ),
    h('section', {class: 'section', 'aria-labelledby': 'bp-steer-h'},
      h('h2', {class: 'section__title', id: 'bp-steer-h'}, 'Steering and frequency'),
      az.el, el.el, freq.el,
    ),
  );

  // ── Centre: 3D pattern ─────────────────────────────────────────────────
  const sceneBox = h('div', {class: 'scene-box scene-box--pattern'});
  const empty = h('div', {class: 'bp-empty', hidden: true},
    h('p', {class: 'bp-empty__title'}, 'No microphone coordinates yet'),
    h('p', {class: 'note note--body'}, 'Add positions one by one, import a coordinate file, or start from one of the built-in geometries.'),
    h('div', {class: 'toolbar'},
      h('button', {type: 'button', class: 'btn', onClick: () => customArray.seed('HEIMDALL')}, 'Start from Heimdall'),
      h('button', {type: 'button', class: 'btn', onClick: () => customArray.seed('UCA', state.count, state.radius, state.sep)}, 'Start from a circular ring'),
    ),
  );
  const legend = h('div', {class: 'bp-legend'},
    h('span', {}, `${MINUS}30 dB`),
    h('span', {class: 'bp-legend__bar', 'aria-hidden': 'true', style: {background: rampGradient('to right')}}),
    h('span', {}, '0 dB'),
  );
  const cameraButtons = [['angled', 'Angled'], ['top', 'Top'], ['side', 'Side']].map(([key, label]) =>
    h('button', {type: 'button', class: 'btn btn--sm btn--ghost', onClick: () => scene?.setView(key)}, label));
  const centre = h('section', {class: 'bp-centre', 'aria-labelledby': 'bp-3d-h'},
    h('div', {class: 'section__row'},
      h('div', {class: 'bp-centre__title'},
        h('h2', {class: 'bp-title', id: 'bp-3d-h'}, 'Beam pattern'),
        h('span', {class: 'note note--body'}, 'Gain relative to the steered direction. Drag to rotate.'),
      ),
      h('div', {class: 'toolbar toolbar--tight'}, ...cameraButtons),
    ),
    sceneBox, empty, legend,
  );

  // ── Right: cuts, figures, layout ───────────────────────────────────────
  const cutFigures = [0, 1].map(() => {
    const path = s('path', {class: 'polar__trace', d: ''});
    const ray = s('line', {class: 'polar__ray', x1: 100, y1: 100, x2: 180, y2: 100});
    const left = s('text', {class: 'polar__label', x: 4, y: 104});
    const bottom = s('text', {class: 'polar__label', x: 100, y: 196, 'text-anchor': 'middle'});
    const svg = s('svg', {viewBox: '0 0 200 200', class: 'polar', role: 'img'},
      s('circle', {class: 'polar__ring', cx: 100, cy: 100, r: 80}),
      s('circle', {class: 'polar__ring', cx: 100, cy: 100, r: 53.3}),
      s('circle', {class: 'polar__ring', cx: 100, cy: 100, r: 26.7}),
      s('line', {class: 'polar__ring', x1: 20, y1: 100, x2: 180, y2: 100}),
      s('line', {class: 'polar__ring', x1: 100, y1: 20, x2: 100, y2: 180}),
      ray, path,
      s('text', {class: 'polar__label', x: 186, y: 104}, '0°'),
      s('text', {class: 'polar__label', x: 100, y: 14, 'text-anchor': 'middle'}, '90°'),
      left, bottom,
      s('text', {class: 'polar__db', x: 104, y: 176}, '0 dB'),
      s('text', {class: 'polar__db', x: 104, y: 150}, `${MINUS}10`),
      s('text', {class: 'polar__db', x: 104, y: 124}, `${MINUS}20`),
    );
    const title = h('span', {class: 'figure-label figure-label--sm'});
    const width = h('span', {class: 'mono'});
    return {path, ray, left, bottom, svg, title, width,
      el: h('figure', {class: 'polar-figure'}, svg, h('figcaption', {}, title, width))};
  });
  const figureList = h('dl', {class: 'figures'});
  const aliasNote = h('p', {class: 'note note--body'});
  const layoutTitle = h('h2', {class: 'section__title', id: 'bp-layout-h'}, 'Layout');
  const layoutSvg = s('svg', {viewBox: '0 0 300 150', class: 'layout-plot', role: 'img', 'aria-label': 'Microphone positions'});
  const right = h('aside', {class: 'bp-figures', 'aria-label': 'Beam figures'},
    h('section', {class: 'section', 'aria-labelledby': 'bp-cuts-h'},
      h('h2', {class: 'section__title', id: 'bp-cuts-h'}, 'Cuts through the steered beam'),
      h('div', {class: 'polar-pair'}, cutFigures[0].el, cutFigures[1].el),
    ),
    h('section', {class: 'section', 'aria-labelledby': 'bp-fig-h'},
      h('h2', {class: 'section__title', id: 'bp-fig-h'}, 'Array figures'),
      figureList, aliasNote,
    ),
    h('section', {class: 'section', 'aria-labelledby': 'bp-layout-h'}, layoutTitle, layoutSvg),
  );

  container.append(h('div', {class: 'bp'}, left, centre, right));

  function currentMics() {
    if (state.geo === 'CUSTOM') return customArray.positions;
    if (state.geo === 'HEIMDALL') return heimdallToolPositions();
    return buildGeometry(state.geo, state.count, state.radius, state.sep);
  }

  function syncControls() {
    const builtIn = state.geo !== 'CUSTOM' && state.geo !== 'HEIMDALL';
    count.el.hidden = !builtIn;
    radius.el.hidden = !builtIn;
    sep.el.hidden = state.geo !== 'CYLINDER';
    editor.hidden = state.geo !== 'CUSTOM';
    heimdallNote.hidden = state.geo !== 'HEIMDALL';
    radius.el.querySelector('label').textContent = state.geo === 'ULA' || state.geo === 'CROSS' ? 'Half-length' : 'Ring radius';
  }

  function polarPath(samples) {
    let d = '';
    samples.forEach((sample, i) => {
      const r = Math.max(0, (sample.db + 30) / 30) * 80;
      const a = sample.angle * Math.PI / 180;
      d += `${i ? 'L' : 'M'}${(100 + r * Math.cos(a)).toFixed(1)} ${(100 - r * Math.sin(a)).toFixed(1)} `;
    });
    return d + 'Z';
  }

  function drawLayout(mics) {
    const spread = [0, 1, 2].map((k) => Math.max(...mics.map((m) => m[k])) - Math.min(...mics.map((m) => m[k])));
    const axes = [0, 1, 2].sort((a, b) => spread[b] - spread[a]).slice(0, 2).sort();
    const top = axes[0] === 0 && axes[1] === 1;
    setText(layoutTitle, top ? 'Layout, seen from above' : 'Layout, seen from the front');
    const pts = mics.map((m) => [m[axes[0]], m[axes[1]]]);
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const minX = Math.min(...xs); const maxX = Math.max(...xs);
    const minY = Math.min(...ys); const maxY = Math.max(...ys);
    const scale = Math.min(260 / Math.max(0.02, maxX - minX), 100 / Math.max(0.02, maxY - minY));
    const cx = 150 - (minX + maxX) / 2 * scale;
    const cy = 62 + (minY + maxY) / 2 * scale;
    const bar = [0.02, 0.05, 0.1, 0.2, 0.5, 1].find((b) => b * scale >= 36) || 1;
    const selected = state.geo === 'CUSTOM' ? customArray.selected : -1;
    layoutSvg.replaceChildren(
      ...pts.map((p, i) => s('circle', {cx: (cx + p[0] * scale).toFixed(1), cy: (cy - p[1] * scale).toFixed(1),
        r: i === selected ? 4.5 : 3, class: i === selected ? 'layout-plot__dot is-selected' : 'layout-plot__dot'})),
      s('line', {class: 'layout-plot__bar', x1: 20, y1: 140, x2: (20 + bar * scale).toFixed(1), y2: 140}),
      s('text', {class: 'layout-plot__label', x: (20 + bar * scale / 2).toFixed(1), y: 136, 'text-anchor': 'middle'},
        bar >= 1 ? `${bar} m` : `${Math.round(bar * 100)} cm`),
    );
  }

  let queued = false;
  function update() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; draw(); });
  }

  function draw() {
    const mics = currentMics();
    const ok = mics.length >= 2;
    sceneBox.hidden = !ok;
    legend.hidden = !ok;
    empty.hidden = ok;
    const gain = makeGain(mics, state.freq, state.az, state.el);
    if (ok) {
      scene?.render({grid: patternGrid(gain, 5), mics, steerAz: state.az, steerEl: state.el});
      const c = cuts(gain, state.az, state.el);
      const describe = [
        [c.azimuth, `Azimuth cut at elevation ${num(state.el, 0)}°`, state.az, '180°', '270°'],
        [c.elevation, `Elevation cut at azimuth ${state.az}°`, state.el, 'back', `${MINUS}90°`],
      ];
      describe.forEach(([cut, title, rayDeg, leftLabel, bottomLabel], i) => {
        const fig = cutFigures[i];
        fig.path.setAttribute('d', polarPath(cut.samples));
        const a = rayDeg * Math.PI / 180;
        fig.ray.setAttribute('x2', (100 + 80 * Math.cos(a)).toFixed(1));
        fig.ray.setAttribute('y2', (100 - 80 * Math.sin(a)).toFixed(1));
        setText(fig.left, leftLabel);
        setText(fig.bottom, bottomLabel);
        setText(fig.title, title);
        const w = width3(cut);
        setText(fig.width, w === null ? 'No −3 dB edge' : `−3 dB width ${w}°`);
        fig.svg.setAttribute('aria-label', `${title}. ${w === null ? 'No −3 dB edge' : `−3 dB width ${w} degrees`}.`);
      });
      const f = figures(mics, state.freq, gain);
      const rows = [
        ['Microphones', String(f.count)],
        ['Aperture', `${(f.aperture * 100).toFixed(1)} cm`],
        ['Nearest spacing', f.nearest ? `${(f.nearest * 100).toFixed(1)} cm` : '—'],
        ['Wavelength', `${(f.wavelength * 100).toFixed(1)} cm`],
        ['Spatial-alias limit', f.aliasHz ? `${Math.round(f.aliasHz)} Hz` : '—'],
        ['Beam solid angle', `≈ ${Math.round(f.solidAngle)} deg²`],
      ];
      figureList.replaceChildren(...rows.flatMap(([label, value]) => [h('dt', {}, label), h('dd', {class: 'mono'}, value)]));
      const aliased = f.aliasHz && state.freq > f.aliasHz;
      setText(aliasNote, aliased
        ? `${state.freq} Hz is above the spatial-alias limit. Expect grating lobes.`
        : `${state.freq} Hz is below the spatial-alias limit.`);
      aliasNote.classList.toggle('is-warn', Boolean(aliased));
      drawLayout(mics);
    } else {
      scene?.render({grid: null, mics: []});
      figureList.replaceChildren(h('dt', {}, 'Microphones'), h('dd', {class: 'mono'}, String(mics.length)));
      setText(aliasNote, 'Add at least two microphones to see figures.');
      aliasNote.classList.remove('is-warn');
      layoutSvg.replaceChildren();
      for (const fig of cutFigures) { fig.path.setAttribute('d', ''); setText(fig.width, '—'); }
    }
  }

  customArray.addEventListener('change', () => { if (state.geo === 'CUSTOM') update(); });
  import('./scene.js').then(({PatternScene}) => {
    scene = new PatternScene(sceneBox);
    if (visible) scene.start();
    update();
  });
  syncControls();
  update();

  return {
    show() { visible = true; scene?.start(); update(); },
    hide() { visible = false; scene?.stop(); },
    setGeometry(geo) { state.geo = geo; geoSelect.value = geo; syncControls(); update(); },
  };
}
