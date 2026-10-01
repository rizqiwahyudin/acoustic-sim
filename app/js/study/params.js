/**
 * params.js — Study · Parameters: block browser (from the register map),
 * editors (taper, filters, level detector, raw register) and the changes
 * column (pending, apply, history, snapshots). Editors only stage words;
 * nothing is written until Apply.
 */

import { apiJson } from '../shared/api.js';
import { h, options, s, segmented, setText } from '../shared/dom.js';
import { MINUS, clockTime, num } from '../shared/format.js';
import { HEIMDALL_POSITIONS_MM } from '../shared/heimdall-array.js';
import { ONE, dsp, hexWord, toFloat, toWord } from './dsp.js';

const FS = 48000;
const R_MAX = Math.max(...HEIMDALL_POSITIONS_MM.map(([x, y]) => Math.hypot(x, y)));
const dspName = (i) => `DSP${String(i).padStart(2, '0')}`;
const sum = (a) => a.reduce((x, y) => x + y, 0);

function taperWeights(kind, norm) {
  const base = HEIMDALL_POSITIONS_MM.map(([x, y]) => {
    const r = Math.hypot(x, y) / R_MAX;
    return kind === 'hann' ? 0.2 + 0.4 * (1 + Math.cos(Math.PI * r)) : kind === 'gauss' ? Math.exp(-2 * r * r) : 1;
  });
  const k = norm === 'peak' ? Math.max(...base) : sum(base) / base.length;
  return base.map((v) => v / k);
}

function azimuthCut(weights, freq) {
  const k = 2 * Math.PI * freq / 343000;
  const total = sum(weights) || 1;
  const out = [];
  for (let i = 0; i <= 360; i++) {
    const sn = Math.sin((i * 0.5 - 90) * Math.PI / 180);
    let re = 0; let im = 0;
    HEIMDALL_POSITIONS_MM.forEach(([x], m) => {
      if (!weights[m]) return;
      const phase = k * x * sn;
      re += weights[m] * Math.cos(phase);
      im += weights[m] * Math.sin(phase);
    });
    out.push(20 * Math.log10(Math.max(Math.hypot(re, im) / total, 1e-4)));
  }
  return out;
}

function cutMetrics(db, weights) {
  const c = 180;
  const edge = (dir) => {
    let i = c;
    while (i + dir >= 0 && i + dir <= 360 && db[i + dir] > -3) i += dir;
    if (i + dir < 0 || i + dir > 360) return 90;
    return (Math.abs(i - c) + (db[i] + 3) / (db[i] - db[i + dir])) * 0.5;
  };
  const valley = (dir) => { let j = c; while (j + dir >= 0 && j + dir <= 360 && db[j + dir] <= db[j]) j += dir; return j; };
  const right = valley(1); const left = valley(-1);
  let side = -Infinity;
  for (let i = 0; i <= 360; i++) if (i > right || i < left) side = Math.max(side, db[i]);
  const s1 = sum(weights); const s2 = sum(weights.map((w) => w * w));
  return {width: edge(1) + edge(-1), sidelobe: side, wng: s2 ? 10 * Math.log10(s1 * s1 / s2) : NaN};
}

function firResponseDb(t1, t2, f) {
  const mag = (taps) => {
    let re = 0; let im = 0;
    taps.forEach((v, k) => { const p = -2 * Math.PI * f * k / FS; re += v * Math.cos(p); im += v * Math.sin(p); });
    return Math.hypot(re, im);
  };
  return 20 * Math.log10(Math.max(mag(t1) * mag(t2), 1e-6));
}

const tauFromAlpha = (alpha) => (alpha > 0 && alpha < 1 ? -1000 / (FS * Math.log(1 - alpha)) : NaN);
const alphaFromTau = (ms) => 1 - Math.exp(-1 / (ms * FS / 1000));

export function parametersScreen({canWrite}) {
  const state = {editor: 'taper', group: 'gains', showAll: false, query: '', taper: null, norm: 'level', freq: 2000, sel: 17,
    firPreset: null, firStages: 'both', rawAddr: '1179', rawVal: '', reads: [], snapName: ''};

  // ── Block browser ──────────────────────────────────────────────────────
  const query = h('input', {type: 'search', class: 'input', placeholder: 'Name or address', 'aria-label': 'Filter blocks by name or address',
    onInput: () => { state.query = query.value.trim().toLowerCase(); render(); }});
  const groupList = h('ul', {class: 'blocks'});
  const browser = h('aside', {class: 'st-browser', 'aria-labelledby': 'st-blocks-h'},
    h('h2', {class: 'section__title', id: 'st-blocks-h'}, 'Blocks in the loaded program'),
    query, groupList,
    h('div', {class: 'st-legend'},
      h('span', {}, h('span', {class: 'dot dot--modified'}), 'Differs from the flashed program'),
      h('span', {}, h('span', {class: 'dot dot--pending'}), 'Pending edit, not written yet'),
      h('span', {}, h('span', {class: 'dot dot--locked'}), 'Read-only'),
      h('p', {class: 'note'}, 'Detector state, safeload registers and program memory can be read but not written.'),
    ),
  );

  // ── Editors ────────────────────────────────────────────────────────────
  const tabs = segmented([['taper', 'Taper and mutes'], ['filter', 'Filters'], ['detector', 'Level detector'], ['raw', 'Raw register']],
    {label: 'Editors', role: 'tablist', onPick: (key) => { state.editor = key; render(); }});
  const panels = {taper: taperPanel(), filter: filterPanel(), detector: detectorPanel(), raw: rawPanel()};
  const centre = h('section', {class: 'st-editor', 'aria-label': 'Editor'}, tabs.el, ...Object.values(panels).map((p) => p.el));

  // ── Changes ────────────────────────────────────────────────────────────
  const pendingList = h('ul', {class: 'pending'});
  const pendingCount = h('span', {class: 'mono note'});
  const summary = h('p', {class: 'pending__summary'});
  const precondition = h('p', {class: 'note'});
  const applyButton = h('button', {type: 'button', class: 'btn btn--primary btn--lg', onClick: () => apply()}, 'Apply');
  const discardButton = h('button', {type: 'button', class: 'btn btn--lg', onClick: () => dsp.discard()}, 'Discard');
  const historyList = h('ol', {class: 'history'});
  const snapList = h('ul', {class: 'snapshots'});
  const snapName = h('input', {class: 'input input--sm', placeholder: 'Snapshot name', 'aria-label': 'Snapshot name'});
  const message = h('p', {class: 'note', role: 'status'});
  const changes = h('aside', {class: 'st-changes', 'aria-label': 'Changes'},
    h('section', {class: 'section'},
      h('div', {class: 'section__row'}, h('h2', {class: 'section__title'}, 'Pending changes'), pendingCount),
      pendingList, summary, precondition,
      h('div', {class: 'toolbar toolbar--tight'}, applyButton, discardButton),
      message,
    ),
    h('section', {class: 'section'}, h('h2', {class: 'section__title'}, 'Applied this session'), historyList),
    h('section', {class: 'section'},
      h('h2', {class: 'section__title'}, 'Snapshots'),
      snapList,
      h('div', {class: 'inline-field'}, snapName,
        h('button', {type: 'button', class: 'btn btn--sm', onClick: () => {
          dsp.saveSnapshot(snapName.value.trim() || `Snapshot ${dsp.snapshots.length + 1}`);
          snapName.value = '';
        }}, 'Save device values')),
      h('button', {type: 'button', class: 'btn btn--sm btn--ghost', onClick: () => dsp.stageFlashed()}, 'Stage a return to flashed'),
      h('p', {class: 'note'}, 'Loading a snapshot fills the pending list. Snapshots are kept in this browser.'),
    ),
  );

  const el = h('div', {class: 'st-params'}, browser, centre, changes);

  async function apply() {
    const labels = dsp.pendingGroups().map((g) => g.label);
    if (dsp.setting) labels.push(dsp.setting.label);
    applyButton.disabled = true;
    setText(message, 'Writing…');
    try {
      const result = await dsp.apply(labels.join(', '));
      setText(message, result ? `Written with ${result.command_count} commands and read back.` : 'Setting applied.');
    } catch (error) {
      setText(message, `Not applied: ${error.message}`);
    }
    render();
  }

  // ── Taper and mutes ────────────────────────────────────────────────────
  function taperPanel() {
    const presets = segmented([['uniform', 'Uniform'], ['hann', 'Radial Hann'], ['gauss', 'Radial Gaussian']], {
      label: 'Taper', onPick: (key) => { state.taper = key; stageTaper(); }});
    const norm = h('select', {class: 'select select--sm', 'aria-label': 'Scale', onChange: () => { state.norm = norm.value; if (state.taper) stageTaper(); }},
      options([['level', 'Keep the on-axis level'], ['peak', 'Largest gain is 1.0']], 'level'));
    const freq = segmented([[1000, '1 kHz'], [2000, '2 kHz'], [4000, '4 kHz']], {label: 'Frequency', onPick: (f) => { state.freq = f; render(); }});
    const layout = s('svg', {viewBox: '0 0 300 290', class: 'mic-layout', role: 'img', 'aria-label': 'Proposed gain of each microphone'});
    const cut = s('svg', {viewBox: '0 0 520 246', class: 'chart', role: 'img'});
    const metrics = h('div', {class: 'metric-grid'});
    const micTitle = h('h3', {class: 'st-mic__title'});
    const micMeta = h('span', {class: 'note note--body'});
    const micGain = h('span', {class: 'mono'});
    const muteBox = h('input', {type: 'checkbox', onChange: () => stageMute(state.sel, muteBox.checked)});
    const muteNote = h('span', {class: 'note'});
    const el = h('div', {class: 'st-panel', role: 'tabpanel'},
      h('div', {class: 'st-panel__head'}, h('h2', {class: 'page-title'}, 'Gain taper and mutes'),
        h('p', {class: 'lede'}, 'One gain and one mute per microphone. A taper lowers the sidelobes and widens the main beam; muting single microphones checks channel order and left/right.')),
      h('div', {class: 'toolbar'}, presets.el, h('span', {class: 'inline-field'}, h('label', {}, 'Scale'), norm)),
      h('div', {class: 'st-figures'},
        h('figure', {class: 'st-figure st-figure--layout'}, h('figcaption', {class: 'figure-caption'}, 'Proposed gain per microphone, seen from behind. Click one to inspect it.'), layout),
        h('figure', {class: 'st-figure'},
          h('div', {class: 'section__row'}, h('figcaption', {class: 'figure-caption'}, 'Predicted response across azimuth at 0° elevation'), freq.el),
          cut,
          h('div', {class: 'chart-legend'}, h('span', {class: 'lg-line lg-line--device'}), 'On the device now', h('span', {class: 'lg-line lg-line--proposed'}), 'Proposed')),
      ),
      h('div', {class: 'st-bottom'},
        metrics,
        h('section', {class: 'st-mic'},
          h('div', {class: 'st-mic__text'}, micTitle, micMeta, micGain),
          h('div', {class: 'st-mic__mute'}, h('label', {class: 'check'}, muteBox, 'Mute this microphone'), muteNote))),
      h('p', {class: 'note'}, 'Predicted from the microphone positions for a distant source. Measure the real response under Measurements.'),
    );

    function stageTaper() {
      const weights = taperWeights(state.taper, state.norm);
      const writes = weights.map((w, i) => ({address: dsp.param('gains', i)?.address, word: toWord(w, '8.24')})).filter((w) => w.address !== undefined);
      dsp.unstage((address, entry) => entry.label.startsWith('Gain taper'));
      dsp.stage(writes, `Gain taper, ${{uniform: 'uniform', hann: 'radial Hann', gauss: 'radial Gaussian'}[state.taper]}`);
    }

    function stageMute(i, mute) {
      const param = dsp.param('mutes', i);
      if (!param) return;
      dsp.stage([{address: param.address, word: mute ? 0 : ONE}], `${mute ? 'Mute' : 'Unmute'} ${dspName(i)}`);
    }

    function render() {
      presets.set(state.taper);
      freq.set(state.freq);
      const gains = HEIMDALL_POSITIONS_MM.map((_, i) => dsp.param('gains', i));
      const mutes = HEIMDALL_POSITIONS_MM.map((_, i) => dsp.param('mutes', i));
      const device = gains.map((p, i) => (p ? dsp.deviceWord(p.address) / ONE : 1) * (mutes[i] && dsp.deviceWord(mutes[i].address) === 0 ? 0 : 1));
      const proposed = gains.map((p, i) => (p ? dsp.proposedWord(p.address) / ONE : 1) * (mutes[i] && dsp.proposedWord(mutes[i].address) === 0 ? 0 : 1));
      const maxW = Math.max(...proposed, 1e-6);
      layout.replaceChildren(
        s('circle', {cx: 150, cy: 145, r: 59, class: 'mic-layout__ring'}),
        s('circle', {cx: 150, cy: 145, r: 118, class: 'mic-layout__ring'}),
        ...HEIMDALL_POSITIONS_MM.map(([x, y], i) => {
          const muted = mutes[i] && dsp.proposedWord(mutes[i].address) === 0;
          const dot = s('circle', {
            cx: (150 + x * 0.5).toFixed(1), cy: (145 - y * 0.5).toFixed(1), r: 9,
            class: `mic-layout__dot ${i === state.sel ? 'is-selected' : ''} ${muted ? 'is-muted' : ''}`,
            'fill-opacity': muted ? 0 : (0.16 + 0.84 * proposed[i] / maxW).toFixed(2),
            onClick: () => { state.sel = i; render(); },
          }, s('title', {}, `${dspName(i)}, gain ${proposed[i].toFixed(3)}${muted ? ', muted' : ''}`));
          return dot;
        }),
        s('text', {x: (150 + HEIMDALL_POSITIONS_MM[state.sel][0] * 0.5).toFixed(1), y: (145 - HEIMDALL_POSITIONS_MM[state.sel][1] * 0.5 - 14).toFixed(1),
          class: 'mic-layout__label', 'text-anchor': 'middle'}, dspName(state.sel)),
      );
      const dbDevice = azimuthCut(device, state.freq);
      const dbProposed = azimuthCut(proposed, state.freq);
      const path = (db) => db.map((v, i) => `${i ? 'L' : 'M'}${(48 + i / 360 * 452).toFixed(1)} ${(16 + Math.min(40, -v) / 40 * 200).toFixed(1)}`).join(' ');
      cut.replaceChildren(
        ...[0, -10, -20, -30, -40].flatMap((d) => [s('line', {class: 'grid', x1: 48, x2: 500, y1: 16 - d * 5, y2: 16 - d * 5}),
          s('text', {class: 'tick', x: 40, y: 20 - d * 5, 'text-anchor': 'end'}, num(d, 0))]),
        ...[-90, -60, -30, 0, 30, 60, 90].map((d) => s('text', {class: 'tick', x: (48 + (d + 90) / 180 * 452).toFixed(1), y: 236, 'text-anchor': 'middle'}, `${num(d, 0)}°`)),
        s('line', {class: 'guide', x1: 48, x2: 500, y1: 31, y2: 31}),
        s('path', {d: path(dbDevice), fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1.5, 'stroke-dasharray': '5 4'}),
        s('path', {d: path(dbProposed), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2}),
      );
      const mDev = cutMetrics(dbDevice, device);
      const mNew = cutMetrics(dbProposed, proposed);
      const rows = [
        [`−3 dB beam width at ${state.freq / 1000} kHz`, `${num(mDev.width, 1)}°`, `${num(mNew.width, 1)}°`],
        ['Highest sidelobe', `${num(mDev.sidelobe, 1)} dB`, `${num(mNew.sidelobe, 1)} dB`],
        ['White-noise array gain', `${num(mDev.wng, 1)} dB`, `${num(mNew.wng, 1)} dB`],
      ];
      metrics.replaceChildren(h('span', {class: 'metric-grid__head'}, 'Predicted'), h('span', {class: 'metric-grid__head'}, 'On device'), h('span', {class: 'metric-grid__head'}, 'Proposed'),
        ...rows.flatMap(([label, a, b]) => [h('span', {}, label), h('span', {class: 'mono'}, a), h('span', {class: `mono ${a !== b ? 'is-changed' : ''}`}, b)]));
      const gain = gains[state.sel];
      const mute = mutes[state.sel];
      const r = Math.hypot(...HEIMDALL_POSITIONS_MM[state.sel]);
      setText(micTitle, `${dspName(state.sel)} · ${gain?.block || 'no gain block'}`);
      setText(micMeta, `Gain at ${gain?.address ?? '—'} · ${r.toFixed(1)} mm from the centre`);
      setText(micGain, gain ? `Device ${(dsp.deviceWord(gain.address) / ONE).toFixed(3)} · Proposed ${(dsp.proposedWord(gain.address) / ONE).toFixed(3)}` : '');
      muteBox.checked = Boolean(mute && dsp.proposedWord(mute.address) === 0);
      muteBox.disabled = !mute;
      setText(muteNote, mute ? `Mute word at ${mute.address}, ${mute.block} · ${mute.name}` : 'No mute block found for this microphone.');
    }
    return {el, render};
  }

  // ── Filters ────────────────────────────────────────────────────────────
  function filterPanel() {
    const presets = segmented([['flashed', 'As flashed'], ['unity', 'Flashed shape at 0 dB'], ['bypass', 'Bypass']], {
      label: 'Filter', onPick: (key) => { state.firPreset = key; stageFir(); }});
    const stages = segmented([['stage1', 'Stage 1'], ['stage2', 'Stage 2'], ['both', 'Both']], {
      label: 'Stages', onPick: (key) => { state.firStages = key; if (state.firPreset) stageFir(); else render(); }});
    const response = s('svg', {viewBox: '0 0 520 250', class: 'chart', role: 'img'});
    const readouts = h('div', {class: 'readouts'});
    const taps = h('div', {class: 'taps'});
    const el = h('div', {class: 'st-panel', role: 'tabpanel'},
      h('div', {class: 'st-panel__head'}, h('h2', {class: 'page-title'}, 'FIR filters'),
        h('p', {class: 'lede'}, 'Every microphone has two FIR stages. Presets apply to all 44 microphones.')),
      h('div', {class: 'toolbar'}, presets.el, stages.el),
      h('figure', {class: 'st-figure'}, h('figcaption', {class: 'figure-caption'}, 'Response of both stages together (DSP00)'), response,
        h('div', {class: 'chart-legend'}, h('span', {class: 'lg-line lg-line--device'}), 'On the device now', h('span', {class: 'lg-line lg-line--proposed'}), 'Proposed')),
      readouts, taps,
      h('p', {class: 'note'}, 'Assumes coefficient k sits at the block address + k; the flashed taps are symmetric, so confirm the order with the DSP owner before writing asymmetric taps. Eleven taps at 48 kHz cannot make a sharp edge.'),
    );

    function presetTaps(defaults) {
      if (state.firPreset === 'bypass') return defaults.map((_, k) => (k === Math.floor(defaults.length / 2) ? 1 : 0));
      if (state.firPreset === 'unity') { const total = sum(defaults); return defaults.map((v) => (total ? v / total : v)); }
      return defaults.slice();
    }

    function stageFir() {
      dsp.unstage((address, entry) => entry.label.startsWith('FIR'));
      const which = state.firStages === 'both' ? ['fir1', 'fir2'] : [state.firStages === 'stage1' ? 'fir1' : 'fir2'];
      const writes = [];
      for (const group of which) {
        for (const param of dsp.byGroup.get(group) || []) {
          const defaults = param.default_words.map((w) => w / ONE);
          presetTaps(defaults).forEach((v, k) => writes.push({address: param.address + k, word: toWord(v, '8.24')}));
        }
      }
      const stageText = which.length === 2 ? 'stages 1 and 2' : which[0] === 'fir1' ? 'stage 1' : 'stage 2';
      dsp.stage(writes, `FIR ${stageText}, ${{flashed: 'as flashed', unity: 'flashed shape at 0 dB', bypass: 'bypass'}[state.firPreset]}`);
    }

    function render() {
      presets.set(state.firPreset);
      stages.set(state.firStages);
      const p1 = dsp.param('fir1', 0);
      const p2 = dsp.param('fir2', 0);
      if (!p1 || !p2) { response.replaceChildren(); return; }
      const words = (param, fn) => Array.from({length: param.words}, (_, k) => fn(param.address + k) / ONE);
      const dev = [words(p1, (a) => dsp.deviceWord(a)), words(p2, (a) => dsp.deviceWord(a))];
      const pro = [words(p1, (a) => dsp.proposedWord(a)), words(p2, (a) => dsp.proposedWord(a))];
      const fx = (f) => 56 + Math.log10(f / 50) / Math.log10(24000 / 50) * 444;
      const fy = (d) => 16 + (6 - Math.max(-60, Math.min(6, d))) / 66 * 200;
      const path = (t) => Array.from({length: 121}, (_, i) => { const f = 50 * Math.pow(480, i / 120); return `${i ? 'L' : 'M'}${fx(f).toFixed(1)} ${fy(firResponseDb(t[0], t[1], f)).toFixed(1)}`; }).join(' ');
      response.replaceChildren(
        ...[0, -20, -40, -60].flatMap((d) => [s('line', {class: 'grid', x1: 56, x2: 500, y1: fy(d), y2: fy(d)}), s('text', {class: 'tick', x: 48, y: fy(d) + 4, 'text-anchor': 'end'}, num(d, 0))]),
        ...[[100, '100'], [300, '300'], [1000, '1k'], [3000, '3k'], [10000, '10k'], [20000, '20k']].map(([f, t]) => s('text', {class: 'tick', x: fx(f), y: 236, 'text-anchor': 'middle'}, t)),
        s('text', {class: 'tick', x: 500, y: 250, 'text-anchor': 'end'}, 'Hz'),
        s('path', {d: path(dev), fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1.5, 'stroke-dasharray': '5 4'}),
        s('path', {d: path(pro), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2}),
      );
      readouts.replaceChildren(...[[500, 'At 500 Hz'], [2000, 'At 2 kHz'], [6000, 'At 6 kHz']].map(([f, label]) =>
        h('div', {class: 'readout-item'}, h('span', {class: 'figure-label'}, label),
          h('span', {class: 'mono'}, `${num(firResponseDb(dev[0], dev[1], f), 1)} → ${num(firResponseDb(pro[0], pro[1], f), 1)} dB`))));
      const cells = [h('span', {class: 'taps__head'}, 'Tap'), ...dev[0].map((_, k) => h('span', {class: 'taps__head'}, String(k)))];
      [['Stage 1, device', dev[0], null], ['Stage 1, proposed', pro[0], dev[0]], ['Stage 2, device', dev[1], null], ['Stage 2, proposed', pro[1], dev[1]]].forEach(([label, values, base]) => {
        cells.push(h('span', {class: 'taps__label'}, label));
        values.forEach((v, k) => cells.push(h('span', {class: `mono ${base && Math.abs(v - base[k]) > 1e-9 ? 'is-changed' : base ? '' : 'is-muted'}`}, v.toFixed(4))));
      });
      taps.style.gridTemplateColumns = `124px repeat(${dev[0].length}, minmax(0, 1fr))`;
      taps.replaceChildren(...cells);
    }
    return {el, render};
  }

  // ── Level detector ─────────────────────────────────────────────────────
  function detectorPanel() {
    const fields = {};
    const field = (key, label, hint) => {
      const input = h('input', {class: 'input input--mono', type: 'text', inputmode: 'decimal', style: {width: '160px'},
        onChange: () => stageDetector(key, input.value)});
      const note = h('span', {class: 'field__hint'}, hint);
      const error = h('span', {class: 'field__error', role: 'alert'});
      fields[key] = {input, note, error};
      return h('div', {class: 'field'}, h('label', {}, label), input, note, error);
    };
    const chart = s('svg', {viewBox: '0 0 520 236', class: 'chart', role: 'img'});
    const readouts = h('div', {class: 'readouts'});
    const el = h('div', {class: 'st-panel', role: 'tabpanel'},
      h('div', {class: 'st-panel__head'}, h('h2', {class: 'page-title'}, 'Level detector'),
        h('p', {class: 'lede'}, 'The level detector measures the summed beam. For every sector the firmware steers, waits the settle time, then reads this level.')),
      h('div', {class: 'field-grid'},
        field('tau', 'Time constant (ms)', ''),
        field('hold', 'Hold (samples)', ''),
        field('decay', 'Decay (per sample)', ''),
        field('settle', 'Firmware settle before each read (µs)', 'A firmware setting sent with SET, not a DSP register.'),
      ),
      h('figure', {class: 'st-figure'}, h('figcaption', {class: 'figure-caption'}, 'How far a level step has settled when the firmware reads it'), chart,
        h('div', {class: 'chart-legend'}, h('span', {class: 'lg-line lg-line--device'}), 'On the device now', h('span', {class: 'lg-line lg-line--proposed'}), 'Proposed')),
      readouts,
    );

    const detectorParam = (name) => dsp.param('detector', undefined, name);

    function stageDetector(key, text) {
      const value = Number(String(text).replace(MINUS, '-'));
      const error = fields[key].error;
      setText(error, '');
      if (key === 'settle') {
        if (!(value >= 50 && value <= 20000)) { setText(error, 'Enter 50 to 20000 µs.'); return; }
        dsp.stageSetting('settle_us', value, 'Firmware settle time');
        return;
      }
      const param = detectorParam({tau: 'TCONST', hold: 'hold', decay: 'decay'}[key]);
      if (!param) return;
      let word;
      if (key === 'tau') {
        if (!(value >= 0.1 && value <= 500)) { setText(error, 'Enter a time from 0.1 to 500 ms.'); return; }
        word = toWord(alphaFromTau(value), '8.24');
      } else if (key === 'hold') {
        if (!(Number.isInteger(value) && value >= 0 && value <= 48000)) { setText(error, 'Enter a whole number of samples up to 48000.'); return; }
        word = value;
      } else {
        if (!(value >= 0 && value < 1)) { setText(error, 'Enter a value from 0 to 1.'); return; }
        word = toWord(value, '8.24');
      }
      dsp.stage([{address: param.address, word}], `Detector ${param.name}`);
    }

    function render() {
      const tc = detectorParam('TCONST');
      const hold = detectorParam('hold');
      const decay = detectorParam('decay');
      if (!tc) return;
      const tauDev = tauFromAlpha(dsp.deviceWord(tc.address) / ONE);
      const tauNew = tauFromAlpha(dsp.proposedWord(tc.address) / ONE);
      const settleDev = (dsp.status?.settings?.settle_us ?? 1250) / 1000;
      const settleNew = (dsp.setting?.key === 'settle_us' ? dsp.setting.value : dsp.status?.settings?.settle_us ?? 1250) / 1000;
      const focused = document.activeElement;
      const setField = (key, value) => { if (focused !== fields[key].input) fields[key].input.value = value; };
      setField('tau', tauNew.toFixed(2));
      setField('hold', String(hold ? dsp.proposedWord(hold.address) : ''));
      setField('decay', decay ? String(+(dsp.proposedWord(decay.address) / ONE).toPrecision(6)) : '');
      setField('settle', String(Math.round(settleNew * 1000)));
      setText(fields.tau.note, `On the device ${num(tauDev, 2)} ms. ${tc.name} at ${tc.address}, 8.24 ${hexWord(dsp.deviceWord(tc.address))}${dsp.pending.has(tc.address) ? ` → ${hexWord(dsp.proposedWord(tc.address))}` : ''}.`);
      if (hold) setText(fields.hold.note, `${(dsp.proposedWord(hold.address) / 48).toFixed(1)} ms. A falling level is held this long before it decays (as read from the block; confirm with the DSP owner). hold at ${hold.address}.`);
      if (decay) setText(fields.decay.note, `decay at ${decay.address}, 8.24 ${hexWord(dsp.proposedWord(decay.address))}.`);
      setText(fields.settle.note, `A firmware setting sent with SET, not a DSP register. On the device ${Math.round(settleDev * 1000)} µs.`);

      const tx = (t) => 48 + Math.min(20, t) / 20 * 452;
      const vy = (v) => 16 + (1 - v) * 180;
      const curve = (tau) => Array.from({length: 101}, (_, i) => `${i ? 'L' : 'M'}${tx(i * 0.2).toFixed(1)} ${vy(1 - Math.exp(-i * 0.2 / tau)).toFixed(1)}`).join(' ');
      const reached = (settle, tau) => 1 - Math.exp(-settle / tau);
      const rDev = reached(settleDev, tauDev);
      const rNew = reached(settleNew, tauNew);
      chart.replaceChildren(
        ...[[1, '100%'], [0.5, '50%'], [0, '0%']].flatMap(([v, t]) => [s('line', {class: 'grid', x1: 48, x2: 500, y1: vy(v), y2: vy(v)}), s('text', {class: 'tick', x: 40, y: vy(v) + 4, 'text-anchor': 'end'}, t)]),
        ...[0, 5, 10, 15, 20].map((t) => s('text', {class: 'tick', x: tx(t), y: 216, 'text-anchor': 'middle'}, String(t))),
        s('text', {class: 'tick', x: 500, y: 232, 'text-anchor': 'end'}, 'ms after the beam moves'),
        s('line', {x1: tx(settleNew), x2: tx(settleNew), y1: 16, y2: 196, stroke: 'var(--text)', 'stroke-dasharray': '3 3'}),
        s('text', {x: tx(settleNew) + 6, y: 12, class: 'tick'}, `Firmware reads at ${num(settleNew, 2)} ms`),
        s('path', {d: curve(tauDev), fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1.5, 'stroke-dasharray': '5 4'}),
        s('path', {d: curve(tauNew), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2}),
        s('circle', {cx: tx(settleDev), cy: vy(rDev), r: 4, fill: 'var(--muted)'}),
        s('circle', {cx: tx(settleNew), cy: vy(rNew), r: 4, fill: 'var(--accent)'}),
      );
      const rate = (settleMs) => 1000 / (36 * (settleMs + 2.22));
      readouts.replaceChildren(
        h('div', {class: 'readout-item'}, h('span', {class: 'figure-label'}, 'Settled when read'), h('span', {class: 'mono'}, `${Math.round(rDev * 100)}% → ${Math.round(rNew * 100)}%`)),
        h('div', {class: 'readout-item'}, h('span', {class: 'figure-label'}, 'Full-scan rate (36 sectors, estimate)'), h('span', {class: 'mono'}, `${rate(settleDev).toFixed(1)} → ${rate(settleNew).toFixed(1)} Hz`)),
        h('div', {class: 'readout-item'}, h('span', {class: 'figure-label'}, 'Settle needed for 95%'), h('span', {class: 'mono'}, `${(3 * tauNew).toFixed(1)} ms`)),
      );
    }
    return {el, render};
  }

  // ── Raw register ───────────────────────────────────────────────────────
  function rawPanel() {
    const address = h('input', {class: 'input input--mono', value: state.rawAddr, style: {width: '140px'}, 'aria-label': 'Address',
      onInput: () => { state.rawAddr = address.value; render(); }});
    const hexAddr = h('span', {class: 'mono note'});
    const readButton = h('button', {type: 'button', class: 'btn', onClick: () => read()}, 'Read');
    const info = h('dl', {class: 'raw-info'});
    const value = h('input', {class: 'input input--mono', style: {width: '180px'}, 'aria-label': 'New value', onInput: () => { state.rawVal = value.value; render(); }});
    const encoded = h('span', {class: 'mono note'});
    const stageButton = h('button', {type: 'button', class: 'btn', onClick: () => stageRaw()}, 'Add to pending');
    const reads = h('div', {class: 'reads'});
    const quick = h('div', {class: 'toolbar toolbar--tight'});
    const el = h('div', {class: 'st-panel', role: 'tabpanel'},
      h('div', {class: 'st-panel__head'}, h('h2', {class: 'page-title'}, 'Raw register'),
        h('p', {class: 'lede'}, 'Reads or writes one parameter word by address. Only words that the register map marks as writable can be written.')),
      h('div', {class: 'form-row'}, h('div', {class: 'field'}, h('label', {}, 'Address'), address), hexAddr, readButton),
      quick, info,
      h('div', {class: 'form-row'}, h('div', {class: 'field'}, h('label', {}, 'New value'), value), encoded, stageButton),
      h('p', {class: 'note'}, 'Enter a decimal value, or a raw 32-bit word as 0x…'),
      h('section', {class: 'section'}, h('h3', {class: 'section__title'}, 'Recent reads'), reads),
    );

    const parseAddress = () => {
      const t = state.rawAddr.trim();
      return /^0x[0-9a-f]+$/i.test(t) ? parseInt(t, 16) : /^\d+$/.test(t) ? parseInt(t, 10) : NaN;
    };
    const encode = (entry) => {
      const t = state.rawVal.trim().replace(MINUS, '-');
      if (/^0x[0-9a-f]{1,8}$/i.test(t)) { const n = parseInt(t, 16); return n >= 2 ** 31 ? n - 2 ** 32 : n; }
      if (t === '' || !Number.isFinite(Number(t))) return null;
      const v = Number(t);
      if (entry.param.type === 'int32') return Number.isInteger(v) ? v : null;
      return v >= -128 && v < 128 ? toWord(v, '8.24') : null;
    };
    async function read() {
      const a = parseAddress();
      const entry = dsp.index.get(a);
      let word = entry ? dsp.deviceWord(a) : null;
      let source = 'export default';
      if (dsp.status?.supported) {
        try { word = (await apiJson('/dsp/read', {method: 'POST', body: {address: a, count: 1}})).words[0]; source = 'device'; } catch (_) { /* keep */ }
      }
      if (word === null) return;
      state.reads.unshift({time: new Date(), address: a, name: entry ? `${entry.param.block} · ${entry.param.name}` : 'unknown', word, type: entry?.param.type || 'int32', source});
      state.reads = state.reads.slice(0, 6);
      render();
    }
    function stageRaw() {
      const a = parseAddress();
      const entry = dsp.index.get(a);
      const word = entry ? encode(entry) : null;
      if (!entry || !entry.param.writable || word === null) return;
      dsp.stage([{address: a, word}], `Raw write at ${a}`);
    }
    function render() {
      quick.replaceChildren(h('span', {class: 'note'}, 'Try'), ...['1179', '1120', '1123', '719'].map((a) =>
        h('button', {type: 'button', class: 'chip-button', onClick: () => { state.rawAddr = a; address.value = a; render(); }}, a)));
      const a = parseAddress();
      const entry = Number.isFinite(a) ? dsp.index.get(a) : null;
      const stateWord = Number.isFinite(a) ? dsp.registry?.states.find((st) => st.address === a) : null;
      setText(hexAddr, Number.isFinite(a) ? '0x' + a.toString(16).toUpperCase().padStart(4, '0') : '');
      const rows = entry ? [
        ['Block', `${entry.param.block} · ${entry.param.name}${entry.param.words > 1 ? `[${entry.offset}]` : ''}`],
        ['Type', entry.param.type === 'int32' ? '32-bit integer' : '8.24 fixed point'],
        ['Access', entry.param.writable ? 'Writable' : `Read-only: ${entry.param.reason}`],
        ['On the device', `${hexWord(dsp.deviceWord(a))} = ${num(toFloat(dsp.deviceWord(a), entry.param.type), entry.param.type === 'int32' ? 0 : 7)}`],
      ] : stateWord ? [['Block', stateWord.name], ['Access', 'Read-only: detector or block state']]
        : [['Block', Number.isFinite(a) ? 'Not a parameter in the loaded program' : 'Enter an address in decimal or as 0x…']];
      info.replaceChildren(...rows.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {class: k === 'Access' && !entry?.param.writable ? 'is-warn' : ''}, v)]));
      const word = entry ? encode(entry) : null;
      setText(encoded, word === null ? 'Not a valid value' : `Encodes as ${hexWord(word)}`);
      stageButton.disabled = !(entry && entry.param.writable && word !== null);
      readButton.disabled = !entry && !stateWord;
      reads.replaceChildren(...state.reads.map((r) => h('div', {class: 'reads__row'},
        h('span', {class: 'mono note'}, clockTime(r.time)), h('span', {class: 'mono'}, String(r.address)), h('span', {class: 'note'}, r.name),
        h('span', {class: 'mono'}, hexWord(r.word)), h('span', {class: 'mono'}, num(toFloat(r.word, r.type), r.type === 'int32' ? 0 : 7)))));
    }
    return {el, render};
  }

  // ── Block browser rendering ────────────────────────────────────────────
  function groupRows(key) {
    const params = dsp.byGroup.get(key) || [];
    const perMic = ['gains', 'mutes', 'fir1', 'fir2'].includes(key);
    return params.map((p) => {
      const addresses = Array.from({length: p.words}, (_, k) => p.address + k);
      const changed = addresses.some((a) => dsp.pending.has(a));
      let value;
      if (p.words === 1) {
        const dev = toFloat(dsp.deviceWord(p.address), p.type);
        const pro = toFloat(dsp.proposedWord(p.address), p.type);
        const fmt = (v) => (key === 'mutes' ? (v === 0 ? 'muted' : 'on') : p.type === 'int32' ? String(v) : v.toFixed(3));
        value = fmt(dev) + (changed ? ` → ${fmt(pro)}` : '');
      } else {
        const modified = addresses.some((a) => dsp.deviceWord(a) !== dsp.defaultWord(a));
        value = changed ? 'pending' : modified ? 'modified' : 'as flashed';
      }
      return {name: perMic && p.mic !== null ? dspName(p.mic) : p.name, address: p.words > 1 ? `${p.address}–${p.address + p.words - 1}` : String(p.address), value, changed};
    });
  }

  function renderBrowser() {
    const groups = dsp.registry?.groups || [];
    const q = state.query;
    const qNum = /^\d+$/.test(q) ? Number(q) : /^0x[0-9a-f]+$/.test(q) ? parseInt(q, 16) : NaN;
    const visible = groups.filter((g) => !q || (Number.isFinite(qNum) ? dsp.index.get(qNum)?.param.group === g.key : g.label.toLowerCase().includes(q) || g.key.includes(q)));
    const editorFor = {gains: 'taper', mutes: 'taper', fir1: 'filter', fir2: 'filter', detector: 'detector'};
    groupList.replaceChildren(...visible.map((g) => {
      const params = dsp.byGroup.get(g.key) || [];
      const words = params.flatMap((p) => Array.from({length: p.words}, (_, k) => p.address + k));
      const modified = words.some((a) => dsp.deviceWord(a) !== dsp.defaultWord(a));
      const pending = words.some((a) => dsp.pending.has(a));
      const expanded = state.group === g.key || (q && visible.length === 1);
      const item = h('li', {class: 'blocks__group'},
        h('button', {type: 'button', class: `blocks__head ${expanded ? 'is-open' : ''}`, 'aria-expanded': String(Boolean(expanded)), onClick: () => {
          state.group = state.group === g.key ? null : g.key;
          state.showAll = false;
          if (editorFor[g.key]) state.editor = editorFor[g.key];
          else { state.editor = 'raw'; state.rawAddr = String(g.min_address); }
          render();
        }},
        h('span', {class: 'blocks__name'}, g.label,
          modified ? h('span', {class: 'dot dot--modified', 'aria-label': 'differs from flashed'}) : null,
          pending ? h('span', {class: 'dot dot--pending', 'aria-label': 'pending edits'}) : null,
          g.writable_words === 0 ? h('span', {class: 'dot dot--locked', 'aria-label': 'read-only'}) : null),
        h('span', {class: 'blocks__meta'}, `${g.parameters} blocks · ${g.words} words${g.contiguous ? '' : ' · scattered'}`),
        h('span', {class: 'blocks__range mono'}, `${g.min_address}–${g.max_address}`)),
      );
      if (expanded) {
        if (g.writable_words === 0) {
          const reason = params[0]?.reason || 'read-only';
          item.append(h('p', {class: 'note blocks__note'}, `Read-only: ${reason}.${g.key === 'delays' ? ' A host-steered fine sweep needs the firmware to accept delay writes while idle (see docs/study-protocol.md).' : ''}`));
        } else {
          const rows = groupRows(g.key);
          const shown = state.showAll ? rows : rows.slice(0, 8);
          item.append(h('div', {class: 'blocks__rows'}, ...shown.map((r) => h('div', {class: 'blocks__row mono'},
            h('span', {}, r.name), h('span', {class: 'note'}, r.address), h('span', {class: r.changed ? 'is-changed' : ''}, r.value))),
          rows.length > 8 ? h('button', {type: 'button', class: 'link-button', onClick: () => { state.showAll = !state.showAll; render(); }},
            state.showAll ? 'Show fewer' : `Show all ${rows.length}`) : null));
        }
      }
      return item;
    }));
  }

  function renderChanges() {
    const groups = dsp.pendingGroups();
    const items = groups.map(({label, addresses}) => {
      const first = dsp.index.get(addresses[0]);
      let change = '';
      if (addresses.length === 1 && first) {
        const t = first.param.type;
        change = `${num(toFloat(dsp.deviceWord(addresses[0]), t), t === 'int32' ? 0 : 4)} → ${num(toFloat(dsp.proposedWord(addresses[0]), t), t === 'int32' ? 0 : 4)}`;
      }
      const span = addresses.length > 1 ? `${addresses.length} words · ${addresses[0]}–${addresses.at(-1)}` : `1 word · ${first?.param.block} · ${addresses[0]}`;
      return h('li', {class: 'pending__item'},
        h('span', {class: 'pending__title'}, label),
        h('button', {type: 'button', class: 'btn btn--sm btn--quiet', 'aria-label': `Remove ${label}`, onClick: () => dsp.unstage((a, e) => e.label === label)}, '×'),
        h('span', {class: 'note'}, span),
        change ? h('span', {class: 'mono pending__change'}, change) : null);
    });
    if (dsp.setting) {
      items.push(h('li', {class: 'pending__item'},
        h('span', {class: 'pending__title'}, dsp.setting.label),
        h('button', {type: 'button', class: 'btn btn--sm btn--quiet', 'aria-label': 'Remove the setting change', onClick: () => { dsp.setting = null; dsp.pendingChanged(); }}, '×'),
        h('span', {class: 'note'}, 'Firmware setting · SET,settle_us'),
        h('span', {class: 'mono pending__change'}, `${dsp.status?.settings?.settle_us ?? 1250} → ${dsp.setting.value} µs`)));
    }
    pendingList.replaceChildren(...(items.length ? items : [h('li', {class: 'note'}, 'Nothing pending. Edits wait here until you apply them.')]));
    setText(pendingCount, items.length ? String(items.length) : '');
    const plan = dsp.plan;
    if (!dsp.pending.size && !dsp.setting) setText(summary, '');
    else if (plan?.error) setText(summary, plan.error);
    else if (plan) setText(summary, `${plan.word_count} words in ${plan.command_count} commands${dsp.setting ? ' and one setting' : ''}, about ${Math.max(1, Math.round(plan.estimate_ms))} ms at 921600 baud`);
    const status = dsp.status;
    const ok = canWrite();
    const any = dsp.pending.size > 0 || Boolean(dsp.setting);
    let text = 'Written with safeload, up to five consecutive words per command, then read back.';
    if (!status || status.offline) text = 'The backend is not reachable.';
    else if (!status.supported) text = `${status.reason} You can browse and plan changes here.`;
    else if (!status.idle) text = 'Stop the scan first. The device refuses writes while the scan engine runs.';
    else if (!ok) text = 'Writes are off. Turn on Enable writes to apply.';
    setText(precondition, text);
    applyButton.disabled = !(any && ok && status?.supported && status?.idle);
    setText(applyButton, any ? `Apply ${items.length} ${items.length === 1 ? 'change' : 'changes'}` : 'Apply');
    discardButton.disabled = !any;

    historyList.replaceChildren(...(dsp.history.length ? dsp.history.map((entry) => h('li', {class: 'history__item'},
      h('span', {class: 'mono note'}, clockTime(entry.time).slice(0, 5)),
      h('span', {}, entry.label || 'Changes'),
      entry.reverted ? h('span', {class: 'note'}, 'Reverted')
        : h('button', {type: 'button', class: 'btn btn--sm', disabled: !(ok && status?.supported && status?.idle),
          onClick: async () => { try { await dsp.revert(entry); } catch (error) { setText(message, `Revert failed: ${error.message}`); } render(); }}, 'Revert'),
      h('span', {class: 'note history__meta'}, `${entry.words} words, ${entry.commands} commands`),
    )) : [h('li', {class: 'note'}, 'Nothing applied yet.')]));

    snapList.replaceChildren(...(dsp.snapshots.length ? dsp.snapshots.map((snap) => h('li', {class: 'snapshots__item'},
      h('span', {class: 'snapshots__text'}, h('span', {}, snap.name),
        h('span', {class: 'note'}, `${Object.keys(snap.words).length} words differ from flashed · ${new Date(snap.time).toLocaleString()}`)),
      h('button', {type: 'button', class: 'btn btn--sm', onClick: () => dsp.loadSnapshot(snap)}, 'Load'),
      h('button', {type: 'button', class: 'btn btn--sm btn--quiet', 'aria-label': `Delete ${snap.name}`, onClick: () => dsp.deleteSnapshot(snap)}, '×'),
    )) : [h('li', {class: 'note'}, 'No snapshots yet.')]));
  }

  function render() {
    if (!dsp.registry) return;
    tabs.set(state.editor);
    for (const [key, panel] of Object.entries(panels)) {
      panel.el.hidden = key !== state.editor;
      if (key === state.editor) panel.render();
    }
    renderBrowser();
    renderChanges();
  }

  dsp.addEventListener('change', render);
  dsp.addEventListener('plan', renderChanges);
  return {el, render};
}
