/**
 * measure.js — Study · Measurements: guided procedures that hold a beam,
 * poll its level with M, and turn the captures into a figure (SNR, main-lobe
 * width, dynamic range, detector scale, range estimate). Runs are kept in this
 * browser and export as CSV with a metadata header.
 *
 * Captures use only commands that exist today (X, S,<n>, M). Procedures that
 * need parameter writes on the device are listed but marked as such.
 */

import { h, s, setText } from '../shared/dom.js';
import { clockTime, deg, fileStamp, num } from '../shared/format.js';
import { download } from '../shared/dom.js';
import { dsp } from './dsp.js';

const RUNS_KEY = 'heimdall.study.runs';
const put = (el, ...kids) => el.replaceChildren(...kids.flat().filter(Boolean));

const TEMPLATES = [
  {key: 'snr', name: 'Signal-to-noise ratio', ready: true,
    text: 'Hold one beam on the source. Capture the level with the source on, then off. SNR is the on/off power ratio above the floor.'},
  {key: 'width', name: 'Main-lobe width, move the source', ready: true,
    text: 'Hold one beam and move the speaker through a list of angles. Gives the measured −3 dB and −6 dB widths and the pointing error of a fixed beam.'},
  {key: 'scale', name: 'Detector scale check', ready: true,
    text: 'Raise the source by a known step (for example +6 dB). If the reading rises by the same amount the detector reports amplitude dB; twice as much means power is being counted twice.'},
  {key: 'dynamic', name: 'Dynamic range', ready: true,
    text: 'Capture the floor with the source off, then step the source level up. The range runs from the floor to where the reading stops following the source.'},
  {key: 'range', name: 'Detection range, estimate', ready: true,
    text: 'Extrapolates an SNR measured at a known distance with spherical spreading and air absorption to the distance where SNR falls to your detection threshold.'},
  {key: 'sweep', name: 'Main-lobe width, steer across the source', ready: false,
    text: 'Steers in 1° steps by writing the 44 delays from the host. Needs the firmware to accept delay writes while idle (protocol v3).'},
  {key: 'arraygain', name: 'Array gain', ready: false,
    text: 'Beam SNR minus single-microphone SNR, by muting 43 of the 44 channels. Needs parameter writes on the device; the emulator accepts them but its levels do not react.'},
];

function powerMean(levels) {
  const p = levels.map((l) => 10 ** (l / 10));
  return 10 * Math.log10(p.reduce((a, b) => a + b, 0) / p.length);
}

function stdDb(levels) {
  const m = levels.reduce((a, b) => a + b, 0) / levels.length;
  return Math.sqrt(levels.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, levels.length - 1));
}

function loadRuns() {
  try { return JSON.parse(localStorage.getItem(RUNS_KEY)) || []; } catch (_) { return []; }
}

function saveRuns(runs) {
  try { localStorage.setItem(RUNS_KEY, JSON.stringify(runs.slice(0, 100))); } catch (_) { /* storage unavailable */ }
}

/** Hold `sector`, poll M at `rate`, collect levels for `seconds`. */
async function capture(session, sector, seconds, rate, onProgress) {
  if (!session.ready) throw new Error('Connect to the device or an emulator first (Hardware screen).');
  if (!session.validSector(sector)) throw new Error('Sector out of range.');
  if (session.firmwareMode !== 'IDLE') {
    session.command('X');
    const start = performance.now();
    while (session.firmwareMode !== 'IDLE' && performance.now() - start < 3000) await new Promise((r) => setTimeout(r, 50));
  }
  session.setMonitorRate(rate);
  session.startMonitor(sector);
  const levels = [];
  let lastSeq = null;
  const onFrame = () => {
    const f = session.frame;
    if (!f || f.sequence === lastSeq || f.last_record?.type !== 'measurement') return;
    lastSeq = f.sequence;
    if (f.last_steer?.sector !== sector) return;
    const level = f.levels_db?.[f.last_steer.row]?.[f.last_steer.column];
    if (Number.isFinite(level)) levels.push(level);
  };
  session.addEventListener('frame', onFrame);
  const t0 = performance.now();
  await new Promise((resolve) => {
    const tick = () => {
      const elapsed = (performance.now() - t0) / 1000;
      onProgress?.(Math.min(1, elapsed / seconds), levels.length);
      if (elapsed >= seconds) resolve(); else setTimeout(tick, 100);
    };
    tick();
  });
  session.removeEventListener('frame', onFrame);
  session.stopMonitor();
  if (!levels.length) throw new Error('No level readings arrived. Check that the device answers M.');
  return {n: levels.length, mean_db: powerMean(levels), std_db: stdDb(levels), min_db: Math.min(...levels), max_db: Math.max(...levels)};
}

function widthAt(points, drop) {
  // points: [{angle, level}] sorted by angle. Interpolated width where the level is `drop` below the peak.
  if (points.length < 3) return null;
  const peak = points.reduce((a, b) => (b.level > a.level ? b : a));
  const i0 = points.indexOf(peak);
  const edge = (dir) => {
    for (let i = i0; i + dir >= 0 && i + dir < points.length; i += dir) {
      const a = points[i]; const b = points[i + dir];
      if (b.level <= peak.level - drop) return a.angle + (b.angle - a.angle) * (a.level - (peak.level - drop)) / (a.level - b.level);
    }
    return null;
  };
  const left = edge(-1); const right = edge(1);
  return left === null || right === null ? null : {width: Math.abs(right - left), left, right, peak: peak.angle};
}

function rangeFor(snr0, d0, threshold, alphaDbKm) {
  const snrAt = (d) => snr0 - 20 * Math.log10(d / d0) - alphaDbKm * (d - d0) / 1000;
  if (snrAt(d0) < threshold) return {range: null, snrAt};
  let lo = d0; let hi = d0;
  while (snrAt(hi) >= threshold && hi < 1e5) hi *= 2;
  for (let i = 0; i < 60; i++) { const mid = (lo + hi) / 2; if (snrAt(mid) >= threshold) lo = mid; else hi = mid; }
  return {range: lo, snrAt};
}

export function measurementsScreen({session}) {
  const state = {template: 'snr', runs: loadRuns(), busy: false, draft: null};

  const list = h('ul', {class: 'templates'});
  const left = h('aside', {class: 'st-browser', 'aria-labelledby': 'ms-templates-h'},
    h('h2', {class: 'section__title', id: 'ms-templates-h'}, 'Measurements'), list,
    h('p', {class: 'note'}, 'Levels are dBFS from the level detector until you calibrate with a sound level meter at the array. Record the reference SPL in the notes.'));

  const body = h('div', {class: 'st-panel'});
  const centre = h('section', {class: 'st-editor', 'aria-label': 'Procedure'}, body);

  const runList = h('ul', {class: 'runs'});
  const right = h('aside', {class: 'st-changes', 'aria-label': 'Runs'},
    h('section', {class: 'section'},
      h('div', {class: 'section__row'}, h('h2', {class: 'section__title'}, 'Saved runs'),
        h('button', {type: 'button', class: 'btn btn--sm btn--quiet', onClick: () => exportAll()}, 'Export all')),
      runList,
      h('p', {class: 'note'}, 'Kept in this browser. Each export carries the device, the register-map hash, the firmware settings and the time.')));

  const el = h('div', {class: 'st-params'}, left, centre, right);

  function meta() {
    return {
      device: session.open ? session.transportLabel : 'not connected',
      grid: session.configuration ? `${session.configuration.rows}x${session.configuration.columns}` : '',
      register_map: dsp.registry?.export?.sha256?.slice(0, 16) || '',
      modified_words: dsp.status?.modified_words ?? '',
      settle_us: dsp.status?.settings?.settle_us ?? '',
    };
  }

  function saveRun(template, name, captures, result, notes) {
    state.runs.unshift({id: Date.now(), template, name, time: new Date().toISOString(), meta: meta(), captures, result, notes});
    saveRuns(state.runs);
    renderRuns();
  }

  function runCsv(run) {
    const lines = [`# ${run.name}`, `# template: ${run.template}`, `# time: ${run.time}`,
      ...Object.entries(run.meta || {}).map(([k, v]) => `# ${k}: ${v}`),
      ...(run.notes ? [`# notes: ${String(run.notes).replace(/\n/g, ' ')}`] : []),
      ...Object.entries(run.result || {}).map(([k, v]) => `# result.${k}: ${v}`),
      'label,value,n,mean_dbfs,std_db,min_dbfs,max_dbfs'];
    for (const c of run.captures || []) lines.push([c.label, c.value ?? '', c.n, c.mean_db?.toFixed(2), c.std_db?.toFixed(2), c.min_db?.toFixed(2), c.max_db?.toFixed(2)].join(','));
    return lines.join('\n') + '\n';
  }

  function exportRun(run) {
    download(new Blob([runCsv(run)], {type: 'text/csv'}), `heimdall-${run.template}-${fileStamp(new Date(run.time))}.csv`);
  }

  function exportAll() {
    if (!state.runs.length) return;
    download(new Blob([state.runs.map(runCsv).join('\n')], {type: 'text/csv'}), `heimdall-runs-${fileStamp()}.csv`);
  }

  function renderRuns() {
    runList.replaceChildren(...(state.runs.length ? state.runs.map((run) => h('li', {class: 'runs__item'},
      h('span', {class: 'runs__title'}, run.name),
      h('span', {class: 'note'}, `${new Date(run.time).toLocaleString()} · ${run.meta?.device || ''}`),
      h('span', {class: 'mono'}, run.result?.summary || ''),
      h('div', {class: 'toolbar toolbar--tight'},
        h('button', {type: 'button', class: 'btn btn--sm', onClick: () => exportRun(run)}, 'Export CSV'),
        h('button', {type: 'button', class: 'btn btn--sm btn--quiet', 'aria-label': `Delete ${run.name}`, onClick: () => {
          state.runs = state.runs.filter((r) => r !== run); saveRuns(state.runs); renderRuns();
        }}, '×')),
    )) : [h('li', {class: 'note'}, 'No runs yet.')]));
  }

  function renderTemplates() {
    list.replaceChildren(...TEMPLATES.map((t) => h('li', {},
      h('button', {type: 'button', class: `template ${state.template === t.key ? 'is-open' : ''}`, onClick: () => { if (!state.busy) { state.template = t.key; state.draft = null; renderBody(); renderTemplates(); } }},
        h('span', {class: 'template__name'}, t.name),
        h('span', {class: `chip ${t.ready ? 'chip--plain' : 'chip--warn'}`}, t.ready ? 'Works now' : 'Needs protocol v3')))));
  }

  // Shared inputs
  const numberInput = (label, value, attrs = {}) => {
    const input = h('input', {class: 'input input--mono', type: 'number', value, style: {width: attrs.width || '110px'}, step: attrs.step || 'any'});
    return {input, el: h('div', {class: 'field'}, h('label', {}, label), input), get value() { return Number(input.value); }};
  };
  const notesInput = () => h('textarea', {class: 'input textarea', rows: 2, placeholder: 'Source, distance, level, room, reference SPL…'});

  function captureControls({sectorDefault}) {
    const sector = numberInput('Sector', sectorDefault, {width: '90px', step: 1});
    const seconds = numberInput('Capture (s)', 5, {width: '90px'});
    const rate = h('select', {class: 'select', 'aria-label': 'Read rate'}, h('option', {value: 10}, '10 Hz'), h('option', {value: 20, selected: true}, '20 Hz'), h('option', {value: 50}, '50 Hz'));
    return {sector, seconds, rate, el: h('div', {class: 'form-row'}, sector.el, seconds.el, h('div', {class: 'field'}, h('label', {}, 'Read rate'), rate))};
  }

  function defaultSector() {
    const cfg = session.configuration;
    const answer = session.solution();
    if (answer) return answer.sector;
    return cfg ? Math.floor(cfg.rows / 2) * cfg.columns + Math.floor(cfg.columns / 2) : 0;
  }

  function header(t) {
    return h('div', {class: 'st-panel__head'}, h('h2', {class: 'page-title'}, t.name), h('p', {class: 'lede'}, t.text));
  }

  function renderBody() {
    const t = TEMPLATES.find((x) => x.key === state.template);
    const status = h('p', {class: 'note', role: 'status'});
    const progress = h('progress', {max: 1, value: 0, hidden: true});
    const results = h('div', {class: 'ms-results'});
    const connected = session.ready;
    const needConnection = connected ? null : h('p', {class: 'note is-warn'}, 'Connect to the device or an emulator on the Hardware screen first.');

    const doCapture = async (controls, label, value) => {
      state.busy = true;
      progress.hidden = false;
      setText(status, `Capturing ${label}…`);
      try {
        const c = await capture(session, controls.sector.value, controls.seconds.value, Number(controls.rate.value), (f, n) => { progress.value = f; setText(status, `Capturing ${label}… ${n} readings`); });
        setText(status, `${label}: ${num(c.mean_db, 1)} dBFS ± ${num(c.std_db, 1)} dB from ${c.n} readings.`);
        return {label, value, ...c};
      } catch (error) {
        setText(status, error.message);
        return null;
      } finally {
        state.busy = false;
        progress.hidden = true;
      }
    };

    if (!t.ready) {
      put(body, header(t), h('p', {class: 'note is-warn'}, 'Not available yet: this procedure needs commands the firmware does not have. See docs/study-protocol.md.'));
      return;
    }

    if (t.key === 'snr') {
      const controls = captureControls({sectorDefault: defaultSector()});
      const notes = notesInput();
      const draft = state.draft ||= {on: null, off: null};
      const show = () => {
        const rows = [['Source on', draft.on], ['Source off', draft.off]].map(([k, c]) => h('div', {class: 'ms-row'}, h('span', {}, k), h('span', {class: 'mono'}, c ? `${num(c.mean_db, 1)} dBFS ± ${num(c.std_db, 1)} (${c.n})` : '—')));
        let figure = null;
        if (draft.on && draft.off) {
          const pOn = 10 ** (draft.on.mean_db / 10); const pOff = 10 ** (draft.off.mean_db / 10);
          const snr = pOn > pOff ? 10 * Math.log10((pOn - pOff) / pOff) : -Infinity;
          figure = h('div', {class: 'ms-figure'}, h('span', {class: 'figure-label'}, 'SNR'), h('span', {class: 'ms-figure__value'}, Number.isFinite(snr) ? `${num(snr, 1)} dB` : 'below the floor'),
            h('button', {type: 'button', class: 'btn btn--primary', onClick: () => saveRun('snr', `SNR, sector ${controls.sector.value}`, [draft.on, draft.off], {snr_db: snr.toFixed(2), summary: `SNR ${num(snr, 1)} dB`}, notes.value)}, 'Save run'));
        }
        put(results, ...rows, figure);
      };
      put(body, header(t), needConnection, controls.el,
        h('div', {class: 'toolbar'},
          h('button', {type: 'button', class: 'btn', disabled: !connected, onClick: async () => { draft.on = await doCapture(controls, 'source on') || draft.on; show(); }}, 'Capture with the source on'),
          h('button', {type: 'button', class: 'btn', disabled: !connected, onClick: async () => { draft.off = await doCapture(controls, 'source off') || draft.off; show(); }}, 'Capture with the source off')),
        progress, status, results, h('div', {class: 'field'}, h('label', {}, 'Notes'), notes),
        h('p', {class: 'note'}, 'SNR = 10·log10((P_on − P_off) / P_off) with P = 10^(L/10). If the detector turns out to count power twice (see Detector scale check), halve the dB figures.'));
      show();
      return;
    }

    if (t.key === 'width') {
      const controls = captureControls({sectorDefault: defaultSector()});
      const angles = h('input', {class: 'input input--mono', value: '-30,-20,-15,-10,-5,0,5,10,15,20,30', style: {width: '100%'}});
      const notes = notesInput();
      const draft = state.draft ||= {points: [], index: 0};
      const list = () => angles.value.split(',').map((v) => Number(v.trim())).filter(Number.isFinite);
      const chart = s('svg', {viewBox: '0 0 520 220', class: 'chart', role: 'img', 'aria-label': 'Measured level against source angle'});
      const show = () => {
        const all = list();
        const next = all[draft.index];
        setText(prompt, next === undefined ? 'All angles recorded.' : `Put the source at ${deg(next, 0)} (angle ${draft.index + 1} of ${all.length}), then record.`);
        recordButton.disabled = !connected || next === undefined;
        const pts = [...draft.points].sort((a, b) => a.angle - b.angle);
        const w3 = widthAt(pts, 3); const w6 = widthAt(pts, 6);
        if (pts.length) {
          const lo = Math.min(...all, ...pts.map((p) => p.angle)); const hi = Math.max(...all, ...pts.map((p) => p.angle));
          const top = Math.max(...pts.map((p) => p.level));
          const x = (a) => 48 + (a - lo) / Math.max(1, hi - lo) * 440;
          const y = (l) => 16 + Math.min(24, top - l) / 24 * 180;
          chart.replaceChildren(
            ...[0, -6, -12, -18, -24].flatMap((d) => [s('line', {class: 'grid', x1: 48, x2: 488, y1: y(top + d), y2: y(top + d)}), s('text', {class: 'tick', x: 40, y: y(top + d) + 4, 'text-anchor': 'end'}, num(d, 0))]),
            s('line', {class: 'guide', x1: 48, x2: 488, y1: y(top - 3), y2: y(top - 3)}),
            s('path', {d: pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.angle)} ${y(p.level)}`).join(' '), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2}),
            ...pts.map((p) => s('circle', {cx: x(p.angle), cy: y(p.level), r: 3.5, fill: 'var(--accent)'})),
            ...[lo, (lo + hi) / 2, hi].map((a) => s('text', {class: 'tick', x: x(a), y: 214, 'text-anchor': 'middle'}, deg(a, 0))),
          );
        } else chart.replaceChildren();
        put(results, chart,
          h('div', {class: 'readouts'},
            h('div', {class: 'readout-item'}, h('span', {class: 'figure-label'}, '−3 dB width'), h('span', {class: 'mono'}, w3 ? `${num(w3.width, 1)}°` : '—')),
            h('div', {class: 'readout-item'}, h('span', {class: 'figure-label'}, '−6 dB width'), h('span', {class: 'mono'}, w6 ? `${num(w6.width, 1)}°` : '—')),
            h('div', {class: 'readout-item'}, h('span', {class: 'figure-label'}, 'Peak at'), h('span', {class: 'mono'}, w3 ? deg(w3.peak, 1) : '—'))),
          pts.length >= 3 ? h('button', {type: 'button', class: 'btn btn--primary', onClick: () => saveRun('width', `Main-lobe width, sector ${controls.sector.value}`,
            pts.map((p) => ({...p.capture, label: `source ${p.angle}`, value: p.angle})),
            {width_3db_deg: w3?.width?.toFixed(2) ?? '', width_6db_deg: w6?.width?.toFixed(2) ?? '', peak_deg: w3?.peak ?? '', summary: w3 ? `−3 dB ${num(w3.width, 1)}°` : 'no −3 dB edge'}, notes.value)}, 'Save run') : null);
      };
      const prompt = h('p', {class: 'ms-prompt'});
      const recordButton = h('button', {type: 'button', class: 'btn btn--primary', onClick: async () => {
        const angle = list()[draft.index];
        const c = await doCapture(controls, `source at ${angle}°`, angle);
        if (c) { draft.points = draft.points.filter((p) => p.angle !== angle).concat([{angle, level: c.mean_db, capture: c}]); draft.index += 1; }
        show();
      }}, 'Record this angle');
      put(body, header(t), needConnection, controls.el,
        h('div', {class: 'field'}, h('label', {}, 'Source angles (degrees, relative to the held beam)'), angles),
        prompt, h('div', {class: 'toolbar'}, recordButton,
          h('button', {type: 'button', class: 'btn btn--ghost', onClick: () => { draft.index = Math.max(0, draft.index - 1); show(); }}, 'Back one'),
          h('button', {type: 'button', class: 'btn btn--ghost', onClick: () => { draft.points = []; draft.index = 0; show(); }}, 'Start over')),
        progress, status, results, h('div', {class: 'field'}, h('label', {}, 'Notes'), notes),
        h('p', {class: 'note'}, 'Keep the distance and the source level constant, and measure the angle from the array centre. A turntable under the array works as well as moving the speaker.'));
      angles.addEventListener('change', show);
      show();
      return;
    }

    if (t.key === 'scale' || t.key === 'dynamic') {
      const controls = captureControls({sectorDefault: defaultSector()});
      const setLevel = numberInput(t.key === 'scale' ? 'Source level change (dB)' : 'Source level (dB, relative)', t.key === 'scale' ? 0 : -30, {width: '120px'});
      const notes = notesInput();
      const draft = state.draft ||= {captures: []};
      const show = () => {
        const caps = draft.captures;
        const rows = caps.map((c) => h('div', {class: 'ms-row'}, h('span', {}, c.label), h('span', {class: 'mono'}, `${num(c.mean_db, 1)} dBFS ± ${num(c.std_db, 1)}`)));
        let summary = null;
        let result = null;
        if (t.key === 'scale' && caps.length >= 2) {
          const a = caps[0]; const b = caps.at(-1);
          const step = b.value - a.value;
          const seen = b.mean_db - a.mean_db;
          const ratio = step ? seen / step : NaN;
          const verdict = !Number.isFinite(ratio) ? 'Change the source level between captures.'
            : Math.abs(ratio - 1) < 0.2 ? 'The reading follows the source 1:1. The detector reports amplitude dB.'
              : Math.abs(ratio - 2) < 0.3 ? 'The reading moves twice as far as the source. Power is counted twice; halve dB differences.'
                : `The reading moves ${ratio.toFixed(2)}× the source step. Check for clipping or the noise floor.`;
          summary = h('p', {class: 'ms-prompt'}, `Source ${num(step, 1)} dB, reading ${num(seen, 1)} dB. ${verdict}`);
          result = {step_db: step, seen_db: seen.toFixed(2), ratio: Number.isFinite(ratio) ? ratio.toFixed(3) : '', summary: `ratio ${Number.isFinite(ratio) ? ratio.toFixed(2) : '—'}`};
        }
        if (t.key === 'dynamic' && caps.length >= 2) {
          const floor = caps.find((c) => c.value === null);
          const steps = caps.filter((c) => c.value !== null).sort((a, b) => a.value - b.value);
          let ceiling = steps.at(-1);
          for (let i = 1; i < steps.length; i++) {
            const slope = (steps[i].mean_db - steps[i - 1].mean_db) / Math.max(1e-6, steps[i].value - steps[i - 1].value);
            if (slope < 0.5) { ceiling = steps[i - 1]; break; }
          }
          const range = floor && ceiling ? ceiling.mean_db - floor.mean_db : null;
          summary = h('p', {class: 'ms-prompt'}, floor
            ? `Floor ${num(floor.mean_db, 1)} dBFS, highest level that still follows the source ${num(ceiling.mean_db, 1)} dBFS: dynamic range ${num(range, 1)} dB.`
            : 'Capture the floor with the source off as well.');
          result = floor ? {floor_dbfs: floor.mean_db.toFixed(2), ceiling_dbfs: ceiling.mean_db.toFixed(2), range_db: range.toFixed(2), summary: `range ${num(range, 1)} dB`} : null;
        }
        put(results, ...rows, summary, result ? h('button', {type: 'button', class: 'btn btn--primary', onClick: () => saveRun(t.key, t.name, caps, result, notes.value)}, 'Save run') : null);
      };
      put(body, header(t), needConnection, controls.el, setLevel.el,
        h('div', {class: 'toolbar'},
          h('button', {type: 'button', class: 'btn', disabled: !connected, onClick: async () => {
            const c = await doCapture(controls, `source ${num(setLevel.value, 1)} dB`, setLevel.value);
            if (c) draft.captures.push(c); show();
          }}, 'Capture at this level'),
          t.key === 'dynamic' ? h('button', {type: 'button', class: 'btn', disabled: !connected, onClick: async () => {
            const c = await doCapture(controls, 'floor, source off', null);
            if (c) draft.captures.unshift({...c, value: null}); show();
          }}, 'Capture the floor (source off)') : null,
          h('button', {type: 'button', class: 'btn btn--ghost', onClick: () => { draft.captures = []; show(); }}, 'Start over')),
        progress, status, results, h('div', {class: 'field'}, h('label', {}, 'Notes'), notes));
      show();
      return;
    }

    if (t.key === 'range') {
      const lastSnr = state.runs.find((r) => r.template === 'snr');
      const snr0 = numberInput('Measured SNR (dB)', lastSnr ? Number(lastSnr.result.snr_db).toFixed(1) : 20, {width: '120px'});
      const d0 = numberInput('at distance (m)', 10, {width: '110px'});
      const threshold = numberInput('Detection threshold (dB)', 6, {width: '120px'});
      const alpha = numberInput('Air absorption (dB/km)', 5, {width: '120px'});
      const chart = s('svg', {viewBox: '0 0 520 220', class: 'chart', role: 'img', 'aria-label': 'Predicted SNR against distance'});
      const answer = h('p', {class: 'ms-prompt'});
      const show = () => {
        const {range, snrAt} = rangeFor(snr0.value, Math.max(0.1, d0.value), threshold.value, Math.max(0, alpha.value));
        setText(answer, range ? `Estimated detection range: ${range < 1000 ? `${Math.round(range)} m` : `${(range / 1000).toFixed(2)} km`}. This is an extrapolation, not a measurement.` : 'The measured SNR is already below the threshold at that distance.');
        const dMin = Math.max(0.1, d0.value) / 2; const dMax = Math.max((range || d0.value) * 3, d0.value * 4);
        const x = (d) => 48 + Math.log10(d / dMin) / Math.log10(dMax / dMin) * 440;
        const top = Math.ceil(snr0.value + 10); const bottom = Math.min(threshold.value - 10, 0);
        const y = (v) => 16 + (top - Math.max(bottom, Math.min(top, v))) / (top - bottom) * 180;
        const pts = Array.from({length: 101}, (_, i) => dMin * (dMax / dMin) ** (i / 100));
        chart.replaceChildren(
          s('line', {class: 'guide', x1: 48, x2: 488, y1: y(threshold.value), y2: y(threshold.value)}),
          s('text', {class: 'tick', x: 492, y: y(threshold.value) + 4}, 'threshold'),
          s('path', {d: pts.map((d, i) => `${i ? 'L' : 'M'}${x(d).toFixed(1)} ${y(snrAt(d)).toFixed(1)}`).join(' '), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2}),
          s('circle', {cx: x(Math.max(0.1, d0.value)), cy: y(snr0.value), r: 4, fill: 'var(--text)'}),
          ...[dMin, Math.sqrt(dMin * dMax), dMax].map((d) => s('text', {class: 'tick', x: x(d), y: 214, 'text-anchor': 'middle'}, d < 1000 ? `${Math.round(d)} m` : `${(d / 1000).toFixed(1)} km`)),
          ...[top, threshold.value, bottom].map((v) => s('text', {class: 'tick', x: 40, y: y(v) + 4, 'text-anchor': 'end'}, num(v, 0))),
        );
        put(results, answer, chart, range ? h('button', {type: 'button', class: 'btn btn--primary', onClick: () => saveRun('range', 'Detection range estimate', [],
          {snr_db: snr0.value, distance_m: d0.value, threshold_db: threshold.value, absorption_db_per_km: alpha.value, range_m: Math.round(range), summary: `≈ ${Math.round(range)} m`}, '')}, 'Save estimate') : null);
      };
      for (const f of [snr0, d0, threshold, alpha]) f.input.addEventListener('input', show);
      put(body, header(t), h('div', {class: 'form-row'}, snr0.el, d0.el, threshold.el, alpha.el), results,
        h('p', {class: 'note'}, 'SNR(d) = SNR₀ − 20·log10(d/d₀) − α·(d − d₀). Assumes a steady source and ambient noise that does not change with distance. The real answer is a field log with a drone at known distances.'));
      show();
    }
  }

  renderTemplates();
  renderBody();
  renderRuns();
  return {el, render() { if (!state.busy) { renderTemplates(); renderBody(); } }};
}
