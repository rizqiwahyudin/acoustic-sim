/**
 * view.js — the Hardware screen (style A). Shows the connection panel until a
 * link is open, then the operator console: toolbar, map / 3D / split views,
 * the last 30 seconds, and the status column.
 */

import { h, options, segmented, setText } from '../shared/dom.js';
import { deg, formatAge, num } from '../shared/format.js';
import { SectorMap } from './map.js';
import { HistoryCharts } from './history.js';
import { StatusColumn } from './status.js';
import { AcousticScene, AuditionPanel, ConnectionPanel } from './connection.js';
import { fovText } from './geometry.js';
import { STALE_MS } from './session.js';
import { dsp } from '../study/dsp.js';

const PREFS_KEY = 'heimdall.hardware.view';
const FLOORS = [['-60', '−60 dBFS'], ['-50', '−50 dBFS'], ['-40', '−40 dBFS'], ['-30', '−30 dBFS'], ['-20', '−20 dBFS']];
const RATES = [['10', '10 Hz'], ['20', '20 Hz'], ['50', '50 Hz']];

function loadPrefs() {
  try { return JSON.parse(localStorage.getItem(PREFS_KEY)) || {}; } catch (_) { return {}; }
}

function savePrefs(prefs) {
  try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch (_) { /* storage unavailable */ }
}

export function createView(container, {session}) {
  const prefs = {view: 'map', floor: -40, showValues: false, ...loadPrefs()};
  const state = {hover: null, sectorTouched: false, visible: false};

  const acoustic = new AcousticScene();
  const connection = new ConnectionPanel(session, acoustic);
  const audition = new AuditionPanel(session, acoustic);
  const status = new StatusColumn(session, {extra: audition.el});
  const history = new HistoryCharts();
  const map = new SectorMap({onPick: (sector) => steerTo(sector, ' from the map'), onHover: setHover});
  let scene3d = null;

  // ── Toolbar ────────────────────────────────────────────────────────────
  const modeSeg = segmented([['F', 'Sweep once'], ['C', 'Continuous'], ['G', 'Track']], {
    label: 'Scan mode', size: 'lg',
    onPick: (key) => ({F: () => session.sweepOnce(), C: () => session.continuous(), G: () => session.track()})[key](),
  });
  const stopButton = h('button', {type: 'button', class: 'btn btn--lg', onClick: () => session.stop()},
    h('span', {class: 'btn__stop', 'aria-hidden': 'true'}), 'Stop');
  const sectorInput = h('input', {id: 'hw-sector', class: 'input input--mono input--lg', type: 'number', min: 0, value: 21,
    style: {width: '76px'}, onInput: () => { state.sectorTouched = true; }});
  const steerButton = h('button', {type: 'button', class: 'btn btn--lg',
    onClick: () => steerTo(parseInt(sectorInput.value, 10), '')}, 'Steer');
  const monitorButton = h('button', {type: 'button', class: 'btn btn--lg', 'aria-pressed': 'false',
    onClick: () => {
      if (session.monitoring) session.stopMonitor();
      else session.startMonitor(parseInt(sectorInput.value, 10));
      render();
    }}, 'Monitor');
  const rateSelect = h('select', {class: 'select select--sm', 'aria-label': 'Monitor rate',
    onChange: () => session.setMonitorRate(parseInt(rateSelect.value, 10))}, options(RATES, String(session.monitorRate)));
  const toolbar = h('div', {class: 'toolbar', role: 'toolbar', 'aria-label': 'Scan controls'},
    modeSeg.el, stopButton,
    h('span', {class: 'toolbar__divider', 'aria-hidden': 'true'}),
    h('label', {for: 'hw-sector', class: 'toolbar__label'}, 'Sector'), sectorInput, steerButton,
    monitorButton, rateSelect,
  );

  // ── View bar ───────────────────────────────────────────────────────────
  const viewSeg = segmented([['map', 'Map'], ['3d', '3D'], ['split', 'Split']], {
    label: 'View', onPick: (key) => { prefs.view = key; savePrefs(prefs); layout(); render(); },
  });
  const showValues = h('input', {type: 'checkbox', checked: prefs.showValues,
    onChange: () => { prefs.showValues = showValues.checked; savePrefs(prefs); render(); }});
  const floorSelect = h('select', {id: 'hw-floor', class: 'select select--sm',
    onChange: () => { prefs.floor = parseInt(floorSelect.value, 10); savePrefs(prefs); render(true); }},
  options(FLOORS, String(prefs.floor)));
  const viewbar = h('div', {class: 'hw-viewbar'},
    viewSeg.el,
    h('div', {class: 'hw-viewbar__right'},
      h('label', {class: 'check'}, showValues, 'Show all levels'),
      h('span', {class: 'inline-field'}, h('label', {for: 'hw-floor'}, 'Colour floor'), floorSelect),
      h('a', {class: 'btn', href: '#exhibit', title: 'Full-screen presentation view'}, 'Present'),
    ),
  );

  // ── Map, 3D and history ────────────────────────────────────────────────
  // One stage grid holds the map, the 3D view, a readout line and the charts.
  // The map and the 3D box start at the same height; the charts sit below
  // (or beside them on ultrawide screens).
  const mapReadout = h('span', {class: 'readout', 'aria-live': 'polite'});
  const legend = h('span', {class: 'freshness'},
    h('span', {class: 'freshness__item'}, h('span', {class: 'freshness__swatch'}), 'Measured in the last 5 s'),
    h('span', {class: 'freshness__item'}, h('span', {class: 'freshness__swatch freshness__swatch--stale'}), 'Older, kept from an earlier pass'),
  );
  const readoutRow = h('div', {class: 'hw-readout'}, mapReadout, legend);
  const mapSection = h('section', {class: 'hw-map', 'aria-label': 'Sector map'}, map.el);

  const sceneBox = h('div', {class: 'scene-box'});
  const cameraButtons = [['behind', 'Behind'], ['above', 'Above'], ['side', 'Side']].map(([key, label]) =>
    h('button', {type: 'button', class: 'btn btn--sm', onClick: () => scene3d?.setView(key)}, label));
  const sceneSection = h('section', {class: 'hw-scene', 'aria-label': '3D view'},
    h('span', {class: 'hw-scene__hint'}, 'Sectors around the array · drag to rotate, click a tile to hold the beam'),
    sceneBox,
  );
  const sceneTools = h('div', {class: 'scene-box__tools'}, ...cameraButtons);

  const waiting = h('p', {class: 'hw-waiting', hidden: true});
  const stage = h('div', {class: 'hw-stage'}, mapSection, sceneSection, readoutRow, history.el);
  const controls = h('div', {class: 'hw-controls'}, toolbar, viewbar);
  const main = h('div', {class: 'hw-main'}, controls, waiting, stage);
  const body = h('div', {class: 'hw-body'}, main, status.el);

  const footLeft = h('span');
  const footRight = h('span');
  const footer = h('footer', {class: 'app-footer'}, footLeft, footRight);

  const bannerText = h('span', {class: 'note note--body'});
  const banner = h('div', {class: 'banner', role: 'status', hidden: true},
    h('div', {class: 'banner__text'}, h('span', {class: 'banner__title'}, 'DSP parameters differ from the flashed program'), bannerText),
    h('div', {class: 'toolbar toolbar--tight'},
      h('a', {class: 'btn', href: '#study/parameters'}, 'Review in Study'),
      h('button', {type: 'button', class: 'btn btn--ghost', onClick: () => { dsp.stageFlashed(); window.location.hash = 'study/parameters'; }}, 'Stage a return to flashed')),
  );

  const connectWrap = h('div', {class: 'hw-connect'}, connection.el);
  container.append(banner, connectWrap, body, footer);

  function renderBanner() {
    const modified = session.open ? dsp.status?.modified_words || 0 : 0;
    banner.hidden = modified === 0;
    const last = dsp.history.find((entry) => !entry.reverted);
    setText(bannerText, `${modified} ${modified === 1 ? 'word differs' : 'words differ'}${last ? `. Last change: ${last.label}` : ''}. Levels measured now reflect the changed parameters.`);
  }
  dsp.addEventListener('change', renderBanner);
  let dspPoll = null;

  // ── Behaviour ──────────────────────────────────────────────────────────
  function steerTo(sector, source) {
    if (!Number.isInteger(sector)) return;
    sectorInput.value = sector;
    state.sectorTouched = true;
    session.steer(sector, source);
  }

  function setHover(sector) {
    state.hover = sector;
    renderReadout();
  }

  function ensureScene() {
    if (scene3d) return;
    return import('./scene3d.js').then(({SectorScene}) => {
      scene3d = new SectorScene(sceneBox, {onPick: (sector) => steerTo(sector, ' from the 3D view'), onHover: setHover});
      sceneBox.append(sceneTools);
      if (state.visible) scene3d.start();
      render(true);
    });
  }

  function layout() {
    const view = prefs.view;
    viewSeg.set(view);
    body.dataset.view = view;
    mapSection.hidden = view === '3d';
    sceneSection.hidden = view === 'map';
    if (view !== 'map') ensureScene();
  }

  function renderReadout() {
    const frame = session.frame;
    const cfg = frame?.configuration;
    let text = 'Hover a sector to read its level. Click one to hold the beam there.';
    if (cfg && state.hover !== null && state.hover !== undefined) {
      const row = Math.floor(state.hover / cfg.columns);
      const column = state.hover % cfg.columns;
      const level = frame.levels_db?.[row]?.[column];
      const age = session.cellAge(row, column);
      const freshness = !Number.isFinite(level) ? 'not measured yet'
        : Number.isFinite(age) && age <= STALE_MS ? 'measured in the last 5 s' : 'older than 5 s';
      text = `Sector ${state.hover} · azimuth ${deg(frame.azimuth_deg?.[column])}, elevation ${deg(frame.elevation_deg?.[row])}`
        + (Number.isFinite(level) ? ` · ${num(level, 1)} dBFS` : '') + ` · ${freshness}`;
    }
    setText(mapReadout, text);
  }

  function renderFooter() {
    const frame = session.frame;
    const cfg = frame?.configuration || session.configuration;
    if (!session.open) {
      setText(footLeft, 'Not connected');
      setText(footRight, '');
      return;
    }
    const age = session.newestSampleAge();
    const parts = [
      `Pass ${frame?.scan_count ?? 0}`,
      `last sample ${Number.isFinite(age) ? formatAge(age) + ' ago' : 'none yet'}`,
      `${frame?.protocol_errors ?? 0} protocol warnings`,
    ];
    if (session.frozen) parts.push('display frozen');
    setText(footLeft, parts.join(' · '));
    const source = {
      firmware_reported: 'firmware table',
      fast_emulator: 'emulator grid',
      deployment_contract: 'acoustic room with the firmware delays',
      exploratory_simulation: 'acoustic room with simulated delays',
    }[frame?.grid_source] || 'grid';
    const fov = fovText(frame);
    setText(footRight, cfg ? `${fov ? `Field of view ${fov} · ` : ''}${source} ${cfg.rows} × ${cfg.columns}` : '');
  }

  /**
   * Largest square cell that fits the height of the stage row and the share
   * of the width the map may take, so the whole map is always on screen.
   */
  function cellSize(cfg) {
    const split = prefs.view === 'split';
    const height = mapSection.clientHeight || 520;
    const width = (stage.clientWidth || 1100) * (split ? 0.46 : 0.62);
    const byHeight = (height - 66) / cfg.rows - 4;
    const byWidth = (width - 132) / cfg.columns - 4;
    return Math.max(12, Math.min(120, Math.floor(Math.min(byHeight, byWidth))));
  }

  new ResizeObserver(() => render(true)).observe(stage);

  let queued = false;
  let forceNext = false;
  function render(force = false) {
    forceNext = forceNext || force === true;
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      const forced = forceNext;
      forceNext = false;
      draw(forced);
    });
  }

  function draw(force) {
    if (!state.visible) return;
    const open = session.open;
    connectWrap.hidden = open;
    body.hidden = !open;
    const frame = session.frame;
    const cfg = session.configuration;
    const ready = session.ready;

    modeSeg.setDisabled(!ready);
    const active = session.monitoring ? null : session.firmwareMode;
    modeSeg.set(['F', 'C', 'G'].includes(active) ? active : null);
    stopButton.disabled = !open;
    steerButton.disabled = !ready;
    monitorButton.disabled = !ready;
    sectorInput.disabled = !ready;
    monitorButton.setAttribute('aria-pressed', String(session.monitoring));
    monitorButton.textContent = session.monitoring ? 'Stop monitoring' : 'Monitor';
    if (cfg) {
      sectorInput.max = cfg.sectors - 1;
      if (!state.sectorTouched) sectorInput.value = Math.floor(cfg.rows / 2) * cfg.columns + Math.floor(cfg.columns / 2);
    }

    const hasLevels = Boolean(frame?.configuration && frame.levels_db);
    waiting.hidden = hasLevels;
    stage.hidden = !hasLevels;
    if (!hasLevels) {
      setText(waiting, open
        ? `Connected to ${session.transportLabel}. Waiting for the firmware to report its beam table…`
        : '');
    } else {
      if (!mapSection.hidden) map.render(frame, {floor: prefs.floor, showValues: prefs.showValues, session, cellPx: cellSize(cfg)});
      if (!sceneSection.hidden && scene3d) scene3d.render(frame, {floor: prefs.floor, session});
      history.render(session, {floor: prefs.floor, force});
    }
    status.render();
    audition.update();
    renderReadout();
    renderFooter();
  }

  session.addEventListener('frame', () => render());
  session.addEventListener('init', () => render(true));
  session.addEventListener('link', () => render());

  let ticker = null;
  layout();
  return {
    show() {
      state.visible = true;
      scene3d?.start();
      ticker = setInterval(() => render(), 500);
      const refreshDsp = () => { if (session.open) dsp.load().then(() => dsp.refresh()).catch(() => {}); };
      refreshDsp();
      dspPoll = setInterval(refreshDsp, 5000);
      render(true);
    },
    hide() {
      state.visible = false;
      scene3d?.stop();
      clearInterval(ticker);
      clearInterval(dspPoll);
    },
  };
}
