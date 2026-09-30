/**
 * view.js — Exhibition (style D): a full-screen presentation of the same
 * Heimdall session for a booth or a second screen. Big answer, simple
 * Track / Scan / Stop, a mini map, and a way back to the operator console.
 * Reached with the Present button or directly at #exhibit; Esc returns.
 */

import { h, setText } from '../shared/dom.js';
import { levelToUnit, rampColor, rgbCss } from '../shared/colormap.js';
import { num } from '../shared/format.js';
import { operatingMode } from '../hardware/status.js';

const FLOOR = -40;

function words(value, positive, negative, zero) {
  if (!Number.isFinite(value)) return '—';
  const rounded = Math.round(Math.abs(value));
  if (rounded === 0) return `0° ${zero}`;
  return `${rounded}° ${value > 0 ? positive : negative}`;
}

export function createView(container, {session, navigate}) {
  container.classList.add('theme-exhibit');
  const stage = h('div', {class: 'ex-stage'});
  const liveDot = h('span', {class: 'ex-live__dot', 'aria-hidden': 'true'});
  const liveText = h('span', {}, 'Not connected');
  const subtitle = h('span', {class: 'ex-subtitle'}, 'Acoustic drone tracking');
  const brand = h('div', {class: 'ex-brand'},
    h('div', {class: 'ex-brand__name'}, h('span', {class: 'ex-title'}, 'Heimdall'), subtitle),
    h('span', {class: 'ex-live', role: 'status'}, liveDot, liveText),
  );

  const label = h('span', {class: 'ex-answer__label'}, 'Paused');
  const azBig = h('span', {class: 'ex-answer__big'}, '—');
  const elBig = h('span', {class: 'ex-answer__big'}, '');
  const detail = h('span', {class: 'ex-answer__detail'}, 'Press Track to find and follow a sound source.');
  const answer = h('div', {class: 'ex-answer', 'aria-live': 'polite'}, label, azBig, elBig, detail);

  const pill = (text, onClick) => h('button', {type: 'button', class: 'ex-pill', 'aria-pressed': 'false', onClick}, text);
  const trackButton = pill('Track', () => session.track());
  const scanButton = pill('Scan', () => session.continuous());
  const stopButton = pill('Stop', () => session.stop());
  const controls = h('div', {class: 'ex-controls', role: 'toolbar', 'aria-label': 'Mode'}, trackButton, scanButton, stopButton);

  const miniTitle = h('span', {class: 'ex-mini__title'}, 'All sectors');
  const mini = h('div', {class: 'ex-mini__grid', 'aria-hidden': 'true'});
  const fullscreen = h('button', {type: 'button', class: 'ex-link', onClick: () => toggleFullscreen()}, 'Full screen');
  const back = h('a', {class: 'ex-link', href: '#hardware'}, 'Operator console →');
  const corner = h('div', {class: 'ex-corner'},
    h('div', {class: 'ex-mini'}, miniTitle, mini),
    h('div', {class: 'ex-links'}, fullscreen, back),
  );
  const offline = h('div', {class: 'ex-offline', hidden: true},
    h('p', {}, 'Heimdall is not connected.'),
    h('a', {class: 'ex-link', href: '#hardware'}, 'Connect in the operator console →'),
  );

  container.append(stage, brand, answer, controls, corner, offline);

  let scene = null;
  let visible = false;
  let miniKey = '';
  let miniCells = [];

  import('../hardware/scene3d.js').then(({SectorScene}) => {
    scene = new SectorScene(stage, {theme: 'exhibit'});
    scene.setOrbit(20, 10, 2.5);
    if (visible) scene.start();
    render();
  });

  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen?.();
    else document.documentElement.requestFullscreen?.().catch(() => {});
  }

  function buildMini(cfg) {
    const key = `${cfg.rows}x${cfg.columns}`;
    if (key === miniKey) return;
    miniKey = key;
    const size = Math.max(8, Math.min(22, Math.floor(150 / cfg.columns)));
    mini.style.gridTemplateColumns = `repeat(${cfg.columns}, ${size}px)`;
    miniCells = [];
    const nodes = [];
    for (let displayRow = 0; displayRow < cfg.rows; displayRow++) {
      const row = cfg.rows - 1 - displayRow;
      for (let column = 0; column < cfg.columns; column++) {
        const cell = h('span', {class: 'ex-mini__cell', style: {width: `${size}px`, height: `${size}px`}});
        miniCells[row * cfg.columns + column] = cell;
        nodes.push(cell);
      }
    }
    mini.replaceChildren(...nodes);
  }

  let queued = false;
  function render() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; draw(); });
  }

  function draw() {
    if (!visible) return;
    const frame = session.frame;
    const cfg = frame?.configuration || session.configuration;
    const online = session.open;
    offline.hidden = online;
    controls.hidden = !online;
    corner.querySelector('.ex-mini').hidden = !cfg;

    liveDot.dataset.state = session.link;
    setText(liveText, online ? `Live · ${session.transportLabel}` : 'Not connected');
    setText(subtitle, cfg ? `Acoustic drone tracking, ${cfg.microphones} microphones` : 'Acoustic drone tracking');

    const mode = operatingMode(session);
    trackButton.setAttribute('aria-pressed', String(mode === 'track'));
    scanButton.setAttribute('aria-pressed', String(mode === 'continuous'));
    stopButton.setAttribute('aria-pressed', String(mode === 'idle'));
    for (const button of [trackButton, scanButton, stopButton]) button.disabled = !session.ready;

    const solution = session.solution(frame);
    const showAnswer = solution && (solution.kind === 'target' || mode === 'continuous' || mode === 'once' || mode === 'fixed');
    if (showAnswer) {
      setText(label, solution.kind === 'target' ? 'Tracking a source' : solution.kind === 'beam' ? 'Listening in one direction' : 'Loudest direction');
      label.classList.add('is-live');
      setText(azBig, words(solution.azimuth, 'right', 'left', 'ahead'));
      setText(elBig, words(solution.elevation, 'up', 'down', 'level'));
      const margin = Number.isFinite(frame.margin_db) ? `, ${num(frame.margin_db, 1)} dB above the next direction` : '';
      setText(detail, `Level ${num(solution.level, 1)} dBFS${margin}.`);
    } else {
      label.classList.remove('is-live');
      setText(label, mode === 'track' ? 'Searching' : 'Paused');
      setText(azBig, '—');
      setText(elBig, '');
      setText(detail, mode === 'track' ? 'Listening across the whole field for a source.' : 'Press Track to find and follow a sound source.');
    }

    if (cfg && frame?.levels_db) {
      buildMini(cfg);
      setText(miniTitle, `All ${cfg.sectors} sectors`);
      const marker = solution?.sector;
      for (let sector = 0; sector < miniCells.length; sector++) {
        const cell = miniCells[sector];
        if (!cell) continue;
        const row = Math.floor(sector / cfg.columns);
        const column = sector % cfg.columns;
        const level = frame.levels_db[row]?.[column];
        const age = session.cellAge(row, column);
        const fresh = Number.isFinite(level) && Number.isFinite(age) && age <= 5000;
        cell.style.background = Number.isFinite(level) ? rgbCss(rampColor(levelToUnit(level, FLOOR), 'ember')) : 'var(--raised)';
        cell.style.opacity = fresh ? '1' : '0.4';
        cell.style.boxShadow = sector === marker && showAnswer ? 'inset 0 0 0 2px #ffffff' : 'none';
      }
      scene?.render(frame, {floor: FLOOR, session});
    }
  }

  function onKey(event) {
    if (event.key === 'Escape' && visible && !document.fullscreenElement) navigate('hardware');
  }

  session.addEventListener('frame', render);
  session.addEventListener('link', render);
  session.addEventListener('init', render);
  document.addEventListener('keydown', onKey);

  return {
    show() {
      visible = true;
      scene?.start();
      render();
    },
    hide() {
      visible = false;
      scene?.stop();
      if (document.fullscreenElement) document.exitFullscreen?.();
    },
  };
}
