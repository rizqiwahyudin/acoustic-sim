/**
 * view.js — Exhibition (style D): a full-screen presentation of the same
 * Heimdall session for a booth or a second screen. Big answer, simple
 * Track / Scan / Stop, a mini map, and a way back to the operator console.
 * Reached with the Present button or directly at #exhibit; Esc returns.
 */

import { h, setText } from '../shared/dom.js';
import { levelToUnit, rampColor, rgbCss } from '../shared/colormap.js';
import { num } from '../shared/format.js';
import { look } from '../shared/look.js';
import { buildField, fieldKind, gateMarker } from '../hardware/field.js';
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
  const detail = h('span', {class: 'ex-answer__detail'}, 'Press Live to find a sound source.');
  const answer = h('div', {class: 'ex-answer', 'aria-live': 'polite'}, label, azBig, elBig, detail);

  const pill = (text, onClick) => h('button', {type: 'button', class: 'ex-pill', 'aria-pressed': 'false', onClick}, text);
  const liveButton = pill('Live', () => session.stare());
  const trackButton = pill('Track', () => session.track());
  const scanButton = pill('Scan', () => session.continuous());
  const stopButton = pill('Stop', () => session.stop());
  const controls = h('div', {class: 'ex-controls', role: 'toolbar', 'aria-label': 'Mode'},
    liveButton, trackButton, scanButton, stopButton);

  const miniTitle = h('span', {class: 'ex-mini__title'}, 'All sectors');
  const mini = h('div', {class: 'ex-mini__grid', 'aria-hidden': 'true'});
  const fullscreen = h('button', {type: 'button', class: 'ex-link', onClick: () => toggleFullscreen()}, 'Full screen');
  const back = h('a', {class: 'ex-link', href: '#hardware'}, 'Operator console →');
  // The Arasaka look shows the field as beams (a ray per sector) by default, as its mockup, with tiles one click away.
  let display = look === 'arasaka' ? 'beams' : 'tiles';
  const displayLink = look === 'arasaka' ? h('button', {type: 'button', class: 'ex-link', onClick: () => {
    display = display === 'beams' ? 'tiles' : 'beams';
    setText(displayLink, display === 'beams' ? 'Show tiles' : 'Show beams');
    scene?.setDisplay(display);
    render();
  }}, 'Show tiles') : null;
  const corner = h('div', {class: 'ex-corner'},
    h('div', {class: 'ex-mini'}, miniTitle, mini),
    h('div', {class: 'ex-links'}, displayLink, fullscreen, back),
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
  let build = null;          // the full 3D build plays every time Present opens
  let markerKind = null;
  let markerState = null;

  import('../hardware/scene3d.js').then(({SectorScene}) => {
    scene = new SectorScene(stage, {theme: 'exhibit', beams: look === 'arasaka'});
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
    if (!online) {
      markerKind = null;
      markerState = null;
    }
    offline.hidden = online;
    controls.hidden = !online;
    corner.querySelector('.ex-mini').hidden = !cfg;

    liveDot.dataset.state = session.link;
    setText(liveText, online ? `Live · ${session.transportLabel}` : 'Not connected');
    setText(subtitle, cfg ? `Acoustic drone tracking, ${cfg.microphones} microphones` : 'Acoustic drone tracking');

    const mode = operatingMode(session);
    liveButton.setAttribute('aria-pressed', String(mode === 'stare'));
    trackButton.setAttribute('aria-pressed', String(mode === 'track'));
    scanButton.setAttribute('aria-pressed', String(mode === 'continuous'));
    stopButton.setAttribute('aria-pressed', String(mode === 'idle'));
    for (const button of [liveButton, trackButton, scanButton, stopButton]) button.disabled = !session.ready || Boolean(session.studyRun);

    let solution = session.solution(frame);
    if (frame?.levels_db) {
      markerKind = fieldKind(frame, {previous: markerKind, monitoring: session.monitoring});
      const field = buildField(frame, markerKind, {cellAge: (row, column) => session.cellAge(row, column)});
      if (field) {
        const gated = gateMarker(field, markerState);
        markerState = gated.previous;
        if (gated.marker) {
          solution = {
            ...solution,
            kind: gated.marker.kind,
            row: gated.marker.cell?.row,
            column: gated.marker.cell?.column,
            sector: gated.marker.cell?.index,
            azimuth: gated.marker.azimuth,
            elevation: gated.marker.elevation,
            level: gated.marker.cell?.level ?? null,
          };
        } else if (gated.suppressed) {
          solution = null;
        }
      }
    }
    const showAnswer = solution && (solution.kind === 'target' || mode === 'continuous' || mode === 'once' || mode === 'fixed');
    if (showCount()) {
      // counting the microphones in (count below)
    } else if (showAnswer) {
      setText(label, solution.kind === 'target' ? 'Tracking a source' : solution.kind === 'beam' ? 'Listening in one direction' : 'Loudest direction');
      label.classList.add('is-live');
      setText(azBig, words(solution.azimuth, 'right', 'left', 'ahead'));
      setText(elBig, words(solution.elevation, 'up', 'down', 'level'));
      const margin = Number.isFinite(frame.margin_db) ? `, ${num(frame.margin_db, 1)} dB above the next direction` : '';
      setText(detail, `Level ${num(solution.level, 1)} dBFS${margin}.`);
    } else {
      label.classList.remove('is-live');
      const searching = mode === 'track' || mode === 'stare';
      setText(label, searching ? 'Searching' : 'Paused');
      setText(azBig, '—');
      setText(elBig, '');
      setText(detail, searching ? 'Listening across the whole field for a source.' : 'Press Live to find a sound source.');
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
      // Idle motion while stopped; in the Beams display the glow wave also runs while searching, as the mockup.
      const searching = (mode === 'track' || mode === 'stare') && !showAnswer;
      scene?.render(frame, {floor: FLOOR, session, idle: mode === 'idle', searching: display === 'beams' && searching, marker: solution});
      if (build && scene && online) {
        scene.play(build);
        build = null;
        requestAnimationFrame(count);
      }
    }
  }

  /** During the build the answer counts the microphones in, in DSP channel order (an animation: no test runs). */
  function showCount() {
    const channel = scene?.buildChannel?.();
    if (!channel) return false;
    label.classList.add('is-live');
    setText(label, `Microphones 01–${String(scene.rig?.parts.mics.count ?? 44).padStart(2, '0')}`);
    setText(azBig, String(channel).padStart(2, '0'));
    setText(elBig, '');
    setText(detail, '');
    return true;
  }

  function count() {
    if (!visible || !scene?.building) { render(); return; }
    showCount();
    requestAnimationFrame(count);
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
      build = 'full';
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
