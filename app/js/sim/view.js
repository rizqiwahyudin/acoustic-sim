/**
 * view.js — Simulator (style A): scenario settings with a sticky Run button,
 * the room in 3D, and the result column. Presets save, load and share as
 * before (#preset=… links still open here).
 */

import { apiJson } from '../shared/api.js';
import { download, h, setText } from '../shared/dom.js';
import { fileStamp } from '../shared/format.js';
import { customArray } from '../shared/custom-array.js';
import { buildControls } from './controls.js';
import { ResultsColumn } from './results.js';

const MARGIN = 0.3;

function previewScene(p) {
  const room = [p.room_length, p.room_width, p.room_height];
  const center = [room[0] / 2, room[1] / 2, 1.0];
  const a = p.source_az_deg * Math.PI / 180;
  const e = p.source_el_deg * Math.PI / 180;
  const clip = (v, i) => Math.min(Math.max(v, MARGIN), room[i] - MARGIN);
  const source = [
    center[0] + p.source_distance * Math.cos(e) * Math.cos(a),
    center[1] + p.source_distance * Math.cos(e) * Math.sin(a),
    center[2] + p.source_distance * Math.sin(e),
  ].map(clip);
  return {room, center, source};
}

export function createView(container, {preset} = {}) {
  let scene = null;
  let visible = false;
  let lastRun = null;
  let running = false;

  const results = new ResultsColumn();
  const caption = h('span', {class: 'note note--body'});
  const staleNote = h('p', {class: 'note is-warn', hidden: true}, 'Settings changed since the last run. Run again to update the result.');
  const controls = buildControls({onChange: () => settingsChanged()});

  const runButton = h('button', {type: 'button', class: 'btn btn--primary btn--lg sim-run__button', onClick: () => run()}, 'Run simulation');
  const presetFile = h('input', {type: 'file', accept: 'application/json', hidden: true});
  const presetStatus = h('span', {class: 'note', role: 'status'});
  const presets = h('div', {class: 'sim-presets'},
    h('button', {type: 'button', class: 'btn btn--sm btn--quiet', onClick: () => savePreset()}, 'Save preset'),
    h('button', {type: 'button', class: 'btn btn--sm btn--quiet', onClick: () => presetFile.click()}, 'Load preset'),
    h('button', {type: 'button', class: 'btn btn--sm', onClick: () => copyLink()}, 'Copy link'),
    presetStatus, presetFile,
  );

  const settings = h('aside', {class: 'sim-settings', 'aria-label': 'Scenario settings'},
    presets,
    controls.el,
    h('div', {class: 'sim-run'}, runButton),
  );

  const sceneBox = h('div', {class: 'scene-box scene-box--room'});
  const cameraButtons = [['angled', 'Angled'], ['plan', 'Plan']].map(([key, label]) =>
    h('button', {type: 'button', class: 'btn btn--sm btn--ghost', onClick: () => scene?.setView(key)}, label));
  const legend = h('ul', {class: 'scene-legend', 'aria-label': 'Legend'},
    h('li', {}, h('span', {class: 'lg lg--array'}), 'Array'),
    h('li', {}, h('span', {class: 'lg lg--drone'}), 'Drone'),
    h('li', {}, h('span', {class: 'lg lg--truth'}), 'True direction'),
    h('li', {}, h('span', {class: 'lg lg--estimate'}), 'Estimated direction'),
    h('li', {}, h('span', {class: 'lg lg--crowd'}), 'Crowd talkers'),
    h('li', {}, h('span', {class: 'lg lg--pa'}), 'PA speakers'),
  );
  const centre = h('section', {class: 'sim-centre', 'aria-labelledby': 'sim-scene-h'},
    h('div', {class: 'section__row'},
      h('div', {class: 'bp-centre__title'}, h('h2', {class: 'bp-title', id: 'sim-scene-h'}, 'Scene'), caption),
      h('div', {class: 'toolbar toolbar--tight'}, ...cameraButtons),
    ),
    sceneBox, staleNote, legend,
  );

  container.append(h('div', {class: 'sim'}, settings, centre, results.el));

  function settingsChanged() {
    const p = controls.params();
    setText(caption, `${p.room_length} × ${p.room_width} × ${p.room_height} m room. Drag to rotate.`);
    const custom = controls.get('simGeo').value === 'CUSTOM';
    runButton.disabled = running || (custom && customArray.positions.length < 2);
    if (lastRun) {
      const same = JSON.stringify(stripAudio(p)) === JSON.stringify(stripAudio(lastRun.params));
      staleNote.hidden = same;
      results.el.classList.toggle('is-stale', !same);
      if (!same) scene?.render(previewScene(p));
      else scene?.render(runScene(lastRun.data, lastRun.params));
    } else {
      scene?.render(previewScene(p));
    }
  }

  function stripAudio(p) {
    const {seed, ...rest} = p;
    return rest;
  }

  function runScene(data, p) {
    return {
      room: data.room_dim, center: data.array_center, source: data.source_pos,
      trajectory: data.trajectory || [], crowd: data.crowd_positions || [], pa: data.pa_positions || [],
      images: data.image_sources || [], estimate: {az: data.est_az_deg, el: data.est_el_deg},
      distance: p.source_distance,
    };
  }

  async function run() {
    const p = controls.params();
    if (p.geometry === 'CUSTOM' && (p.custom_mic_positions || []).length < 2) {
      results.showError('Custom coordinates need at least two microphones');
      return;
    }
    running = true;
    runButton.disabled = true;
    setText(runButton, 'Running…');
    results.setBusy(true);
    try {
      const data = await apiJson('/simulate', {method: 'POST', body: p});
      lastRun = {data, params: p};
      results.render(data, p);
      staleNote.hidden = true;
      results.el.classList.remove('is-stale');
      scene?.render(runScene(data, p));
    } catch (error) {
      results.showError(error.message);
    } finally {
      running = false;
      results.setBusy(false);
      setText(runButton, 'Run simulation');
      settingsChanged();
    }
  }

  function savePreset() {
    const state = controls.captureState();
    download(new Blob([JSON.stringify(state, null, 2)], {type: 'application/json'}), `preset-${fileStamp()}.json`);
  }

  presetFile.addEventListener('change', () => {
    const file = presetFile.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        controls.applyState(JSON.parse(reader.result));
        setText(presetStatus, 'Preset loaded.');
      } catch (error) {
        setText(presetStatus, `Not a valid preset: ${error.message}`);
      }
    };
    reader.readAsText(file);
    presetFile.value = '';
  });

  async function copyLink() {
    const encoded = btoa(unescape(encodeURIComponent(JSON.stringify(controls.captureState()))));
    const url = `${window.location.origin}${window.location.pathname}#preset=${encoded}`;
    try {
      await navigator.clipboard.writeText(url);
      setText(presetStatus, 'Link copied.');
    } catch (_) {
      window.prompt('Copy this link:', url);
    }
  }

  async function loadMaterials() {
    try {
      const data = await apiJson('/materials');
      if (Array.isArray(data.choices) && data.choices.length) controls.populateMaterials(data.choices, data.exhibition_hall || undefined);
    } catch (_) { /* backend offline: keep the built-in list */ }
  }

  function applyHashPreset(encoded) {
    if (!encoded) return;
    try {
      controls.applyState(JSON.parse(decodeURIComponent(escape(atob(encoded)))));
      setText(presetStatus, 'Preset loaded from the link.');
    } catch (error) {
      console.warn('Could not decode the preset in the link:', error);
    }
  }

  loadMaterials().then(() => applyHashPreset(preset));
  import('./scene.js').then(({RoomScene}) => {
    scene = new RoomScene(sceneBox);
    if (visible) scene.start();
    settingsChanged();
  });
  settingsChanged();

  return {
    show() { visible = true; scene?.start(); },
    hide() { visible = false; scene?.stop(); },
    applyPreset: applyHashPreset,
  };
}

