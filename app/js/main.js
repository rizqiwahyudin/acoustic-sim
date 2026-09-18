/**
 * main.js — Application entry point.
 * Imports foundation modules and contains all mode logic.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { MICCANVAS_POSITIONS, el } from './state.js';

// ── Mutable application state (local to main) ──
let mode = 'beam';
let currentMics = [];
let currentPower = [];
let customMicPositions = [];
let customSelectedIdx = -1;
let customSeedGeometry = 'UCA';

import { buildGeometry } from './geometry.js';

import {
  C, GRID_AZ, GRID_COLAT, N_AZ, N_COLAT,
  colorLow, colorMid, colorHigh,
  dirVec, computeBeamPattern, maxMicSpacing, minAdjacentSpacing,
  estimateBeamwidth, buildPatternMesh
} from './beam-math.js';

import {
  renderer, scene, camera, controls, canvas,
  beamGroup, micGroup, steerGroup, trueDirGroup, roomGroup, sourceGroup,
  hwRingGroup, hwArrayMarkerGroup,
  renderMics, renderSteerDir, renderTrueDir, renderRoom, renderSources, renderRIR,
  setOnFrameCallback
} from './three-scene.js';

// Load SRP data
const SRP_DATA = await fetch('srp_spectra.json').then(r => r.json());

// ── Aliases for compatibility with original code ──
const elSelGeo = el.selGeo;
const elMicCount = el.micCount;
const elRadius = el.arrayRadius;
const elSep = el.ringSep;
const elSteerAz = el.steerAz;
const elSteerEl = el.steerEl;
const elFreq = el.freq;
const elDispRadius = el.dispRadius;
const elOpacity = el.opacity;
const elInfo = el.infoText;
const elMips = el.mipsText;
const elHeatmapWrap = el.heatmapWrap;
const elHeatmapCanvas = el.heatmapCanvas;
const elHeatmapRange = el.heatmapRange;
const elTopCand = el.topCandidates;
const elBeamPolarWrap = el.beamPolarWrap;
const elBeamPolar = el.beamPolar;
const elBeamPolarRange = el.beamPolarRange;

// ── SRP data setup ──
const srpGeoSet = [...new Set(SRP_DATA.spectra.map(s => s.geometry))];
const srpCondSet = [...new Set(SRP_DATA.spectra.map(s => s.condition))];
const srpGeoSel = document.getElementById('srpGeo');
const srpCondSel = document.getElementById('srpCond');
srpGeoSet.forEach(g => { const o = document.createElement('option'); o.value=g; o.textContent=g; srpGeoSel.appendChild(o); });
srpCondSet.forEach(c => { const o = document.createElement('option'); o.value=c; o.textContent = c==='rt60_1.0'?'RT60=1.0s (clean)':'Diffuse (crowd+PA)'; srpCondSel.appendChild(o); });

function getSRPSpectrum(geo, cond) {
  return SRP_DATA.spectra.find(s => s.geometry===geo && s.condition===cond);
}

function srpToPowerGrid(spec) {
  const azArr = spec.az_rad;
  const colatArr = spec.colat_rad;
  const raw = spec.power;
  return raw;
}

// (State variables declared above)
// MICCANVAS_POSITIONS imported from state.js

// DOM refs provided via aliases above

function fmtCoord(v) {
  return Number(v).toFixed(4).replace(/\.?0+$/, '');
}

function validateMicTriple(p) {
  return Array.isArray(p) && p.length === 3 && p.every(v => Number.isFinite(Number(v)));
}

function clearCustomClipNotes() {
  const banner = document.getElementById('customClipBanner');
  if (!banner) return;
  banner.classList.add('hidden');
  banner.innerHTML = '';
}

function setCustomClipNotes(notes) {
  const banner = document.getElementById('customClipBanner');
  if (!banner) return;
  if (!notes || notes.length === 0) {
    clearCustomClipNotes();
    return;
  }
  const lines = notes.map(n => {
    if (n.axis) {
      const micNo = (Number.isInteger(n.idx) ? n.idx + 1 : '?');
      return `Mic #${micNo} clipped on ${n.axis}: ${Number(n.before).toFixed(3)} -> ${Number(n.after).toFixed(3)} m`;
    }
    const micNo = (Number.isInteger(n.idx) ? n.idx + 1 : '?');
    const angle = (n.angle_deg !== undefined) ? ` at ${Number(n.angle_deg).toFixed(0)}°` : '';
    return `FracDelay clamp${angle}: mic #${micNo} needed ${Number(n.before_samples).toFixed(2)} samples, MAX=${Number(n.max_samples).toFixed(0)}`;
  });
  banner.innerHTML = `<button type="button" class="custom-clip-dismiss" id="btnCustomClipDismiss">x</button>${lines.join('<br>')}`;
  banner.classList.remove('hidden');
  document.getElementById('btnCustomClipDismiss').addEventListener('click', clearCustomClipNotes);
}

function renderMipsReadout(mips) {
  if (!mips) {
    elMips.textContent = '';
    return;
  }
  const pct = Number(mips.budget_used_pct || 0).toFixed(1);
  const ops = Number(mips.ops_per_sample || 0).toFixed(0);
  const scan = (mips.scan_latency_s !== null && mips.scan_latency_s !== undefined)
    ? ` | scan ${Number(mips.scan_latency_s).toFixed(2)}s`
    : '';
  elMips.textContent = `MIPS ${mips.method}: ${ops} ops/sample (${pct}% of 6144)${scan}`;
}

function renderBeamPolar(scan) {
  if (!scan || !Array.isArray(scan.angles_deg) || !scan.angles_deg.length) {
    elBeamPolarWrap.classList.add('hidden');
    return;
  }
  elBeamPolarWrap.classList.remove('hidden');
  const angles = scan.angles_deg;
  const powers = scan.powers_db || [];
  const argmax = Number(scan.argmax_idx || 0);
  const ctx = elBeamPolar.getContext('2d');
  const W = elBeamPolar.width, H = elBeamPolar.height;
  const cx = W / 2, cy = H / 2;
  const rMax = Math.min(W, H) * 0.42;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(10,10,20,0.95)';
  ctx.fillRect(0, 0, W, H);

  const pMin = Math.min(...powers);
  const pMax = Math.max(...powers);
  const span = Math.max(pMax - pMin, 1e-9);
  elBeamPolarRange.textContent = `${pMax.toFixed(1)}..${pMin.toFixed(1)} dB`;

  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  for (let i = 1; i <= 4; i++) {
    ctx.beginPath();
    ctx.arc(cx, cy, (rMax * i) / 4, 0, Math.PI * 2);
    ctx.stroke();
  }

  const barWidth = (2 * Math.PI) / angles.length;
  for (let i = 0; i < angles.length; i++) {
    const a = (angles[i] * Math.PI / 180);
    const norm = (powers[i] - pMin) / span;
    const r = rMax * (0.15 + 0.85 * norm);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, a - barWidth * 0.45, a + barWidth * 0.45);
    ctx.closePath();
    ctx.fillStyle = (i === argmax) ? 'rgba(126,207,255,0.9)' : 'rgba(180,180,180,0.35)';
    ctx.fill();
  }
}

function updateBeamMethodUi() {
  const method = document.getElementById('simBeamMethod').value;
  document.getElementById('simScanAnglesRow').classList.toggle('hidden', method !== 'steered_das');
  document.getElementById('simBeamKRow').classList.toggle('hidden', method !== 'beam_bank_das');
}

function updateRunEnabled() {
  const btn = document.getElementById('btnRun');
  if (!btn) return;
  const isCustom = document.getElementById('simGeo').value === 'CUSTOM';
  btn.disabled = isCustom && customMicPositions.length < 2;
}

function updateCustomArrayVisibility() {
  const block = document.getElementById('customArrayBlock');
  if (!block) return;

  const beamCustom = mode === 'beam' && elSelGeo.value === 'CUSTOM';
  const liveCustom = mode === 'sim' && document.getElementById('simGeo').value === 'CUSTOM';
  if (beamCustom) {
    document.getElementById('beamGeoRow').insertAdjacentElement('afterend', block);
  } else if (liveCustom) {
    document.getElementById('simGeoRow').insertAdjacentElement('afterend', block);
  }
  block.classList.toggle('hidden', !(beamCustom || liveCustom));

  document.getElementById('beamMicCountRow').classList.toggle('hidden', beamCustom);
  document.getElementById('beamRadiusRow').classList.toggle('hidden', beamCustom);
  document.getElementById('sepRow').classList.toggle('hidden', beamCustom || elSelGeo.value !== 'CYLINDER');

  document.getElementById('simMicCountRow').classList.toggle('hidden', liveCustom);
  document.getElementById('simRadiusRow').classList.toggle('hidden', liveCustom);
  document.getElementById('simSepRow').classList.toggle('hidden', liveCustom || document.getElementById('simGeo').value !== 'CYLINDER');
  updateRunEnabled();
}

function renderCustomScatter() {
  const canvas = document.getElementById('customScatter');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#08080d';
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(W / 2, 10); ctx.lineTo(W / 2, H - 10);
  ctx.moveTo(10, H / 2); ctx.lineTo(W - 10, H / 2);
  ctx.stroke();
  ctx.fillStyle = '#666';
  ctx.font = '10px monospace';
  ctx.fillText('x', W - 14, H / 2 - 4);
  ctx.fillText('y', W / 2 + 5, 16);

  const maxAbs = Math.max(
    0.05,
    ...customMicPositions.flatMap(p => [Math.abs(p[0]), Math.abs(p[1])])
  );
  const scale = (Math.min(W, H) * 0.42) / maxAbs;
  customMicPositions.forEach((p, idx) => {
    const x = W / 2 + p[0] * scale;
    const y = H / 2 - p[1] * scale;
    ctx.beginPath();
    ctx.arc(x, y, idx === customSelectedIdx ? 5 : 4, 0, 2 * Math.PI);
    ctx.fillStyle = idx === customSelectedIdx ? '#ffdd44' : '#7ecfff';
    ctx.fill();
    ctx.fillStyle = '#aaa';
    ctx.fillText(String(idx + 1), x + 6, y - 6);
  });
}

function renderCustomList() {
  const list = document.getElementById('customList');
  if (!list) return;
  document.getElementById('customMicCount').textContent = String(customMicPositions.length);
  list.innerHTML = '';
  if (customMicPositions.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'custom-list-empty';
    empty.textContent = 'No custom mics yet. Add [x, y, z] offsets in metres.';
    list.appendChild(empty);
  } else {
    customMicPositions.forEach((p, idx) => {
      const row = document.createElement('div');
      row.className = 'custom-row' + (idx === customSelectedIdx ? ' selected' : '');
      row.innerHTML = `<span>#${idx + 1}</span><span>[${fmtCoord(p[0])}, ${fmtCoord(p[1])}, ${fmtCoord(p[2])}] m</span>`;
      const rm = document.createElement('button');
      rm.type = 'button';
      rm.className = 'custom-remove';
      rm.textContent = 'x';
      rm.addEventListener('click', (ev) => {
        ev.stopPropagation();
        customRemove(idx);
      });
      row.appendChild(rm);
      row.addEventListener('click', () => setCustomSelected(idx));
      list.appendChild(row);
    });
  }
  renderCustomScatter();
  updateRunEnabled();
}

function refreshAfterCustomMutation() {
  clearCustomClipNotes();
  renderCustomList();
  if (mode === 'beam' && elSelGeo.value === 'CUSTOM') refresh();
}

function customAdd(x, y, z) {
  const p = [Number(x), Number(y), Number(z)];
  if (!validateMicTriple(p)) {
    alert('Enter finite numeric x, y, z mic offsets.');
    return;
  }
  for (const q of customMicPositions) {
    const dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
    if (Math.sqrt(dx*dx + dy*dy + dz*dz) < 1e-6) {
      alert('That mic position already exists.');
      return;
    }
  }
  customMicPositions.push(p);
  customSelectedIdx = customMicPositions.length - 1;
  document.getElementById('customAddX').value = '';
  document.getElementById('customAddY').value = '';
  document.getElementById('customAddZ').value = '';
  refreshAfterCustomMutation();
}

function customRemove(idx) {
  customMicPositions.splice(idx, 1);
  if (customSelectedIdx === idx) customSelectedIdx = -1;
  else if (customSelectedIdx > idx) customSelectedIdx -= 1;
  refreshAfterCustomMutation();
}

function customClear() {
  customMicPositions = [];
  customSelectedIdx = -1;
  refreshAfterCustomMutation();
}

function customSeedFromPreset() {
  let nMics, radius, sep;
  if (mode === 'beam') {
    nMics = parseInt(elMicCount.value);
    radius = parseFloat(elRadius.value);
    sep = parseFloat(elSep.value);
  } else {
    nMics = parseInt(document.getElementById('simMicCount').value);
    radius = parseFloat(document.getElementById('simRadius').value);
    sep = parseFloat(document.getElementById('simSep').value);
  }
  customMicPositions = buildGeometry(customSeedGeometry, nMics, radius, sep)
    .map(p => [Number(p[0]), Number(p[1]), Number(p[2])]);
  customSelectedIdx = customMicPositions.length ? 0 : -1;
  refreshAfterCustomMutation();
}

function customParseFile(text) {
  const trimmed = text.trim();
  if (!trimmed) return [];
  const normalize = arr => {
    if (validateMicTriple(arr)) return [arr.map(Number)];
    if (Array.isArray(arr) && arr.every(validateMicTriple)) return arr.map(p => p.map(Number));
    throw new Error('Expected [x,y,z] or [[x,y,z], ...].');
  };
  try {
    return normalize(JSON.parse(trimmed));
  } catch (_) {
    try {
      return normalize(JSON.parse(`[${trimmed}]`));
    } catch (_) {
      const rows = trimmed.split(/\r?\n/)
        .map(line => line.trim())
        .filter(line => line && !line.startsWith('#'))
        .map(line => line.replace(/[\[\],]/g, ' ').trim().split(/\s+/).map(Number));
      return normalize(rows);
    }
  }
}

function setCustomSelected(idx) {
  customSelectedIdx = idx;
  renderCustomList();
  if ((mode === 'beam' && elSelGeo.value === 'CUSTOM') ||
      (mode === 'sim' && document.getElementById('simGeo').value === 'CUSTOM')) {
    renderMics(currentMics, customSelectedIdx);
  }
}

function getParams() {
  return {
    geo: elSelGeo.value,
    nMics: parseInt(elMicCount.value),
    radius: parseFloat(elRadius.value),
    sep: parseFloat(elSep.value),
    steerAz: parseFloat(elSteerAz.value),
    steerEl: parseFloat(elSteerEl.value),
    freq: parseFloat(elFreq.value),
    dispRadius: parseFloat(elDispRadius.value),
    opacity: parseFloat(elOpacity.value),
  };
}

function refreshBeam() {
  const p = getParams();
  if (p.geo !== 'CUSTOM') customSeedGeometry = p.geo;
  document.getElementById('micCountVal').textContent = p.nMics;
  document.getElementById('radiusVal').textContent = p.radius.toFixed(2);
  document.getElementById('sepVal').textContent = p.sep.toFixed(2);
  document.getElementById('steerAzVal').textContent = p.steerAz.toFixed(0);
  document.getElementById('steerElVal').textContent = p.steerEl.toFixed(0);
  document.getElementById('freqVal').textContent = p.freq.toFixed(0);
  document.getElementById('dispRadVal').textContent = p.dispRadius.toFixed(1);
  document.getElementById('opacityVal').textContent = p.opacity.toFixed(2);

  updateCustomArrayVisibility();

  currentMics = buildGeometry(p.geo, p.nMics, p.radius, p.sep, customMicPositions);
  if (p.geo === 'CUSTOM' && currentMics.length === 0) {
    micGroup.clear();
    beamGroup.clear();
    steerGroup.clear();
    elInfo.innerHTML = `<span class="val">CUSTOM</span> -- 0 mics<br>Add mic coordinates to preview a beam pattern.`;
    return;
  }
  const steerAzRad = p.steerAz * Math.PI / 180;
  const steerElRad = p.steerEl * Math.PI / 180;

  currentPower = computeBeamPattern(currentMics, steerAzRad, steerElRad, p.freq);
  renderMics(currentMics, p.geo === 'CUSTOM' ? customSelectedIdx : -1);
  renderSteerDir(steerAzRad, steerElRad, p.dispRadius * 1.15);
  buildPatternMesh(currentPower, p.dispRadius, p.opacity, beamGroup);

  const wavelength = C / p.freq;
  const adjSpacing = minAdjacentSpacing(currentMics);
  const aliasFreq = C / (2 * adjSpacing);
  const aperture = maxMicSpacing(currentMics);

  const steerColatIdx = Math.round((Math.PI/2 - steerElRad) / (Math.PI/18));
  const steerAzIdx = Math.round(steerAzRad / (2*Math.PI/72)) % 72;
  const bw = estimateBeamwidth(currentPower, steerAzIdx, Math.min(steerColatIdx, N_COLAT-1));

  elInfo.innerHTML =
    `<span class="val">${p.geo}</span> -- ${currentMics.length} mics<br>` +
    (p.geo==='CUSTOM'
      ? `Custom coordinates: <span class="val">centre-relative</span><br>`
      : `Radius: <span class="val">${p.radius.toFixed(2)} m</span>` +
        (p.geo==='CYLINDER' ? ` / sep: <span class="val">${p.sep.toFixed(2)} m</span>` : '') + `<br>`) +
    `Aperture: <span class="val">${aperture.toFixed(3)} m</span><br>` +
    `Adj. spacing: <span class="val">${(adjSpacing*100).toFixed(1)} cm</span><br>` +
    `Freq: <span class="val">${p.freq} Hz</span> (λ = ${wavelength.toFixed(2)} m)<br>` +
    `Steer: az=<span class="val">${p.steerAz}°</span> el=<span class="val">${p.steerEl}°</span><br>` +
    `<hr class="sep">` +
    `Alias freq: <span class="val">${Number.isFinite(aliasFreq) ? aliasFreq.toFixed(0) : 'n/a'} Hz</span>` +
    (p.freq > aliasFreq ? ` <span style="color:#ff4444">⚠ ALIASED</span>` : ` <span style="color:#44cc44">OK</span>`) + `<br>` +
    `≈ Beam solid angle: <span class="val">${bw}°²</span>`;
}

function refreshSRP() {
  const geo = srpGeoSel.value;
  const cond = srpCondSel.value;
  const spec = getSRPSpectrum(geo, cond);
  if (!spec) return;

  const p = getParams();
  document.getElementById('dispRadVal').textContent = p.dispRadius.toFixed(1);
  document.getElementById('opacityVal').textContent = p.opacity.toFixed(2);

  const mics = SRP_DATA.mic_positions[geo];
  micGroup.clear();
  if (mics) {
    const g = new THREE.SphereGeometry(0.02, 8, 8);
    const mt = new THREE.MeshPhongMaterial({color: 0x7ecfff, emissive: 0x112233});
    for (const pt of mics) {
      const m = new THREE.Mesh(g, mt);
      m.position.set(pt[0]-SRP_DATA.array_center[0], pt[2]-SRP_DATA.array_center[2], pt[1]-SRP_DATA.array_center[1]);
      micGroup.add(m);
    }
  }

  const power = srpToPowerGrid(spec);
  buildPatternMesh(power, p.dispRadius, p.opacity, beamGroup);

  const estAzRad = spec.est_az_deg * Math.PI / 180;
  const estElRad = spec.est_el_deg * Math.PI / 180;
  renderSteerDir(estAzRad, estElRad, p.dispRadius * 1.15);

  const trueAz = SRP_DATA.true_az_deg;
  const trueEl = SRP_DATA.true_el_deg;
  const azErr = Math.abs(((spec.est_az_deg - trueAz) + 180) % 360 - 180);
  const elErr = Math.abs(spec.est_el_deg - trueEl);

  elInfo.innerHTML =
    `<span class="val">${geo}</span> -- ${cond==='rt60_1.0'?'RT60=1.0s':'Diffuse'}<br>` +
    `True: az=<span class="val">${trueAz}°</span> el=<span class="val">${trueEl}°</span><br>` +
    `Est: az=<span class="val">${spec.est_az_deg.toFixed(1)}°</span> el=<span class="val">${spec.est_el_deg.toFixed(1)}°</span><br>` +
    `Error: az=<span class="val">${azErr.toFixed(1)}°</span> el=<span class="val">${elErr.toFixed(1)}°</span>`;
}

function refresh() {
  if (mode === 'beam') refreshBeam();
  else if (mode === 'srp') refreshSRP();
  trueDirGroup.clear();
  roomGroup.clear();
  sourceGroup.clear();
  document.getElementById('rirWrap').classList.remove('visible');
  if (mode !== 'sim' && mode !== 'realtime') {
    renderBeamPolar(null);
    renderMipsReadout(null);
  }
}

// ── Mode toggle ──
function setMode(m) {
  mode = m;
  ['btnBeam','btnSRP','btnSimulator','btnRealtime','btnHardware'].forEach(id => {
    document.getElementById(id).classList.remove('active');
  });
  ['beamControls','srpControls','simControls','realtimeControls','hardwareControls'].forEach(id => {
    document.getElementById(id).classList.add('hidden');
  });
  document.getElementById('btnRun').classList.toggle('hidden', m !== 'sim');
  document.body.classList.toggle('realtime-mode', m === 'realtime');
  document.body.classList.toggle('hardware-mode', m === 'hardware');
  document.body.classList.remove('hardware-3d-active', 'hardware-split', 'hardware-3d-only');
  document.getElementById('realtimeWrap').classList.toggle('hidden', m !== 'realtime');
  document.getElementById('hardwareWrap').classList.toggle('hidden', m !== 'hardware');
  if (m === 'hardware') {
    const vmode = document.getElementById('hwViewMode').value;
    if (vmode === '3d') {
      document.body.classList.add('hardware-3d-active', 'hardware-3d-only');
    } else if (vmode === 'split') {
      document.body.classList.add('hardware-3d-active', 'hardware-split');
    }
    document.getElementById('hwBirdsEye').style.display = (vmode === '3d') ? 'none' : 'block';
    if (vmode !== '2d') frameHardwareDome();
  }

  if (m === 'realtime') stopOneShotExtras();
  else stopRealtimeSession();

  if (m !== 'hardware') stopHardwareSession();

  if (m === 'beam') {
    document.getElementById('btnBeam').classList.add('active');
    document.getElementById('beamControls').classList.remove('hidden');
    refresh();
  } else if (m === 'srp') {
    document.getElementById('btnSRP').classList.add('active');
    document.getElementById('srpControls').classList.remove('hidden');
    refresh();
  } else if (m === 'sim') {
    document.getElementById('btnSimulator').classList.add('active');
    document.getElementById('simControls').classList.remove('hidden');
    updateLiveLabels();
  } else if (m === 'realtime') {
    document.getElementById('btnRealtime').classList.add('active');
    document.getElementById('realtimeControls').classList.remove('hidden');
    startRealtimeSession();
  } else if (m === 'hardware') {
    document.getElementById('btnHardware').classList.add('active');
    document.getElementById('hardwareControls').classList.remove('hidden');
    discoverHardwareTransport();
  }
  updateCustomArrayVisibility();
  renderCustomList();
}
document.getElementById('btnBeam').addEventListener('click', () => setMode('beam'));
document.getElementById('btnSRP').addEventListener('click', () => setMode('srp'));
document.getElementById('btnSimulator').addEventListener('click', () => setMode('sim'));
document.getElementById('btnRealtime').addEventListener('click', () => setMode('realtime'));
document.getElementById('btnHardware').addEventListener('click', () => setMode('hardware'));

// ── Live sim label updaters ──
const simSliders = {
  simMicCount: 'simMicVal', simRadius: 'simRadVal', simSep: 'simSepVal',
  simRoomL: 'simRoomLVal', simRoomW: 'simRoomWVal', simRoomH: 'simRoomHVal',
  simRT60: 'simRT60Val', simSrcAz: 'simSrcAzVal', simSrcEl: 'simSrcElVal',
  simSrcDist: 'simSrcDistVal',
  simDroneSPL: 'simDroneSPLVal', simCrowdSPL: 'simCrowdSPLVal',
  simPASPL: 'simPASPLVal', simMicFloor: 'simMicFloorVal',
  simSeed: 'simSeedVal',
  simCrowd: 'simCrowdVal', simPA: 'simPAVal', simInt: 'simIntVal',
  simTemp: 'simTempVal', simHumidity: 'simHumidityVal',
  simTempGrad: 'simTempGradVal',
  simSpeed: 'simSpeedVal', simHeading: 'simHeadingVal',
  simChunks: 'simChunksVal',
  simBeamK: 'simBeamKVal',
  simFracMax: 'simFracMaxVal',
  simFracTaps: 'simFracTapsVal',
  simTargetFs: 'simTargetFsVal',
};
function updateLiveLabels() {
  for (const [sliderId, valId] of Object.entries(simSliders)) {
    const el = document.getElementById(sliderId);
    if (el) document.getElementById(valId).textContent = el.value;
  }
  const geo = document.getElementById('simGeo').value;
  if (geo !== 'CUSTOM') customSeedGeometry = geo;
  updateCustomArrayVisibility();
  const rand = document.getElementById('simRandSeed').checked;
  document.getElementById('simSeedRow').classList.toggle('hidden', rand);
  // Heading is only meaningful for straight trajectories -- hide it for arc.
  const traj = document.getElementById('simTrajType');
  const headLabel = document.getElementById('simHeadingLabel');
  const headSlider = document.getElementById('simHeading');
  if (traj && headLabel && headSlider) {
    const isArc = traj.value === 'arc';
    headLabel.style.display = isArc ? 'none' : 'block';
    headSlider.style.display = isArc ? 'none' : 'block';
  }
}
Object.keys(simSliders).forEach(id => {
  document.getElementById(id).addEventListener('input', updateLiveLabels);
});
document.getElementById('simGeo').addEventListener('change', () => {
  updateLiveLabels();
  renderCustomList();
});
document.getElementById('simRandSeed').addEventListener('change', updateLiveLabels);
document.getElementById('simBeamMethod').addEventListener('change', updateBeamMethodUi);

// ── Live simulation ──
let audioUrls = {raw: null, unsteered: null, beam: null, ml: null};
let activeAudioMode = 'beam';

function revokeAudioUrls() {
  for (const k of Object.keys(audioUrls)) {
    if (audioUrls[k]) {
      URL.revokeObjectURL(audioUrls[k]);
      audioUrls[k] = null;
    }
  }
}
function b64ToWavUrl(b64) {
  const raw = atob(b64);
  const buf = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) buf[i] = raw.charCodeAt(i);
  return URL.createObjectURL(new Blob([buf], {type: 'audio/wav'}));
}
function setAudioMode(mode) {
  activeAudioMode = mode;
  const mapping = {raw: 'btnAudioRaw', unsteered: 'btnAudioUnsteered',
                   beam: 'btnAudioBeam', ml: 'btnAudioMl'};
  for (const [m, id] of Object.entries(mapping)) {
    const btn = document.getElementById(id);
    if (btn) btn.classList.toggle('active', m === mode);
  }
  const url = audioUrls[mode];
  const player = document.getElementById('audioPlayer');
  if (url) {
    const wasPlaying = !player.paused;
    const t = player.currentTime;
    player.src = url;
    player.currentTime = Math.min(t, 0.0) || 0.0;
    if (wasPlaying) player.play().catch(() => {});
  }
}
document.getElementById('btnAudioRaw').addEventListener('click', () => setAudioMode('raw'));
document.getElementById('btnAudioUnsteered').addEventListener('click', () => setAudioMode('unsteered'));
document.getElementById('btnAudioBeam').addEventListener('click', () => setAudioMode('beam'));
document.getElementById('btnAudioMl').addEventListener('click', () => setAudioMode('ml'));

function parseScanAnglesDeg(raw) {
  const vals = String(raw || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)
    .map(Number)
    .filter(v => Number.isFinite(v))
    .map(v => ((v % 360) + 360) % 360);
  const uniq = [...new Set(vals.map(v => Number(v.toFixed(6))))];
  return uniq.length ? uniq : [0,45,90,135,180,225,270,315];
}

async function runOneShotSimulation() {
  const btn = document.getElementById('btnRun');
  const simGeo = document.getElementById('simGeo').value;
  if (simGeo === 'CUSTOM' && customMicPositions.length < 2) {
    updateRunEnabled();
    elInfo.innerHTML = '<span style="color:#ffdd44">CUSTOM requires at least 2 mic positions before running.</span>';
    return;
  }
  btn.disabled = true;
  btn.textContent = 'Running...';
  document.getElementById('spinner').classList.add('active');
  document.getElementById('audioWrap').classList.remove('visible');
  revokeAudioUrls();
  elInfo.innerHTML = '<span class="val">Running simulation...</span>';

  const useRand = document.getElementById('simRandSeed').checked;
  const beamMethod = document.getElementById('simBeamMethod').value;
  const beamK = parseInt(document.getElementById('simBeamK').value);
  const bankAngles = (beamMethod === 'beam_bank_das')
    ? Array.from({length: beamK}, (_, i) => i * (360 / beamK))
    : parseScanAnglesDeg(document.getElementById('simScanAngles').value);
  const params = {
    geometry: simGeo,
    custom_mic_positions: simGeo === 'CUSTOM' ? customMicPositions.map(p => [p[0], p[1], p[2]]) : null,
    beam_method: beamMethod,
    beam_bank_angles_deg: bankAngles,
    frac_delay_max_samples: parseInt(document.getElementById('simFracMax').value),
    frac_delay_taps: parseInt(document.getElementById('simFracTaps').value),
    target_fs_hz: parseInt(document.getElementById('simTargetFs').value),
    mic_count: parseInt(document.getElementById('simMicCount').value),
    radius: parseFloat(document.getElementById('simRadius').value),
    ring_separation: parseFloat(document.getElementById('simSep').value),
    room_length: parseFloat(document.getElementById('simRoomL').value),
    room_width: parseFloat(document.getElementById('simRoomW').value),
    room_height: parseFloat(document.getElementById('simRoomH').value),
    rt60: parseFloat(document.getElementById('simRT60').value),
    source_az_deg: parseFloat(document.getElementById('simSrcAz').value),
    source_el_deg: parseFloat(document.getElementById('simSrcEl').value),
    source_distance: parseFloat(document.getElementById('simSrcDist').value),
    drone_spl_db: parseFloat(document.getElementById('simDroneSPL').value),
    crowd_spl_db: parseFloat(document.getElementById('simCrowdSPL').value),
    pa_spl_db: parseFloat(document.getElementById('simPASPL').value),
    mic_noise_floor_db: parseFloat(document.getElementById('simMicFloor').value),
    seed: useRand ? -1 : parseInt(document.getElementById('simSeed').value),
    diffuse: document.getElementById('simDiffuse').checked,
    crowd_count: parseInt(document.getElementById('simCrowd').value),
    pa_count: parseInt(document.getElementById('simPA').value),
    mic_mismatch: document.getElementById('simMismatch').checked,
    crosstalk: document.getElementById('simCrosstalk').checked,
    crosstalk_db: parseFloat(document.getElementById('simCrosstalkDb').value),
    quantization: document.getElementById('simQuant').checked,
    bit_depth: parseInt(document.getElementById('simBitDepth').value),
    absorption_mode: document.querySelector('input[name="absMode"]:checked').value,
    floor_material: document.getElementById('matFloor').value,
    ceiling_material: document.getElementById('matCeiling').value,
    east_material: document.getElementById('matEast').value,
    west_material: document.getElementById('matWest').value,
    south_material: document.getElementById('matSouth').value,
    north_material: document.getElementById('matNorth').value,
    integration_ms: parseInt(document.getElementById('simInt').value),
    temperature_c: parseFloat(document.getElementById('simTemp').value),
    humidity_pct: parseFloat(document.getElementById('simHumidity').value),
    temp_gradient_c_per_m: parseFloat(document.getElementById('simTempGrad').value),
    moving_source: document.getElementById('simMoving').checked,
    trajectory_type: document.getElementById('simTrajType').value,
    speed_mps: parseFloat(document.getElementById('simSpeed').value),
    heading_deg: parseFloat(document.getElementById('simHeading').value),
    n_trajectory_chunks: parseInt(document.getElementById('simChunks').value),
    crowd_model: document.getElementById('simCrowdModel').value,
    n_plane_waves: parseInt(document.getElementById('simNPlaneWaves').value),
    crosstalk_model: document.getElementById('simCrosstalkModel').value,
    crosstalk_corner_hz: parseFloat(document.getElementById('simCrosstalkCorner').value),
    ml_preview: document.getElementById('simMlPreview').checked,
    ml_bit_depth: parseInt(document.getElementById('simMlBitDepth').value),
    ml_feature_bit_depth: parseInt(document.getElementById('simMlFeatBitDepth').value),
    ml_n_mels: parseInt(document.getElementById('simMlNMels').value),
    fmin_hz: parseFloat(document.getElementById('simFmin').value),
    fmax_hz: parseFloat(document.getElementById('simFmax').value),
    harmonic_comb: document.getElementById('simHarmonicComb').checked,
    drone_fundamental_hz: parseFloat(document.getElementById('simFundamental').value),
    normalize_audio: document.getElementById('simNormalizeAudio').checked,
  };

  // Phase 3+: forbid an empty DOA band (fmax must exceed fmin). If the user
  // dragged the sliders past each other we nudge fmax up so the server never
  // gets a degenerate band that would fall back silently.
  if (params.fmax_hz <= params.fmin_hz) {
    params.fmax_hz = params.fmin_hz + 50;
    const fmaxEl = document.getElementById('simFmax');
    fmaxEl.value = params.fmax_hz;
    document.getElementById('simFmaxVal').textContent = params.fmax_hz.toFixed(0);
  }

  try {
    const resp = await fetch('http://127.0.0.1:8766/simulate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params),
    });
    if (!resp.ok) {
      let msg = `Server error: ${resp.status}`;
      try {
        const errBody = await resp.json();
        if (errBody && errBody.detail) msg += ` (${errBody.detail})`;
      } catch (_) { /* keep status-only error */ }
      throw new Error(msg);
    }
    const data = await resp.json();
    const mergedClipNotes = [...(data.mic_clip_notes || [])];
    if (data.beam_scan && Array.isArray(data.beam_scan.clamp_notes)) {
      mergedClipNotes.push(...data.beam_scan.clamp_notes);
    }
    setCustomClipNotes(mergedClipNotes);

    const p = getParams();
    if (Array.isArray(data.power) && data.power.length) {
      buildPatternMesh(data.power, p.dispRadius, p.opacity, beamGroup);
    } else {
      beamGroup.clear();
    }

    currentMics = data.mic_positions || [];
    renderMics(currentMics, params.geometry === 'CUSTOM' ? customSelectedIdx : -1);

    const sceneScale = renderRoom(data.room_dim, data.array_center);
    renderSources(data.source_pos, data.crowd_positions || [], data.pa_positions || [],
                  data.image_sources || [], data.array_center, sceneScale,
                  data.trajectory || []);
    renderRIR(data.rir);

    const estAzRad = data.est_az_deg * Math.PI / 180;
    const estElRad = data.est_el_deg * Math.PI / 180;
    renderSteerDir(estAzRad, estElRad, p.dispRadius * 1.15);

    const trueAzRad = data.true_az_deg * Math.PI / 180;
    const trueElRad = data.true_el_deg * Math.PI / 180;
    renderTrueDir(trueAzRad, trueElRad, p.dispRadius * 1.15);

    const azErr = Math.abs(((data.est_az_deg - data.true_az_deg) + 180) % 360 - 180);
    const elErr = Math.abs(data.est_el_deg - data.true_el_deg);

    const seedUsed = (data.seed_used !== undefined && data.seed_used !== null) ? data.seed_used : params.seed;
    const seedStr = useRand ? `random (#${seedUsed})` : `#${seedUsed}`;
    const absLine = params.absorption_mode === 'materials'
      ? `Absorption: <span class="val">materials</span>` +
        (data.rt60_actual !== null && data.rt60_actual !== undefined
          ? ` · measured RT60: <span class="val">${data.rt60_actual}s</span>` : '')
      : `RT60: <span class="val">${params.rt60}s</span>`;
    const atmBias = data.atmospheric_bias_deg;
    const atmLine = (atmBias !== undefined && atmBias !== null && Math.abs(atmBias) >= 0.05)
      ? `Atm bias el: <span class="val">${atmBias > 0 ? '+' : ''}${atmBias.toFixed(2)}°</span><br>`
      : '';
    const atmPrefixLine = `Atmos: T=<span class="val">${params.temperature_c}&deg;C</span> RH=<span class="val">${params.humidity_pct}%</span> dT/dz=<span class="val">${params.temp_gradient_c_per_m}</span>&deg;C/m<br>`;
    const movLine = params.moving_source
      ? `Moving: <span class="val">${params.trajectory_type}</span> @ <span class="val">${params.speed_mps}</span> m/s, ${params.n_trajectory_chunks} chunks` +
        (params.trajectory_type === 'straight' ? `, heading <span class="val">${params.heading_deg}&deg;</span>` : '') + '<br>'
      : '';
    elInfo.innerHTML =
      `<span class="val">${params.geometry}</span> -- ${data.mic_positions.length} mics<br>` +
      `Beam method: <span class="val">${(data.beam_method || params.beam_method)}</span><br>` +
      `Room: ${params.room_length}x${params.room_width}x${params.room_height} m<br>` +
      `${absLine} · seed: <span class="val">${seedStr}</span><br>` +
      `Drone <span class="val">${params.drone_spl_db}</span> / Crowd <span class="val">${params.crowd_spl_db}</span> / PA <span class="val">${params.pa_spl_db}</span> / Floor <span class="val">${params.mic_noise_floor_db}</span> dB<br>` +
      (params.diffuse ? `Diffuse: ${params.crowd_count} crowd + ${params.pa_count} PA<br>` : '') +
      atmPrefixLine +
      movLine +
      `HW: ${[params.mic_mismatch&&'mismatch', params.crosstalk&&`xtalk ${params.crosstalk_db}dB (${params.crosstalk_model==='fir_capacitive'?'FIR '+params.crosstalk_corner_hz+'Hz':'flat'})`, params.quantization&&`${params.bit_depth}-bit`].filter(Boolean).join(' · ') || '<span style="color:#555">none</span>'}<br>` +
      (params.diffuse && params.crowd_model === 'plane_wave'
        ? `Crowd: <span class="val">plane-wave</span> (${params.n_plane_waves} planes)<br>` : '') +
      `DOA band: <span class="val">${params.fmin_hz}-${params.fmax_hz}</span> Hz` +
      (params.harmonic_comb
        ? ` · comb <span class="val">f0=${params.drone_fundamental_hz} Hz</span>` : '') +
      ((data.n_freq_bins !== undefined)
        ? ` (${data.n_freq_bins} bins)` : '') + `<br>` +
      `<hr class="sep">` +
      `True: az=<span class="val">${data.true_az_deg}°</span> el=<span class="val">${data.true_el_deg}°</span><br>` +
      `Est: az=<span class="val">${data.est_az_deg}°</span> el=<span class="val">${data.est_el_deg}°</span><br>` +
      `Error: az=<span class="val">${azErr.toFixed(1)}°</span> el=<span class="val">${elErr.toFixed(1)}°</span><br>` +
      atmLine +
      ((data.ml_path_snr_db !== undefined && data.ml_path_snr_db !== null)
        ? `ML path SNR: <span class="val">${data.ml_path_snr_db} dB</span> (${params.ml_bit_depth}-bit audio)<br>` : '') +
      ((data.feature_snr_db !== undefined && data.feature_snr_db !== null)
        ? `ML feature SNR: <span class="val">${data.feature_snr_db} dB</span> (${params.ml_feature_bit_depth}-bit log-mel)<br>` : '') +
      `<hr class="sep">` +
      ((data.image_sources || []).length > 0 ? `Reflections: <span class="val">${data.image_sources.length}</span> image src<br>` : '') +
      `Computed in <span class="val">${data.elapsed_s}s</span>`;

    const isSrp = (data.beam_method || 'srp_phat') === 'srp_phat';
    if (isSrp) {
      renderHeatmapAndTopN(data);
      renderBeamPolar(null);
    } else {
      renderHeatmapAndTopN(null);
      renderBeamPolar(data.beam_scan || null);
    }
    renderMipsReadout(data.mips || null);

    if (data.audio_b64) {
      audioUrls.beam = b64ToWavUrl(data.audio_b64);
      if (data.raw_audio_b64) audioUrls.raw = b64ToWavUrl(data.raw_audio_b64);
      if (data.unsteered_audio_b64) audioUrls.unsteered = b64ToWavUrl(data.unsteered_audio_b64);
      if (data.ml_audio_b64) audioUrls.ml = b64ToWavUrl(data.ml_audio_b64);
      setAudioMode('beam');
      document.getElementById('audioWrap').classList.add('visible');
      const tsTag = `${params.geometry}_${seedUsed}`;
      const dlRaw = document.getElementById('dlRaw');
      const dlUns = document.getElementById('dlUnsteered');
      const dlBf  = document.getElementById('dlBeam');
      const dlMl  = document.getElementById('dlMl');
      const btnMl = document.getElementById('btnAudioMl');
      if (dlRaw) { dlRaw.href = audioUrls.raw || '#';       dlRaw.download = `raw_${tsTag}.wav`; }
      if (dlUns) { dlUns.href = audioUrls.unsteered || '#'; dlUns.download = `unsteered_${tsTag}.wav`; }
      if (dlBf)  { dlBf.href  = audioUrls.beam || '#';      dlBf.download  = `beamformed_${tsTag}.wav`; }
      const mlPresent = !!audioUrls.ml;
      if (dlMl) {
        dlMl.classList.toggle('hidden', !mlPresent);
        dlMl.href = audioUrls.ml || '#';
        dlMl.download = `ml_preview_${tsTag}.wav`;
      }
      if (btnMl) btnMl.classList.toggle('hidden', !mlPresent);

      const mlWrap = document.getElementById('mlSpectrogramWrap');
      const mlImg  = document.getElementById('mlSpectrogramImg');
      if (data.ml_spectrogram_png_b64) {
        mlImg.src = `data:image/png;base64,${data.ml_spectrogram_png_b64}`;
        mlWrap.classList.remove('hidden');
      } else {
        mlWrap.classList.add('hidden');
        mlImg.removeAttribute('src');
      }
    }
  } catch (err) {
    elInfo.innerHTML = `<span style="color:#ff4444">Error: ${err.message}</span><br>` +
      `Make sure sim_server.py is running on port 8766`;
    trueDirGroup.clear();
  } finally {
    document.getElementById('spinner').classList.remove('active');
    btn.textContent = 'Run Simulation';
    updateRunEnabled();
  }
}

document.getElementById('btnRun').addEventListener('click', runOneShotSimulation);
document.getElementById('btnLoadMiccanvas').addEventListener('click', () => {
  document.getElementById('simGeo').value = 'CUSTOM';
  customMicPositions = MICCANVAS_POSITIONS.map(p => [p[0], p[1], p[2]]);
  customSelectedIdx = customMicPositions.length ? 0 : -1;
  updateLiveLabels();
  renderCustomList();
  updateCustomArrayVisibility();
});
document.getElementById('btnCustomAdd').addEventListener('click', () => {
  customAdd(
    document.getElementById('customAddX').value,
    document.getElementById('customAddY').value,
    document.getElementById('customAddZ').value
  );
});
['customAddX','customAddY','customAddZ'].forEach(id => {
  document.getElementById(id).addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') {
      ev.preventDefault();
      document.getElementById('btnCustomAdd').click();
    }
  });
});
document.getElementById('btnCustomSeed').addEventListener('click', customSeedFromPreset);
document.getElementById('btnCustomClear').addEventListener('click', customClear);
document.getElementById('btnCustomImport').addEventListener('click', () => {
  document.getElementById('customFile').click();
});
document.getElementById('customFile').addEventListener('change', (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      customMicPositions = customParseFile(e.target.result);
      customSelectedIdx = customMicPositions.length ? 0 : -1;
      refreshAfterCustomMutation();
    } catch (err) {
      alert('Invalid custom mic file: ' + err.message);
    }
  };
  reader.readAsText(file);
  ev.target.value = '';
});

// ── Phase 2a: materials / impairments / presets / heatmap ──

const MATERIAL_FALLBACK = [
  "hard_surface","rough_concrete","unpainted_concrete","brickwork","marble_floor",
  "concrete_floor","linoleum_on_concrete","wood_1.6cm","carpet_thin","carpet_hairy",
  "carpet_tufted_9.5mm","plasterboard","gypsum_board","wooden_lining",
  "ceiling_plasterboard","ceiling_fissured_tile","ceiling_metal_panel",
  "ceiling_perforated_gypsum_board","mineral_wool_50mm_40kgm3",
  "glass_window","double_glazing_30mm","curtains_0.2","curtains_cotton_0.33",
  "curtains_cotton_0.5","curtains_velvet","audience_1_m2","chairs_medium_upholstered",
];
const EXHIBITION_HALL_FALLBACK = {
  floor:"carpet_hairy", ceiling:"ceiling_fissured_tile",
  east:"plasterboard", west:"plasterboard",
  south:"curtains_cotton_0.5", north:"plasterboard",
};
const MATERIAL_SEL_IDS = {
  floor:'matFloor', ceiling:'matCeiling',
  east:'matEast', west:'matWest',
  south:'matSouth', north:'matNorth',
};

function populateMaterialSelects(choices, defaults) {
  for (const [wall, selId] of Object.entries(MATERIAL_SEL_IDS)) {
    const sel = document.getElementById(selId);
    sel.innerHTML = '';
    for (const name of choices) {
      const o = document.createElement('option');
      o.value = name; o.textContent = name;
      sel.appendChild(o);
    }
    sel.value = defaults[wall] || choices[0];
  }
}

async function initMaterials() {
  let choices = MATERIAL_FALLBACK;
  let defaults = EXHIBITION_HALL_FALLBACK;
  try {
    const r = await fetch('http://127.0.0.1:8766/materials');
    if (r.ok) {
      const d = await r.json();
      if (Array.isArray(d.choices) && d.choices.length) choices = d.choices;
      if (d.exhibition_hall) defaults = d.exhibition_hall;
    }
  } catch (_) { /* server offline -- fall back to hard-coded */ }
  populateMaterialSelects(choices, defaults);
}
const materialsReady = initMaterials();

function setAbsorptionMode(mode) {
  const grid = document.getElementById('materialGrid');
  const btn  = document.getElementById('btnExhibitionHall');
  const rt60Row = document.getElementById('simRT60').parentElement;
  if (mode === 'materials') {
    grid.classList.remove('hidden');
    btn.classList.remove('hidden');
    rt60Row.style.opacity = 0.4;
    rt60Row.style.pointerEvents = 'none';
  } else {
    grid.classList.add('hidden');
    btn.classList.add('hidden');
    rt60Row.style.opacity = '';
    rt60Row.style.pointerEvents = '';
  }
}
document.getElementById('absRT60').addEventListener('change', () => setAbsorptionMode('rt60'));
document.getElementById('absMaterials').addEventListener('change', () => setAbsorptionMode('materials'));

document.getElementById('btnExhibitionHall').addEventListener('click', () => {
  for (const [wall, selId] of Object.entries(MATERIAL_SEL_IDS)) {
    const sel = document.getElementById(selId);
    const want = EXHIBITION_HALL_FALLBACK[wall];
    if ([...sel.options].some(o => o.value === want)) sel.value = want;
  }
  // Phase 2b: realistic hall atmosphere -- warm ceiling under stage lights,
  // mild upward temperature gradient. Moving source stays off; it's a
  // what-if toggle.
  const atm = {simTemp: 22, simHumidity: 55, simTempGrad: 1.5};
  for (const [id, v] of Object.entries(atm)) {
    const el = document.getElementById(id);
    if (el) { el.value = v; el.dispatchEvent(new Event('input')); }
  }
  // Phase 3: keep Exhibition Hall as the "known baseline" -- it applies
  // the default models (point_source crowd, simple crosstalk, no ML
  // preview) so the user can toggle Phase 3 items independently.
  document.getElementById('simCrowdModel').value = 'point_source';
  document.getElementById('simCrosstalkModel').value = 'simple';
  document.getElementById('simMlPreview').checked = false;
  // Phase 3+: also reset DOA-band knobs so Exhibition Hall matches the
  // original Phase 1 / Phase 2 band (200-2000 Hz, flat, no comb).
  const bandReset = {simFmin: 200, simFmax: 2000, simFundamental: 200};
  for (const [id, v] of Object.entries(bandReset)) {
    const el = document.getElementById(id);
    if (el) { el.value = v; el.dispatchEvent(new Event('input')); }
  }
  document.getElementById('simHarmonicComb').checked = false;
  document.getElementById('simNormalizeAudio').checked = true;
  syncImpairmentDetails();
});

// Impairment enable/disable of nested sliders
function syncImpairmentDetails() {
  document.getElementById('crosstalkDetail').classList.toggle('active',
    document.getElementById('simCrosstalk').checked);
  document.getElementById('quantDetail').classList.toggle('active',
    document.getElementById('simQuant').checked);
  document.getElementById('movingDetail').classList.toggle('active',
    document.getElementById('simMoving').checked);
  // Phase 3: crowd-model sub-slider shows only for plane_wave; ML preview
  // panel follows its checkbox; crosstalk corner controls only show in
  // fir_capacitive mode.
  const crowdPlane = document.getElementById('simCrowdModel').value === 'plane_wave';
  document.getElementById('crowdModelDetail').classList.toggle('active', crowdPlane);
  document.getElementById('mlPreviewDetail').classList.toggle('active',
    document.getElementById('simMlPreview').checked);
  const firMode = document.getElementById('simCrosstalkModel').value === 'fir_capacitive';
  document.getElementById('simCrosstalkCornerLabel').style.display = firMode ? 'block' : 'none';
  document.getElementById('simCrosstalkCorner').style.display      = firMode ? 'block' : 'none';
  // Phase 3+: harmonic-comb detail (fundamental slider) only when comb is on.
  document.getElementById('harmonicCombDetail').classList.toggle('active',
    document.getElementById('simHarmonicComb').checked);
}
document.getElementById('simCrosstalk').addEventListener('change', syncImpairmentDetails);
document.getElementById('simQuant').addEventListener('change', syncImpairmentDetails);
document.getElementById('simMoving').addEventListener('change', syncImpairmentDetails);
document.getElementById('simMlPreview').addEventListener('change', syncImpairmentDetails);
document.getElementById('simCrowdModel').addEventListener('change', syncImpairmentDetails);
document.getElementById('simCrosstalkModel').addEventListener('change', syncImpairmentDetails);
document.getElementById('simTrajType').addEventListener('change', updateLiveLabels);
document.getElementById('simCrosstalkDb').addEventListener('input', (e) => {
  document.getElementById('simCrosstalkDbVal').textContent = e.target.value;
});
document.getElementById('simCrosstalkCorner').addEventListener('input', (e) => {
  document.getElementById('simCrosstalkCornerVal').textContent = e.target.value;
});
document.getElementById('simNPlaneWaves').addEventListener('input', (e) => {
  document.getElementById('simNPlaneWavesVal').textContent = e.target.value;
});
document.getElementById('simMlNMels').addEventListener('input', (e) => {
  document.getElementById('simMlNMelsVal').textContent = e.target.value;
});
document.getElementById('simHarmonicComb').addEventListener('change', syncImpairmentDetails);
document.getElementById('simFmin').addEventListener('input', (e) => {
  document.getElementById('simFminVal').textContent = e.target.value;
});
document.getElementById('simFmax').addEventListener('input', (e) => {
  document.getElementById('simFmaxVal').textContent = e.target.value;
});
document.getElementById('simFundamental').addEventListener('input', (e) => {
  document.getElementById('simFundamentalVal').textContent = e.target.value;
});
syncImpairmentDetails();
updateLiveLabels();
updateBeamMethodUi();

document.getElementById('btnHwPreset').addEventListener('click', () => {
  document.getElementById('simMismatch').checked  = true;
  document.getElementById('simCrosstalk').checked = true;
  document.getElementById('simCrosstalkDb').value = -40;
  document.getElementById('simCrosstalkDbVal').textContent = '-40';
  document.getElementById('simQuant').checked    = true;
  document.getElementById('simBitDepth').value   = '16';
  syncImpairmentDetails();
});

// ── SRP-PHAT heatmap + top-N candidates ──
function renderHeatmapAndTopN(data) {
  if (!data || !data.power || !data.power.length) {
    elHeatmapWrap.classList.add('hidden');
    return;
  }
  const grid = data.power;               // [n_colat][n_az]
  const nC = grid.length;
  const nA = grid[0].length;
  let vmax = -Infinity, vmin = Infinity;
  for (let c = 0; c < nC; c++) {
    for (let a = 0; a < nA; a++) {
      const v = grid[c][a];
      if (v > vmax) vmax = v;
      if (v < vmin) vmin = v;
    }
  }
  const span = Math.max(vmax - vmin, 1e-12);
  elHeatmapCanvas.width = nA;
  elHeatmapCanvas.height = nC;
  const ctx = elHeatmapCanvas.getContext('2d');
  const img = ctx.createImageData(nA, nC);
  for (let c = 0; c < nC; c++) {
    for (let a = 0; a < nA; a++) {
      const t = (grid[c][a] - vmin) / span;
      const r = Math.round(10 + 245 * Math.max(0, Math.min(1, t * 1.4 - 0.2)));
      const g = Math.round(20 + 140 * Math.max(0, Math.min(1, 1 - Math.abs(t - 0.5) * 2)));
      const b = Math.round(80 + 140 * Math.max(0, Math.min(1, 1 - t)));
      const i = (c * nA + a) * 4;
      img.data[i] = r; img.data[i+1] = g; img.data[i+2] = b; img.data[i+3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  elHeatmapRange.textContent = `max ${vmax.toFixed(2)}`;

  const peaks = data.top_peaks || [];
  elTopCand.innerHTML = peaks.map((p, i) => {
    const cls = (i === 0) ? 'cand-main' : '';
    return `<div class="cand-line ${cls}">` +
           `#${i+1}  az=${p.az_deg}°  el=${p.el_deg}°  ` +
           `<span style="color:#888">(${p.rel_db.toFixed(1)} dB)</span>` +
           `</div>`;
  }).join('');
  elHeatmapWrap.classList.remove('hidden');
}

// ── Save / load / share presets ──
const PRESET_SLIDER_IDS = [
  'simMicCount','simRadius','simSep',
  'simRoomL','simRoomW','simRoomH','simRT60',
  'simSrcAz','simSrcEl','simSrcDist',
  'simDroneSPL','simCrowdSPL','simPASPL','simMicFloor',
  'simSeed','simCrowd','simPA','simInt','simCrosstalkDb',
  'simTemp','simHumidity','simTempGrad',
  'simSpeed','simHeading','simChunks',
  'simNPlaneWaves','simCrosstalkCorner','simMlNMels',
  'simFmin','simFmax','simFundamental',
  'simBeamK','simFracMax','simFracTaps','simTargetFs',
];
const PRESET_SELECT_IDS = [
  'simGeo','simBitDepth','simTrajType','simBeamMethod',
  'matFloor','matCeiling','matEast','matWest','matSouth','matNorth',
  'simCrowdModel','simCrosstalkModel',
  'simMlBitDepth','simMlFeatBitDepth',
];
const PRESET_CHECK_IDS = [
  'simRandSeed','simDiffuse','simMismatch','simCrosstalk','simQuant',
  'simMoving','simMlPreview','simHarmonicComb','simNormalizeAudio',
];

function captureState() {
  const s = {v: 1, sliders: {}, selects: {}, checks: {}, absMode: '', customMicPositions: []};
  for (const id of PRESET_SLIDER_IDS) s.sliders[id] = document.getElementById(id).value;
  for (const id of PRESET_SELECT_IDS) s.selects[id] = document.getElementById(id).value;
  for (const id of PRESET_CHECK_IDS)  s.checks[id]  = document.getElementById(id).checked;
  s.absMode = document.querySelector('input[name="absMode"]:checked').value;
  s.customMicPositions = customMicPositions.map(p => [p[0], p[1], p[2]]);
  s.simScanAngles = document.getElementById('simScanAngles').value;
  return s;
}

function _legacySimId(id) {
  if (document.getElementById(id)) return id;
  if (id.startsWith('sim')) return 'live' + id.slice(3);
  return id;
}

function applyState(s) {
  if (!s) return;
  const sliders = {...(s.sliders || {})};
  for (const [id, v] of Object.entries(s.sliders || {})) {
    if (id.startsWith('live') && !sliders['sim' + id.slice(4)]) {
      sliders['sim' + id.slice(4)] = v;
    }
  }
  for (const [id, v] of Object.entries(sliders)) {
    const el = document.getElementById(id) || document.getElementById(_legacySimId(id));
    if (el) { el.value = v; el.dispatchEvent(new Event('input')); }
  }
  const selects = {...(s.selects || {})};
  for (const [id, v] of Object.entries(s.selects || {})) {
    if (id.startsWith('live') && !selects['sim' + id.slice(4)]) {
      selects['sim' + id.slice(4)] = v;
    }
  }
  for (const [id, v] of Object.entries(selects)) {
    const el = document.getElementById(id) || document.getElementById(_legacySimId(id));
    if (el) el.value = v;
  }
  if (typeof s.simScanAngles === 'string') {
    document.getElementById('simScanAngles').value = s.simScanAngles;
  } else if (typeof s.liveScanAngles === 'string') {
    document.getElementById('simScanAngles').value = s.liveScanAngles;
  }
  for (const [id, v] of Object.entries(s.checks || {})) {
    const el = document.getElementById(id);
    if (el) { el.checked = !!v; el.dispatchEvent(new Event('change')); }
  }
  if (s.absMode === 'materials') {
    document.getElementById('absMaterials').checked = true;
    setAbsorptionMode('materials');
  } else {
    document.getElementById('absRT60').checked = true;
    setAbsorptionMode('rt60');
  }
  if (Array.isArray(s.customMicPositions)) {
    customMicPositions = s.customMicPositions
      .filter(validateMicTriple)
      .map(p => p.map(Number));
    customSelectedIdx = customMicPositions.length ? 0 : -1;
  } else {
    customMicPositions = [];
    customSelectedIdx = -1;
  }
  syncImpairmentDetails();
  updateBeamMethodUi();
  updateLiveLabels();
  renderCustomList();
  updateCustomArrayVisibility();
}

document.getElementById('btnPresetSave').addEventListener('click', () => {
  const s = captureState();
  const blob = new Blob([JSON.stringify(s, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  a.href = url; a.download = `preset-${ts}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
});

document.getElementById('btnPresetLoad').addEventListener('click', () => {
  document.getElementById('presetFile').click();
});
document.getElementById('presetFile').addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const rd = new FileReader();
  rd.onload = (ev) => {
    try { applyState(JSON.parse(ev.target.result)); }
    catch (err) { alert('Invalid preset JSON: ' + err.message); }
  };
  rd.readAsText(f);
  e.target.value = '';
});

document.getElementById('btnPresetShare').addEventListener('click', async () => {
  const s = captureState();
  const hash = btoa(unescape(encodeURIComponent(JSON.stringify(s))));
  const url = window.location.origin + window.location.pathname + '#preset=' + hash;
  try { await navigator.clipboard.writeText(url); }
  catch (_) { prompt('Copy this URL:', url); return; }
  const btn = document.getElementById('btnPresetShare');
  const prev = btn.textContent;
  btn.textContent = 'Copied!';
  setTimeout(() => { btn.textContent = prev; }, 1200);
});

function applyUrlHashPreset() {
  const m = /#preset=([^&]+)/.exec(window.location.hash || '');
  if (!m) return;
  try {
    const json = decodeURIComponent(escape(atob(m[1])));
    applyState(JSON.parse(json));
  } catch (err) {
    console.warn('Could not decode preset from URL hash:', err);
  }
}

// ── Event listeners ──
[elSelGeo, elMicCount, elRadius, elSep, elSteerAz, elSteerEl, elFreq, elDispRadius, elOpacity].forEach(el => {
  el.addEventListener('input', refresh);
});
elSelGeo.addEventListener('change', refresh);
srpGeoSel.addEventListener('change', refresh);
srpCondSel.addEventListener('change', refresh);

// ── Init ──
renderCustomList();
updateCustomArrayVisibility();
refresh();
materialsReady.then(applyUrlHashPreset);

// ── Realtime WebSocket + bird's-eye view ──
const RT_WS_URL = (location.protocol === 'https:' ? 'wss:' : 'ws:') + '//127.0.0.1:8766/realtime';
let rtWs = null;
let rtInit = null;
let rtLastFrame = null;
let rtPowerHistory = [];
const RT_HISTORY_MAX = 200;
let rtDroneXY = null;
let rtDragging = false;
let rtPaused = false;

const elBirdsEye = document.getElementById('birdsEye');
const elRtMetrics = document.getElementById('rtMetrics');
const elRtGeoOverlay = document.getElementById('rtGeoOverlay');
const elRtTimeline = document.getElementById('rtTimelineCanvas');
const elRtStatus = document.getElementById('rtStatus');

function stopOneShotExtras() {}

function rtSend(obj) {
  if (rtWs && rtWs.readyState === WebSocket.OPEN) {
    rtWs.send(JSON.stringify(obj));
  }
}

function stopRealtimeSession() {
  if (rtWs) {
    rtWs.close();
    rtWs = null;
  }
  rtInit = null;
  rtLastFrame = null;
  rtPowerHistory = [];
}

function resizeBirdsEye() {
  if (!elBirdsEye) return;
  elBirdsEye.width = window.innerWidth;
  elBirdsEye.height = Math.max(200, window.innerHeight - 90);
  if (elRtTimeline) {
    elRtTimeline.width = window.innerWidth;
    elRtTimeline.height = 90;
  }
  if (rtInit || rtLastFrame) drawBirdsEye(rtInit, rtLastFrame);
}

function roomToCanvas(x, y, roomDim, W, H, pad = 40) {
  const rw = roomDim[0], rh = roomDim[1];
  const sx = (W - 2 * pad) / rw;
  const sy = (H - 2 * pad) / rh;
  const s = Math.min(sx, sy);
  const ox = (W - rw * s) / 2;
  const oy = (H - rh * s) / 2;
  return { px: ox + x * s, py: H - (oy + y * s), s, ox, oy };
}

function drawBirdsEye(init, frame) {
  if (!elBirdsEye || !init) return;
  const ctx = elBirdsEye.getContext('2d');
  const W = elBirdsEye.width, H = elBirdsEye.height;
  ctx.fillStyle = '#020805';
  ctx.fillRect(0, 0, W, H);
  const room = init.room_dim;
  const ac = init.array_center;
  const p0 = roomToCanvas(0, 0, room, W, H);
  const p1 = roomToCanvas(room[0], room[1], room, W, H);
  ctx.strokeStyle = 'rgba(95,210,138,0.25)';
  ctx.lineWidth = 1;
  ctx.strokeRect(p0.px, p1.py, p1.px - p0.px, p0.py - p1.py);
  const pc = roomToCanvas(ac[0], ac[1], room, W, H);
  for (let r = 5; r <= Math.min(room[0], room[1]); r += 5) {
    ctx.beginPath();
    ctx.arc(pc.px, pc.py, r * pc.s, 0, Math.PI * 2);
    ctx.stroke();
  }
  (init.crowd_positions || []).forEach(p => {
    const q = roomToCanvas(p[0], p[1], room, W, H);
    ctx.fillStyle = 'rgba(255,136,51,0.55)';
    ctx.beginPath();
    ctx.arc(q.px, q.py, 3, 0, Math.PI * 2);
    ctx.fill();
  });
  (init.pa_positions || []).forEach(p => {
    const q = roomToCanvas(p[0], p[1], room, W, H);
    ctx.fillStyle = 'rgba(170,68,255,0.7)';
    ctx.fillRect(q.px - 3, q.py - 3, 6, 6);
  });
  ctx.fillStyle = '#5fd28a';
  ctx.beginPath();
  ctx.arc(pc.px, pc.py, 6, 0, Math.PI * 2);
  ctx.fill();
  if (frame && frame.beam_scan) {
    const scan = frame.beam_scan;
    const powers = scan.powers_db || [];
    const angles = scan.angles_deg || [];
    const pMin = Math.min(...powers);
    const pMax = Math.max(...powers);
    const span = Math.max(pMax - pMin, 1e-6);
    const fanR = Math.min(W, H) * 0.38;
    const wedge = (2 * Math.PI) / Math.max(angles.length, 1);
    angles.forEach((deg, i) => {
      const t = (powers[i] - pMin) / span;
      const a0 = (deg * Math.PI / 180) - wedge * 0.48;
      const a1 = (deg * Math.PI / 180) + wedge * 0.48;
      const isMax = i === scan.argmax_idx;
      ctx.beginPath();
      ctx.moveTo(pc.px, pc.py);
      ctx.arc(pc.px, pc.py, fanR * (0.2 + 0.8 * t), a0, a1);
      ctx.closePath();
      ctx.fillStyle = isMax
        ? `rgba(210,161,74,${0.35 + 0.5 * t})`
        : `rgba(95,210,138,${0.08 + 0.25 * t})`;
      ctx.fill();
    });
  }
  const src = (frame && frame.source_pos) ? frame.source_pos : init.source_pos;
  if (src) {
    const ds = roomToCanvas(src[0], src[1], room, W, H);
    if (frame) {
      const az = frame.true_az_deg * Math.PI / 180;
      const len = 50 + 30 * ((frame.argmax_db || 0) + 40) / 40;
      ctx.strokeStyle = 'rgba(210,161,74,0.9)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(pc.px, pc.py);
      const lockAz = frame.est_az_deg * Math.PI / 180;
      ctx.lineTo(pc.px + len * Math.cos(lockAz), pc.py - len * Math.sin(lockAz));
      ctx.stroke();
      ctx.strokeStyle = 'rgba(68,204,68,0.85)';
      ctx.beginPath();
      ctx.moveTo(pc.px, pc.py);
      ctx.lineTo(pc.px + 40 * Math.cos(az), pc.py - 40 * Math.sin(az));
      ctx.stroke();
    }
    ctx.fillStyle = '#ff4444';
    ctx.beginPath();
    ctx.arc(ds.px, ds.py, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#d2a14a';
    ctx.font = '11px DM Mono, monospace';
    ctx.fillText(`el ${(frame && frame.true_el_deg !== undefined) ? frame.true_el_deg : 0}°`, ds.px + 12, ds.py - 8);
    rtDroneXY = { px: ds.px, py: ds.py, roomX: src[0], roomY: src[1] };
  }
  const geo = (frame && frame.geometry) || init.geometry || 'UCA';
  const nm = (frame && frame.n_mics) || init.n_mics || '?';
  elRtGeoOverlay.textContent = `${geo} · ${nm} mics`;
}

function renderRtMetrics(frame) {
  if (!frame || !elRtMetrics) return;
  const scan = frame.beam_scan || {};
  const mips = frame.mips || {};
  const lat = mips.scan_latency_s;
  const latLine = (lat !== null && lat !== undefined && mips.method === 'steered_das')
    ? `Scan latency: <span style="color:#5fd28a">${Number(lat).toFixed(2)} s</span><br>`
    : (mips.method === 'beam_bank_das'
      ? `Mux: <span style="color:#5fd28a">instant</span><br>` : '');
  elRtMetrics.innerHTML =
    `Method: <span style="color:#5fd28a">${frame.beam_method}</span><br>` +
    `Lock: az=<span style="color:#5fd28a">${frame.est_az_deg}°</span> ` +
    `(${frame.argmax_db} dB, margin ${frame.margin_db} dB)<br>` +
    `True: az=${frame.true_az_deg}° el=${frame.true_el_deg}°<br>` +
    latLine +
    `MIPS: ${mips.ops_per_sample || 0} ops/sample (${(mips.budget_used_pct || 0).toFixed(1)}%)<br>` +
    `Frame #${frame.frame_idx} · t=${frame.t_sim_s}s`;
  renderMipsReadout(mips);
}

function renderRtTimeline() {
  if (!elRtTimeline || rtPowerHistory.length < 2) return;
  const ctx = elRtTimeline.getContext('2d');
  const W = elRtTimeline.width, H = elRtTimeline.height;
  ctx.fillStyle = '#05100a';
  ctx.fillRect(0, 0, W, H);
  const nK = rtPowerHistory[0].powers.length;
  const cols = ['#3a5a44','#4a6a54','#5a7a64','#6a8a74','#7a9a84','#8aaa94','#9aba9f','#aacaaa'];
  const tMin = rtPowerHistory[0].t;
  const tMax = rtPowerHistory[rtPowerHistory.length - 1].t;
  const tSpan = Math.max(tMax - tMin, 0.01);
  rtPowerHistory.forEach((row, idx) => {
    const x = ((row.t - tMin) / tSpan) * (W - 20) + 10;
    row.powers.forEach((p, ki) => {
      const y = H - 10 - (ki + 1) * (H - 20) / (nK + 1);
      const isMax = ki === row.argmax;
      ctx.fillStyle = isMax ? '#d2a14a' : cols[ki % cols.length];
      ctx.fillRect(x - 1, y - 2, 3, 4);
    });
  });
}

function onRealtimeMessage(msg) {
  if (msg.type === 'init') {
    rtInit = msg;
    if (!rtDroneXY && msg.source_pos) {
      rtDroneXY = { roomX: msg.source_pos[0], roomY: msg.source_pos[1] };
    }
    drawBirdsEye(rtInit, rtLastFrame);
    elRtStatus.textContent = 'Streaming';
    return;
  }
  if (msg.type === 'regenerating_ambient') {
    if (msg.state === 'start') {
      elRtStatus.textContent = 'Streaming (regenerating ambient...)';
    } else if (msg.state === 'done') {
      elRtStatus.textContent = 'Streaming';
    }
    return;
  }
  if (msg.type === 'frame') {
    rtLastFrame = msg;
    renderRtMetrics(msg);
    if (msg.beam_scan && msg.beam_scan.powers_db) {
      rtPowerHistory.push({
        t: msg.t_sim_s,
        powers: msg.beam_scan.powers_db.slice(),
        argmax: msg.beam_scan.argmax_idx,
      });
      if (rtPowerHistory.length > RT_HISTORY_MAX) rtPowerHistory.shift();
      renderRtTimeline();
    }
    drawBirdsEye(rtInit, rtLastFrame);
  }
}

function startRealtimeSession() {
  stopRealtimeSession();
  resizeBirdsEye();
  elRtStatus.textContent = 'Connecting...';
  rtWs = new WebSocket(RT_WS_URL);
  rtWs.onopen = () => {
    elRtStatus.textContent = 'Connected';
    rtSend({ type: 'resume' });
    pushRealtimeSettings();
  };
  rtWs.onmessage = (ev) => {
    try { onRealtimeMessage(JSON.parse(ev.data)); }
    catch (e) { console.warn('rt parse', e); }
  };
  rtWs.onclose = () => {
    if (mode === 'realtime') elRtStatus.textContent = 'Disconnected (is sim_server.py running?)';
  };
  rtWs.onerror = () => {
    elRtStatus.textContent = 'WebSocket error';
  };
}

function pushRealtimeSettings() {
  const k = parseInt(document.getElementById('rtBeamK').value);
  rtSend({ type: 'set_method', beam_method: document.getElementById('rtBeamMethod').value });
  rtSend({ type: 'set_beam_bank', k });
  rtSend({ type: 'set_integration_ms', integration_ms: parseInt(document.getElementById('rtInt').value) });
  rtSend({
    type: 'set_crowd',
    diffuse: document.getElementById('rtDiffuse').checked,
    crowd_count: parseInt(document.getElementById('rtCrowd').value),
  });
  const geo = document.getElementById('rtGeo').value;
  const payload = {
    type: 'set_geometry',
    geometry: geo,
    mic_count: parseInt(document.getElementById('simMicCount')?.value || 12),
    radius: parseFloat(document.getElementById('simRadius')?.value || 0.15),
    ring_separation: parseFloat(document.getElementById('simSep')?.value || 0.12),
  };
  if (geo === 'CUSTOM') {
    payload.custom_mic_positions = customMicPositions.map(p => [p[0], p[1], p[2]]);
  }
  rtSend(payload);
  if (rtDroneXY) {
    rtSend({
      type: 'set_drone',
      pos: [rtDroneXY.roomX, rtDroneXY.roomY],
      el_deg: parseFloat(document.getElementById('rtEl').value),
    });
  }
}

function canvasToRoom(px, py, roomDim, W, H) {
  const rw = roomDim[0], rh = roomDim[1];
  const pad = 40;
  const sx = (W - 2 * pad) / rw;
  const sy = (H - 2 * pad) / rh;
  const s = Math.min(sx, sy);
  const ox = (W - rw * s) / 2;
  const oy = (H - rh * s) / 2;
  const x = (px - ox) / s;
  const y = (H - py - oy) / s;
  return [x, y];
}

elBirdsEye.addEventListener('mousedown', (ev) => {
  if (!rtInit || mode !== 'realtime') return;
  const rect = elBirdsEye.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  if (rtDroneXY && Math.hypot(mx - rtDroneXY.px, my - rtDroneXY.py) < 18) rtDragging = true;
});
window.addEventListener('mousemove', (ev) => {
  if (!rtDragging || !rtInit) return;
  const rect = elBirdsEye.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  const W = elBirdsEye.width, H = elBirdsEye.height;
  const [rx, ry] = canvasToRoom(mx, my, rtInit.room_dim, W, H);
  rtDroneXY = { roomX: rx, roomY: ry };
  rtSend({
    type: 'set_drone',
    pos: [rx, ry],
    el_deg: parseFloat(document.getElementById('rtEl').value),
  });
  drawBirdsEye(rtInit, rtLastFrame);
});
window.addEventListener('mouseup', () => { rtDragging = false; });

document.getElementById('btnRtPause').addEventListener('click', () => {
  rtPaused = !rtPaused;
  document.getElementById('btnRtPause').textContent = rtPaused ? 'Resume' : 'Pause';
  rtSend({ type: rtPaused ? 'pause' : 'resume' });
});

['rtBeamK','rtInt','rtEl','rtCrowd'].forEach(id => {
  document.getElementById(id).addEventListener('input', (e) => {
    const valId = id + 'Val';
    const vel = document.getElementById(valId);
    if (vel) vel.textContent = e.target.value;
    if (mode === 'realtime') pushRealtimeSettings();
  });
});
document.getElementById('rtBeamMethod').addEventListener('change', () => {
  if (mode === 'realtime') pushRealtimeSettings();
});
document.getElementById('rtDiffuse').addEventListener('change', () => {
  if (mode === 'realtime') pushRealtimeSettings();
});
document.getElementById('rtGeo').addEventListener('change', () => {
  if (mode === 'realtime') pushRealtimeSettings();
});
document.getElementById('btnRtMiccanvas').addEventListener('click', () => {
  document.getElementById('rtGeo').value = 'CUSTOM';
  customMicPositions = MICCANVAS_POSITIONS.map(p => [p[0], p[1], p[2]]);
  if (mode === 'realtime') pushRealtimeSettings();
});

window.addEventListener('resize', resizeBirdsEye);

// ── Animate ──
function animate() {
  requestAnimationFrame(animate);

  if (mode === 'hardware' && hwLastFrame) {
    const now = performance.now();
    if (now - hwLastUiTick > 500) {
      hwLastUiTick = now;
      updateHwReadiness(hwLastFrame);
      renderHwSolution(hwLastFrame);
      hwNeedsRedraw = true;
    }
    if (hwNeedsRedraw) {
      hwNeedsRedraw = false;
      drawHwBirdsEye(hwLastFrame);
      updateHw3dRing(hwLastFrame);
    }
    // Sonar: lerp toward target angle (one full rev per scan)
    const sonarDiff = hwSonarTargetAngle - hwSonarAngle;
    hwSonarAngle += sonarDiff * 0.08;
    renderHwSonar();
    // Update jet particles
    updateJetParticles();
  }

  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  if (mode === 'hardware') {
    resizeHardwareViews();
    return;
  }
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ── Hardware Mode ─────────────────────────────────────────────────────────────

const HW_WS_URL = 'ws://127.0.0.1:8766/realtime_hw';
let hwWs = null;
let hwInit = null;
let hwLastFrame = null;
let hwPowerHistory = [];
const HW_HISTORY_MAX = 500;
let hwPaused = false;
let hwMonitoring = false;
let hwMonitorTimer = null;
let hwConnectionState = 'disconnected';
let hwCellTimestamps = [];
let hwEventLog = [];
let hwLastLoggedSequence = -1;
let hwLastTargetSector = null;
let hwLastUiTick = 0;
let hwAcousticCacheId = null;
let hwAcousticPollTimer = null;
let hwAuditionUrls = {};
const HW_EVENT_LOG_MAX = 50;
const HW_STALE_MS = 5000;

// -- Interpolation state for 60 FPS smooth animation --
let hwNeedsRedraw = false;
let hwSonarTargetAngle = 0;
let hwAfterglowHistory = []; // stores last 5 interpVals arrays for phosphor decay

// -- 3D ring mesh for hardware mode --
let hw3dActive = false;

// -- Jet particle system (plasma ejections) --
let hwJetParticles = []; // {x,y,z, vx,vy,vz, life, maxLife, r,g,b}
const HW_JET_MAX = 300;
let hwJetPoints = null;
let hwJetGeo = null;

function hwTransportOpen() {
  return Boolean(hwWs && hwWs.readyState === WebSocket.OPEN);
}

function setHwConnectionState(state, detail) {
  hwConnectionState = state;
  const link = document.getElementById('hbLink');
  if (link) {
    link.className = `hw-state ${state}`;
    link.textContent = state === 'online' ? 'ONLINE'
      : state === 'connecting' ? 'CONNECTING'
      : state === 'fault' ? 'FAULT' : 'OFFLINE';
  }
  if (detail) document.getElementById('hwStatus').textContent = detail;
  if (state === 'disconnected') {
    document.getElementById('hbPort').textContent = '--';
    document.getElementById('hbArray').textContent = '--';
    document.getElementById('hbStatus').textContent = 'IDLE';
    document.getElementById('hbAge').textContent = '--';
    document.getElementById('hbWarnings').textContent = '0';
  }
  syncHwControls();
}

function syncHwControls() {
  const linkOpen = hwTransportOpen() && hwConnectionState === 'online';
  const ready = linkOpen && Boolean(hwInit?.configuration);
  for (const id of ['btnHwOnce', 'btnHwContinuous', 'btnHwAdaptive', 'btnHwSteer', 'btnHwMonitor']) {
    const control = document.getElementById(id);
    if (control) control.disabled = !ready;
  }
  for (const id of ['btnHwStop', 'btnHwPause']) {
    const control = document.getElementById(id);
    if (control) control.disabled = !linkOpen;
  }
  for (const id of ['btnHwConnect', 'btnHwEmulator']) {
    const control = document.getElementById(id);
    if (control) control.disabled = hwConnectionState === 'connecting' || linkOpen;
  }
  document.getElementById('btnHwDisconnect').disabled = !hwTransportOpen();

  const activeMode = hwLastFrame?.firmware_mode;
  document.getElementById('btnHwOnce').classList.toggle('active', activeMode === 'F');
  document.getElementById('btnHwContinuous').classList.toggle('active', activeMode === 'C');
  document.getElementById('btnHwAdaptive').classList.toggle('active', activeMode === 'G');
}

function addHwEvent(kind, detail, tone = '') {
  const previous = hwEventLog.at(-1);
  if (previous && previous.kind === kind && previous.detail === detail) return;
  hwEventLog.push({time: new Date(), kind, detail, tone});
  if (hwEventLog.length > HW_EVENT_LOG_MAX) hwEventLog.shift();
  renderHwEventLog();
}

function renderHwEventLog() {
  const log = document.getElementById('hwEventLog');
  if (!log) return;
  log.replaceChildren();
  if (hwEventLog.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'hw-event-empty';
    empty.textContent = 'No session events';
    log.appendChild(empty);
    return;
  }
  for (const event of [...hwEventLog].reverse()) {
    const row = document.createElement('div');
    row.className = `hw-event ${event.tone}`.trim();
    const time = document.createElement('span');
    time.className = 'hw-event-time';
    time.textContent = event.time.toLocaleTimeString([], {hour12: false});
    const kind = document.createElement('span');
    kind.className = 'hw-event-kind';
    kind.textContent = event.kind;
    const detail = document.createElement('span');
    detail.className = 'hw-event-detail';
    detail.textContent = event.detail;
    row.append(time, kind, detail);
    log.appendChild(row);
  }
}

function resetHwFreshness(configuration = null) {
  hwCellTimestamps = configuration
    ? Array.from({length: configuration.rows}, () => Array(configuration.columns).fill(null))
    : [];
}

function updateHwCellTimestamps(frame, previousFrame) {
  const configuration = frame.configuration;
  if (!configuration) return;
  if (hwCellTimestamps.length !== configuration.rows ||
      hwCellTimestamps.some(row => row.length !== configuration.columns)) {
    resetHwFreshness(configuration);
  }
  const receivedAt = Number.isFinite(frame.timestamp_s) ? frame.timestamp_s * 1000 : Date.now();
  const record = frame.last_record;
  if (record?.type === 'measurement' &&
      hwCellTimestamps[record.row]?.[record.column] !== undefined) {
    hwCellTimestamps[record.row][record.column] = receivedAt;
  }
  for (let row = 0; row < configuration.rows; row++) {
    for (let column = 0; column < configuration.columns; column++) {
      const value = frame.levels_raw?.[row]?.[column];
      const previousValue = previousFrame?.levels_raw?.[row]?.[column];
      if (value != null && value !== previousValue) hwCellTimestamps[row][column] = receivedAt;
    }
  }
}

function hwCellAgeMs(row, column) {
  const timestamp = hwCellTimestamps[row]?.[column];
  return Number.isFinite(timestamp) ? Math.max(0, Date.now() - timestamp) : null;
}

function formatHwAge(ageMs) {
  if (!Number.isFinite(ageMs)) return '--';
  if (ageMs < 1000) return `${Math.round(ageMs)} ms`;
  if (ageMs < 60000) return `${(ageMs / 1000).toFixed(1)} s`;
  return `${Math.floor(ageMs / 60000)}m ${Math.floor((ageMs % 60000) / 1000)}s`;
}

function hwStrongestObservation(frame) {
  let strongest = null;
  for (let row = 0; row < (frame.configuration?.rows || 0); row++) {
    for (let column = 0; column < (frame.configuration?.columns || 0); column++) {
      const level = frame.levels_db?.[row]?.[column];
      if (Number.isFinite(level) && (!strongest || level > strongest.level)) {
        strongest = {row, column, sector: row * frame.configuration.columns + column, level};
      }
    }
  }
  return strongest;
}

function hwSolution(frame) {
  if (!frame?.configuration) return null;
  let marker;
  let label;
  let stateClass;
  if (frame.target) {
    marker = frame.target;
    label = 'TRACKED TARGET';
    stateClass = 'locked';
  } else if (frame.last_steer) {
    marker = frame.last_steer;
    label = hwMonitoring ? 'FIXED BEAM MONITOR' : 'FIXED BEAM';
    stateClass = 'monitoring';
  } else {
    marker = hwStrongestObservation(frame);
    label = marker ? 'STRONGEST OBSERVATION' : 'NO OBSERVATION';
    stateClass = marker ? 'observation' : '';
  }
  if (!marker) return null;
  const azimuth = Number.isFinite(marker.azimuth_deg)
    ? marker.azimuth_deg : frame.azimuth_deg?.[marker.column];
  const elevation = Number.isFinite(marker.elevation_deg)
    ? marker.elevation_deg : frame.elevation_deg?.[marker.row];
  const level = frame.levels_db?.[marker.row]?.[marker.column];
  return {...marker, azimuth, elevation, level, label, stateClass,
    ageMs: hwCellAgeMs(marker.row, marker.column)};
}

function renderHwSolution(frame) {
  const state = document.getElementById('hwSolutionState');
  const direction = document.getElementById('hwSolutionDirection');
  const meta = document.getElementById('hwSolutionMeta');
  if (!state || !direction || !meta) return;
  renderHwTruth(frame);
  const solution = hwSolution(frame);
  if (!solution) {
    state.className = 'hw-solution-state';
    state.textContent = 'NO OBSERVATION';
    direction.textContent = '--.-° / --.-°';
    meta.textContent = 'Waiting for measurements';
    renderHwRateComparison(frame);
    return;
  }
  state.className = `hw-solution-state ${solution.stateClass}`.trim();
  state.textContent = solution.label;
  const azimuth = Number.isFinite(solution.azimuth) ? `${solution.azimuth.toFixed(1)}°` : '--.-°';
  const elevation = Number.isFinite(solution.elevation) ? `${solution.elevation.toFixed(1)}°` : '--.-°';
  direction.textContent = `${azimuth} / ${elevation}`;
  const level = Number.isFinite(solution.level) ? `${solution.level.toFixed(1)} dBFS` : '--';
  const margin = Number.isFinite(frame.margin_db) ? ` · peak-to-next ${frame.margin_db.toFixed(1)} dB` : '';
  meta.textContent = `Sector ${solution.sector} · row ${solution.row}, column ${solution.column} · ${level}${margin} · age ${formatHwAge(solution.ageMs)}`;
  renderHwRateComparison(frame);
}

function renderHwTruth(frame) {
  const panel = document.getElementById('hwTruth');
  const direction = document.getElementById('hwTruthDirection');
  const meta = document.getElementById('hwTruthMeta');
  const state = document.getElementById('hwTruthState');
  if (!panel || !direction || !meta || !state) return;
  const truth = frame?.emulator_truth;
  panel.classList.toggle('hidden', !truth);
  if (!truth) return;
  const active = truth.source_active !== false;
  state.textContent = active ? 'ACTIVE' : 'ACOUSTIC DROPOUT';
  state.classList.toggle('dropout', !active);
  direction.textContent = `${truth.azimuth_deg.toFixed(1)}° / ${truth.elevation_deg.toFixed(1)}°`;
  const position = truth.room_position_m.map(value => value.toFixed(1)).join(', ');
  const estimateError = Number.isFinite(frame.est_az_deg) && Number.isFinite(frame.est_el_deg)
    ? ` · estimate error ${Math.hypot(frame.est_az_deg - truth.azimuth_deg, frame.est_el_deg - truth.elevation_deg).toFixed(1)}°`
    : '';
  meta.textContent = `Range ${truth.range_m.toFixed(1)} m · room [${position}] m · loop ${truth.loop_elapsed_s.toFixed(1)} s${estimateError}`;
}

function renderHwRateComparison(frame) {
  const trackRateEl = document.getElementById('hwTrackRate');
  const fullRateEl = document.getElementById('hwFullScanRate');
  const gainEl = document.getElementById('hwRateGain');
  if (!trackRateEl || !fullRateEl || !gainEl) return;

  const fullRate = frame?.scan_rate_hz;
  const trackRate = frame?.target_update_rate_hz;
  const updateAge = frame?.target_update_age_s;
  const trackCurrent = Number.isFinite(trackRate) && Number.isFinite(updateAge) && updateAge < 2.0;
  trackRateEl.textContent = trackCurrent ? `${trackRate.toFixed(1)} Hz` : '--';
  fullRateEl.textContent = Number.isFinite(fullRate) ? `${fullRate.toFixed(2)} Hz` : '--';

  if (trackCurrent && Number.isFinite(fullRate) && fullRate > 0) {
    gainEl.textContent = `${(trackRate / fullRate).toFixed(1)}× faster solution refresh · host observed`;
    gainEl.classList.add('active');
  } else if (frame?.firmware_mode === 'G' && frame?.target) {
    gainEl.textContent = 'Local tracking update cadence settling';
    gainEl.classList.remove('active');
  } else if (frame?.firmware_mode === 'G') {
    gainEl.textContent = 'Global search / confirmation in progress';
    gainEl.classList.remove('active');
  } else {
    gainEl.textContent = 'Waiting for target lock';
    gainEl.classList.remove('active');
  }
}

function logHwRecord(frame) {
  if (!Number.isFinite(frame.sequence) || frame.sequence === hwLastLoggedSequence) return;
  hwLastLoggedSequence = frame.sequence;
  const record = frame.last_record;
  if (!record) return;
  switch (record.type) {
  case 'scan_started':
    addHwEvent('MODE', `${record.mode} started`);
    break;
  case 'scan_done':
    if (frame.firmware_mode === 'IDLE') addHwEvent('MODE', 'Full sweep complete');
    break;
  case 'scan_stopped':
    addHwEvent('MODE', 'Stopped by operator', 'warning');
    break;
  case 'steer_ok':
    addHwEvent('STEER', `Sector ${record.sector} · ${record.azimuth_deg.toFixed(1)}° / ${record.elevation_deg.toFixed(1)}°`);
    break;
  case 'target_acquired':
    hwLastTargetSector = record.sector;
    addHwEvent('TARGET', `Acquired sector ${record.sector}`, 'target');
    break;
  case 'target_updated':
    if (record.sector !== hwLastTargetSector) {
      hwLastTargetSector = record.sector;
      addHwEvent('TARGET', `Moved to sector ${record.sector}`, 'target');
    }
    break;
  case 'target_lost':
    hwLastTargetSector = null;
    addHwEvent('TARGET', `Lost from sector ${record.sector}`, 'warning');
    break;
  case 'steer_error':
    addHwEvent('ERROR', `Steer failed at mic ${record.microphone} · code ${record.code}`, 'error');
    break;
  case 'error':
    addHwEvent('ERROR', [record.reason, ...record.details].join(' · '), 'error');
    break;
  }
}

function logHwStateTransitions(frame, previousFrame) {
  if (!previousFrame) return;
  if (frame.scan_count > previousFrame.scan_count && frame.firmware_mode === 'IDLE') {
    addHwEvent('MODE', `Full sweep complete · pass ${frame.scan_count}`);
  }
  const previousTarget = previousFrame.target;
  const target = frame.target;
  if (!previousTarget && target) {
    hwLastTargetSector = target.sector;
    addHwEvent('TARGET', `Acquired sector ${target.sector}`, 'target');
  } else if (previousTarget && !target) {
    hwLastTargetSector = null;
    addHwEvent('TARGET', `Lock cleared from sector ${previousTarget.sector}`, 'warning');
  } else if (target && target.sector !== previousTarget?.sector) {
    hwLastTargetSector = target.sector;
    addHwEvent('TARGET', `Moved to sector ${target.sector}`, 'target');
  }
  if (frame.last_steer?.sector !== previousFrame.last_steer?.sector && frame.last_steer) {
    addHwEvent('STEER', `Sector ${frame.last_steer.sector} selected`);
  }
  if (frame.last_error && JSON.stringify(frame.last_error) !== JSON.stringify(previousFrame.last_error)) {
    addHwEvent('ERROR', frame.last_error.reason || 'Hardware command failed', 'error');
  }
}

function spawnJetParticles(peakAzRad, power01, dispRadius) {
  const count = Math.floor(3 + 8 * power01);
  for (let i = 0; i < count && hwJetParticles.length < HW_JET_MAX; i++) {
    const spread = (Math.random() - 0.5) * 0.3;
    const az = peakAzRad + spread;
    const speed = dispRadius * (0.008 + 0.015 * power01 * Math.random());
    const ySpeed = (Math.random() - 0.5) * speed * 0.2;
    // Start from outer edge of particle cloud
    const startR = dispRadius * (0.75 + 0.15 * Math.random());
    hwJetParticles.push({
      x: startR * Math.cos(az), y: ySpeed * 3, z: startR * Math.sin(az),
      vx: speed * Math.cos(az), vy: ySpeed, vz: speed * Math.sin(az),
      life: 1.0, maxLife: 0.2 + Math.random() * 0.2,
      r: 1.0, g: 0.4 + 0.3 * Math.random(), b: 0.0,
    });
  }
}

function updateJetParticles() {
  if (hwJetParticles.length === 0 && !hwJetPoints) return;
  const dt = 1 / 60;
  // Update existing
  for (let i = hwJetParticles.length - 1; i >= 0; i--) {
    const p = hwJetParticles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.z += p.vz;
    p.vx *= 0.96; p.vy *= 0.96; p.vz *= 0.96; // drag
    p.life -= dt / p.maxLife;
    if (p.life <= 0) { hwJetParticles.splice(i, 1); }
  }
  // Rebuild geometry
  if (hwJetPoints) { hwRingGroup.remove(hwJetPoints); hwJetPoints = null; }
  if (hwJetParticles.length === 0) return;
  const pos = [], cols = [];
  for (const p of hwJetParticles) {
    pos.push(p.x, p.y, p.z);
    cols.push(p.r * p.life, p.g * p.life, p.b * p.life);
  }
  hwJetGeo = new THREE.BufferGeometry();
  hwJetGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  hwJetGeo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  const mat = new THREE.PointsMaterial({size: 0.06, vertexColors: true, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true});
  hwJetPoints = new THREE.Points(hwJetGeo, mat);
  hwRingGroup.add(hwJetPoints);
}

function clearHardwarePresentation() {
  hwInit = null;
  hwLastFrame = null;
  hwPaused = false;
  hwLastLoggedSequence = -1;
  hwLastTargetSector = null;
  hwGridLayout = null;
  resetHwFreshness();
  hwPowerHistory = [];
  hwAfterglowHistory = [];
  hwNeedsRedraw = false;
  hwRingGroup.clear();
  hwArrayMarkerGroup.visible = false;
  trueDirGroup.clear();
  steerGroup.clear();
  document.getElementById('hwMetrics').textContent = 'No active transport';
  document.getElementById('hwStatsLine').textContent = 'No measurements received';
  renderHwSolution(null);
  updateHwGridSource();
  for (const canvasId of ['hwBirdsEye', 'hwTimelineCanvas']) {
    const canvas = document.getElementById(canvasId);
    canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
  }
}

function clearHwAudition() {
  for (const url of Object.values(hwAuditionUrls)) URL.revokeObjectURL(url);
  hwAuditionUrls = {};
  const player = document.getElementById('hwAuditionPlayer');
  if (player) player.removeAttribute('src');
  document.getElementById('hwAuditionDownload')?.classList.add('hidden');
  const status = document.getElementById('hwAuditionStatus');
  if (status) status.textContent = 'No clip rendered';
}

function selectHwAuditionClip() {
  const mode = document.getElementById('hwAuditionMode').value;
  const url = hwAuditionUrls[mode];
  if (!url) return;
  const player = document.getElementById('hwAuditionPlayer');
  player.src = url;
  const download = document.getElementById('hwAuditionDownload');
  download.href = url;
  download.download = `heimdall-${mode}.wav`;
  download.classList.remove('hidden');
}

function stopHardwareSession() {
  stopHwMonitor();
  if (hwWs) { hwWs.close(); hwWs = null; }
  clearHardwarePresentation();
  hw3dActive = false;
  document.body.classList.remove('hardware-3d-active', 'hardware-split', 'hardware-3d-only');
  const hardwareVisible = mode === 'hardware';
  document.getElementById('hwScanlines').classList.add('hidden');
  document.getElementById('hwHeaderBar').classList.toggle('hidden', !hardwareVisible);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  beamGroup.visible = true;
  sourceGroup.visible = true;
  roomGroup.visible = true;
  const freezeButton = document.getElementById('btnHwPause');
  if (freezeButton) freezeButton.textContent = 'Freeze Display';
  setHwConnectionState('disconnected', 'Disconnected');
}

function hwSend(obj) {
  if (!hwTransportOpen()) {
    document.getElementById('hwStatus').textContent = 'No active Hardware link';
    return false;
  }
  hwWs.send(JSON.stringify(obj));
  return true;
}

function startHardwareSession() {
  stopHardwareSession();
  resizeHwCanvas();
  // Show hardware UI elements
  document.getElementById('hwScanlines').classList.toggle(
    'hidden', !document.getElementById('hwCrtToggle').checked);
  document.getElementById('hwHeaderBar').classList.remove('hidden');
  setHwConnectionState('connecting', 'Connecting to Hardware service...');

  const socket = new WebSocket(HW_WS_URL);
  hwWs = socket;
  socket.onopen = () => {
    if (hwWs !== socket) return;
    setHwConnectionState('online', 'Link open · requesting firmware metadata');
    addHwEvent('LINK', 'WebSocket connected');
    hwSend({ type: 'resume' });
    hwSend({ type: 'set_true_dir', az_deg: null, el_deg: null });
    hwSend({ type: 'request_init' });
  };
  socket.onmessage = (ev) => {
    if (hwWs !== socket) return;
    try { onHardwareMessage(JSON.parse(ev.data)); }
    catch (e) { console.warn('hw parse', e); }
  };
  socket.onclose = () => {
    if (hwWs !== socket) return;
    hwWs = null;
    stopHwMonitor();
    clearHardwarePresentation();
    if (mode === 'hardware') {
      setHwConnectionState('disconnected', 'Disconnected');
      addHwEvent('LINK', 'Transport disconnected', 'warning');
    }
  };
  socket.onerror = () => {
    if (hwWs !== socket) return;
    setHwConnectionState('fault', 'Hardware WebSocket unavailable');
    addHwEvent('ERROR', 'Hardware WebSocket unavailable', 'error');
  };
}

function onHardwareMessage(msg) {
  if (msg.type === 'init') {
    const previousConfig = hwInit?.configuration;
    hwInit = msg;
    const cfg = msg.configuration;
    const changed = cfg && (!previousConfig || previousConfig.rows !== cfg.rows ||
      previousConfig.columns !== cfg.columns || previousConfig.microphones !== cfg.microphones);
    if (changed) resetHwFreshness(cfg);
    setHwConnectionState('online', cfg
      ? `Connected · ${msg.transport} · ${cfg.rows} × ${cfg.columns} · ${cfg.microphones} mics`
      : `Connected · ${msg.transport} · waiting for firmware info`);
    if (cfg) document.getElementById('hwSector').max = cfg.sectors - 1;
    if (changed) addHwEvent('CONFIG', `${cfg.rows} × ${cfg.columns} · ${cfg.sectors} sectors · ${cfg.microphones} microphones`);
    updateHwGridSource(msg);
    updateHwReadiness(msg);
    return;
  }
  if (msg.type === 'frame') {
    const previousFrame = hwLastFrame;
    const telemetryAdvanced = msg.sequence !== previousFrame?.sequence;
    if (telemetryAdvanced) updateHwCellTimestamps(msg, previousFrame);
    hwLastFrame = msg;
    if (!msg.connected) {
      setHwConnectionState('fault', `Link fault · ${msg.transport}`);
    } else {
      setHwConnectionState('online', `${msg.transport} · ${msg.configuration ? 'ready' : 'waiting for metadata'}`);
    }
    if (telemetryAdvanced) {
      logHwRecord(msg);
      logHwStateTransitions(msg, previousFrame);
    }
    updateHwGridSource(msg);
    renderHwMetrics(msg);
    renderHwSolution(msg);
    updateHwReadiness(msg);
    if (msg.levels_db && msg.configuration) {
      hwNeedsRedraw = true;
      if (telemetryAdvanced) hwSonarTargetAngle += Math.PI / Math.max(1, msg.configuration.sectors);
      if (telemetryAdvanced && msg.last_record?.type === 'measurement') {
        hwPowerHistory.push({
          t: msg.timestamp_s,
          azimuth: msg.est_az_deg,
          elevation: msg.est_el_deg,
          level: hwFixedBeamLevel(msg) ?? msg.argmax_db,
        });
        if (hwPowerHistory.length > HW_HISTORY_MAX) hwPowerHistory.shift();
        renderHwTimeline2d();
      }
    }
    return;
  }
  if (msg.type === 'command_error') {
    const detail = String(msg.detail || 'unknown command error');
    document.getElementById('hwStatus').textContent = `Command error · ${detail}`;
    addHwEvent('ERROR', detail, 'error');
  }
}

function resizeHwCanvas() {
  const canvas2d = document.getElementById('hwBirdsEye');
  if (canvas2d) {
    const dpr = window.devicePixelRatio || 1;
    canvas2d.width = canvas2d.clientWidth * dpr;
    canvas2d.height = canvas2d.clientHeight * dpr;
  }
}

function resizeHardwareViews() {
  resizeHwCanvas();
  const canvas3d = document.getElementById('canvas3d');
  if (document.body.classList.contains('hardware-3d-active') && canvas3d.clientWidth > 0) {
    camera.aspect = canvas3d.clientWidth / Math.max(1, canvas3d.clientHeight);
    camera.updateProjectionMatrix();
    renderer.setSize(canvas3d.clientWidth, canvas3d.clientHeight);
    frameHardwareDome();
  }
  hwNeedsRedraw = true;
}

async function discoverHardwareTransport() {
  document.getElementById('hwHeaderBar').classList.remove('hidden');
  resizeHardwareViews();
  setHwConnectionState('connecting', 'Checking Hardware service...');
  try {
    const response = await fetch('http://127.0.0.1:8766/hw_status');
    if (!response.ok) throw new Error('Hardware service unavailable');
    const status = await response.json();
    if (status.available) {
      startHardwareSession();
    } else {
      setHwConnectionState('disconnected', 'Backend ready · select serial or emulator');
    }
  } catch (error) {
    setHwConnectionState('fault', 'Backend unavailable on 127.0.0.1:8766');
    addHwEvent('ERROR', error.message, 'error');
  }
}

function updateHwEmulatorModelUi() {
  const acoustic = document.getElementById('hwEmulatorModel').value === 'acoustic';
  document.getElementById('hwFastScenarioSettings').classList.toggle('hidden', acoustic);
  document.getElementById('hwAcousticScenarioSettings').classList.toggle('hidden', !acoustic);
  document.getElementById('btnHwEmulator').textContent = acoustic ? 'Acoustic' : 'Emulator';
  updateHwGridSource();
}

function updateHwGridSource(frame = hwLastFrame || hwInit) {
  const element = document.getElementById('hwGridSource');
  if (!element) return;
  if (frame?.grid_source === 'deployment_contract') {
    const cfg = frame.configuration;
    element.textContent = `Deployment parity · ${cfg?.rows ?? 7} × ${cfg?.columns ?? 7} / ${cfg?.sectors ?? 49} sectors · exact firmware delay table`;
  } else if (frame?.grid_source === 'exploratory_simulation') {
    const cfg = frame.configuration;
    element.textContent = `Exploratory acoustic grid · ${cfg?.rows ?? '?'} × ${cfg?.columns ?? '?'} / ${cfg?.sectors ?? '?'} sectors · simulated quantized delays, hardware unchanged`;
  } else if (frame?.grid_source === 'firmware_reported') {
    const cfg = frame.configuration;
    element.textContent = cfg
      ? `Firmware reported · ${cfg.rows} × ${cfg.columns} / ${cfg.sectors} sectors · change the generated beam table and reflash to modify`
      : 'Firmware reported · grid and FOV arrive from the device after connection';
  } else {
    const acoustic = document.getElementById('hwEmulatorModel').value === 'acoustic';
    element.textContent = acoustic
      ? 'Acoustic simulation grid · changes rebuild beam levels while reusing the room cache'
      : 'Fast emulator grid · editable below';
  }
}

function renderHwAcousticStatus(status) {
  const statusEl = document.getElementById('hwAcousticStatus');
  const progress = document.getElementById('hwAcousticProgress');
  const prepare = document.getElementById('btnHwAcousticPrepare');
  const cancel = document.getElementById('btnHwAcousticCancel');
  const state = status?.state || 'idle';
  progress.value = Number(status?.overall_percent ?? status?.percent ?? 0);
  prepare.disabled = state === 'preparing';
  cancel.disabled = state !== 'preparing';
  if (state === 'ready') {
    hwAcousticCacheId = status.cache_id;
    const rt60 = Number.isFinite(status.measured_rt60_s)
      ? ` · RT60 ${status.measured_rt60_s.toFixed(2)} s` : '';
    statusEl.textContent = `Ready · ${Number(status.preparation_seconds || 0).toFixed(1)} s${rt60} · uncalibrated`;
  } else if (state === 'preparing') {
    const eta = Number.isFinite(status.eta_s) ? ` · ETA ${Math.ceil(status.eta_s)} s` : '';
    statusEl.textContent = `${status.stage || 'preparing'} · ${Number(status.overall_percent ?? status.percent ?? 0).toFixed(0)}%${eta}`;
  } else if (state === 'failed') {
    hwAcousticCacheId = null;
    statusEl.textContent = `Failed · ${status.error || 'unknown error'}`;
  } else if (state === 'canceled') {
    hwAcousticCacheId = null;
    statusEl.textContent = 'Preparation canceled';
  } else {
    statusEl.textContent = 'No acoustic cache prepared';
  }
}

async function pollHwAcousticStatus() {
  if (hwAcousticPollTimer) clearTimeout(hwAcousticPollTimer);
  try {
    const response = await fetch('http://127.0.0.1:8766/hw_acoustic_status');
    if (!response.ok) throw new Error('status unavailable');
    const status = await response.json();
    renderHwAcousticStatus(status);
    if (status.state === 'preparing') {
      hwAcousticPollTimer = setTimeout(pollHwAcousticStatus, 500);
    }
  } catch (error) {
    document.getElementById('hwAcousticStatus').textContent = `Backend unavailable · ${error.message}`;
  }
}

// -- 2D polar curve renderer (smooth closed polygon, like beam_scan_gui.py) --
function drawHwBirdsEye(frame) {
  const canvas = document.getElementById('hwBirdsEye');
  if (!canvas || canvas.style.display === 'none') return;
  // Resize canvas to match CSS size
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
  }
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#040810';
  ctx.fillRect(0, 0, W, H);

  if (frame.levels_db && frame.configuration) {
    drawHwSectorHeatmap(ctx, W, H, frame);
    return;
  }

  // Draw dot grid on canvas
  ctx.fillStyle = 'rgba(217,75,0,0.08)';
  for (let gx = 0; gx < W; gx += 16) {
    for (let gy = 0; gy < H; gy += 16) {
      ctx.fillRect(gx, gy, 1, 1);
    }
  }

  const cx = W / 2, cy = H / 2;
  const maxR = Math.min(W, H) * 0.42;

  const scan = frame.beam_scan;
  if (!scan || !scan.powers_db) return;

  const powers = scan.powers_db;
  const angles = scan.angles_deg;
  const N = angles.length;
  const dbFloor = parseFloat(document.getElementById('hwDbFloor').value) || -40;

  // Normalize to [0, 1] using relative peak-to-min range (like beam_scan_gui.py)
  const pMax = Math.max(...powers);
  const pMin = Math.min(...powers);
  const pRange = Math.max(pMax - pMin, 0.1);  // avoid div-by-zero
  const vals = powers.map(p => Math.max(0, Math.min(1, (p - pMin) / pRange)));

  // Draw concentric rings with dB labels
  ctx.strokeStyle = 'rgba(217,75,0,0.3)';
  ctx.lineWidth = 1;
  ctx.shadowColor = 'rgba(217,75,0,0.4)';
  ctx.shadowBlur = 0;
  const dbLabels = ['-10', '-20', '-30', '-40'];
  for (let i = 0; i < 4; i++) {
    const r = (i + 1) * 0.25;
    ctx.beginPath();
    ctx.arc(cx, cy, maxR * r, 0, Math.PI * 2);
    ctx.stroke();
    // dB label
    ctx.fillStyle = 'rgba(255,180,100,0.7)';
    ctx.font = '11px Share Tech Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(dbLabels[3 - i] + ' dB', cx + maxR * r + 4, cy - 4);
  }

  // Noise floor dashed circle
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = 'rgba(255,100,0,0.5)';
  ctx.beginPath();
  ctx.arc(cx, cy, maxR * 0.15, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.shadowBlur = 0;

  // Draw radial lines every 30 deg
  ctx.strokeStyle = 'rgba(217,75,0,0.3)';
  ctx.shadowColor = 'rgba(217,75,0,0.3)';
  ctx.shadowBlur = 8;
  for (let a = 0; a < 360; a += 30) {
    const ar = a * Math.PI / 180;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + maxR * 1.05 * Math.cos(ar), cy - maxR * 1.05 * Math.sin(ar));
    ctx.stroke();
  }
  ctx.shadowBlur = 0;

  // Draw angle labels with CRT glow
  ctx.fillStyle = 'rgba(255,200,140,0.95)';
  ctx.shadowColor = 'rgba(217,75,0,0.6)';
  ctx.shadowBlur = 10;
  ctx.font = `${Math.max(12, Math.round(W/60))}px Share Tech Mono, monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let a = 0; a < 360; a += 30) {
    const ar = a * Math.PI / 180;
    ctx.fillText(`${a}°`, cx + maxR * 1.12 * Math.cos(ar), cy - maxR * 1.12 * Math.sin(ar));
  }
  ctx.shadowBlur = 0;

  // Interpolate 4x for smoothness (same as beam_scan_gui.py)
  const interpFactor = 4;
  const Ni = N * interpFactor;
  const valsWrapped = [...vals, vals[0]];
  const anglesWrapped = [...angles.map(a => a * Math.PI / 180), 2 * Math.PI];
  const interpAngles = [];
  const interpVals = [];
  for (let i = 0; i < Ni; i++) {
    const a = (i / Ni) * 2 * Math.PI;
    interpAngles.push(a);
    // Linear interpolation in the wrapped array
    let idx = 0;
    for (let j = 0; j < anglesWrapped.length - 1; j++) {
      if (a >= anglesWrapped[j] && a <= anglesWrapped[j+1]) { idx = j; break; }
    }
    const t = (anglesWrapped[idx+1] - anglesWrapped[idx]) > 1e-9
      ? (a - anglesWrapped[idx]) / (anglesWrapped[idx+1] - anglesWrapped[idx])
      : 0;
    interpVals.push(valsWrapped[idx] * (1 - t) + valsWrapped[idx + 1] * t);
  }

  // Draw afterglow trail (phosphor decay - older scans)
  for (let hi = 0; hi < hwAfterglowHistory.length; hi++) {
    const age = hwAfterglowHistory.length - hi;
    const opacity = 0.5 / age;
    const oldVals = hwAfterglowHistory[hi];
    ctx.beginPath();
    for (let i = 0; i <= Ni; i++) {
      const idx2 = i % Ni;
      const a2 = interpAngles[idx2];
      const r2 = maxR * (0.15 + 0.85 * oldVals[idx2]);
      const x2 = cx + r2 * Math.cos(a2);
      const y2 = cy - r2 * Math.sin(a2);
      if (i === 0) ctx.moveTo(x2, y2); else ctx.lineTo(x2, y2);
    }
    ctx.closePath();
    ctx.shadowColor = `rgba(217,75,0,${opacity * 0.6})`;
    ctx.shadowBlur = 6 + age * 2;
    ctx.strokeStyle = `rgba(217,75,0,${opacity})`;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = `rgba(217,75,0,${opacity * 0.1})`;
    ctx.fill();
  }
  ctx.shadowBlur = 0;

  // Draw filled polar curve (current scan) with color gradient
  // First: filled shape as dim base
  ctx.beginPath();
  for (let i = 0; i <= Ni; i++) {
    const idx = i % Ni;
    const a = interpAngles[idx];
    const r = maxR * (0.15 + 0.85 * interpVals[idx]);
    const x = cx + r * Math.cos(a);
    const y = cy - r * Math.sin(a);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = 'rgba(217,75,0,0.08)';
  ctx.fill();

  // Per-segment gradient stroke (colorLow→colorMid→colorHigh mapped to val)
  function hw2dColor(val, alpha) {
    let r, g, b;
    if (val < 0.5) {
      const t = val * 2;
      r = Math.round(5 + (217 - 5) * t);
      g = Math.round(5 + (75 - 5) * t);
      b = Math.round(5 + (0 - 5) * t);
    } else {
      const t = (val - 0.5) * 2;
      r = Math.round(217 + (255 - 217) * t);
      g = Math.round(75 + (255 - 75) * t);
      b = Math.round(0 + (255 - 0) * t);
    }
    return `rgba(${r},${g},${b},${alpha})`;
  }
  ctx.lineWidth = 2.5;
  for (let i = 0; i < Ni; i++) {
    const idx0 = i, idx1 = (i + 1) % Ni;
    const a0 = interpAngles[idx0], a1 = interpAngles[idx1];
    const v0 = interpVals[idx0], v1 = interpVals[idx1];
    const r0 = maxR * (0.15 + 0.85 * v0), r1 = maxR * (0.15 + 0.85 * v1);
    const x0 = cx + r0 * Math.cos(a0), y0 = cy - r0 * Math.sin(a0);
    const x1 = cx + r1 * Math.cos(a1), y1 = cy - r1 * Math.sin(a1);
    const vMid = (v0 + v1) * 0.5;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.strokeStyle = hw2dColor(vMid, 0.9);
    ctx.shadowColor = hw2dColor(vMid, 0.5);
    ctx.shadowBlur = 6;
    ctx.stroke();
  }
  ctx.shadowBlur = 0;

  // Store current interpolated vals for afterglow
  hwAfterglowHistory.push(interpVals.slice());
  if (hwAfterglowHistory.length > 5) hwAfterglowHistory.shift();

  // Draw peak dot
  const peakIdx = scan.argmax_idx;
  const peakAngle = angles[peakIdx] * Math.PI / 180;
  const peakR = maxR * (0.15 + 0.85 * vals[peakIdx]);
  const peakX = cx + peakR * Math.cos(peakAngle);
  const peakY = cy - peakR * Math.sin(peakAngle);
  ctx.fillStyle = '#ff6600';
  ctx.beginPath();
  ctx.arc(peakX, peakY, 7, 0, Math.PI * 2);
  ctx.fill();

  // Draw peak direction line from center
  ctx.strokeStyle = 'rgba(255,102,0,0.8)';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + maxR * 1.1 * Math.cos(peakAngle), cy - maxR * 1.1 * Math.sin(peakAngle));
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw true direction if set
  if (frame.true_az_deg != null) {
    const trueRad = frame.true_az_deg * Math.PI / 180;
    ctx.strokeStyle = 'rgba(68,204,68,0.8)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + maxR * 0.85 * Math.cos(trueRad), cy - maxR * 0.85 * Math.sin(trueRad));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.shadowBlur = 0;

  // Corner crosshairs / reticle marks
  const reticleLen = 20;
  const reticleInset = 30;
  ctx.strokeStyle = 'rgba(217,75,0,0.4)';
  ctx.lineWidth = 1.5;
  // Top-left
  ctx.beginPath();
  ctx.moveTo(reticleInset, reticleInset + reticleLen); ctx.lineTo(reticleInset, reticleInset); ctx.lineTo(reticleInset + reticleLen, reticleInset);
  ctx.stroke();
  // Top-right
  ctx.beginPath();
  ctx.moveTo(W - reticleInset - reticleLen, reticleInset); ctx.lineTo(W - reticleInset, reticleInset); ctx.lineTo(W - reticleInset, reticleInset + reticleLen);
  ctx.stroke();
  // Bottom-left
  ctx.beginPath();
  ctx.moveTo(reticleInset, H - reticleInset - reticleLen); ctx.lineTo(reticleInset, H - reticleInset); ctx.lineTo(reticleInset + reticleLen, H - reticleInset);
  ctx.stroke();
  // Bottom-right
  ctx.beginPath();
  ctx.moveTo(W - reticleInset - reticleLen, H - reticleInset); ctx.lineTo(W - reticleInset, H - reticleInset); ctx.lineTo(W - reticleInset, H - reticleInset - reticleLen);
  ctx.stroke();
}

function hwHeatColor(value) {
  const clamped = Math.max(0, Math.min(1, value));
  const low = new THREE.Color(0x08131a);
  const mid = new THREE.Color(0x238ba3);
  const high = new THREE.Color(0xf2c96d);
  const color = clamped < 0.5
    ? low.lerp(mid, clamped * 2)
    : mid.lerp(high, (clamped - 0.5) * 2);
  return `rgb(${Math.round(color.r * 255)},${Math.round(color.g * 255)},${Math.round(color.b * 255)})`;
}

function hwLevelNormalized(level) {
  if (!Number.isFinite(level)) return 0;
  const floorDb = parseFloat(document.getElementById('hwDbFloor').value);
  const floor = Number.isFinite(floorDb) ? floorDb : -40;
  return Math.max(0, Math.min(1, (level - floor) / Math.max(1, -floor)));
}

function hwFreshnessFactor(row, column) {
  const age = hwCellAgeMs(row, column);
  if (!Number.isFinite(age)) return 0.25;
  if (age <= 2000) return 1;
  return Math.max(0.32, 1 - (age - 2000) / 28000 * 0.68);
}

function drawHwSectorHeatmap(ctx, W, H, frame) {
  const cfg = frame.configuration;
  const azimuths = frame.azimuth_deg;
  const elevations = frame.elevation_deg;
  const levels = frame.levels_db;
  const marginLeft = Math.max(64, W * 0.09);
  const marginRight = 34;
  const marginTop = 46;
  const marginBottom = 62;
  const gridWidth = W - marginLeft - marginRight;
  const gridHeight = H - marginTop - marginBottom;
  const cellWidth = gridWidth / cfg.columns;
  const cellHeight = gridHeight / cfg.rows;
  hwGridLayout = { left: marginLeft, top: marginTop, width: gridWidth, height: gridHeight,
    rows: cfg.rows, columns: cfg.columns };

  ctx.fillStyle = '#040810';
  ctx.fillRect(0, 0, W, H);
  ctx.font = `${Math.max(10, Math.min(14, cellWidth * 0.15))}px Share Tech Mono, monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  let strongest = null;
  for (let row = 0; row < cfg.rows; row++) {
    for (let column = 0; column < cfg.columns; column++) {
      const level = levels[row][column];
      if (Number.isFinite(level) && (!strongest || level > strongest.level)) {
        strongest = { row, column, level };
      }
      const displayRow = cfg.rows - 1 - row;
      const x = marginLeft + column * cellWidth;
      const y = marginTop + displayRow * cellHeight;
      const normalized = hwLevelNormalized(level);
      const age = hwCellAgeMs(row, column);
      const freshness = hwFreshnessFactor(row, column);
      ctx.globalAlpha = freshness;
      ctx.fillStyle = Number.isFinite(level) ? hwHeatColor(normalized) : '#11151a';
      ctx.fillRect(x + 1, y + 1, cellWidth - 2, cellHeight - 2);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = 'rgba(94,142,165,0.28)';
      ctx.strokeRect(x + 1, y + 1, cellWidth - 2, cellHeight - 2);
      if (Number.isFinite(age) && age > HW_STALE_MS) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(x + 1, y + 1, cellWidth - 2, cellHeight - 2);
        ctx.clip();
        ctx.strokeStyle = 'rgba(185,204,214,0.24)';
        ctx.lineWidth = 1;
        const spacing = Math.max(8, Math.min(16, Math.min(cellWidth, cellHeight) / 3));
        for (let offset = -cellHeight; offset < cellWidth; offset += spacing) {
          ctx.beginPath();
          ctx.moveTo(x + offset, y + cellHeight);
          ctx.lineTo(x + offset + cellHeight, y);
          ctx.stroke();
        }
        ctx.restore();
      }
      if (cellWidth > 54 && cellHeight > 32 && Number.isFinite(level)) {
        ctx.globalAlpha = Math.max(0.55, freshness);
        ctx.fillStyle = normalized > 0.65 ? '#071014' : '#d9edf4';
        ctx.fillText(level.toFixed(1), x + cellWidth / 2, y + cellHeight / 2);
        ctx.globalAlpha = 1;
      }
    }
  }

  ctx.fillStyle = '#b9d6e2';
  for (let column = 0; column < cfg.columns; column++) {
    if (!Number.isFinite(azimuths[column])) continue;
    const x = marginLeft + (column + 0.5) * cellWidth;
    ctx.fillText(`${azimuths[column].toFixed(1)}°`, x, H - marginBottom / 2);
  }
  ctx.textAlign = 'right';
  for (let row = 0; row < cfg.rows; row++) {
    if (!Number.isFinite(elevations[row])) continue;
    const displayRow = cfg.rows - 1 - row;
    const y = marginTop + (displayRow + 0.5) * cellHeight;
    ctx.fillText(`${elevations[row].toFixed(1)}°`, marginLeft - 10, y);
  }
  ctx.textAlign = 'center';
  ctx.fillStyle = '#7e96a4';
  ctx.fillText('AZIMUTH', marginLeft + gridWidth / 2, H - 10);
  ctx.save();
  ctx.translate(16, marginTop + gridHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('ELEVATION', 0, 0);
  ctx.restore();

  const drawMarker = (marker, color, width, inset) => {
    if (!marker) return;
    const displayRow = cfg.rows - 1 - marker.row;
    const x = marginLeft + marker.column * cellWidth;
    const y = marginTop + displayRow * cellHeight;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.strokeRect(x + inset, y + inset, cellWidth - inset * 2, cellHeight - inset * 2);
  };
  drawMarker(strongest, '#ffffff', 1.5, 5);
  drawMarker(frame.last_steer, '#6bcfff', 2.5, 3);
  drawMarker(frame.target, '#f2c96d', 3.5, 1);

  const truth = frame.emulator_truth;
  if (truth) {
    const azimuthEdges = hwAngularEdges(azimuths, 10);
    const elevationEdges = hwAngularEdges(elevations, 10);
    const azimuthRatio = (truth.azimuth_deg - azimuthEdges[0]) /
      (azimuthEdges.at(-1) - azimuthEdges[0]);
    const elevationRatio = (truth.elevation_deg - elevationEdges[0]) /
      (elevationEdges.at(-1) - elevationEdges[0]);
    if (azimuthRatio >= 0 && azimuthRatio <= 1 && elevationRatio >= 0 && elevationRatio <= 1) {
      const x = marginLeft + azimuthRatio * gridWidth;
      const y = marginTop + (1 - elevationRatio) * gridHeight;
      ctx.save();
      ctx.strokeStyle = truth.source_active === false ? '#f2c96d' : '#8dffac';
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.moveTo(x - 14, y); ctx.lineTo(x + 14, y);
      ctx.moveTo(x, y - 14); ctx.lineTo(x, y + 14);
      ctx.stroke();
      ctx.font = '10px Share Tech Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText('TRUE', x + 12, y - 12);
      ctx.restore();
    }
  }
}

// -- Metrics panel --
function computeHwEngMetrics(powers, angles, peakIdx) {
  const N = powers.length;
  const peak = powers[peakIdx];
  const mean = powers.reduce((a, b) => a + b, 0) / N;

  // Directivity Index: 10*log10(peak_linear / mean_linear) ≈ peak_dB - mean_dB for relative
  const DI = (peak - mean).toFixed(1);

  // -3 dB beamwidth: count contiguous angles within 3 dB of peak
  let bwCount = 0;
  for (let off = 0; off < N; off++) {
    if (powers[(peakIdx + off) % N] >= peak - 3) bwCount++;
    else break;
  }
  for (let off = 1; off < N; off++) {
    if (powers[(peakIdx - off + N) % N] >= peak - 3) bwCount++;
    else break;
  }
  const step = 360 / N;
  const bw3db = Math.round(bwCount * step);

  // Sidelobe level: find highest local max that's > 1 beamwidth away from peak
  const minSep = Math.max(3, Math.ceil(bwCount * 1.5)); // min separation from peak
  let sll = -Infinity;
  for (let i = 0; i < N; i++) {
    const dist = Math.min(Math.abs(i - peakIdx), N - Math.abs(i - peakIdx));
    if (dist < minSep) continue;
    if (powers[i] > sll) sll = powers[i];
  }
  const sllRel = sll > -Infinity ? (sll - peak).toFixed(1) : '—';

  // Front-to-back ratio: peak vs. 180° opposite
  const backIdx = (peakIdx + Math.round(N / 2)) % N;
  const fbr = (peak - powers[backIdx]).toFixed(1);

  return { DI, bw3db, sllRel, fbr };
}

function renderHwMetrics(frame) {
  const el = document.getElementById('hwMetrics');
  if (!el) return;
  if (frame.configuration) {
    const rate = Number.isFinite(frame.scan_rate_hz) ? `${frame.scan_rate_hz.toFixed(2)} Hz` : '—';
    const trackRate = Number.isFinite(frame.target_update_rate_hz)
      ? `${frame.target_update_rate_hz.toFixed(1)} Hz` : '—';
    const flight = frame.emulator_flight_profile
      ? `<br>Flight: <span class="val">${frame.emulator_flight_profile}</span> · ${Number(frame.emulator_flight_speed).toFixed(1)}×`
      : '';
    const acoustic = frame.acoustic_cache_id
      ? `<br>Acoustic: <span class="val">${frame.acoustic_scenario}</span> · RT60 ${Number(frame.acoustic_measured_rt60_s).toFixed(2)} s · ${frame.acoustic_calibrated ? 'calibrated' : 'uncalibrated'}` +
        `<br>Detector: <span class="val">intended 1–4 kHz FIR</span> · SigmaStudio export mismatch`
      : '';
    const margin = Number.isFinite(frame.margin_db) ? `${frame.margin_db.toFixed(1)} dB` : '—';
    const azimuthEdges = frame.azimuth_deg.length ? hwAngularEdges(frame.azimuth_deg, 10) : [];
    const elevationEdges = frame.elevation_deg.length ? hwAngularEdges(frame.elevation_deg, 10) : [];
    const azimuthFov = azimuthEdges.length
      ? `${azimuthEdges[0].toFixed(1)}°..${azimuthEdges.at(-1).toFixed(1)}°` : '—';
    const elevationFov = elevationEdges.length
      ? `${elevationEdges[0].toFixed(1)}°..${elevationEdges.at(-1).toFixed(1)}°` : '—';
    const gridSource = frame.grid_source === 'deployment_contract' ? 'deployment contract'
      : frame.grid_source === 'exploratory_simulation' ? 'exploratory acoustic'
      : frame.grid_source === 'firmware_reported' ? 'firmware reported' : 'fast emulator';
    el.innerHTML =
      `Grid <span class="val">${frame.configuration.rows} × ${frame.configuration.columns}</span> · ` +
      `${frame.configuration.sectors} sectors · ${frame.configuration.microphones} mics · ${gridSource}<br>` +
      `FOV: az ${azimuthFov} · el ${elevationFov}<br>` +
      `Peak-to-next: <span class="val">${margin}</span><br>` +
      `Last full pass: <span class="val">${rate}</span><br>` +
      `Track updates: <span class="val">${trackRate}</span> · host observed<br>` +
      `Pass ${frame.scan_count} · protocol warnings ${frame.protocol_errors}${flight}${acoustic}`;
    const stats = document.getElementById('hwStatsLine');
    if (stats) {
      const peak = Number.isFinite(frame.argmax_db) ? `${frame.argmax_db.toFixed(1)} dBFS` : '—';
      stats.textContent = `PEAK ${peak}  |  TRACK UPDATE ${trackRate}  |  FULL SCAN ${rate}  |  PASS ${frame.scan_count}`;
    }
    return;
  }
  const az = String(frame.est_az_deg).padStart(3, '\u00a0');
  const db = String(frame.argmax_db).padStart(6, '\u00a0');
  const margin = String(frame.margin_db).padStart(5, '\u00a0');
  const rate = String(frame.scan_rate_hz).padStart(4, '\u00a0');
  const fnum = String(frame.frame_idx).padStart(5, '\u00a0');
  const t = String(frame.t_sim_s).padStart(7, '\u00a0');

  // Engineering metrics from beam scan
  let engLine = '';
  if (frame.beam_scan && frame.beam_scan.powers_db) {
    const m = computeHwEngMetrics(frame.beam_scan.powers_db, frame.beam_scan.angles_deg, frame.beam_scan.argmax_idx);
    engLine =
      `<br><span style="color:#998070">──────────────────────</span><br>` +
      `BW<sub>-3dB</sub>: <span style="color:#ffcc66">${m.bw3db}°</span> · ` +
      `DI: <span style="color:#ffcc66">${m.DI} dB</span><br>` +
      `SLL: <span style="color:#b89878">${m.sllRel} dB</span> · ` +
      `F/B: <span style="color:#b89878">${m.fbr} dB</span>`;
  }

  el.innerHTML =
    `<span style="color:#d94b00;font-weight:600">HARDWARE BEAM SCAN</span><br>` +
    `Lock: <span style="color:#ff8833">${az}°</span> ` +
    `(${db} dB, Δ${margin} dB)<br>` +
    `Scan rate: <span style="color:#ffcc66">${rate} Hz</span><br>` +
    `Frame #${fnum} · t=${t}s` + engLine;
}

function hwFixedBeamLevel(frame) {
  const steer = frame?.last_steer;
  if (!steer || !frame.levels_db?.[steer.row]) return null;
  const level = frame.levels_db[steer.row][steer.column];
  return Number.isFinite(level) ? level : null;
}

function renderHwTimeline2d() {
  const canvas = document.getElementById('hwTimelineCanvas');
  if (!canvas || hwPowerHistory.length < 2) return;
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#0a0604';
  ctx.fillRect(0, 0, W, H);
  const history = hwPowerHistory.slice(-Math.max(2, Math.floor(W)));
  const azValues = hwLastFrame?.azimuth_deg || [];
  const elValues = hwLastFrame?.elevation_deg || [];
  const azMin = Math.min(...azValues), azMax = Math.max(...azValues);
  const elMin = Math.min(...elValues), elMax = Math.max(...elValues);

  ctx.lineWidth = 1.5;
  for (const [key, minimum, maximum, color] of [
    ['azimuth', azMin, azMax, '#ff6600'],
    ['elevation', elMin, elMax, '#5fd28a'],
  ]) {
    ctx.strokeStyle = color;
    ctx.beginPath();
    history.forEach((item, index) => {
      const value = item[key];
      if (!Number.isFinite(value) || !Number.isFinite(minimum) || maximum === minimum) return;
      const x = index / Math.max(1, history.length - 1) * W;
      const y = H - 14 - ((value - minimum) / (maximum - minimum)) * (H - 28);
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  const levelValues = history.map(item => item.level).filter(Number.isFinite);
  if (levelValues.length > 1) {
    const levelMax = Math.max(...levelValues);
    const levelMin = Math.min(...levelValues);
    const levelRange = Math.max(3, levelMax - levelMin);
    ctx.strokeStyle = '#ffdd66';
    ctx.lineWidth = hwMonitoring ? 2.5 : 1.2;
    ctx.beginPath();
    let started = false;
    history.forEach((item, index) => {
      if (!Number.isFinite(item.level)) return;
      const x = index / Math.max(1, history.length - 1) * W;
      const y = H - 14 - ((item.level - levelMin) / levelRange) * (H - 28);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  ctx.font = '10px Share Tech Mono, monospace';
  ctx.textAlign = 'left';
  ctx.fillStyle = '#ff6600';
  ctx.fillText('AZ', 8, 11);
  ctx.fillStyle = '#5fd28a';
  ctx.fillText('EL', 30, 11);
  ctx.fillStyle = '#ffdd66';
  ctx.fillText(hwMonitoring ? 'FIXED BEAM LEVEL' : 'LEVEL', 52, 11);
}

// -- Timeline strip --
function renderHwTimeline() {
  const canvas = document.getElementById('hwTimelineCanvas');
  if (!canvas || hwPowerHistory.length < 2) return;
  if (canvas.width !== canvas.clientWidth) canvas.width = canvas.clientWidth;
  if (canvas.height !== canvas.clientHeight) canvas.height = canvas.clientHeight;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#0a0604';
  ctx.fillRect(0, 0, W, H);

  // Horizontal waterfall: X = angle (0-360), Y = time (newest at top, scrolls down)
  const nHistory = Math.min(hwPowerHistory.length, H);
  const angleAxisH = 12; // reserved for top labels

  for (let hi = 0; hi < nHistory; hi++) {
    const row = hwPowerHistory[hwPowerHistory.length - 1 - hi];
    const y = angleAxisH + hi;
    const nAngles = row.powers.length;
    const pMax = Math.max(...row.powers);
    const pMin = Math.min(...row.powers);
    const pRange = Math.max(pMax - pMin, 0.1);

    for (let ai = 0; ai < nAngles; ai++) {
      const val = Math.max(0, Math.min(1, (row.powers[ai] - pMin) / pRange));
      const x = (ai / nAngles) * W;
      const colW = Math.ceil(W / nAngles);

      let r, g, b;
      if (val < 0.5) {
        const t = val * 2;
        r = Math.floor(5 + t * 212);
        g = Math.floor(5 + t * 70);
        b = Math.floor(5 + t * -5);
      } else {
        const t = (val - 0.5) * 2;
        r = Math.floor(217 + t * 38);
        g = Math.floor(75 + t * 180);
        b = Math.floor(0 + t * 255);
      }
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(x, y, colW, 1);
    }
  }

  // Angle axis labels (top)
  ctx.fillStyle = 'rgba(255,180,100,0.8)';
  ctx.font = '9px Share Tech Mono, monospace';
  ctx.textAlign = 'center';
  for (let a = 0; a <= 360; a += 90) {
    const x = (a / 360) * W;
    ctx.fillText(a + '°', x, 9);
  }

  // Update stats line
  const last = hwPowerHistory[hwPowerHistory.length - 1];
  if (last) {
    const powers = last.powers;
    const peak = Math.max(...powers);
    const mean = powers.reduce((a, b) => a + b, 0) / powers.length;
    const snr = (peak - mean).toFixed(1);
    const peakIdx = powers.indexOf(peak);
    const nAngles = powers.length;
    // Estimate -3dB beamwidth
    const halfPower = peak - 3;
    let bwCount = 0;
    for (let i = 0; i < nAngles; i++) {
      if (powers[i] >= halfPower) bwCount++;
    }
    const bw3db = Math.round(bwCount * (360 / nAngles));
    // Angular rate (degrees per second between last two frames)
    let rate = 0;
    if (hwPowerHistory.length >= 2) {
      const prev = hwPowerHistory[hwPowerHistory.length - 2];
      const prevPeakIdx = prev.powers.indexOf(Math.max(...prev.powers));
      const dt = last.t - prev.t;
      if (dt > 0) {
        let dAngle = (peakIdx - prevPeakIdx) * (360 / nAngles);
        if (dAngle > 180) dAngle -= 360;
        if (dAngle < -180) dAngle += 360;
        rate = (dAngle / dt).toFixed(0);
      }
    }
    const statsEl = document.getElementById('hwStatsLine');
    if (statsEl) {
      statsEl.innerHTML = `<span style="color:#ff6600">PEAK: ${Math.round(peakIdx * 360 / nAngles)}°</span> | ` +
        `<span style="color:#ffcc66">PWR: ${peak.toFixed(1)} dB</span> | ` +
        `SNR: ${snr} dB | BW₃dB: ${bw3db}° | ` +
        `<span style="color:#ffaa44">Δ: ${rate > 0 ? '+' : ''}${rate}°/s</span>`;
    }
  }
}

// -- Sonar sweep indicator --
let hwSonarAngle = 0;
function renderHwSonar() {
  const canvas = document.getElementById('hwSonar');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2, r = 4.5;
  const angle = hwSonarAngle % (2 * Math.PI);
  ctx.clearRect(0, 0, W, H);
  // Ring
  ctx.strokeStyle = 'rgba(217,75,0,0.4)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();
  // Sweep line
  ctx.strokeStyle = '#ffaa44';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + r * Math.cos(angle), cy + r * Math.sin(angle));
  ctx.stroke();
  // Glow dot at tip
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(cx + r * Math.cos(hwSonarAngle), cy + r * Math.sin(hwSonarAngle), 2.5, 0, Math.PI * 2);
  ctx.fill();
}

// -- Header bar time update --
function updateHwHeader(frame) {
  const timeEl = document.getElementById('hbTime');
  const arrayEl = document.getElementById('hbArray');
  const portEl = document.getElementById('hbPort');
  const statusEl = document.getElementById('hbStatus');
  if (timeEl) timeEl.textContent = new Date().toISOString().slice(11, 19) + 'Z';
  if (arrayEl && frame.configuration) arrayEl.textContent = `${frame.configuration.microphones} MIC · ${frame.configuration.rows}×${frame.configuration.columns}`;
  if (portEl) portEl.textContent = frame.transport || '--';
  if (statusEl) {
    const modeLabel = hwMonitoring ? 'FIXED BEAM'
      : frame.target ? 'TRACKING'
      : ({F: 'FULL SWEEP', C: 'CONTINUOUS', G: 'SEARCHING', IDLE: 'READY'}[frame.firmware_mode] || frame.firmware_mode || 'READY');
    statusEl.textContent = hwPaused ? `DISPLAY FROZEN · ${modeLabel}` : modeLabel;
  }
}

function updateHwReadiness(frame) {
  updateHwHeader(frame || {});
  const ages = hwCellTimestamps.flat().map(timestamp => Number.isFinite(timestamp)
    ? Math.max(0, Date.now() - timestamp) : null).filter(Number.isFinite);
  const age = ages.length ? Math.min(...ages) : null;
  const ageEl = document.getElementById('hbAge');
  const warningsEl = document.getElementById('hbWarnings');
  if (ageEl) ageEl.textContent = formatHwAge(age);
  if (warningsEl) {
    const warnings = frame?.protocol_errors || 0;
    warningsEl.textContent = warnings;
    warningsEl.style.color = warnings > 0 ? 'var(--hw-fault)' : '';
  }
  syncHwControls();
}

// -- 3D beam pattern mesh (torus or sphere, selectable) --
function buildHardwareRingMesh(powers_db, angles_deg, dispRadius, opacity) {
  hwRingGroup.clear();
  if (!powers_db || !angles_deg || angles_deg.length === 0) return;

  const N = angles_deg.length;
  const shape = document.getElementById('hw3dShape')?.value || 'torus';

  // ── Spatial smoothing: 5-tap circular moving average ──
  const smoothed = new Array(N);
  const K = 2;
  for (let i = 0; i < N; i++) {
    let sum = 0;
    for (let k = -K; k <= K; k++) sum += powers_db[(i + k + N) % N];
    smoothed[i] = sum / (2 * K + 1);
  }

  // ── Dynamic range gating ──
  const pMax = Math.max(...smoothed);
  const pMin = Math.min(...smoothed);
  const pRange = pMax - pMin;
  const gateThreshold = 3.0;
  const compression = Math.min(1.0, Math.max(0.0, (pRange - 0.5) / gateThreshold));
  const effectiveRange = Math.max(pRange, 0.1);
  const valsRaw = smoothed.map(p => {
    const raw = (p - pMin) / effectiveRange;
    return raw * compression + 0.5 * (1 - compression);
  });

  // 4x azimuth upsampling for smooth geometry (72 -> 288 vertices)
  const UPSAMPLE = 4;
  const M = N * UPSAMPLE;
  const vals = new Array(M);
  const interpAngles = new Array(M);
  for (let i = 0; i < M; i++) {
    const t = i / M;
    const srcF = t * N;
    const srcI = Math.floor(srcF) % N;
    const srcJ = (srcI + 1) % N;
    const frac = srcF - Math.floor(srcF);
    vals[i] = Math.pow(valsRaw[srcI] * (1 - frac) + valsRaw[srcJ] * frac, 1.0);
    interpAngles[i] = t * 2 * Math.PI;
  }

  const geometry = new THREE.BufferGeometry();
  const vertices = [];
  const colors = [];
  const indices = [];

  if (shape === 'torus') {
    // ── Torus: inner ring fixed, only outer side expands with power ──
    const nTube = 16;
    const R_major = dispRadius * 0.4;
    const r_fixed = dispRadius * 0.1; // small fixed inner tube radius

    for (let ti = 0; ti <= nTube; ti++) {
      const theta = (ti / nTube) * 2 * Math.PI;
      const cosT = Math.cos(theta), sinT = Math.sin(theta);
      // Expansion factor: 0 on inner side (cosT=-1), full on outer side (cosT=+1)
      const expand = Math.max(0, (cosT + 1) * 0.5); // 0..1

      for (let ai = 0; ai <= M; ai++) {
        const aIdx = ai % M;
        const phi = interpAngles[aIdx];
        const val = vals[aIdx];
        // Inner ring: r_fixed. Outer ring: r_fixed + power-driven expansion
        const r_tube = r_fixed + expand * dispRadius * 0.3 * val;

        const x = (R_major + r_tube * cosT) * Math.cos(phi);
        const z = (R_major + r_tube * cosT) * Math.sin(phi);
        const y = r_tube * sinT;
        vertices.push(x, y, z);

        const c = new THREE.Color();
        if (val < 0.5) c.copy(colorLow).lerp(colorMid, val * 2);
        else c.copy(colorMid).lerp(colorHigh, (val - 0.5) * 2);
        colors.push(c.r, c.g, c.b);
      }
    }

    const rowLen = M + 1;
    for (let ti = 0; ti < nTube; ti++) {
      for (let ai = 0; ai < M; ai++) {
        const a = ti * rowLen + ai;
        const b = a + 1;
        const c = (ti + 1) * rowLen + ai;
        const d = c + 1;
        indices.push(a, c, b);
        indices.push(b, c, d);
      }
    }
  } else if (shape === 'sphere') {
    // ── Sphere: radius modulated by azimuthal power, uniform across latitude ──
    // Keeps a solid minimum radius so it never collapses. Equator bulges at peak.
    const nLat = 24;  // latitude segments (pole to pole)
    const nLon = M;   // longitude = azimuth resolution
    const rMin = dispRadius * 0.25; // minimum sphere radius
    const rMod = dispRadius * 0.35; // max expansion at equator

    for (let lat = 0; lat <= nLat; lat++) {
      const theta = (lat / nLat) * Math.PI; // 0=north pole, PI=south pole
      const sinT = Math.sin(theta), cosT = Math.cos(theta);
      // Latitude fade: full modulation at equator (sinT=1), none at poles (sinT=0)
      const latFade = Math.pow(sinT, 1.5);

      for (let lon = 0; lon <= nLon; lon++) {
        const lonIdx = lon % nLon;
        const phi = interpAngles[lonIdx];
        const val = vals[lonIdx];
        const r = rMin + rMod * val * latFade;

        const x = r * sinT * Math.cos(phi);
        const z = r * sinT * Math.sin(phi);
        const y = r * cosT;
        vertices.push(x, y, z);

        // Color: blend by power value, with slight desaturation near poles
        const cv = val * (0.3 + 0.7 * latFade);
        const c = new THREE.Color();
        if (cv < 0.5) c.copy(colorLow).lerp(colorMid, cv * 2);
        else c.copy(colorMid).lerp(colorHigh, (cv - 0.5) * 2);
        colors.push(c.r, c.g, c.b);
      }
    }

    const rowLen = nLon + 1;
    for (let lat = 0; lat < nLat; lat++) {
      for (let lon = 0; lon < nLon; lon++) {
        const a = lat * rowLen + lon;
        const b = a + 1;
        const c = (lat + 1) * rowLen + lon;
        const d = c + 1;
        indices.push(a, c, b);
        indices.push(b, c, d);
      }
    }
  } else if (shape === 'particles') {
    // ── Particle cloud: scatter points radially, density/distance encodes power ──
    const nParticlesPerAngle = 40;
    const positions = [];
    const particleColors = [];

    for (let ai = 0; ai < M; ai++) {
      const phi = interpAngles[ai];
      const val = vals[ai];
      const innerR = dispRadius * 0.28; // larger inner ring
      const outerR = dispRadius * (0.4 + 0.45 * val); // slightly smaller outer
      const count = Math.floor(nParticlesPerAngle * (0.2 + 0.8 * val));

      for (let p = 0; p < count; p++) {
        const r = innerR + (outerR - innerR) * Math.pow(Math.random(), 0.6);
        const ySpread = (Math.random() - 0.5) * dispRadius * 0.2;
        const x = r * Math.cos(phi);
        const z = r * Math.sin(phi);
        positions.push(x, ySpread, z);

        const cv = val * (0.7 + 0.3 * Math.random());
        const c = new THREE.Color();
        if (cv < 0.5) c.copy(colorLow).lerp(colorMid, cv * 2);
        else c.copy(colorMid).lerp(colorHigh, (cv - 0.5) * 2);
        particleColors.push(c.r, c.g, c.b);
      }
    }

    // Spawn jet particles at peak direction
    const peakIdx = vals.indexOf(Math.max(...vals));
    const peakAz = interpAngles[peakIdx];
    const peakVal = vals[peakIdx];
    if (peakVal > 0.6) {
      spawnJetParticles(peakAz, peakVal, dispRadius);
    }

    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    pGeo.setAttribute('color', new THREE.Float32BufferAttribute(particleColors, 3));
    const pMat = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: opacity * 0.8,
      sizeAttenuation: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    hwRingGroup.add(new THREE.Points(pGeo, pMat));
    return; // particles don't need mesh/wireframe below
  } else if (shape === 'bars') {
    // ── Cylindrical bar chart: vertical bars in a circle ──
    const barWidth = 0.04;
    const barDepth = 0.07;
    const ringR = dispRadius * 0.5;
    // Use single shared geometry + instanced approach: every 4th upsampled angle = 72 bars
    const step = 4;

    for (let ai = 0; ai < M; ai += step) {
      const phi = interpAngles[ai];
      const val = vals[ai];
      const barH = dispRadius * (0.1 + 0.8 * val);

      const barGeo = new THREE.BoxGeometry(barWidth, barH, barDepth);
      const c = new THREE.Color();
      if (val < 0.5) c.copy(colorLow).lerp(colorMid, val * 2);
      else c.copy(colorMid).lerp(colorHigh, (val - 0.5) * 2);
      const barMat = new THREE.MeshPhongMaterial({color: c, transparent: true, opacity: opacity * 0.9, shininess: 40, emissive: c, emissiveIntensity: 0.15});
      const bar = new THREE.Mesh(barGeo, barMat);

      bar.position.set(ringR * Math.cos(phi), barH / 2, ringR * Math.sin(phi));
      bar.rotation.y = -phi;
      hwRingGroup.add(bar);
    }
    // Add reference ring at the base
    const ringGeo = new THREE.RingGeometry(ringR - 0.02, ringR + 0.02, 72);
    const ringMat = new THREE.MeshBasicMaterial({color: 0x5fd28a, transparent: true, opacity: 0.3, side: THREE.DoubleSide});
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = -Math.PI / 2;
    hwRingGroup.add(ringMesh);
    return;
  }

  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  const mat = new THREE.MeshPhongMaterial({
    vertexColors: true, transparent: true, opacity,
    side: THREE.DoubleSide, shininess: 60,
    emissive: colorLow, emissiveIntensity: 0.6,
  });
  hwRingGroup.add(new THREE.Mesh(geometry, mat));

  const wireMat = new THREE.MeshBasicMaterial({
    wireframe: true, color: 0xffffff, transparent: true, opacity: opacity * 0.12
  });
  hwRingGroup.add(new THREE.Mesh(geometry.clone(), wireMat));
}

// -- Hardware 3D array marker (small disc at origin) --

function ensureHwArrayMarker() {
  if (hwArrayMarkerGroup.children.length > 0) return;
  // Small disc to represent the array at origin
  const discGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.015, 24);
  const discMat = new THREE.MeshPhongMaterial({color: 0x5fd28a, emissive: 0x112211, transparent: true, opacity: 0.8});
  const disc = new THREE.Mesh(discGeo, discMat);
  hwArrayMarkerGroup.add(disc);
  // Small upward pin
  const pinGeo = new THREE.CylinderGeometry(0.01, 0.01, 0.15, 8);
  const pinMat = new THREE.MeshPhongMaterial({color: 0x5fd28a, emissive: 0x112211});
  const pin = new THREE.Mesh(pinGeo, pinMat);
  pin.position.y = 0.08;
  hwArrayMarkerGroup.add(pin);
}

function updateHw3dRing(frame) {
  const viewMode = document.getElementById('hwViewMode').value;
  if (viewMode === '2d') {
    hwRingGroup.visible = false;
    hwArrayMarkerGroup.visible = false;
    hw3dActive = false;
    return;
  }
  hwRingGroup.visible = true;
  hwArrayMarkerGroup.visible = true;
  hw3dActive = true;
  ensureHwArrayMarker();

  // Hide other mode groups that might be showing
  beamGroup.visible = false;
  sourceGroup.visible = false;
  roomGroup.visible = false;

  if (frame.levels_db && frame.configuration) {
    buildHardwareSectorMesh(frame);
    steerGroup.clear();
    trueDirGroup.clear();
    return;
  }

  const scan = frame.beam_scan;
  if (!scan || !scan.powers_db) return;

  const dispRadius = parseFloat(document.getElementById('hwDisplayRadius')?.value || 2.0);
  const opacity = parseFloat(document.getElementById('hwOpacity')?.value || 0.7);

  buildHardwareRingMesh(scan.powers_db, scan.angles_deg, dispRadius, opacity);

  // Update DOA arrows
  const estAzRad = frame.est_az_deg * Math.PI / 180;
  renderSteerDir(estAzRad, 0, dispRadius * 1.3);
  trueDirGroup.clear();
}

function buildHardwareSectorMesh(frame) {
  hwRingGroup.clear();
  const levels = frame.levels_db;
  const radius = parseFloat(document.getElementById('hwDisplayRadius')?.value || 2.0);
  const opacity = parseFloat(document.getElementById('hwOpacity')?.value || 0.7);
  const azimuthEdges = hwAngularEdges(frame.azimuth_deg, 10);
  const elevationEdges = hwAngularEdges(frame.elevation_deg, 10).map(value => Math.max(-90, Math.min(90, value)));
  const vertices = [];
  const colors = [];
  const indices = [];
  const gridVertices = [];
  let strongest = null;

  const sphericalPoint = (azimuthDeg, elevationDeg, pointRadius) => {
    const azimuth = azimuthDeg * Math.PI / 180;
    const elevation = elevationDeg * Math.PI / 180;
    return new THREE.Vector3(
      pointRadius * Math.cos(elevation) * Math.cos(azimuth),
      pointRadius * Math.sin(elevation),
      pointRadius * Math.cos(elevation) * Math.sin(azimuth),
    );
  };

  for (let row = 0; row < frame.configuration.rows; row++) {
    for (let column = 0; column < frame.configuration.columns; column++) {
      const level = levels[row][column];
      const measured = Number.isFinite(level);
      const value = hwLevelNormalized(level);
      if (measured && (!strongest || level > strongest.level)) strongest = {row, column, level};

      // Stronger incoming energy pulls the tile slightly inward toward the array.
      const tileRadius = radius * (1.0 - 0.12 * value);
      const corners = [
        sphericalPoint(azimuthEdges[column], elevationEdges[row], tileRadius),
        sphericalPoint(azimuthEdges[column + 1], elevationEdges[row], tileRadius),
        sphericalPoint(azimuthEdges[column + 1], elevationEdges[row + 1], tileRadius),
        sphericalPoint(azimuthEdges[column], elevationEdges[row + 1], tileRadius),
      ];
      const base = vertices.length / 3;
      const freshness = hwFreshnessFactor(row, column);
      const color = new THREE.Color(measured ? hwHeatColor(value) : '#11151a');
      color.multiplyScalar(Math.max(0.3, freshness));
      for (const corner of corners) {
        vertices.push(corner.x, corner.y, corner.z);
        colors.push(color.r, color.g, color.b);
      }
      indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
      for (let edge = 0; edge < 4; edge++) {
        const first = corners[edge], second = corners[(edge + 1) % 4];
        gridVertices.push(first.x, first.y, first.z, second.x, second.y, second.z);
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  hwRingGroup.add(new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({
    vertexColors: true,
    transparent: true,
    opacity: Math.max(0.25, opacity),
    side: THREE.DoubleSide,
    shininess: 25,
    emissive: 0x06151a,
    emissiveIntensity: 0.35,
  })));

  const gridGeometry = new THREE.BufferGeometry();
  gridGeometry.setAttribute('position', new THREE.Float32BufferAttribute(gridVertices, 3));
  hwRingGroup.add(new THREE.LineSegments(gridGeometry, new THREE.LineBasicMaterial({
    color: 0x5f91a5, transparent: true, opacity: 0.38,
  })));

  const focus = frame.target || frame.last_steer || strongest;
  if (focus) {
    const focusPoint = sphericalPoint(frame.azimuth_deg[focus.column], frame.elevation_deg[focus.row], radius * 1.04);
    const inward = focusPoint.clone().multiplyScalar(-1).normalize();
    const arrowColor = frame.target ? 0x5fd28a : (frame.last_steer ? 0x6bcfff : 0xffffff);
    hwRingGroup.add(new THREE.ArrowHelper(inward, focusPoint, radius * 0.86, arrowColor,
      radius * 0.08, radius * 0.045));
    const sourceMarker = new THREE.Mesh(
      new THREE.SphereGeometry(radius * 0.035, 16, 12),
      new THREE.MeshBasicMaterial({color: arrowColor}),
    );
    sourceMarker.position.copy(focusPoint);
    hwRingGroup.add(sourceMarker);
    hwRingGroup.add(hwSectorOutline(
      azimuthEdges[focus.column], azimuthEdges[focus.column + 1],
      elevationEdges[focus.row], elevationEdges[focus.row + 1],
      radius * 0.985, arrowColor, sphericalPoint,
    ));
  }

  const truth = frame.emulator_truth;
  if (truth) {
    const truthPoint = sphericalPoint(truth.azimuth_deg, truth.elevation_deg, radius * 1.08);
    const truthColor = truth.source_active === false ? 0xf2c96d : 0x8dffac;
    const truthLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), truthPoint]),
      new THREE.LineBasicMaterial({color: truthColor, transparent: true, opacity: 0.8}),
    );
    hwRingGroup.add(truthLine);
    const truthMarker = new THREE.Mesh(
      new THREE.SphereGeometry(radius * 0.045, 16, 12),
      new THREE.MeshBasicMaterial({color: truthColor}),
    );
    truthMarker.position.copy(truthPoint);
    hwRingGroup.add(truthMarker);
  }
}

function hwAngularEdges(centers, fallbackHalfSpan) {
  if (centers.length === 1) return [centers[0] - fallbackHalfSpan, centers[0] + fallbackHalfSpan];
  const edges = [centers[0] - (centers[1] - centers[0]) / 2];
  for (let index = 0; index < centers.length - 1; index++) {
    edges.push((centers[index] + centers[index + 1]) / 2);
  }
  edges.push(centers[centers.length - 1] + (centers[centers.length - 1] - centers[centers.length - 2]) / 2);
  return edges;
}

function hwSectorOutline(azimuthLow, azimuthHigh, elevationLow, elevationHigh,
                         radius, color, sphericalPoint) {
  const corners = [
    sphericalPoint(azimuthLow, elevationLow, radius),
    sphericalPoint(azimuthHigh, elevationLow, radius),
    sphericalPoint(azimuthHigh, elevationHigh, radius),
    sphericalPoint(azimuthLow, elevationHigh, radius),
  ];
  const points = [];
  for (let edge = 0; edge < 4; edge++) points.push(corners[edge], corners[(edge + 1) % 4]);
  return new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({color, transparent: true, opacity: 1}),
  );
}

// -- Hardware controls event handlers --
document.getElementById('hwDbFloor').addEventListener('input', (e) => {
  document.getElementById('hwDbFloorVal').textContent = e.target.value;
  hwNeedsRedraw = true;
});

document.getElementById('hwCrtToggle').addEventListener('change', (e) => {
  const el = document.getElementById('hwScanlines');
  if (e.target.checked) el.classList.remove('hidden');
  else el.classList.add('hidden');
});

document.getElementById('hwViewMode').addEventListener('change', () => {
  const vmode = document.getElementById('hwViewMode').value;
  const canvas2d = document.getElementById('hwBirdsEye');
  document.body.classList.remove('hardware-3d-active', 'hardware-split', 'hardware-3d-only');
  if (vmode === '2d') {
    canvas2d.style.display = 'block';
    hwRingGroup.visible = false;
  } else if (vmode === '3d') {
    canvas2d.style.display = 'none';
    document.body.classList.add('hardware-3d-active', 'hardware-3d-only');
    hwRingGroup.visible = true;
    frameHardwareDome();
  } else { // split
    canvas2d.style.display = 'block';
    document.body.classList.add('hardware-3d-active', 'hardware-split');
    hwRingGroup.visible = true;
    frameHardwareDome();
  }
  setTimeout(resizeHardwareViews, 50);
});

function frameHardwareDome() {
  const radius = parseFloat(document.getElementById('hwDisplayRadius')?.value || 2.0);
  camera.position.set(-radius * 2.35, radius * 1.15, radius * 2.0);
  controls.target.set(radius * 0.32, 0, 0);
  camera.lookAt(controls.target);
  controls.update();
}

document.getElementById('btnHwPause').addEventListener('click', () => {
  hwPaused = !hwPaused;
  document.getElementById('btnHwPause').textContent = hwPaused ? 'Resume Display' : 'Freeze Display';
  hwSend({ type: hwPaused ? 'pause' : 'resume' });
  addHwEvent('DISPLAY', hwPaused ? 'Presentation frozen; device continues operating' : 'Live presentation resumed', 'warning');
  if (hwLastFrame) updateHwReadiness(hwLastFrame);
});

document.getElementById('hwAdvancedToggle').addEventListener('change', (event) => {
  document.getElementById('hwAdvancedPanel').classList.toggle('hidden', !event.target.checked);
});
document.getElementById('hwAdvancedPanel').classList.toggle(
  'hidden', !document.getElementById('hwAdvancedToggle').checked);
document.getElementById('hwEmulatorModel').addEventListener('change', updateHwEmulatorModelUi);
updateHwEmulatorModelUi();

document.getElementById('hwAcousticScenario').addEventListener('change', () => {
  hwAcousticCacheId = null;
  clearHwAudition();
  renderHwAcousticStatus({state: 'idle'});
});

for (const id of ['hwEmRows', 'hwEmColumns', 'hwEmAzMin', 'hwEmAzMax', 'hwEmElMin', 'hwEmElMax']) {
  document.getElementById(id).addEventListener('change', () => {
    if (document.getElementById('hwEmulatorModel').value === 'acoustic') {
      hwAcousticCacheId = null;
      clearHwAudition();
      renderHwAcousticStatus({state: 'idle'});
      updateHwGridSource();
    }
  });
}

document.getElementById('btnHwAcousticPrepare').addEventListener('click', async () => {
  hwAcousticCacheId = null;
  try {
    const response = await fetch('http://127.0.0.1:8766/hw_acoustic_prepare', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        scenario: document.getElementById('hwAcousticScenario').value,
        rows: parseInt(document.getElementById('hwEmRows').value),
        columns: parseInt(document.getElementById('hwEmColumns').value),
        azimuth_min_deg: parseFloat(document.getElementById('hwEmAzMin').value),
        azimuth_max_deg: parseFloat(document.getElementById('hwEmAzMax').value),
        elevation_min_deg: parseFloat(document.getElementById('hwEmElMin').value),
        elevation_max_deg: parseFloat(document.getElementById('hwEmElMax').value),
      }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || 'preparation failed');
    renderHwAcousticStatus(await response.json());
    pollHwAcousticStatus();
  } catch (error) {
    renderHwAcousticStatus({state: 'failed', error: error.message});
  }
});

document.getElementById('hwAuditionMode').addEventListener('change', selectHwAuditionClip);
document.getElementById('btnHwAudition').addEventListener('click', async () => {
  const button = document.getElementById('btnHwAudition');
  const status = document.getElementById('hwAuditionStatus');
  if (!hwAcousticCacheId) {
    status.textContent = 'Prepare an acoustic scene first';
    return;
  }
  const strongest = hwLastFrame ? hwStrongestObservation(hwLastFrame) : null;
  const selectedSector = hwLastFrame?.target?.sector ?? hwLastFrame?.last_steer?.sector
    ?? strongest?.sector ?? parseInt(document.getElementById('hwSector').value);
  button.disabled = true;
  status.textContent = 'Rendering...';
  try {
    const response = await fetch('http://127.0.0.1:8766/hw_acoustic_audition', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        acoustic_cache_id: hwAcousticCacheId,
        start_s: hwLastFrame?.emulator_truth?.elapsed_s ?? 0,
        duration_s: 3.0,
        selected_sector: selectedSector,
      }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || 'audio render failed');
    const result = await response.json();
    clearHwAudition();
    for (const [name, encoded] of Object.entries(result.clips_b64)) {
      hwAuditionUrls[name] = b64ToWavUrl(encoded);
    }
    selectHwAuditionClip();
    status.textContent = `3.0 s · sector ${result.selected_sector} · joint gain`;
  } catch (error) {
    status.textContent = `Render failed · ${error.message}`;
  } finally {
    button.disabled = false;
  }
});

document.getElementById('btnHwAcousticCancel').addEventListener('click', async () => {
  await fetch('http://127.0.0.1:8766/hw_acoustic_cancel', {method: 'POST'});
  pollHwAcousticStatus();
});

for (const [inputId, valueId, decimals] of [
  ['hwDisplayRadius', 'hwDisplayRadiusVal', 1],
  ['hwOpacity', 'hwOpacityVal', 2],
]) {
  document.getElementById(inputId).addEventListener('input', (event) => {
    document.getElementById(valueId).textContent = Number(event.target.value).toFixed(decimals);
    hwNeedsRedraw = true;
    if (document.body.classList.contains('hardware-3d-active')) frameHardwareDome();
  });
}

document.getElementById('btnHwClearEvents').addEventListener('click', () => {
  hwEventLog = [];
  renderHwEventLog();
});

document.getElementById('btnHwConnect').addEventListener('click', async () => {
  const port = document.getElementById('hwSerialPort').value.trim();
  if (!port) { alert('Enter a serial port (e.g. COM18)'); return; }
  const baud = parseInt(document.getElementById('hwSerialBaud').value);
  const statusEl = document.getElementById('hwStatus');
  setHwConnectionState('connecting', `Opening ${port} at ${baud} baud...`);
  try {
    const response = await fetch('http://127.0.0.1:8766/hw_connect', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({transport: 'serial', port, baud}),
    });
    if (!response.ok) throw new Error((await response.json()).detail || 'connection failed');
    startHardwareSession();
  } catch (err) {
    setHwConnectionState('fault', `Connection failed · ${err.message}`);
    addHwEvent('ERROR', `Serial connection failed: ${err.message}`, 'error');
  }
});

document.getElementById('btnHwEmulator').addEventListener('click', async () => {
  setHwConnectionState('connecting', 'Starting protocol emulator...');
  try {
    const acoustic = document.getElementById('hwEmulatorModel').value === 'acoustic';
    if (acoustic && !hwAcousticCacheId) throw new Error('prepare an acoustic scene first');
    const emulatorConfig = {
      rows: parseInt(document.getElementById('hwEmRows').value),
      columns: parseInt(document.getElementById('hwEmColumns').value),
      azimuth_min_deg: parseFloat(document.getElementById('hwEmAzMin').value),
      azimuth_max_deg: parseFloat(document.getElementById('hwEmAzMax').value),
      elevation_min_deg: parseFloat(document.getElementById('hwEmElMin').value),
      elevation_max_deg: parseFloat(document.getElementById('hwEmElMax').value),
      flight_profile: document.getElementById('hwEmFlightProfile').value,
      flight_speed: parseFloat(document.getElementById('hwEmFlightSpeed').value),
    };
    const response = await fetch('http://127.0.0.1:8766/hw_connect', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(acoustic
        ? {transport: 'acoustic', acoustic_cache_id: hwAcousticCacheId}
        : {transport: 'emulator', ...emulatorConfig}),
    });
    if (!response.ok) throw new Error((await response.json()).detail || 'emulator failed');
    startHardwareSession();
  } catch (err) {
    setHwConnectionState('fault', `Emulator failed · ${err.message}`);
    addHwEvent('ERROR', `Emulator failed: ${err.message}`, 'error');
  }
});

document.getElementById('btnHwDisconnect').addEventListener('click', async () => {
  addHwEvent('LINK', 'Operator requested disconnect', 'warning');
  try {
    await fetch('http://127.0.0.1:8766/hw_disconnect', {method: 'POST'});
  } finally {
    stopHardwareSession();
  }
});

for (const [buttonId, command] of [
  ['btnHwOnce', 'F'], ['btnHwContinuous', 'C'], ['btnHwAdaptive', 'G'], ['btnHwStop', 'X'],
]) {
  document.getElementById(buttonId).addEventListener('click', () => {
    stopHwMonitor();
    if (hwSend({type: 'command', command})) {
      const labels = {F: 'Full sweep requested', C: 'Continuous scan requested', G: 'Adaptive tracking requested', X: 'Stop requested'};
      addHwEvent('MODE', labels[command], command === 'X' ? 'warning' : '');
    }
  });
}

document.getElementById('btnHwSteer').addEventListener('click', () => {
  stopHwMonitor();
  const sector = parseInt(document.getElementById('hwSector').value);
  if (!Number.isInteger(sector) || sector < 0 || (hwInit?.configuration && sector >= hwInit.configuration.sectors)) {
    document.getElementById('hwStatus').textContent = 'Invalid sector';
    return;
  }
  if (hwSend({type: 'command', command: `S,${sector}`})) addHwEvent('STEER', `Sector ${sector} requested`);
});

document.getElementById('btnHwMonitor').addEventListener('click', () => {
  if (hwMonitoring) {
    stopHwMonitor();
    return;
  }
  const sector = parseInt(document.getElementById('hwSector').value);
  if (!Number.isInteger(sector) || sector < 0 ||
      (hwInit?.configuration && sector >= hwInit.configuration.sectors)) {
    document.getElementById('hwStatus').textContent = 'Invalid sector';
    return;
  }
  if (!hwWs || hwWs.readyState !== WebSocket.OPEN) {
    document.getElementById('hwStatus').textContent = 'Connect hardware or emulator first';
    return;
  }
  hwSend({type: 'command', command: `S,${sector}`});
  hwMonitoring = true;
  const button = document.getElementById('btnHwMonitor');
  button.textContent = 'Stop Monitor';
  button.classList.add('active');
  addHwEvent('MODE', `Fixed-beam monitor · sector ${sector}`);
  restartHwMonitorTimer();
});

document.getElementById('hwMonitorRate').addEventListener('change', () => {
  if (hwMonitoring) restartHwMonitorTimer();
});

function restartHwMonitorTimer() {
  if (hwMonitorTimer) clearInterval(hwMonitorTimer);
  const rate = Math.max(1, parseInt(document.getElementById('hwMonitorRate').value) || 20);
  hwMonitorTimer = setInterval(() => hwSend({type: 'command', command: 'M'}), 1000 / rate);
}

function stopHwMonitor() {
  const wasMonitoring = hwMonitoring;
  hwMonitoring = false;
  if (hwMonitorTimer) clearInterval(hwMonitorTimer);
  hwMonitorTimer = null;
  const button = document.getElementById('btnHwMonitor');
  if (button) {
    button.textContent = 'Monitor Beam';
    button.classList.remove('active');
  }
  if (wasMonitoring) addHwEvent('MODE', 'Fixed-beam monitor stopped');
}

let hwGridLayout = null;
document.getElementById('hwBirdsEye').addEventListener('click', (event) => {
  if (!hwGridLayout) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const x = (event.clientX - rect.left) * dpr;
  const y = (event.clientY - rect.top) * dpr;
  if (x < hwGridLayout.left || x >= hwGridLayout.left + hwGridLayout.width ||
      y < hwGridLayout.top || y >= hwGridLayout.top + hwGridLayout.height) return;
  const column = Math.floor((x - hwGridLayout.left) / hwGridLayout.width * hwGridLayout.columns);
  const displayRow = Math.floor((y - hwGridLayout.top) / hwGridLayout.height * hwGridLayout.rows);
  const row = hwGridLayout.rows - 1 - displayRow;
  const sector = row * hwGridLayout.columns + column;
  stopHwMonitor();
  document.getElementById('hwSector').value = sector;
  if (hwSend({type: 'command', command: `S,${sector}`})) addHwEvent('STEER', `Sector ${sector} selected from heatmap`);
});

syncHwControls();
pollHwAcousticStatus();
