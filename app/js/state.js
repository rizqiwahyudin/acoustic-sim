/**
 * state.js — Shared mutable state and DOM ref accessors.
 * All modules that need cross-cutting state import from here.
 */

// ── Application mode ──
export let mode = 'beam';
export function setModeState(m) { mode = m; }

// ── Array state ──
export let currentMics = [];
export let currentPower = [];
export let customMicPositions = [];
export let customSelectedIdx = -1;
export let customSeedGeometry = 'UCA';

export function setCurrentMics(m) { currentMics = m; }
export function setCurrentPower(p) { currentPower = p; }
export function setCustomMicPositions(p) { customMicPositions = p; }
export function setCustomSelectedIdx(i) { customSelectedIdx = i; }
export function setCustomSeedGeometry(g) { customSeedGeometry = g; }

// ── MICCANVAS preset positions (15-mic array) ──
export const MICCANVAS_POSITIONS = [
  [0.000, 0.000, 0.000],
  [0.010, 0.000, 0.000], [0.005, 0.008660254, 0.000], [-0.005, 0.008660254, 0.000],
  [-0.010, 0.000, 0.000], [-0.005, -0.008660254, 0.000], [0.005, -0.008660254, 0.000],
  [0.018477591, 0.007653669, 0.000], [0.007653669, 0.018477591, 0.000],
  [-0.007653669, 0.018477591, 0.000], [-0.018477591, 0.007653669, 0.000],
  [-0.018477591, -0.007653669, 0.000], [-0.007653669, -0.018477591, 0.000],
  [0.007653669, -0.018477591, 0.000], [0.018477591, -0.007653669, 0.000],
];

// ── DOM refs (lazy getters to avoid issues with import order vs DOM readiness) ──
export const el = {
  get selGeo() { return document.getElementById('selGeo'); },
  get micCount() { return document.getElementById('micCount'); },
  get arrayRadius() { return document.getElementById('arrayRadius'); },
  get ringSep() { return document.getElementById('ringSep'); },
  get steerAz() { return document.getElementById('steerAz'); },
  get steerEl() { return document.getElementById('steerEl'); },
  get freq() { return document.getElementById('freq'); },
  get dispRadius() { return document.getElementById('dispRadius'); },
  get opacity() { return document.getElementById('opacity'); },
  get infoText() { return document.getElementById('infoText'); },
  get mipsText() { return document.getElementById('mipsText'); },
  get heatmapWrap() { return document.getElementById('heatmapWrap'); },
  get heatmapCanvas() { return document.getElementById('heatmapCanvas'); },
  get heatmapRange() { return document.getElementById('heatmapRange'); },
  get topCandidates() { return document.getElementById('topCandidates'); },
  get beamPolarWrap() { return document.getElementById('beamPolarWrap'); },
  get beamPolar() { return document.getElementById('beamPolar'); },
  get beamPolarRange() { return document.getElementById('beamPolarRange'); },
};
