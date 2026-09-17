/**
 * geometry.js — Array geometry builders.
 * Pure functions: no imports, no side effects.
 */

export function makeUCA(nMics, radius) {
  const pts = [];
  for (let i = 0; i < nMics; i++) {
    const a = 2 * Math.PI * i / nMics;
    pts.push([radius * Math.cos(a), radius * Math.sin(a), 0]);
  }
  return pts;
}

export function makeCross(nMics, halfLen) {
  const perArm = Math.floor(nMics / 2);
  const offsets = [];
  const n = perArm + 1;
  for (let i = 0; i < n; i++) {
    const o = -halfLen + 2 * halfLen * i / (n - 1);
    if (Math.abs(o) > 1e-9) offsets.push(o);
  }
  const pts = [];
  for (const o of offsets) pts.push([o, 0, 0]);
  for (const o of offsets) pts.push([0, 0, o]);
  return pts;
}

export function makeULA(nMics, length) {
  const pts = [];
  for (let i = 0; i < nMics; i++) {
    pts.push([-length / 2 + length * i / (nMics - 1), 0, 0]);
  }
  return pts;
}

export function makeCylinder(nMics, radius, separation) {
  const perRing = Math.floor(nMics / 2);
  const halfSep = separation / 2;
  const pts = [];
  for (let i = 0; i < perRing; i++) {
    const a = 2 * Math.PI * i / perRing;
    pts.push([radius * Math.cos(a), radius * Math.sin(a), -halfSep]);
  }
  for (let i = 0; i < perRing; i++) {
    const a = 2 * Math.PI * i / perRing + Math.PI / perRing;
    pts.push([radius * Math.cos(a), radius * Math.sin(a), halfSep]);
  }
  return pts;
}

export function buildGeometry(name, nMics, radius, separation, customMicPositions = []) {
  switch (name) {
    case 'UCA': return makeUCA(nMics, radius);
    case 'CROSS': return makeCross(nMics, radius);
    case 'ULA': return makeULA(nMics, radius * 2);
    case 'CYLINDER': return makeCylinder(nMics, radius, separation);
    case 'CUSTOM': return customMicPositions.map(p => [p[0], p[1], p[2]]);
    default: return makeUCA(nMics, radius);
  }
}
