/**
 * beam-math.js — Beam pattern computation, constants, and color definitions.
 */

import * as THREE from 'three';

// ── Physical constants ──
export const C = 343.0;

// ── Angular grids ──
export const GRID_AZ = [];
for (let i = 0; i < 72; i++) GRID_AZ.push(i * 2 * Math.PI / 72);
export const GRID_COLAT = [];
for (let i = 0; i < 19; i++) GRID_COLAT.push(i * Math.PI / 18);
export const N_AZ = GRID_AZ.length;
export const N_COLAT = GRID_COLAT.length;

// ── 3D color ramp ──
export const colorLow = new THREE.Color(0x050505);
export const colorMid = new THREE.Color(0xd94b00);
export const colorHigh = new THREE.Color(0xffffff);

// ── Beam pattern computation ──
export function dirVec(az, colat) {
  const sinC = Math.sin(colat), cosC = Math.cos(colat);
  return [sinC * Math.cos(az), sinC * Math.sin(az), cosC];
}

export function computeBeamPattern(mics, steerAzRad, steerElRad, freq) {
  const steerColat = Math.PI / 2 - steerElRad;
  const dSteer = dirVec(steerAzRad, steerColat);
  const k = 2 * Math.PI * freq / C;
  const M = mics.length;
  const power = [];

  for (let ci = 0; ci < N_COLAT; ci++) {
    const row = [];
    for (let ai = 0; ai < N_AZ; ai++) {
      const dLook = dirVec(GRID_AZ[ai], GRID_COLAT[ci]);
      const dx = dLook[0] - dSteer[0];
      const dy = dLook[1] - dSteer[1];
      const dz = dLook[2] - dSteer[2];
      let re = 0, im = 0;
      for (let m = 0; m < M; m++) {
        const phase = k * (mics[m][0] * dx + mics[m][1] * dy + mics[m][2] * dz);
        re += Math.cos(phase);
        im += Math.sin(phase);
      }
      row.push((re * re + im * im) / (M * M));
    }
    power.push(row);
  }
  return power;
}

// ── Spacing utilities ──
export function maxMicSpacing(mics) {
  let maxD = 0;
  for (let i = 0; i < mics.length; i++) {
    for (let j = i + 1; j < mics.length; j++) {
      const dx = mics[i][0] - mics[j][0], dy = mics[i][1] - mics[j][1], dz = mics[i][2] - mics[j][2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d > maxD) maxD = d;
    }
  }
  return maxD;
}

export function minAdjacentSpacing(mics) {
  let minD = Infinity;
  for (let i = 0; i < mics.length; i++) {
    for (let j = i + 1; j < mics.length; j++) {
      const dx = mics[i][0] - mics[j][0], dy = mics[i][1] - mics[j][1], dz = mics[i][2] - mics[j][2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d < minD) minD = d;
    }
  }
  return minD;
}

export function estimateBeamwidth(power, steerAzIdx, steerColatIdx) {
  const peak = power[steerColatIdx][steerAzIdx];
  const halfPower = peak * 0.5;
  let count = 0, total = 0;
  for (let ci = 0; ci < N_COLAT; ci++) {
    for (let ai = 0; ai < N_AZ; ai++) {
      total++;
      if (power[ci][ai] >= halfPower) count++;
    }
  }
  return (count / total * 4 * Math.PI * (180 / Math.PI) * (180 / Math.PI) / (Math.PI)).toFixed(0);
}

// ── 3D beam pattern mesh builder ──
export function buildPatternMesh(power, dispRadius, opacity, beamGroup) {
  beamGroup.clear();
  const geometry = new THREE.BufferGeometry();
  const vertices = [];
  const colors = [];
  const indices = [];

  let maxVal = 0;
  for (let ci = 0; ci < N_COLAT; ci++)
    for (let ai = 0; ai < N_AZ; ai++)
      if (power[ci][ai] > maxVal) maxVal = power[ci][ai];
  if (maxVal < 1e-12) maxVal = 1;

  for (let ci = 0; ci < N_COLAT; ci++) {
    const colat = GRID_COLAT[ci];
    for (let ai = 0; ai <= N_AZ; ai++) {
      const aIdx = ai % N_AZ;
      const az = GRID_AZ[aIdx];
      const val = power[ci][aIdx] / maxVal;
      const r = dispRadius * (0.25 + 0.75 * val);

      const sinC = Math.sin(colat), cosC = Math.cos(colat);
      vertices.push(r * sinC * Math.cos(az), r * cosC, r * sinC * Math.sin(az));

      const c = new THREE.Color();
      if (val < 0.5) c.copy(colorLow).lerp(colorMid, val * 2);
      else c.copy(colorMid).lerp(colorHigh, (val - 0.5) * 2);
      colors.push(c.r, c.g, c.b);
    }
  }

  for (let ci = 0; ci < N_COLAT - 1; ci++) {
    for (let ai = 0; ai < N_AZ; ai++) {
      const a = ci * (N_AZ + 1) + ai;
      const b = a + 1;
      const cc = (ci + 1) * (N_AZ + 1) + ai;
      const d = cc + 1;
      indices.push(a, cc, b);
      indices.push(b, cc, d);
    }
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
  beamGroup.add(new THREE.Mesh(geometry, mat));

  const wireMat = new THREE.MeshBasicMaterial({
    wireframe: true, color: 0xffffff, transparent: true, opacity: opacity * 0.15,
  });
  beamGroup.add(new THREE.Mesh(geometry.clone(), wireMat));
}
