/**
 * scene.js — the 3D beam-pattern surface. Radius and colour follow the gain
 * in dB (−30 dB at the centre, 0 dB at the outer radius), with a thin grid
 * every 10°, the axes, the microphones (scaled to fit) and the steering ray.
 */

import * as THREE from 'three';
import { Viewport3D, clearGroup } from '../shared/viewport3d.js';
import { rampColor } from '../shared/colormap.js';
import { h, setText } from '../shared/dom.js';
import { deg } from '../shared/format.js';
import { direction, toDb } from './pattern.js';

const BR = 1.35;
const FLOOR_DB = -30;
export const VIEWS = {angled: {yaw: 38, pitch: 24}, top: {yaw: 0, pitch: 85}, side: {yaw: 0, pitch: 2}};

/** Tool axes (x, y, z-up) to Three.js axes (x, y-up, z): a proper rotation. */
function toThree(v, scale = 1) {
  return new THREE.Vector3(v[0] * scale, v[2] * scale, -v[1] * scale);
}

export function unitOf(db) {
  return Math.max(0, (db - FLOOR_DB) / -FLOOR_DB);
}

export class PatternScene {
  constructor(container) {
    this.viewport = new Viewport3D(container, {background: '#1a1c20', fov: 38});
    this.labels = h('div', {class: 'scene-labels', 'aria-hidden': 'true'});
    container.append(this.labels);
    this.surface = new THREE.Group();
    this.guides = new THREE.Group();
    this.array = new THREE.Group();
    this.ray = new THREE.Group();
    this.viewport.scene.add(this.guides, this.array, this.surface, this.ray);
    this.labelNodes = [];
    this.buildGuides();
    this.viewport.onFrame = () => this.placeLabels();
    this.setView('angled');
  }

  start() { this.viewport.start(); }
  stop() { this.viewport.stop(); }

  setView(name) {
    this.viewport.orbitTo(new THREE.Vector3(0, 0, 0), {...(VIEWS[name] || VIEWS.angled), distance: 4.6});
  }

  label(text, point, className = '') {
    const el = h('span', {class: `scene-label scene-label--center ${className}`.trim()}, text);
    this.labels.append(el);
    const node = {el, point};
    this.labelNodes.push(node);
    return node;
  }

  buildGuides() {
    const equator = [];
    for (let i = 0; i <= 72; i++) {
      const a = i * 5 * Math.PI / 180;
      equator.push(toThree([Math.cos(a) * BR, Math.sin(a) * BR, 0]));
    }
    const ring = new THREE.Line(new THREE.BufferGeometry().setFromPoints(equator),
      new THREE.LineDashedMaterial({color: 0x3a3f46, dashSize: 0.05, gapSize: 0.06}));
    ring.computeLineDistances();
    this.guides.add(ring);
    for (const [name, end] of [['x', [1.8, 0, 0]], ['y', [0, 1.8, 0]], ['z', [0, 0, 1.6]]]) {
      this.guides.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), toThree(end)]),
        new THREE.LineBasicMaterial({color: 0x878d94}),
      ));
      this.label(name, toThree(end, 1.08), 'scene-label--muted');
    }
    this.steerLabel = this.label('', new THREE.Vector3(), 'scene-label--accent');
  }

  placeLabels() {
    for (const node of this.labelNodes) {
      const p = this.viewport.project(node.point);
      node.el.style.left = `${p.x.toFixed(1)}px`;
      node.el.style.top = `${p.y.toFixed(1)}px`;
      node.el.hidden = !p.visible || !node.el.textContent;
    }
  }

  render({grid, mics, steerAz, steerEl}) {
    clearGroup(this.surface);
    clearGroup(this.array);
    clearGroup(this.ray);
    if (!grid || mics.length < 2) {
      setText(this.steerLabel.el, '');
      this.viewport.requestRender();
      return;
    }

    const nA = grid.az.length;
    const nE = grid.el.length;
    const positions = new Float32Array(nA * nE * 3);
    const colors = new Float32Array(nA * nE * 3);
    const color = new THREE.Color();
    for (let i = 0; i < nA; i++) {
      for (let j = 0; j < nE; j++) {
        const unit = unitOf(toDb(grid.values[i][j]));
        const p = toThree(direction(grid.az[i], grid.el[j]), Math.max(0.02, unit) * BR);
        const k = (i * nE + j) * 3;
        positions[k] = p.x; positions[k + 1] = p.y; positions[k + 2] = p.z;
        const rgb = rampColor(unit);
        color.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, THREE.SRGBColorSpace);
        colors[k] = color.r; colors[k + 1] = color.g; colors[k + 2] = color.b;
      }
    }
    const index = [];
    for (let i = 0; i < nA - 1; i++) {
      for (let j = 0; j < nE - 1; j++) {
        const a = i * nE + j;
        const b = (i + 1) * nE + j;
        index.push(a, b, b + 1, a, b + 1, a + 1);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setIndex(index);
    this.surface.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.9,
    })));

    // Thin grid every 10° over the surface.
    const segments = [];
    const vertex = (i, j) => new THREE.Vector3(positions[(i * nE + j) * 3], positions[(i * nE + j) * 3 + 1], positions[(i * nE + j) * 3 + 2]);
    const stepA = Math.max(1, Math.round(10 / (grid.az[1] - grid.az[0])));
    const stepE = Math.max(1, Math.round(10 / (grid.el[1] - grid.el[0])));
    for (let i = 0; i < nA; i += stepA) for (let j = 0; j < nE - 1; j++) segments.push(vertex(i, j), vertex(i, j + 1));
    for (let j = 0; j < nE; j += stepE) for (let i = 0; i < nA - 1; i++) segments.push(vertex(i, j), vertex(i + 1, j));
    this.surface.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(segments),
      new THREE.LineBasicMaterial({color: 0x1a1c20, transparent: true, opacity: 0.45})));

    let maxR = 0;
    for (const m of mics) maxR = Math.max(maxR, Math.hypot(m[0], m[1], m[2]));
    const scale = maxR > 0 ? 0.32 / maxR : 1;
    const dots = new THREE.InstancedMesh(new THREE.SphereGeometry(0.018, 10, 8),
      new THREE.MeshBasicMaterial({color: 0xe9eaec}), mics.length);
    const matrix = new THREE.Matrix4();
    mics.forEach((m, i) => {
      const p = toThree(m, scale);
      matrix.makeTranslation(p.x, p.y, p.z);
      dots.setMatrixAt(i, matrix);
    });
    this.array.add(dots);

    const tip = toThree(direction(steerAz, steerEl), BR * 1.18);
    this.ray.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), tip]),
      new THREE.LineBasicMaterial({color: 0x7aa7ff})));
    const end = new THREE.Mesh(new THREE.SphereGeometry(0.035, 16, 12), new THREE.MeshBasicMaterial({color: 0x7aa7ff}));
    end.position.copy(tip);
    this.ray.add(end);
    this.steerLabel.point = tip.clone().multiplyScalar(1.1);
    setText(this.steerLabel.el, `Steered ${deg(steerAz, 0)}, ${deg(steerEl, 0)}`);
    this.viewport.requestRender();
  }
}
