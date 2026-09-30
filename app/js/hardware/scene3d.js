/**
 * scene3d.js — sectors drawn as tiles on a sphere in front of the array.
 * Tiles are built from the angles the firmware reports, coloured like the
 * map, and clickable to hold the beam. Drag to orbit.
 *
 * Three.js world axes: x = LEFT, y = up, z = forward. Mirroring x keeps the
 * contract's "right" on the right of the screen when looking forward from
 * behind the array (a right-handed camera looking down +z shows −x on the right).
 */

import * as THREE from 'three';
import { Viewport3D, clearGroup } from '../shared/viewport3d.js';
import { levelToUnit, rampColor } from '../shared/colormap.js';
import { h, setText } from '../shared/dom.js';
import { deg } from '../shared/format.js';
import { angularEdges, direction } from './geometry.js';
import { STALE_MS } from './session.js';

const RADIUS = 1.25;
const TARGET = new THREE.Vector3(0, 0, 0.7);
const DISTANCE = 2.9;
export const CAMERA_VIEWS = {
  behind: {yaw: 24, pitch: 14},
  above: {yaw: 0, pitch: 62},
  side: {yaw: 78, pitch: 8},
};

const COLORS = {
  line: new THREE.Color('#2c3036'),
  surface: new THREE.Color('#1a1c20'),
  text: new THREE.Color('#e9eaec'),
  accent: new THREE.Color('#7aa7ff'),
  ok: new THREE.Color('#4cc38a'),
  warn: new THREE.Color('#e2b454'),
};

function toScene(vec, radius = 1) {
  return new THREE.Vector3(-vec[0] * radius, vec[1] * radius, vec[2] * radius);
}

function srgb(rgb) {
  return new THREE.Color().setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, THREE.SRGBColorSpace);
}

function tileGeometry(a0, a1, e0, e1, radius, steps = 6) {
  const positions = [];
  const index = [];
  for (let i = 0; i <= steps; i++) {
    for (let j = 0; j <= steps; j++) {
      const a = a0 + (a1 - a0) * j / steps;
      const e = e0 + (e1 - e0) * i / steps;
      const p = toScene(direction(a, e), radius);
      positions.push(p.x, p.y, p.z);
    }
  }
  const row = steps + 1;
  for (let i = 0; i < steps; i++) {
    for (let j = 0; j < steps; j++) {
      const k = i * row + j;
      index.push(k, k + 1, k + row, k + 1, k + row + 1, k + row);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(index);
  return geometry;
}

function outlinePoints(a0, a1, e0, e1, radius, steps = 8) {
  const points = [];
  for (let i = 0; i <= steps; i++) points.push(toScene(direction(a0 + (a1 - a0) * i / steps, e0), radius));
  for (let i = 1; i <= steps; i++) points.push(toScene(direction(a1, e0 + (e1 - e0) * i / steps), radius));
  for (let i = 1; i <= steps; i++) points.push(toScene(direction(a1 - (a1 - a0) * i / steps, e1), radius));
  for (let i = 1; i <= steps; i++) points.push(toScene(direction(a0, e1 - (e1 - e0) * i / steps), radius));
  return points;
}

export class SectorScene {
  constructor(container, {onPick, onHover} = {}) {
    this.container = container;
    this.onPick = onPick;
    this.onHover = onHover;
    this.viewport = new Viewport3D(container, {background: '#1a1c20'});
    this.labels = h('div', {class: 'scene-labels', 'aria-hidden': 'true'});
    container.append(this.labels);

    this.floor = new THREE.Group();
    this.tiles = new THREE.Group();
    this.highlights = new THREE.Group();
    this.array = new THREE.Group();
    this.pointer = new THREE.Group();
    this.viewport.scene.add(this.floor, this.array, this.tiles, this.highlights, this.pointer);
    this.buildFloor();

    this.key = '';
    this.layoutKey = '';
    this.tileMeshes = [];
    this.edges = {az: [], el: []};
    this.hover = null;
    this.labelNodes = [];
    this.markerLabel = null;

    this.viewport.onFrame = () => this.placeLabels();
    this.bindPointer();
    this.setView('behind');
  }

  start() { this.viewport.start(); }
  stop() { this.viewport.stop(); }

  setView(name) {
    const view = CAMERA_VIEWS[name] || CAMERA_VIEWS.behind;
    this.viewport.orbitTo(TARGET, {...view, distance: DISTANCE});
  }

  buildFloor() {
    const points = [];
    for (let i = 0; i <= 6; i++) {
      const x = -1.2 + i * 0.4;
      points.push(new THREE.Vector3(x, -0.9, -0.3), new THREE.Vector3(x, -0.9, 1.5));
      const z = -0.3 + i * 0.3;
      points.push(new THREE.Vector3(-1.2, -0.9, z), new THREE.Vector3(1.2, -0.9, z));
    }
    this.floor.add(new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({color: COLORS.line}),
    ));
  }

  setArray(layout) {
    const key = layout ? JSON.stringify(layout.positions) : '';
    if (key === this.layoutKey) return;
    this.layoutKey = key;
    clearGroup(this.array);
    if (!layout?.positions?.length) return;
    const n = layout.positions.length;
    const mean = [0, 0];
    for (const [x, y] of layout.positions) { mean[0] += x / n; mean[1] += y / n; }
    const mesh = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.011, 10, 8),
      new THREE.MeshBasicMaterial({color: COLORS.text, transparent: true, opacity: 0.85}),
      n,
    );
    const matrix = new THREE.Matrix4();
    layout.positions.forEach(([x, y, z = 0], i) => {
      const p = toScene([(x - mean[0]) / 1000, (y - mean[1]) / 1000, z / 1000]);
      matrix.makeTranslation(p.x, p.y, p.z);
      mesh.setMatrixAt(i, matrix);
    });
    this.array.add(mesh);
    this.viewport.requestRender();
  }

  build(frame) {
    const cfg = frame.configuration;
    const key = `${cfg.rows}x${cfg.columns}|${frame.azimuth_deg.join(',')}|${frame.elevation_deg.join(',')}`;
    if (key === this.key) return;
    this.key = key;
    clearGroup(this.tiles);
    this.tileMeshes = [];
    const az = angularEdges(frame.azimuth_deg);
    const el = angularEdges(frame.elevation_deg).map((v) => Math.max(-89, Math.min(89, v)));
    this.edges = {az, el};
    const inset = Math.min(0.6, (az[1] - az[0]) * 0.05, (el[1] - el[0]) * 0.05);
    for (let row = 0; row < cfg.rows; row++) {
      for (let column = 0; column < cfg.columns; column++) {
        const mesh = new THREE.Mesh(
          tileGeometry(az[column] + inset, az[column + 1] - inset, el[row] + inset, el[row + 1] - inset, RADIUS),
          new THREE.MeshBasicMaterial({color: COLORS.surface, transparent: true, opacity: 0.95, side: THREE.DoubleSide}),
        );
        mesh.userData = {row, column, sector: row * cfg.columns + column};
        this.tiles.add(mesh);
        this.tileMeshes.push(mesh);
      }
    }
    this.buildTickLabels();
  }

  buildTickLabels() {
    const {az, el} = this.edges;
    this.labelNodes.forEach((node) => node.el.remove());
    this.labelNodes = [];
    const add = (vec, text, align) => {
      const el = h('span', {class: `scene-label scene-label--${align}`}, text);
      this.labels.append(el);
      this.labelNodes.push({el, point: toScene(vec, RADIUS)});
    };
    const azTicks = [az[0], 0, az.at(-1)].filter((v, i, all) => v >= az[0] && v <= az.at(-1) && all.indexOf(v) === i);
    const elTicks = [el[0], 0, el.at(-1)].filter((v, i, all) => v >= el[0] && v <= el.at(-1) && all.indexOf(v) === i);
    for (const a of azTicks) add(direction(a, el[0] - 6), `az ${deg(a, 0)}`, 'center');
    for (const e of elTicks) add(direction(az[0] - 5, e), `el ${deg(e, 0)}`, 'end');
    const arrayLabel = h('span', {class: 'scene-label scene-label--center scene-label--muted'}, 'Array');
    this.labels.append(arrayLabel);
    this.labelNodes.push({el: arrayLabel, point: new THREE.Vector3(0, -0.22, 0)});
    this.markerLabel = h('span', {class: 'scene-label scene-label--marker', hidden: true});
    this.labels.append(this.markerLabel);
    this.markerPoint = null;
  }

  placeLabels() {
    for (const node of this.labelNodes) {
      const p = this.viewport.project(node.point);
      node.el.style.left = `${p.x.toFixed(1)}px`;
      node.el.style.top = `${p.y.toFixed(1)}px`;
      node.el.hidden = !p.visible;
    }
    if (this.markerLabel && this.markerPoint) {
      const p = this.viewport.project(this.markerPoint);
      this.markerLabel.style.left = `${(p.x + 14).toFixed(1)}px`;
      this.markerLabel.style.top = `${(p.y - 12).toFixed(1)}px`;
    }
  }

  render(frame, {floor, session}) {
    const cfg = frame?.configuration;
    if (!cfg || !frame.levels_db || !frame.azimuth_deg?.length || !frame.elevation_deg?.length) return;
    this.build(frame);
    this.setArray(session.arrayLayout);

    for (const mesh of this.tileMeshes) {
      const {row, column} = mesh.userData;
      const level = frame.levels_db?.[row]?.[column];
      const measured = Number.isFinite(level);
      const age = session.cellAge(row, column);
      const fresh = measured && Number.isFinite(age) && age <= STALE_MS;
      mesh.material.color.copy(measured ? srgb(rampColor(levelToUnit(level, floor))) : COLORS.line);
      mesh.material.opacity = measured ? (fresh ? 0.95 : 0.4) : 0.35;
    }

    clearGroup(this.highlights);
    clearGroup(this.pointer);
    const {az, el} = this.edges;
    const outline = (row, column, color, dashed = false, width = 1) => {
      const points = outlinePoints(az[column], az[column + 1], el[row], el[row + 1], RADIUS * 0.992);
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = dashed
        ? new THREE.LineDashedMaterial({color, dashSize: 0.04, gapSize: 0.03, linewidth: width})
        : new THREE.LineBasicMaterial({color, linewidth: width});
      const line = new THREE.Line(geometry, material);
      if (dashed) line.computeLineDistances();
      this.highlights.add(line);
    };

    const target = frame.target;
    const steer = !target ? frame.last_steer : null;
    const strongest = !target && !steer ? session.strongest(frame) : null;
    const marker = target || steer || strongest;
    if (this.hover !== null) {
      const {row, column} = this.tileMeshes[this.hover]?.userData || {};
      if (row !== undefined) outline(row, column, COLORS.text);
    }
    if (steer) outline(steer.row, steer.column, COLORS.accent, true);
    if (marker) {
      const a = frame.azimuth_deg[marker.column];
      const e = frame.elevation_deg[marker.row];
      const tip = toScene(direction(a, e), RADIUS);
      if (!steer) outline(marker.row, marker.column, new THREE.Color('#ffffff'));
      const bearing = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), tip]),
        new THREE.LineDashedMaterial({color: steer ? COLORS.accent : COLORS.text, dashSize: 0.06, gapSize: 0.06}),
      );
      bearing.computeLineDistances();
      this.pointer.add(bearing);
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.022, 16, 12), new THREE.MeshBasicMaterial({color: 0xffffff}));
      dot.position.copy(tip);
      this.pointer.add(dot);
      this.markerPoint = tip;
      if (this.markerLabel) {
        this.markerLabel.hidden = false;
        setText(this.markerLabel, `${deg(a)}, ${deg(e)}`);
      }
    } else {
      this.markerPoint = null;
      if (this.markerLabel) this.markerLabel.hidden = true;
    }

    const truth = frame.emulator_truth;
    if (truth) {
      const color = truth.source_active === false ? COLORS.warn : COLORS.ok;
      const point = toScene(direction(truth.azimuth_deg, truth.elevation_deg), RADIUS * 1.06);
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), point]),
        new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.7}),
      );
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.03, 16, 12), new THREE.MeshBasicMaterial({color}));
      dot.position.copy(point);
      this.pointer.add(line, dot);
    }
    this.viewport.requestRender();
  }

  bindPointer() {
    const canvas = this.viewport.canvas;
    let down = null;
    canvas.addEventListener('pointerdown', (event) => { down = {x: event.clientX, y: event.clientY}; });
    canvas.addEventListener('pointermove', (event) => {
      if (event.buttons) return;
      const hit = this.viewport.pick(event, this.tileMeshes)[0];
      const sector = hit ? hit.object.userData.sector : null;
      if (sector !== this.hover) {
        this.hover = sector;
        canvas.style.cursor = sector === null ? 'grab' : 'pointer';
        this.onHover?.(sector);
      }
    });
    canvas.addEventListener('pointerleave', () => {
      if (this.hover !== null) {
        this.hover = null;
        this.onHover?.(null);
      }
    });
    canvas.addEventListener('pointerup', (event) => {
      if (!down || Math.hypot(event.clientX - down.x, event.clientY - down.y) > 4) { down = null; return; }
      down = null;
      const hit = this.viewport.pick(event, this.tileMeshes)[0];
      if (hit) this.onPick?.(hit.object.userData.sector);
    });
  }
}
