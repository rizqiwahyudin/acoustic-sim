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
import { look, sceneColor } from '../shared/look.js';
import { angularEdges, direction } from './geometry.js';
import { STALE_MS } from './session.js';
import { buildRig } from './rig3d.js';
import { HALO_OUTSET, floorMesh, haloMaterial, setHalo, setTile, tileMaterial } from './look3d.js';
import { ditherStep, meterFill } from './lookparts.js';
import { Motion, prefersReducedMotion, tileHandle } from './build3d.js';

const RADIUS = 1.25;
const TARGET = new THREE.Vector3(0, 0, 0.7);
const DISTANCE = 2.9;
export const CAMERA_VIEWS = {
  behind: {yaw: 24, pitch: 14},
  above: {yaw: 0, pitch: 62},
  side: {yaw: 78, pitch: 8},
};

const THEMES = {
  instrument: {
    background: '#1a1c20', ramp: 'inferno', line: '#2c3036', text: '#e9eaec', accent: '#7aa7ff',
    labels: true, trail: false, ring: false, fov: 40, micSize: 0.011, floor: {half: 1.2, depth: [-0.3, 1.5], y: -0.9},
  },
  exhibit: {
    background: '#08090b', ramp: 'ember', line: '#1a1d21', text: '#f4f4f2', accent: '#ff5a36',
    labels: false, trail: true, ring: true, fov: 52, micSize: 0.014, floor: {half: 1.6, depth: [-0.4, 1.6], y: -0.95},
  },
};

const OK = new THREE.Color('#4cc38a');
const WARN = new THREE.Color('#e2b454');
const MARKER = sceneColor('marker', '#ffffff');

/** A theme in the current look: its scene colours swapped for the look's (look.js); the ramp follows colormap.js. */
function lookTheme(theme) {
  return {...theme, background: sceneColor('background', theme.background), line: sceneColor('line', theme.line),
    text: sceneColor('text', theme.text), accent: sceneColor('accent', theme.accent)};
}

function toScene(vec, radius = 1) {
  return new THREE.Vector3(-vec[0] * radius, vec[1] * radius, vec[2] * radius);
}

function srgb(rgb) {
  return new THREE.Color().setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, THREE.SRGBColorSpace);
}

function tileGeometry(a0, a1, e0, e1, radius, steps = 6) {
  const positions = [];
  const uvs = [];
  const index = [];
  for (let i = 0; i <= steps; i++) {
    for (let j = 0; j <= steps; j++) {
      const a = a0 + (a1 - a0) * j / steps;
      const e = e0 + (e1 - e0) * i / steps;
      const p = toScene(direction(a, e), radius);
      positions.push(p.x, p.y, p.z);
      uvs.push(j / steps, i / steps);
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
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
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

const RAY_ALPHA = [1, 0.82, 0.58, 0.45, 0.45, 0.45];   // Arasaka's rays by dither step: quiet ones no fainter than 45 %
const SWAY = {degrees: 16, period: 36};                 // the hidden looks' slow camera sway on Present, until a drag

/** A ray of unit length along +y from the origin, fading from 30 % at the array to full at its tip. */
function rayGeometry() {
  const geometry = new THREE.CylinderGeometry(0.0035, 0.0035, 1, 6, 1, true).translate(0, 0.5, 0);
  const y = geometry.getAttribute('position');
  const colors = [];
  for (let i = 0; i < y.count; i++) colors.push(1, 1, 1, 0.3 + 0.7 * y.getY(i));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 4));
  return geometry;
}

export class SectorScene {
  constructor(container, {onPick, onHover, theme = 'instrument', beams = false} = {}) {
    this.container = container;
    this.onPick = onPick;
    this.onHover = onHover;
    this.theme = lookTheme(THEMES[theme] || THEMES.instrument);
    this.colors = {
      line: new THREE.Color(this.theme.line),
      text: new THREE.Color(this.theme.text),
      accent: new THREE.Color(this.theme.accent),
    };
    this.viewport = new Viewport3D(container, {background: this.theme.background, fov: this.theme.fov});
    this.labels = h('div', {class: 'scene-labels', 'aria-hidden': 'true'});
    container.append(this.labels);

    this.floor = new THREE.Group();
    this.tiles = new THREE.Group();
    this.highlights = new THREE.Group();
    this.array = new THREE.Group();
    this.pointer = new THREE.Group();
    this.rays = new THREE.Group();
    this.tips = new THREE.Group();
    this.viewport.scene.add(this.floor, this.array, this.tiles, this.highlights, this.pointer, this.rays, this.tips);
    this.buildFloor();
    // Arasaka's Beams display (look.js): a ray per sector from the array instead of the tiles' fill; `display` picks it.
    this.beamsAllowed = beams;
    this.display = beams ? 'beams' : 'tiles';
    this.rayParts = [];

    this.key = '';
    this.layoutKey = '';
    this.tileMeshes = [];
    this.edges = {az: [], el: []};
    this.hover = null;
    this.labelNodes = [];
    this.markerLabel = null;

    // Startup and idle motion in every look (build3d.js); reaches scaled to this dome (1.25 m against Hardware's 3.5 m).
    this.grid = {columns: 1, rows: 1};
    this.motion = new Motion({look: look || 'default', viewport: this.viewport, parts: () => this.motionParts(),
      reach: {wave: 2.25, ring: 1.34, pulse: 0.68, jitter: 0.013}});
    this.sway = null;
    const building = this.viewport.animating;
    this.viewport.animating = () => building() || this.swaying();
    this.viewport.onFrame = () => {
      if (this.swaying()) this.swayStep();
      this.motion?.apply();
      this.placeLabels();
    };
    this.bindPointer();
    this.setView('behind');
  }

  swaying() {
    return Boolean(this.sway) && !this.sway.manual && !prefersReducedMotion();
  }

  swayStep() {
    const {yaw, pitch, distance, t0} = this.sway;
    const t = performance.now() / 1000 - t0;
    this.viewport.orbitTo(TARGET, {yaw: yaw + SWAY.degrees * Math.sin(t * 2 * Math.PI / SWAY.period), pitch, distance});
  }

  /** 'beams' (Arasaka's rays, where allowed) or 'tiles'. */
  setDisplay(name) {
    this.display = this.beamsAllowed && name === 'beams' ? 'beams' : 'tiles';
    this.viewport.requestRender();
  }

  start() { this.viewport.start(); }
  stop() { this.viewport.stop(); }

  /** Play the 3D build: 'full' or 'quick'. */
  play(kind) { this.motion?.play(kind); }

  /** The microphone the build has reached (1–44), or null outside a build. */
  buildChannel() { return this.motion?.channel() ?? null; }

  get building() { return Boolean(this.motion?.build); }

  motionParts() {
    return {
      rig: this.rig,
      floor: this.floorMesh,
      tiles: this.tileMeshes.map((mesh, i) => tileHandle(mesh, this.halos?.[i])),
      columns: this.grid.columns,
      rows: this.grid.rows,
      marks: [this.pointer, this.highlights],
      labels: [this.markerLabel],
      rays: this.display === 'beams' ? this.rays : null,
      tips: this.display === 'beams' ? this.tips : null,
    };
  }

  setView(name) {
    const view = CAMERA_VIEWS[name] || CAMERA_VIEWS.behind;
    this.viewport.orbitTo(TARGET, {...view, distance: DISTANCE});
  }

  /** Place the camera; it then sways slowly round that place until the view is dragged (Present's motion). */
  setOrbit(yaw, pitch, distance = DISTANCE) {
    this.viewport.orbitTo(TARGET, {yaw, pitch, distance});
    this.sway = {yaw, pitch, distance, t0: performance.now() / 1000, manual: false};
  }

  buildFloor() {
    if (look) {
      // The hidden looks' floor (look3d.js), scaled to this smaller scene: the dome here is 1.25 m, not 3.5 m.
      const {y} = this.theme.floor;
      this.floorMesh = floorMesh(look, {y, centre: [0, 0.45], reach: 2.6, size: 6});
      this.floorMesh.material.uniforms.uSpacing.value *= 0.45;
      this.floorMesh.material.uniforms.uDot.value *= 0.6;
      this.floor.add(this.floorMesh);
      return;
    }
    const {half, depth: [near, far], y} = this.theme.floor;
    const points = [];
    const steps = Math.round(half / 0.2);
    for (let i = 0; i <= steps; i++) {
      const x = -half + i * (2 * half / steps);
      points.push(new THREE.Vector3(x, y, near), new THREE.Vector3(x, y, far));
      const z = near + i * ((far - near) / steps);
      points.push(new THREE.Vector3(-half, y, z), new THREE.Vector3(half, y, z));
    }
    // Fades in during the build, so transparent; drawn first, as when it was opaque.
    this.floorMesh = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({color: this.colors.line, transparent: true}));
    this.floorMesh.renderOrder = -1;
    this.floor.add(this.floorMesh);
  }

  /** The array as its real boards on the CAD mount and desk (rig3d.js), standing on this scene's floor. */
  setArray(layout) {
    const key = layout ? JSON.stringify(layout.positions) : '';
    if (key === this.layoutKey) return;
    this.layoutKey = key;
    clearGroup(this.array);
    this.rig = null;
    if (!layout?.positions?.length) return;
    this.rig = buildRig({look: look || 'default', layout, floorY: this.theme.floor.y});
    this.array.add(this.rig.group);
    this.viewport.requestRender();
  }

  build(frame) {
    const cfg = frame.configuration;
    const key = `${cfg.rows}x${cfg.columns}|${frame.azimuth_deg.join(',')}|${frame.elevation_deg.join(',')}`;
    if (key === this.key) return;
    this.key = key;
    clearGroup(this.tiles);
    this.tileMeshes = [];
    this.halos = [];
    this.grid = {columns: cfg.columns, rows: cfg.rows};
    const az = angularEdges(frame.azimuth_deg);
    const el = angularEdges(frame.elevation_deg).map((v) => Math.max(-89, Math.min(89, v)));
    this.edges = {az, el};
    const inset = Math.min(0.6, (az[1] - az[0]) * 0.05, (el[1] - el[0]) * 0.05);
    for (let row = 0; row < cfg.rows; row++) {
      for (let column = 0; column < cfg.columns; column++) {
        const [a0, a1, e0, e1] = [az[column] + inset, az[column + 1] - inset, el[row] + inset, el[row + 1] - inset];
        const mesh = new THREE.Mesh(
          tileGeometry(a0, a1, e0, e1, RADIUS),
          look ? tileMaterial(look) : new THREE.MeshBasicMaterial({color: this.colors.line, transparent: true, opacity: 0.95, side: THREE.DoubleSide}),
        );
        mesh.userData = {row, column, sector: row * cfg.columns + column, opacity: 0.95};
        this.tiles.add(mesh);
        this.tileMeshes.push(mesh);
        if (look) {
          // The hidden looks ring the target and glow the loudest tiles from a halo just behind each tile (look3d.js).
          const da = (a1 - a0) * HALO_OUTSET;
          const de = (e1 - e0) * HALO_OUTSET;
          const halo = new THREE.Mesh(tileGeometry(a0 - da, a1 + da, e0 - de, e1 + de, RADIUS * 1.004), haloMaterial(look));
          halo.visible = false;
          this.tiles.add(halo);
          this.halos.push(halo);
        }
      }
    }
    if (this.beamsAllowed) this.buildRays(frame);
    this.buildTickLabels();
  }

  /** Arasaka's Beams display: a ray from the array toward each sector's centre, with a square tip facing the array. */
  buildRays(frame) {
    clearGroup(this.rays);
    clearGroup(this.tips);
    this.rayParts = [];
    const geometry = rayGeometry();
    const tip = new THREE.RingGeometry(0.016, 0.02, 4, 1, Math.PI / 4);
    const up = new THREE.Vector3(0, 1, 0);
    for (const mesh of this.tileMeshes) {
      const {row, column} = mesh.userData;
      const along = toScene(direction(frame.azimuth_deg[column], frame.elevation_deg[row])).normalize();
      const ray = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({color: 0xf2122a, vertexColors: true, transparent: true, depthWrite: false}));
      ray.quaternion.setFromUnitVectors(up, along);
      const square = new THREE.Mesh(tip, new THREE.MeshBasicMaterial({color: 0xf2122a, transparent: true, side: THREE.DoubleSide}));
      this.rays.add(ray);
      this.tips.add(square);
      this.rayParts.push({ray, tip: square, along});
    }
  }

  /** Ray lengths and strengths: further and brighter the more a sector stands above the median; the target's in ink. */
  renderRays(frame, floor, marker) {
    const levels = this.tileMeshes.map(({userData: {row, column}}) => frame.levels_db?.[row]?.[column]);
    const measured = levels.filter(Number.isFinite).sort((a, b) => a - b);
    const median = measured.length ? measured[Math.floor(measured.length / 2)] : floor;
    this.rayParts.forEach(({ray, tip, along}, i) => {
      const level = levels[i];
      const {row, column} = this.tileMeshes[i].userData;
      const hot = Boolean(marker) && marker.row === row && marker.column === column;
      const relief = Number.isFinite(level) ? Math.max(0, Math.min(1, (level - median) / 24)) : 0;
      const length = RADIUS * (0.3 + 0.7 * relief);
      const alpha = hot ? 1 : RAY_ALPHA[ditherStep(level, floor) ?? 5];
      ray.scale.set(1, length, 1);
      ray.material.color.set(hot ? '#e9e3dc' : '#f2122a');
      ray.material.opacity = alpha;
      tip.position.copy(along).multiplyScalar(length);
      tip.lookAt(0, 0, 0);
      tip.material.color.set(hot ? '#e9e3dc' : '#f2122a');
      tip.material.opacity = alpha;
    });
  }

  buildTickLabels() {
    if (!this.theme.labels) return;
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

  /** state: {floor, session, idle: nothing running, searching: the glow wave alone (the hidden looks' motion)} */
  render(frame, {floor, session, idle = false, searching = false, marker: markerOverride} = {}) {
    const cfg = frame?.configuration;
    if (!cfg || !frame.levels_db || !frame.azimuth_deg?.length || !frame.elevation_deg?.length) return;
    this.motion?.setIdle(idle, idle || searching);
    this.build(frame);
    this.setArray(session.arrayLayout);

    const target = frame.target;
    const steer = !target ? frame.last_steer : null;
    const strongest = !target && !steer ? session.strongest(frame) : null;
    const marker = markerOverride === undefined ? target || steer || strongest : markerOverride;
    const pixel = Math.max(1, Math.round(this.viewport.renderer.getPixelRatio()));
    const beams = this.display === 'beams';
    this.rays.visible = beams;
    this.tips.visible = beams && !this.motion?.build;
    if (beams) this.renderRays(frame, floor, marker);
    this.tileMeshes.forEach((mesh, i) => {
      const {row, column} = mesh.userData;
      const level = frame.levels_db?.[row]?.[column];
      const measured = Number.isFinite(level);
      const age = session.cellAge(row, column);
      const fresh = measured && Number.isFinite(age) && age <= STALE_MS;
      if (look) {
        const state = {step: ditherStep(level, floor), fill: meterFill(level, floor),
          target: Boolean(marker) && marker.row === row && marker.column === column};
        setTile(mesh.material, look, {...state, opacity: beams ? 1 : measured ? (fresh ? 1 : 0.55) : 0.5});
        mesh.material.uniforms.uPixel.value = pixel;
        mesh.material.uniforms.uWire.value = beams ? 1 : 0;
        setHalo(this.halos[i], look, state);
        if (beams) this.halos[i].visible = false;
        return;
      }
      // Kept on the tile too: the build's fade and the idle glow wave go on top of them (build3d.js tileHandle).
      mesh.userData.color = measured ? srgb(rampColor(levelToUnit(level, floor), this.theme.ramp)) : this.colors.line.clone();
      mesh.userData.opacity = measured ? (fresh ? 0.96 : 0.38) : 0.35;
      mesh.material.color.copy(mesh.userData.color);
      mesh.material.opacity = mesh.userData.opacity;
    });

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

    if (this.hover !== null) {
      const {row, column} = this.tileMeshes[this.hover]?.userData || {};
      if (row !== undefined) outline(row, column, this.colors.text);
    }
    if (marker?.kind === 'beam') outline(marker.row, marker.column, this.colors.accent, true);
    if (marker && beams) {
      // In the Beams display the target's own ray, in ink, is the pointer.
      const hot = this.rayParts[this.tileMeshes.findIndex((m) => m.userData.row === marker.row && m.userData.column === marker.column)];
      this.markerPoint = hot ? hot.tip.position.clone() : null;
      if (this.markerLabel) {
        this.markerLabel.hidden = !hot;
        setText(this.markerLabel, `${deg(frame.azimuth_deg[marker.column])}, ${deg(frame.elevation_deg[marker.row])}`);
      }
    } else if (marker) {
      const a = frame.azimuth_deg[marker.column];
      const e = frame.elevation_deg[marker.row];
      const tip = toScene(direction(a, e), RADIUS);
      if (!steer && !look) outline(marker.row, marker.column, new THREE.Color(MARKER));
      const bearing = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), tip]),
        new THREE.LineDashedMaterial({color: steer ? this.colors.accent : this.colors.text, dashSize: 0.06, gapSize: 0.06}),
      );
      bearing.computeLineDistances();
      this.pointer.add(bearing);
      const dot = new THREE.Mesh(new THREE.SphereGeometry(this.theme.ring ? 0.018 : 0.022, 16, 12), new THREE.MeshBasicMaterial({color: MARKER}));
      dot.position.copy(tip);
      this.pointer.add(dot);
      if (this.theme.ring) {
        const ring = new THREE.Mesh(new THREE.RingGeometry(0.05, 0.062, 40), new THREE.MeshBasicMaterial({color: MARKER, side: THREE.DoubleSide}));
        ring.position.copy(tip);
        ring.lookAt(0, 0, 0);
        this.pointer.add(ring);
      }
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
      const color = truth.source_active === false ? WARN : OK;
      const point = toScene(direction(truth.azimuth_deg, truth.elevation_deg), RADIUS * 1.06);
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), point]),
        new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.7}),
      );
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.03, 16, 12), new THREE.MeshBasicMaterial({color}));
      dot.position.copy(point);
      this.pointer.add(line, dot);
    }
    if (this.theme.trail) this.drawTrail(session, frame);
    this.motion?.apply();
    this.viewport.requestRender();
  }

  /** Where the answer has been over the last 15 s, as a tube on the sphere. */
  drawTrail(session, frame) {
    const latest = Number.isFinite(frame.timestamp_s) ? frame.timestamp_s : Date.now() / 1000;
    const points = session.history
      .filter((item) => latest - item.t <= 15 && Number.isFinite(item.azimuth) && Number.isFinite(item.elevation))
      .map((item) => toScene(direction(item.azimuth, item.elevation), RADIUS * 1.02));
    const unique = points.filter((p, i) => i === 0 || p.distanceTo(points[i - 1]) > 1e-4);
    if (unique.length < 2) return;
    const curve = new THREE.CatmullRomCurve3(unique, false, 'centripetal');
    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, Math.min(400, unique.length * 6), 0.008, 6, false),
      new THREE.MeshBasicMaterial({color: this.colors.accent, transparent: true, opacity: 0.55}),
    );
    this.pointer.add(tube);
  }

  bindPointer() {
    const canvas = this.viewport.canvas;
    let down = null;
    canvas.addEventListener('pointerdown', (event) => {
      down = {x: event.clientX, y: event.clientY};
      if (this.sway) this.sway.manual = true;
    });
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
