/**
 * scene.js — the simulated room: walls, floor grid, the array, the drone
 * (and its trajectory), crowd talkers, PA speakers, first reflections, and
 * the true and estimated directions. Room coordinates are metres with z up.
 */

import * as THREE from 'three';
import { Viewport3D, clearGroup } from '../shared/viewport3d.js';
import { h, setText } from '../shared/dom.js';

const VIEWS = {angled: {yaw: 32, pitch: 28}, plan: {yaw: 0, pitch: 88}};
const COLOR = {
  line: 0x2c3036, edge: 0x5a6068, text: 0xe9eaec, muted: 0xa3a9b0, accent: 0x7aa7ff, faint: 0x5a6068,
};

export class RoomScene {
  constructor(container) {
    this.viewport = new Viewport3D(container, {background: '#1a1c20', fov: 40});
    this.labels = h('div', {class: 'scene-labels', 'aria-hidden': 'true'});
    container.append(this.labels);
    this.group = new THREE.Group();
    this.viewport.scene.add(this.group);
    this.labelNodes = [];
    this.scale = 1;
    this.viewport.onFrame = () => this.placeLabels();
    this.view = 'angled';
  }

  start() { this.viewport.start(); }
  stop() { this.viewport.stop(); }

  setView(name) {
    this.view = name;
    this.viewport.orbitTo(new THREE.Vector3(0, 0.5, 0), {...VIEWS[name], distance: 11});
  }

  /** Room point (x, y, z-up) relative to the array centre, scaled to ~6 units. */
  point(p, center) {
    return new THREE.Vector3((p[0] - center[0]) * this.scale, (p[2] - center[2]) * this.scale, -(p[1] - center[1]) * this.scale);
  }

  placeLabels() {
    for (const node of this.labelNodes) {
      const p = this.viewport.project(node.point);
      node.el.style.left = `${p.x.toFixed(1)}px`;
      node.el.style.top = `${p.y.toFixed(1)}px`;
      node.el.hidden = !p.visible;
    }
  }

  addLabel(text, point, className) {
    const el = h('span', {class: `scene-label ${className}`}, text);
    this.labels.append(el);
    this.labelNodes.push({el, point});
  }

  /**
   * scene: {room: [L, W, H], center: [x, y, z], source, trajectory, crowd, pa,
   *         images, truth: {az, el}, estimate: {az, el}, distance}
   */
  render(scene) {
    clearGroup(this.group);
    this.labels.replaceChildren();
    this.labelNodes = [];
    const {room, center} = scene;
    this.scale = 6 / Math.max(room[0], room[1], room[2]);
    const at = (p) => this.point(p, center);

    const box = new THREE.BoxGeometry(room[0] * this.scale, room[2] * this.scale, room[1] * this.scale);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({color: COLOR.edge}));
    edges.position.copy(at([room[0] / 2, room[1] / 2, room[2] / 2]));
    this.group.add(edges);
    box.dispose();

    const grid = [];
    const step = room[0] > 40 || room[1] > 40 ? 10 : 5;
    for (let x = 0; x <= room[0] + 1e-6; x += step) grid.push(at([x, 0, 0]), at([x, room[1], 0]));
    for (let y = 0; y <= room[1] + 1e-6; y += step) grid.push(at([0, y, 0]), at([room[0], y, 0]));
    this.group.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(grid), new THREE.LineBasicMaterial({color: COLOR.line})));

    const dot = (p, radius, color, opacity = 1) => {
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 14, 10),
        new THREE.MeshBasicMaterial({color, transparent: opacity < 1, opacity}));
      mesh.position.copy(at(p));
      this.group.add(mesh);
      return mesh;
    };

    for (const p of scene.images || []) dot(p, 0.05, COLOR.faint, 0.35);
    for (const p of scene.crowd || []) dot(p, 0.04, COLOR.muted);
    for (const p of scene.pa || []) {
      const cube = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.09, 0.09), new THREE.MeshBasicMaterial({color: COLOR.muted}));
      cube.position.copy(at(p));
      this.group.add(cube);
    }

    const origin = at(center);
    dot(center, 0.07, COLOR.text);
    this.addLabel('Array', origin.clone().add(new THREE.Vector3(0, -0.3, 0)), 'scene-label--center scene-label--below');

    if (scene.trajectory?.length > 1) {
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(scene.trajectory.map(at)),
        new THREE.LineDashedMaterial({color: COLOR.text, dashSize: 0.08, gapSize: 0.06}));
      line.computeLineDistances();
      this.group.add(line);
    }

    if (scene.source) {
      const drone = at(scene.source);
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.1, 0.012, 8, 32), new THREE.MeshBasicMaterial({color: COLOR.text}));
      ring.position.copy(drone);
      ring.lookAt(this.viewport.camera.position);
      this.group.add(ring);
      dot(scene.source, 0.035, COLOR.text);
      this.group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([origin, drone]),
        new THREE.LineBasicMaterial({color: COLOR.text})));
      this.addLabel('Drone', drone.clone().add(new THREE.Vector3(0, 0.22, 0)), 'scene-label--center scene-label--above');
    }

    if (scene.estimate && Number.isFinite(scene.estimate.az)) {
      const a = scene.estimate.az * Math.PI / 180;
      const e = scene.estimate.el * Math.PI / 180;
      const d = scene.distance || 4;
      const tip = at([center[0] + d * Math.cos(e) * Math.cos(a), center[1] + d * Math.cos(e) * Math.sin(a), center[2] + d * Math.sin(e)]);
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([origin, tip]),
        new THREE.LineDashedMaterial({color: COLOR.accent, dashSize: 0.12, gapSize: 0.08}));
      line.computeLineDistances();
      this.group.add(line);
      const end = new THREE.Mesh(new THREE.SphereGeometry(0.05, 14, 10), new THREE.MeshBasicMaterial({color: COLOR.accent}));
      end.position.copy(tip);
      this.group.add(end);
      this.addLabel('Estimate', tip.clone().add(new THREE.Vector3(0.18, -0.12, 0)), 'scene-label--accent');
    }
    if (!this.framed) {
      this.framed = true;
      this.setView(this.view);
    }
    this.viewport.requestRender();
  }
}

export function describeRoom(room) {
  return `${room.map((v) => Number(v).toFixed(0)).join(' × ')} m room`;
}

export function setCaption(el, room) {
  setText(el, `${describeRoom(room)}. Drag to rotate.`);
}
