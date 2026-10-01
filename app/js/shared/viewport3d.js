/**
 * viewport3d.js — a Three.js view that lives inside one container instead of
 * taking over the window. It renders only when something changed and only
 * while the container is visible, so several views can coexist.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { currentZoom, onZoom } from './scale.js';

export class Viewport3D {
  constructor(container, {background = '#1a1c20', fov = 40, near = 0.02, far = 200} = {}) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({antialias: true});
    this.applyPixelRatio();
    this.renderer.setClearColor(new THREE.Color(background));
    this.canvas = this.renderer.domElement;
    this.canvas.className = 'viewport3d__canvas';
    container.append(this.canvas);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(fov, 1, near, far);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.addEventListener('change', () => this.requestRender());

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const key = new THREE.DirectionalLight(0xffffff, 0.55);
    key.position.set(-2, 4, -3);
    this.scene.add(key);

    this.onFrame = null;
    this.dirty = true;
    this.running = false;
    this.width = 0;
    this.height = 0;
    this.pointer = new THREE.Vector2();
    this.raycaster = new THREE.Raycaster();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.offZoom = onZoom(() => { this.applyPixelRatio(); this.resize(true); });
    this.loop = this.loop.bind(this);
  }

  /** The app is zoomed to fit the window, so render at the zoomed resolution. */
  applyPixelRatio() {
    this.renderer.setPixelRatio(Math.min((window.devicePixelRatio || 1) * currentZoom(), 3));
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.resize();
    requestAnimationFrame(this.loop);
  }

  stop() {
    this.running = false;
  }

  requestRender() {
    this.dirty = true;
  }

  loop() {
    if (!this.running) return;
    requestAnimationFrame(this.loop);
    const moving = this.controls.update();
    if (!(moving || this.dirty) || this.width === 0 || this.height === 0) return;
    this.dirty = false;
    this.onFrame?.();
    this.renderer.render(this.scene, this.camera);
  }

  resize(force = false) {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (!width || !height) return;
    if (!force && width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    this.renderer.setSize(width, height, false);
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  /** Screen position (CSS px, container-relative) of a world point. */
  project(point) {
    const v = point.clone().project(this.camera);
    return {x: (v.x + 1) / 2 * this.width, y: (1 - v.y) / 2 * this.height, visible: v.z > -1 && v.z < 1};
  }

  /** Objects under a pointer event, nearest first. */
  pick(event, objects) {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(this.pointer, this.camera);
    return this.raycaster.intersectObjects(objects, false);
  }

  /** Place the camera on a sphere around `target` (angles in degrees). */
  orbitTo(target, {yaw, pitch, distance}) {
    const yawRad = yaw * Math.PI / 180;
    const pitchRad = pitch * Math.PI / 180;
    this.controls.target.copy(target);
    this.camera.position.set(
      target.x + distance * Math.sin(yawRad) * Math.cos(pitchRad),
      target.y + distance * Math.sin(pitchRad),
      target.z - distance * Math.cos(yawRad) * Math.cos(pitchRad),
    );
    this.camera.lookAt(target);
    this.controls.update();
    this.requestRender();
  }

  dispose() {
    this.stop();
    this.resizeObserver.disconnect();
    this.offZoom?.();
    this.controls.dispose();
    this.renderer.dispose();
    this.canvas.remove();
  }
}

/** Dispose every geometry/material under a group, then empty it. */
export function clearGroup(group) {
  group.traverse((object) => {
    object.geometry?.dispose?.();
    const material = object.material;
    if (Array.isArray(material)) material.forEach((m) => m.dispose?.());
    else material?.dispose?.();
  });
  group.clear();
}
