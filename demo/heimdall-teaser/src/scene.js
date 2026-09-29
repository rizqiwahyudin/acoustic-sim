/**
 * HEIMDALL teaser, one continuous world driven by time alone.
 *
 * A valley whose ridges are the drone's real spectrum. A watchman and the
 * array on its mast. Sparks rise into the 44 microphones; a wavefront falls
 * from the sky and crosses them in delay order; an A.T.-field-like hex ripple;
 * the wires converge into one beam; a sector of the sky locks; the camera
 * climbs the beam to the drone, whose spectrum unrolls behind it as a second
 * valley: the landscape was its sound all along.
 *
 * The array's own frame faces +z; it stands at ARRAY_POS tilted up by TILT.
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import {
  W, H, clamp, lerp, range01, smooth, easeInOut, easeOut, window01, hash, WIRE, VIOLET, MAGENTA, SIGNAL,
} from './util.js';

// The story (T and the camera keys) is blocked out in 30 s of story time. PACE maps
// wall-clock seconds to story seconds so it plays slower on screen: most of all
// through the lock and the climb to the drone, while the title keeps its tempo.
// The first knot is the black intro that carries the epigraph (see overlay.js).
const STORY = 30;
const PACE = [[0, 0], [11.0, 4.6], [21.0, 10.6], [28.0, 15.0], [32.8, 17.4], [37.4, 20.0], [43.4, 23.8], [44.0, 24.2], [50.0, STORY]];
export const DURATION = PACE.at(-1)[0];
/** Wall-clock seconds of black before the world fades in. */
export const INTRO = PACE[1][0];

// Monotone cubic (Fritsch-Butland) slopes, so playback speed never jumps.
const PACE_SLOPES = PACE.map(([x, y], k) => {
  const secant = (i) => (PACE[i + 1][1] - PACE[i][1]) / (PACE[i + 1][0] - PACE[i][0]);
  if (k === 0) return secant(0);
  if (k === PACE.length - 1) return secant(k - 1);
  const h0 = x - PACE[k - 1][0]; const h1 = PACE[k + 1][0] - x; const d0 = secant(k - 1); const d1 = secant(k);
  return d0 * d1 <= 0 ? 0 : (3 * (h0 + h1)) / ((2 * h1 + h0) / d0 + (h1 + 2 * h0) / d1);
});

/** Story time shown at wall-clock time t. */
export function storyTime(t) {
  const x = clamp(t, 0, DURATION);
  let k = 0;
  while (k < PACE.length - 2 && x > PACE[k + 1][0]) k++;
  const [x0, y0] = PACE[k]; const [x1, y1] = PACE[k + 1]; const h = x1 - x0; const s = (x - x0) / h;
  return (2 * s ** 3 - 3 * s ** 2 + 1) * y0 + (s ** 3 - 2 * s ** 2 + s) * h * PACE_SLOPES[k]
    + (-2 * s ** 3 + 3 * s ** 2) * y1 + (s ** 3 - s ** 2) * h * PACE_SLOPES[k + 1];
}

/** Wall-clock time at which story time s is shown. */
export function wallTime(s) {
  let lo = 0; let hi = DURATION;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (storyTime(mid) < s) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

export const T = {
  fadeIn: 4.6, subtitle: 6.2, subtitleOut: 9.2,
  frameDraw: 8.8, ignite: 9.0,
  listen: 10.6, listenOut: 12.6,
  wave: 12.6, cross: 15.0,
  beam: 15.7, dome: 15.6, lock: 16.5,
  capture: 16.6, captureOut: 18.4,
  ascend: 17.4, drone: 19.2, print: 20.0, label: 20.8,
  fadeOut: 23.0, black: 23.8,
  title: 24.2, end: STORY,
};
export const SOURCE = { az: 30, el: 20 };
export const ARRAY_POS = new THREE.Vector3(0, 2.3, -12);
const TILT = (20 * Math.PI) / 180;
const X_AXIS = new THREE.Vector3(1, 0, 0);
const PLATE_R = 0.297;
const DOME_R = 8;
const DRONE_DIST = 16;
const WAVE_START = 2.4;
const WAVE_SPEED = 1.0;
const LAND = { rows: 72, cols: 300, zNear: 5, dz: 0.68, xHalf: 15, amp: 3.0 };
const FIGURE_POS = new THREE.Vector3(-1.25, 0, -11.6);
// The drone's spectral wake: slices of its live spectrum leave the drone and drift back along the beam.
const WAKE = { dt: 0.075, life: 2.6, speed: 0.9, start: -0.25, half: 1.1, base: -0.45, amp: 0.34, points: 120 };

const scale = (c, k) => [c[0] * k, c[1] * k, c[2] * k];
const mix = (a, b, k) => [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k)];

/** Direction in the array's own frame (faces +z; array right = -x). */
export function direction(azDeg, elDeg) {
  const az = (azDeg * Math.PI) / 180; const el = (elDeg * Math.PI) / 180;
  return new THREE.Vector3(-Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
}
export const toWorldDir = (v) => v.clone().applyAxisAngle(X_AXIS, -TILT);
export const toWorld = (v) => toWorldDir(v).add(ARRAY_POS);

/** Fat additive line segments rebuilt every frame. Brightness lives in the colour. */
class Wires {
  constructor(scene, width, renderOrder = 2) {
    this.geometry = new LineSegmentsGeometry();
    this.material = new LineMaterial({
      linewidth: width, vertexColors: true, transparent: true, depthWrite: false,
      blending: THREE.AdditiveBlending, worldUnits: false,
    });
    this.material.resolution.set(W, H);
    this.object = new LineSegments2(this.geometry, this.material);
    this.object.frustumCulled = false;
    this.object.renderOrder = renderOrder;
    scene.add(this.object);
    this.p = []; this.c = [];
  }

  begin() { this.p.length = 0; this.c.length = 0; return this; }

  seg(a, b, ca, cb = ca) {
    this.p.push(a.x, a.y, a.z, b.x, b.y, b.z);
    this.c.push(ca[0], ca[1], ca[2], cb[0], cb[1], cb[2]);
  }

  polyline(points, colorAt, closed = false, fraction = 1) {
    const count = closed ? points.length : points.length - 1;
    const limit = count * clamp(fraction);
    for (let i = 0; i < count && i < limit; i++) {
      const a = points[i]; let b = points[(i + 1) % points.length];
      if (i + 1 > limit) b = a.clone().lerp(b, limit - i);
      this.seg(a, b, colorAt(i / count), colorAt((i + 1) / count));
    }
  }

  end() {
    const n = this.p.length;
    if (!n) { this.object.visible = false; return; }
    this.object.visible = true;
    if (!this.capacity || n > this.capacity) {
      this.capacity = Math.max(n, 2 * (this.capacity || 0));
      this.positions = new Float32Array(this.capacity);
      this.colors = new Float32Array(this.capacity);
      this.geometry.setPositions(this.positions);
      this.geometry.setColors(this.colors);
      // three.js caches the drawable instance count on first use; forget it when buffers grow.
      this.geometry._maxInstanceCount = undefined;
    }
    this.positions.set(this.p);
    this.colors.set(this.c);
    this.geometry.attributes.instanceStart.data.needsUpdate = true;
    this.geometry.attributes.instanceColorStart.data.needsUpdate = true;
    this.geometry.instanceCount = n / 6;
  }
}

function canvasTexture(width, height, draw) {
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  draw(canvas.getContext('2d'));
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

const glowTexture = () => canvasTexture(128, 128, (ctx) => {
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(0.18, 'rgba(255,255,255,0.5)');
  g.addColorStop(0.5, 'rgba(255,255,255,0.07)'); g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g; ctx.fillRect(0, 0, 128, 128);
});

function hexPoints(radius, rotation = 0) {
  return Array.from({ length: 6 }, (_, i) => {
    const a = rotation + (i * Math.PI) / 3;
    return new THREE.Vector3(radius * Math.cos(a), radius * Math.sin(a), 0);
  });
}

// A hooded watchman, 1.78 m, as a flat silhouette.
const FIGURE = [
  [-0.07, 1.78], [0.07, 1.78], [0.11, 1.7], [0.11, 1.6], [0.08, 1.53], [0.22, 1.46], [0.26, 1.3], [0.27, 1.0],
  [0.25, 0.8], [0.12, 0.78], [0.1, 0.0], [0.03, 0.0], [0.02, 0.72], [-0.02, 0.72], [-0.03, 0.0], [-0.1, 0.0],
  [-0.12, 0.78], [-0.25, 0.8], [-0.27, 1.0], [-0.26, 1.3], [-0.22, 1.46], [-0.08, 1.53], [-0.11, 1.6], [-0.11, 1.7],
];

export class Teaser {
  constructor(canvas, array, spectra) {
    this.canvas = canvas;
    this.array = array;
    this.spec = spectra.drone;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
    renderer.setPixelRatio(1);
    renderer.setSize(W, H, false);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    this.renderer = renderer;
    this.scene = new THREE.Scene();
    this.scene.background = canvasTexture(4, 512, (ctx) => {
      const g = ctx.createLinearGradient(0, 0, 0, 512);
      g.addColorStop(0, '#020308'); g.addColorStop(0.42, '#07111a');
      g.addColorStop(0.55, '#0d1a26'); g.addColorStop(0.7, '#060a12'); g.addColorStop(1, '#020306');
      ctx.fillStyle = g; ctx.fillRect(0, 0, 4, 512);
    });
    this.camera = new THREE.PerspectiveCamera(32, W / H, 0.02, 200);
    this.glow = glowTexture();

    this.micsLocal = array.microphones_mm.map(([x, y]) => new THREE.Vector3(-x / 1000, y / 1000, 0));
    this.mics = this.micsLocal.map((p) => toWorld(p));
    this.uLocal = direction(SOURCE.az, SOURCE.el);
    this.u = toWorldDir(this.uLocal);
    this.forward = toWorldDir(new THREE.Vector3(0, 0, 1));
    this.D = ARRAY_POS.clone().add(this.u.clone().multiplyScalar(DRONE_DIST));
    this.F = ARRAY_POS.clone().add(this.u.clone().multiplyScalar(0.9));
    const byRadius = this.micsLocal.map((_, i) => i).sort((a, b) => this.micsLocal[a].length() - this.micsLocal[b].length());
    this.riseStart = []; this.arrive = [];
    byRadius.forEach((mic, k) => {
      this.riseStart[mic] = T.ignite + (k / 43) * 1.0 + 0.2 * hash(mic, 11);
      this.arrive[mic] = this.riseStart[mic] + 1.4;
    });
    this.waveTime = this.micsLocal.map((p) => T.wave + (WAVE_START - p.dot(this.uLocal)) / WAVE_SPEED);

    this.buildAtmosphere();
    this.buildLand();
    this.buildArray();
    this.buildDynamics();
    this.buildDome();
    this.buildDrone();

    const target = new THREE.WebGLRenderTarget(W, H, { type: THREE.HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(renderer, target);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(new UnrealBloomPass(new THREE.Vector2(W, H), 0.8, 0.65, 0.32));
    this.composer.addPass(new OutputPass());
  }

  // ── Data ─────────────────────────────────────────────────────────────────
  mag(frame, band) {
    const frames = this.spec.frames;
    const f = ((Math.floor(frame) % frames) + frames) % frames;
    const b0 = clamp(Math.floor(band), 0, 62); const k = clamp(band - b0);
    const d = this.spec.data;
    const v = d[b0 * frames + f] * (1 - k) + d[(b0 + 1) * frames + f] * k;
    return Math.pow(v / 255, 2.2);
  }

  landHeight(x, row, t) {
    const ax = Math.abs(x);
    const valley = smooth(range01(ax, 0.9, 3.2));
    const envelope = 0.25 + 0.75 * Math.exp(-((x / 8) ** 2));
    return LAND.amp * valley * envelope * this.mag(row * 3 + t * 6, 1.5 + (ax / LAND.xHalf) * 56);
  }

  fog(point, near = 3, depth = 22) {
    return Math.exp(-Math.max(0, point.distanceTo(this.camera.position) - near) / depth);
  }

  // ── Construction ─────────────────────────────────────────────────────────
  buildAtmosphere() {
    this.halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glow, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    this.halo.position.set(0, 4.5, -34);
    this.halo.scale.setScalar(30);
    this.scene.add(this.halo);
    this.rain = new Wires(this.scene, 1.0, 1);
    const positions = [];
    for (let i = 0; i < 1400; i++) {
      const theta = hash(i, 1) * Math.PI * 2; const phi = Math.acos(2 * hash(i, 2) - 1); const r = 40 + 30 * hash(i, 3);
      positions.push(r * Math.sin(phi) * Math.cos(theta), Math.abs(r * Math.cos(phi)) * 0.8 + 6, r * Math.sin(phi) * Math.sin(theta) - 12);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    this.stars = new THREE.Points(geometry, new THREE.PointsMaterial({
      color: new THREE.Color(0.5, 0.6, 0.75), size: 1.4, sizeAttenuation: false,
      transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    this.scene.add(this.stars);
  }

  buildLand() {
    const { rows, cols } = LAND;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(rows * (cols + 1) * 2 * 3), 3));
    const index = [];
    for (let r = 0; r < rows; r++) {
      const base = r * (cols + 1) * 2;
      for (let c = 0; c < cols; c++) {
        const a = base + c * 2;
        index.push(a, a + 1, a + 2, a + 2, a + 1, a + 3);
      }
    }
    geometry.setIndex(index);
    // Black curtains under each ridge give hidden-line removal.
    this.curtains = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      color: 0x000000, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    }));
    this.curtains.frustumCulled = false;
    this.scene.add(this.curtains);
    this.ridges = new Wires(this.scene, 1.2);
  }

  buildArray() {
    const group = new THREE.Group();
    group.position.copy(ARRAY_POS);
    group.rotation.x = -TILT;
    this.scene.add(group);
    const plateShape = new THREE.Shape(hexPoints(PLATE_R).map((p) => new THREE.Vector2(p.x, p.y)));
    const plate = new THREE.Mesh(new THREE.ShapeGeometry(plateShape), new THREE.MeshBasicMaterial({ color: 0x000000, side: THREE.DoubleSide }));
    plate.position.z = -0.004;
    group.add(plate);
    const pod = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.05, 6, 1, false, Math.PI / 2).rotateX(Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x000000 }));
    pod.position.z = -0.035;
    group.add(pod);
    const base = new THREE.Vector3(ARRAY_POS.x, 0, ARRAY_POS.z + 0.05);
    this.mastTop = toWorld(new THREE.Vector3(0, -0.28, -0.03));
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 1, 12), new THREE.MeshBasicMaterial({ color: 0x000000 }));
    mast.position.copy(base).lerp(this.mastTop, 0.5);
    mast.scale.y = base.distanceTo(this.mastTop);
    this.scene.add(mast);
    this.mastBase = base;

    const figure = new THREE.Mesh(new THREE.ShapeGeometry(new THREE.Shape(FIGURE.map(([x, y]) => new THREE.Vector2(x, y)))), new THREE.MeshBasicMaterial({ color: 0x000000, side: THREE.DoubleSide }));
    figure.position.copy(FIGURE_POS);
    figure.rotation.y = 0.25;
    this.scene.add(figure);
    this.figure = figure;

    this.structWires = new Wires(this.scene, 1.5);
    this.detailWires = new Wires(this.scene, 1.0);
    this.ringWires = new Wires(this.scene, 1.35);
    this.micGlows = this.mics.map((p) => {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glow, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
      sprite.position.copy(p).add(this.forward.clone().multiplyScalar(0.004));
      sprite.renderOrder = 3;
      this.scene.add(sprite);
      return sprite;
    });
    this.routes = this.micsLocal.map((mic) => {
      const angle = Math.atan2(mic.y, mic.x);
      const start = new THREE.Vector3(Math.cos(angle) * 0.053, Math.sin(angle) * 0.053, 0);
      const dx = mic.x - start.x; const dy = mic.y - start.y;
      const bend = Math.abs(dx) > Math.abs(dy)
        ? new THREE.Vector3(start.x + Math.sign(dx) * Math.abs(dy), mic.y, 0)
        : new THREE.Vector3(mic.x, start.y + Math.sign(dy) * Math.abs(dx), 0);
      const toPort = mic.clone().sub(bend);
      const stop = toPort.length() > 0.012 ? mic.clone().sub(toPort.normalize().multiplyScalar(0.0115)) : bend;
      return [stop, bend, start].map((p) => toWorld(p.setZ(0.002)));
    });
  }

  buildDynamics() {
    this.particleWires = new Wires(this.scene, 1.4, 3);
    this.heads = this.mics.map(() => {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glow, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
      sprite.scale.setScalar(0.07);
      sprite.renderOrder = 4;
      this.scene.add(sprite);
      return sprite;
    });
    this.waveWires = new Wires(this.scene, 1.0);
    this.hitWires = new Wires(this.scene, 2.4, 3);
    this.fieldWires = new Wires(this.scene, 2.0, 3);
    this.convergeWires = new Wires(this.scene, 1.1, 3);
    this.beamWires = new Wires(this.scene, 3.2, 3);
    this.beamCore = new Wires(this.scene, 1.1, 4);
    this.focusGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glow, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    this.focusGlow.position.copy(this.F);
    this.focusGlow.scale.setScalar(0.3);
    this.scene.add(this.focusGlow);
  }

  buildDome() {
    const edges = (centers) => {
      const step = centers[1] - centers[0];
      return [...centers.map((c) => c - step / 2), centers.at(-1) + step / 2];
    };
    this.azEdges = edges(this.array.azimuth_deg);
    this.elEdges = edges(this.array.elevation_deg);
    this.domeWires = new Wires(this.scene, 1.0);
    this.cursorWires = new Wires(this.scene, 2.2, 3);
    const col = this.azEdges.findIndex((e, i) => SOURCE.az >= e && SOURCE.az < this.azEdges[i + 1]);
    const row = this.elEdges.findIndex((e, i) => SOURCE.el >= e && SOURCE.el < this.elEdges[i + 1]);
    this.lockSector = row * this.array.columns + col;
    const positions = []; const index = []; const n = 8;
    for (let j = 0; j <= n; j++) {
      for (let i = 0; i <= n; i++) {
        const p = toWorld(direction(lerp(this.azEdges[col], this.azEdges[col + 1], i / n), lerp(this.elEdges[row], this.elEdges[row + 1], j / n)).multiplyScalar(DOME_R));
        positions.push(p.x, p.y, p.z);
      }
    }
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const a = j * (n + 1) + i; index.push(a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setIndex(index);
    this.lockFill = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      color: 0x000000, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
    }));
    this.scene.add(this.lockFill);
    this.sectorCenter = (k) => {
      const r = Math.floor(k / this.array.columns); const c = k % this.array.columns;
      return toWorld(direction(this.array.azimuth_deg[c], this.array.elevation_deg[r]).multiplyScalar(DOME_R));
    };
  }

  sectorOutline(k, radius) {
    const row = Math.floor(k / this.array.columns); const col = k % this.array.columns;
    const a0 = this.azEdges[col]; const a1 = this.azEdges[col + 1]; const e0 = this.elEdges[row]; const e1 = this.elEdges[row + 1];
    const points = [];
    for (const [fa, fe, ta, te] of [[a0, e0, a1, e0], [a1, e0, a1, e1], [a1, e1, a0, e1], [a0, e1, a0, e0]]) {
      for (let s = 0; s < 8; s++) points.push(toWorld(direction(lerp(fa, ta, s / 8), lerp(fe, te, s / 8)).multiplyScalar(radius)));
    }
    return points;
  }

  buildDrone() {
    this.drone = new THREE.Group();
    this.drone.position.copy(this.D);
    this.drone.rotation.set(0.1, 0.6, -0.08);
    this.drone.scale.setScalar(1.6);
    this.scene.add(this.drone);
    this.drone.add(new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.02, 0.06), new THREE.MeshBasicMaterial({ color: 0x000000 })));
    this.discs = [];
    for (const [sx, sz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
      const disc = new THREE.Mesh(new THREE.CircleGeometry(0.055, 40).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({
        color: 0x000000, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
      }));
      disc.position.set(sx * 0.085, 0.012, sz * 0.085);
      this.drone.add(disc);
      this.discs.push(disc);
    }
    this.droneHalo = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glow, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    this.droneHalo.position.copy(this.D).add(this.u.clone().multiplyScalar(2.5));
    this.droneHalo.scale.setScalar(4.5);
    this.scene.add(this.droneHalo);
    this.droneWires = new Wires(this.scene, 1.4, 3);

    // Frame for the wake: `back` runs from the drone towards the array, `lift` is up relative to the beam.
    this.back = this.u.clone().negate();
    this.across = new THREE.Vector3().crossVectors(this.u, new THREE.Vector3(0, 1, 0)).normalize();
    this.lift = new THREE.Vector3().crossVectors(this.across, this.u).normalize();
    const slices = Math.ceil(WAKE.life / WAKE.dt) + 2;
    const columns = WAKE.points;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(slices * (columns + 1) * 2 * 3), 3));
    const index = [];
    for (let r = 0; r < slices; r++) {
      const base = r * (columns + 1) * 2;
      for (let c = 0; c < columns; c++) {
        const a = base + c * 2;
        index.push(a, a + 1, a + 2, a + 2, a + 1, a + 3);
      }
    }
    geometry.setIndex(index);
    this.wakeCurtains = new THREE.Mesh(geometry, this.curtains.material);
    this.wakeCurtains.frustumCulled = false;
    this.scene.add(this.wakeCurtains);
    this.wakeSlices = slices;
    this.wakeWires = new Wires(this.scene, 1.3, 3);
  }

  // ── Camera ───────────────────────────────────────────────────────────────
  keys() {
    if (this._keys) return this._keys;
    const v = (x, y, z) => new THREE.Vector3(x, y, z);
    const A = ARRAY_POS; const u = this.u; const D = this.D;
    const side = new THREE.Vector3().crossVectors(u, v(0, 1, 0)).normalize();
    const lift = new THREE.Vector3().crossVectors(side, u).normalize();
    const along = (d) => A.clone().add(u.clone().multiplyScalar(d));
    this._keys = [
      { t: T.fadeIn, p: v(0, 1.0, 4.0), l: v(0, 2.1, -12), fov: 32 },
      { t: 7.4, p: v(0.08, 1.45, -2.2), l: v(0, 2.2, -12), fov: 32 },
      { t: 9.6, p: v(0.35, 2.15, -8.1), l: A.clone(), fov: 32 },
      { t: 11.8, p: v(0.95, 2.45, -9.55), l: A.clone(), fov: 32 },
      { t: 14.0, p: v(2.2, 2.95, -10.4), l: A.clone().add(v(0, 0.1, 0)), fov: 34 },
      { t: 15.3, p: v(1.95, 2.7, -10.25), l: A.clone().add(v(0, 0.1, 0)), fov: 34 },
      { t: 16.7, p: v(1.5, 1.3, -12.75), l: along(5.5), fov: 46 },
      { t: 17.6, p: v(1.25, 1.35, -13.1), l: along(7), fov: 46 },
      // Above the beam, looking down its spectral wake at the drone.
      { t: 20.0, p: along(12.6).addScaledVector(side, -0.35).addScaledVector(lift, 0.55), l: D.clone().addScaledVector(lift, -0.25).addScaledVector(side, 0.1), fov: 34 },
      { t: T.black, p: along(13.3).addScaledVector(side, -0.25).addScaledVector(lift, 0.45), l: D.clone().addScaledVector(lift, -0.22).addScaledVector(side, 0.08), fov: 34 },
    ];
    return this._keys;
  }

  pathCamera(t) {
    const keys = this.keys();
    const tt = clamp(t, keys[0].t, keys.at(-1).t);
    let i = 0;
    while (i < keys.length - 2 && tt > keys[i + 1].t) i++;
    const k0 = keys[i - 1]; const k1 = keys[i]; const k2 = keys[i + 1]; const k3 = keys[i + 2];
    const dt = k2.t - k1.t; const s = (tt - k1.t) / dt;
    const h00 = 2 * s ** 3 - 3 * s ** 2 + 1; const h10 = s ** 3 - 2 * s ** 2 + s;
    const h01 = -2 * s ** 3 + 3 * s ** 2; const h11 = s ** 3 - s ** 2;
    const interp = (key) => {
      const m1 = k0 ? k2[key].clone().sub(k0[key]).multiplyScalar(dt / (k2.t - k0.t)) : new THREE.Vector3();
      const m2 = k3 ? k3[key].clone().sub(k1[key]).multiplyScalar(dt / (k3.t - k1.t)) : new THREE.Vector3();
      return k1[key].clone().multiplyScalar(h00).add(m1.multiplyScalar(h10)).add(k2[key].clone().multiplyScalar(h01)).add(m2.multiplyScalar(h11));
    };
    return { p: interp('p'), l: interp('l'), fov: lerp(k1.fov, k2.fov, smooth(s)) };
  }

  setCamera(pose) {
    this.camera.position.copy(pose.p);
    this.camera.fov = pose.fov;
    this.camera.aspect = W / H;
    this.camera.updateProjectionMatrix();
    this.camera.lookAt(pose.l);
    this.camera.updateMatrixWorld();
  }

  // ── Frame update ─────────────────────────────────────────────────────────
  /** t is story time; wall is clock time, for things with a physical rate (rain, rotors). */
  update(t, wall = t) {
    this.halo.material.color.setRGB(0.06, 0.13, 0.19);
    this.updateRain(wall);
    this.updateLand(t);
    this.updateArray(t);
    this.updateParticles(t);
    this.updateWave(t);
    this.updateBeam(t);
    this.updateDome(t);
    this.updateDrone(t, wall);
  }

  updateRain(t) {
    const rain = this.rain.begin();
    const cam = this.camera.position;
    const wrap = (v, span) => ((((v % span) + span) % span) - span / 2);
    for (let i = 0; i < 700; i++) {
      const x = cam.x + wrap(hash(i, 21) * 40 - cam.x, 16);
      const z = cam.z - 3 + wrap(hash(i, 22) * 40 - cam.z, 16);
      const y = cam.y + 6 - ((t * 5.5 + hash(i, 23) * 12) % 12);
      const top = new THREE.Vector3(x, y, z);
      const k = 0.14 * this.fog(top, 1, 7);
      rain.seg(top, new THREE.Vector3(x - 0.03, y - 0.42, z), scale(WIRE, 0), scale(WIRE, k));
    }
    rain.end();
  }

  updateLand(t) {
    const { rows, cols, zNear, dz, xHalf } = LAND;
    const ridges = this.ridges.begin();
    const positions = this.curtains.geometry.attributes.position;
    let v = 0;
    const appear = smooth(range01(t, T.fadeIn, 6.4));
    for (let r = 0; r < rows; r++) {
      const z = zNear - r * dz;
      const rowFog = this.fog(new THREE.Vector3(this.camera.position.x, 0, z), 3, 20);
      const points = [];
      for (let c = 0; c <= cols; c++) {
        const x = -xHalf + (2 * xHalf * c) / cols;
        const h = this.landHeight(x, r, t);
        points.push([x, h]);
        positions.setXYZ(v++, x, h, z);
        positions.setXYZ(v++, x, -0.05, z);
      }
      const bright = appear * rowFog;
      if (bright < 0.004) continue;
      for (let c = 0; c < cols; c++) {
        const [x0, y0] = points[c]; const [x1, y1] = points[c + 1];
        const color = (y) => mix(scale(WIRE, 0.42), scale(MAGENTA, 1.5), smooth(range01(y / LAND.amp, 0.12, 0.55))).map((q) => q * bright);
        ridges.seg(new THREE.Vector3(x0, y0, z), new THREE.Vector3(x1, y1, z), color(y0), color(y1));
      }
    }
    positions.needsUpdate = true;
    ridges.end();
  }

  micState(m, t) {
    const arrive = this.arrive[m];
    if (t < arrive) return [[0, 0, 0], 0];
    const pop = Math.exp(-(t - arrive) / 0.16);
    let color = mix(scale(WIRE, 0.85), scale(WIRE, 2.2), pop);
    let glow = 0.3 + 0.6 * pop;
    const hit = this.waveTime[m];
    if (t > hit - 0.1) {
      const flash = Math.exp(-(((t - hit) / 0.05) ** 2));
      const armed = t < T.fadeOut ? smooth(range01(t, hit, hit + 0.15)) : 0;
      color = mix(color, scale(SIGNAL, 1.3), armed);
      color = mix(color, [1.9, 1.3, 1.6], flash);
      glow += 0.8 * flash + 0.25 * armed;
    }
    return [color, glow];
  }

  updateArray(t) {
    const struct = this.structWires.begin(); const detail = this.detailWires.begin(); const rings = this.ringWires.begin();
    const lit = easeInOut(range01(t, T.frameDraw, 10.2));
    const silhouette = 0.22 * this.fog(ARRAY_POS, 2, 14);
    const frameK = lerp(silhouette, 0.95, lit);
    const plate = (r) => hexPoints(r).map((p) => toWorld(p.setZ(0.003)));
    struct.polyline(plate(PLATE_R), () => scale(WIRE, frameK), true);
    struct.polyline(plate(PLATE_R - 0.02), () => scale(WIRE, frameK * 0.4), true);
    struct.polyline(hexPoints(PLATE_R).map((p) => toWorld(p.setZ(-0.012))), () => scale(WIRE, frameK * 0.6), true);
    // Mast and the watchman, rim-lit from the valley glow.
    const mastK = 0.22 * this.fog(this.mastBase, 2, 16);
    for (const dx of [-0.03, 0.03]) struct.seg(this.mastBase.clone().add(new THREE.Vector3(dx, 0, 0)), this.mastTop.clone().add(new THREE.Vector3(dx, 0, 0)), scale(WIRE, mastK));
    for (let i = 0; i < 3; i++) {
      const a = (i / 3) * Math.PI * 2 + 0.4;
      struct.seg(this.mastBase.clone().add(new THREE.Vector3(0, 0.55, 0)), this.mastBase.clone().add(new THREE.Vector3(0.45 * Math.cos(a), 0, 0.45 * Math.sin(a))), scale(WIRE, mastK * 0.8));
    }
    const figureK = 0.3 * this.fog(FIGURE_POS, 2, 16) * smooth(range01(t, T.fadeIn, 7));
    this.figure.updateMatrixWorld();
    const outline = FIGURE.map(([x, y]) => new THREE.Vector3(x, y, 0.002).applyMatrix4(this.figure.matrixWorld));
    struct.polyline(outline, () => scale(WIRE, figureK), true);
    detail.polyline(hexPoints(0.058, Math.PI / 6).map((p) => toWorld(p.setZ(0.003))), () => scale(WIRE, 0.85 * lit), true);
    detail.polyline(hexPoints(0.032, Math.PI / 6).map((p) => toWorld(p.setZ(0.003))), () => scale(SIGNAL, 1.1 * lit), true);
    this.mics.forEach((mic, m) => {
      const [color, glow] = this.micState(m, t);
      const arrive = this.arrive[m];
      const sprite = this.micGlows[m];
      sprite.visible = t >= arrive && t < T.black;
      sprite.material.color.setRGB(color[0] * glow, color[1] * glow, color[2] * glow);
      sprite.scale.setScalar(0.03 + 0.03 * Math.min(1.2, glow));
      if (t < arrive) return;
      const grow = easeOut(range01(t, arrive, arrive + 0.2));
      const radius = 0.0095 * (grow + 0.5 * Math.exp(-(t - arrive) / 0.1));
      const ring = Array.from({ length: 20 }, (_, i) => {
        const a = (i / 20) * Math.PI * 2;
        return toWorld(this.micsLocal[m].clone().add(new THREE.Vector3(radius * Math.cos(a), radius * Math.sin(a), 0.003)));
      });
      rings.polyline(ring, () => color, true);
      detail.polyline(this.routes[m], () => scale(WIRE, 0.3), false, easeOut(range01(t, arrive, arrive + 0.45)));
    });
    struct.end(); detail.end(); rings.end();
  }

  bezier(m, s) {
    const end = this.mics[m];
    const start = new THREE.Vector3(lerp(-2.4, 2.4, hash(m, 3)), 0.05, ARRAY_POS.z + lerp(-2.5, 3.2, hash(m, 7)));
    const c1 = start.clone().add(new THREE.Vector3(0, 1.6, 0));
    const c2 = end.clone().add(this.forward.clone().multiplyScalar(0.9)).add(new THREE.Vector3(0, 0.2, 0));
    const a = 1 - s;
    return start.multiplyScalar(a * a * a).add(c1.multiplyScalar(3 * a * a * s)).add(c2.multiplyScalar(3 * a * s * s)).add(end.clone().multiplyScalar(s * s * s));
  }

  updateParticles(t) {
    const wires = this.particleWires.begin();
    this.mics.forEach((_, m) => {
      const t0 = this.riseStart[m];
      const head = this.heads[m];
      const s = easeInOut(range01(t, t0, t0 + 1.4));
      const active = t >= t0 && t <= t0 + 1.42;
      head.visible = active;
      if (!active) return;
      const k = smooth(range01(t, t0, t0 + 0.2));
      const trail = [];
      for (let i = 0; i <= 14; i++) trail.push(this.bezier(m, Math.max(0, s - 0.24 * (1 - i / 14))));
      wires.polyline(trail, (q) => mix(scale(VIOLET, 0), scale(WIRE, 1.9 * k), q * q));
      head.position.copy(trail.at(-1));
      head.material.color.setRGB(0.55 * k, 0.95 * k, 1.2 * k);
    });
    wires.end();
  }

  updateWave(t) {
    const wave = this.waveWires.begin(); const hit = this.hitWires.begin(); const field = this.fieldWires.begin();
    const vis = window01(t, T.wave, T.wave + 0.5, T.cross + 0.6, T.cross + 1.1);
    if (vis > 0.002) {
      const u = this.u;
      const e1 = new THREE.Vector3().crossVectors(u, new THREE.Vector3(0, 1, 0)).normalize();
      const e2 = new THREE.Vector3().crossVectors(e1, u).normalize();
      const d = WAVE_START - WAVE_SPEED * (t - T.wave);
      const center = ARRAY_POS.clone().add(u.clone().multiplyScalar(d));
      const n = 20; const half = 1.5; const pieces = 26;
      const at = (a, b) => center.clone().add(e1.clone().multiplyScalar(a)).add(e2.clone().multiplyScalar(b));
      const fade = (a, b) => 0.55 * vis * Math.exp(-(((a * a + b * b) / (1.25 * 1.25)) ** 2));
      for (let i = 0; i <= n; i++) {
        const fixed = -half + (2 * half * i) / n;
        for (let p = 0; p < pieces; p++) {
          const q0 = -half + (2 * half * p) / pieces; const q1 = -half + (2 * half * (p + 1)) / pieces;
          wave.seg(at(fixed, q0), at(fixed, q1), scale(VIOLET, fade(fixed, q0)), scale(VIOLET, fade(fixed, q1)));
          wave.seg(at(q0, fixed), at(q1, fixed), scale(VIOLET, fade(q0, fixed)), scale(VIOLET, fade(q1, fixed)));
        }
      }
      // Where the wavefront currently cuts the board (array frame, then to world).
      const ul = this.uLocal; const n2 = ul.x * ul.x + ul.y * ul.y;
      const p0 = new THREE.Vector3((d * ul.x) / n2, (d * ul.y) / n2, 0.005);
      const r0 = p0.length(); const reach = PLATE_R - 0.004;
      if (r0 < reach) {
        const along = new THREE.Vector3(-ul.y, ul.x, 0).normalize().multiplyScalar(Math.sqrt(reach * reach - r0 * r0));
        hit.seg(toWorld(p0.clone().sub(along)), toWorld(p0.clone().add(along)), scale(MAGENTA, 3.0 * vis));
      }
    }
    // A.T.-field-like hexagonal ripple as the wave crosses the array's centre.
    for (let k = 0; k < 5; k++) {
      const age = t - (T.cross + k * 0.14);
      if (age < 0 || age > 1.4) continue;
      const r = PLATE_R + 0.03 + 2.4 * easeOut(age / 1.4);
      const intensity = 1.5 * Math.pow(1 - age / 1.4, 1.6);
      field.polyline(hexPoints(r).map((p) => toWorld(p.setZ(0.01))), () => scale(mix(SIGNAL, MAGENTA, k / 4), intensity), true);
    }
    wave.end(); hit.end(); field.end();
  }

  updateBeam(t) {
    const lines = this.convergeWires.begin(); const beam = this.beamWires.begin(); const core = this.beamCore.begin();
    const fade = 1 - smooth(range01(t, T.fadeOut, T.black));
    const settle = 1 - 0.6 * smooth(range01(t, 16.4, 17.4));
    this.mics.forEach((mic, m) => {
      const start = this.waveTime[m] + 0.1;
      const q = easeOut(range01(t, start, start + 0.5));
      if (q <= 0) return;
      const end = mic.clone().lerp(this.F, q);
      const points = Array.from({ length: 7 }, (_, i) => mic.clone().lerp(end, i / 6));
      lines.polyline(points, (k) => mix(scale(WIRE, 0.45 * fade * settle), scale(SIGNAL, 1.4 * fade * settle), k * q));
    });
    const reach = easeInOut(range01(t, T.beam, T.beam + 1.1));
    if (reach > 0) {
      const tip = this.F.clone().lerp(this.D, reach);
      const points = Array.from({ length: 40 }, (_, i) => this.F.clone().lerp(tip, i / 39));
      beam.polyline(points, (k) => scale(SIGNAL, (3.2 - 1.2 * k) * fade));
      core.polyline(points, () => [2.2 * fade, 1.9 * fade, 1.7 * fade]);
    }
    const focus = smooth(range01(t, T.cross + 0.2, T.cross + 0.7)) * fade;
    this.focusGlow.visible = focus > 0.001;
    this.focusGlow.material.color.setRGB(2.2 * focus, 1.0 * focus, 0.35 * focus);
    lines.end(); beam.end(); core.end();
  }

  updateDome(t) {
    const dome = this.domeWires.begin(); const cursor = this.cursorWires.begin();
    const vis = window01(t, T.dome, T.dome + 0.8, 17.9, 18.7);
    if (vis > 0.002) {
      const sectors = this.array.rows * this.array.columns;
      for (let k = 0; k < sectors; k++) dome.polyline(this.sectorOutline(k, DOME_R), () => scale(WIRE, 0.3 * vis), true);
      let k = this.lockSector;
      if (t < T.lock) k = Math.floor((t - T.dome) / 0.02) % sectors;
      const locked = t >= T.lock;
      cursor.polyline(this.sectorOutline(Math.max(0, k), DOME_R * 0.998), () => (locked ? scale(SIGNAL, 2.8 * vis) : scale(WIRE, 1.4 * vis)), true);
    }
    const fill = t >= T.lock ? vis * (0.2 + 0.7 * Math.exp(-(t - T.lock) / 0.3)) : 0;
    this.lockFill.visible = fill > 0.001;
    this.lockFill.material.color.setRGB(SIGNAL[0] * fill, SIGNAL[1] * fill, SIGNAL[2] * fill);
    dome.end(); cursor.end();
  }

  updateDrone(t, wall) {
    const wires = this.droneWires.begin();
    const vis = smooth(range01(t, T.drone, T.drone + 0.9)) * (1 - smooth(range01(t, T.fadeOut, T.black)));
    this.drone.visible = vis > 0.002;
    this.droneHalo.visible = this.drone.visible;
    this.droneHalo.material.color.setRGB(0.05 * vis, 0.1 * vis, 0.14 * vis);
    if (this.drone.visible) {
      this.drone.updateMatrixWorld(true);
      const local = (x, y, z) => new THREE.Vector3(x, y, z).applyMatrix4(this.drone.matrixWorld);
      const white = scale(WIRE, 1.1 * vis);
      const b = 0.03; const h = 0.01;
      const corners = [[-b, -h, -b], [b, -h, -b], [b, -h, b], [-b, -h, b], [-b, h, -b], [b, h, -b], [b, h, b], [-b, h, b]].map(([x, y, z]) => local(x, y, z));
      for (const [i, j] of [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]) wires.seg(corners[i], corners[j], white);
      for (const [sx, sz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
        wires.seg(local(sx * 0.02, 0, sz * 0.02), local(sx * 0.085, 0.008, sz * 0.085), white);
        const ring = Array.from({ length: 36 }, (_, i) => {
          const a = (i / 36) * Math.PI * 2;
          return local(sx * 0.085 + 0.055 * Math.cos(a), 0.012, sz * 0.085 + 0.055 * Math.sin(a));
        });
        wires.polyline(ring, () => scale(WIRE, 0.85 * vis), true);
        const spin = wall * 37 + sx * 1.3 + sz * 0.7;
        wires.seg(local(sx * 0.085 + 0.05 * Math.cos(spin), 0.012, sz * 0.085 + 0.05 * Math.sin(spin)),
          local(sx * 0.085 - 0.05 * Math.cos(spin), 0.012, sz * 0.085 - 0.05 * Math.sin(spin)), scale(WIRE, 0.55 * vis));
      }
      this.discs.forEach((disc) => disc.material.color.setRGB(0.05 * vis, 0.06 * vis, 0.08 * vis));
    }
    wires.end();
    this.updateWake(t, vis);
  }

  /** Spectrum slices born at the drone every WAKE.dt, drifting back along the beam as ridgelines. */
  updateWake(t, vis) {
    const ridges = this.wakeWires.begin();
    const positions = this.wakeCurtains.geometry.attributes.position;
    const { dt, life, speed, start, half, base, amp, points } = WAKE;
    const born0 = T.print - 0.1;   // the wake starts as the camera settles above the beam
    const newest = Math.floor((t - born0) / dt);
    let v = 0;
    for (let s = 0; s < this.wakeSlices; s++) {
      const j = newest - s;
      const age = t - (born0 + j * dt);
      const alive = vis > 0.002 && j >= 0 && age >= 0 && age <= life;
      const origin = this.D.clone().addScaledVector(this.back, start + age * speed).addScaledVector(this.lift, base);
      const line = [];
      for (let c = 0; c <= points; c++) {
        const x = -1 + (2 * c) / points; const ax = Math.abs(x);
        // A canyon: quiet high bands along the beam, the loud low bands rising into walls on either side.
        const h = alive ? amp * smooth(range01(age, 0, 0.35)) * smooth(range01(ax, 0.06, 0.32)) * (1 - smooth(range01(ax, 0.8, 1)))
          * this.mag(j * 2, 1.5 + clamp((1 - ax) / 0.92) * 52) : 0;
        const top = origin.clone().addScaledVector(this.across, x * half).addScaledVector(this.lift, h);
        const bottom = alive ? origin.clone().addScaledVector(this.across, x * half).addScaledVector(this.lift, -0.03) : top;
        positions.setXYZ(v++, top.x, top.y, top.z);
        positions.setXYZ(v++, bottom.x, bottom.y, bottom.z);
        line.push([top, h]);
      }
      if (!alive) continue;
      const bright = vis * smooth(range01(age, 0, 0.3)) * (1 - smooth(range01(age, life * 0.55, life)));
      const color = (h) => mix(scale(WIRE, 0.35), scale(mix(SIGNAL, MAGENTA, 0.3), 1.05), smooth(range01(h / amp, 0.1, 0.6))).map((q) => q * bright);
      for (let c = 0; c < points; c++) ridges.seg(line[c][0], line[c + 1][0], color(line[c][1]), color(line[c + 1][1]));
    }
    positions.needsUpdate = true;
    ridges.end();
  }

  render() { this.composer.render(); }

  project(point) {
    const v = point.clone().project(this.camera);
    return { x: ((v.x + 1) / 2) * W, y: ((1 - v.y) / 2) * H, visible: v.z < 1 };
  }

  /** On-screen size in pixels of a world-space radius r around point. */
  screenRadius(point, r) {
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0);
    const a = this.project(point); const b = this.project(point.clone().addScaledVector(right, r));
    return Math.hypot(b.x - a.x, b.y - a.y);
  }

  // ── Sound cues ───────────────────────────────────────────────────────────
  audioEvents() {
    const events = [];
    const add = (type, t, extra = {}) => events.push({ type, t: +t.toFixed(4), ...extra });
    add('shimmer', 0.2, { t1: T.fadeIn + 0.4 });
    add('pad', T.fadeIn - 0.4, { t1: T.black + 0.2 });
    add('rain', 0.6, { t1: T.black });
    this.micsLocal.forEach((p, m) => add('pluck', this.arrive[m], { pan: clamp(p.x / 0.3, -1, 1), index: m }));
    add('swell', T.listen, { t1: T.listenOut });
    add('whoosh', T.wave, { t1: T.cross + 0.4 });
    for (let k = 0; k < 5; k++) add('ring', T.cross + k * 0.14, { index: k });
    add('riser', T.beam, { t1: T.lock });
    add('lock', T.lock);
    add('swell', T.ascend, { t1: T.drone });
    add('heartbeat', T.drone - 0.4, { t1: T.fadeOut + 0.2 });
    add('drone', T.drone - 0.6, { t1: T.black });
    add('chime', T.label);
    add('boom', T.title);
    add('shimmer', T.title, { t1: T.end });
    add('pad', T.title, { t1: T.end, key: 'title' });
    return events.sort((a, b) => a.t - b.t);
  }
}
