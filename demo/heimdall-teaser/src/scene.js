/**
 * HEIMDALL teaser: a wireframe world driven by time alone.
 *
 * silence line -> drone spectrum terrain -> 44 points rise into the array ->
 * a wavefront crosses the microphones in delay order -> their wires converge
 * into one beam -> sector dome -> the drone and its spectral fingerprint.
 *
 * The array faces +z. Microphone positions, the 6x6 sector grid and the drone
 * spectrum are the repository's real data.
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { W, H, clamp, lerp, range01, smooth, easeInOut, easeOut, window01, hash, WIRE, SIGNAL } from './util.js';

export const T = {
  lineIn: 0.3, textA: 1.1, spread: 2.9, terrain: 3.0,
  rise: 7.1, frameDraw: 8.4, textB: 9.7,
  wave: 10.8, converge: 12.1, dome: 12.8, lock: 13.45, beam: 13.0,
  drone: 14.2, print: 14.9, title: 16.8, titleFull: 17.6, end: 20.0,
};

const BASE_Y = -0.95;
const ROWS = 60;
const COLS = 200;
const Z_NEAR = 2.2;
const DZ = 0.085;
const X_HALF = 2.6;
const AMP = 0.5;
const PLATE_R = 0.297;
const DOME_R = 2.5;
const DRONE_DIST = 4.2;
export const SOURCE = { az: 30, el: 20 };

const scale = (c, k) => [c[0] * k, c[1] * k, c[2] * k];
const mix = (a, b, k) => [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k)];

/** Unit vector for an azimuth/elevation seen from the array (array right = world -x). */
export function direction(azDeg, elDeg) {
  const az = (azDeg * Math.PI) / 180; const el = (elDeg * Math.PI) / 180;
  return new THREE.Vector3(-Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
}

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

function glowTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 128;
  const ctx = canvas.getContext('2d');
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(0.18, 'rgba(255,255,255,0.55)');
  g.addColorStop(0.5, 'rgba(255,255,255,0.08)'); g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g; ctx.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(canvas);
}

function hexPoints(radius, z, rotation = 0) {
  return Array.from({ length: 6 }, (_, i) => {
    const a = rotation + (i * Math.PI) / 3;
    return new THREE.Vector3(radius * Math.cos(a), radius * Math.sin(a), z);
  });
}

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
    this.scene.background = new THREE.Color(0x000000);
    this.camera = new THREE.PerspectiveCamera(32, W / H, 0.01, 100);
    this.glow = glowTexture();

    this.mics = array.microphones_mm.map(([x, y]) => new THREE.Vector3(-x / 1000, y / 1000, 0));
    this.u = direction(SOURCE.az, SOURCE.el);
    this.D = this.u.clone().multiplyScalar(DRONE_DIST);
    this.F = this.u.clone().multiplyScalar(0.8);
    this.byRadius = this.mics.map((_, i) => i).sort((a, b) => this.mics[a].length() - this.mics[b].length());
    this.riseOrder = this.byRadius.map((_, k) => k);
    this.riseStart = []; this.arrive = [];
    this.byRadius.forEach((mic, k) => {
      this.riseStart[mic] = T.rise + (k / 43) * 0.9 + 0.15 * hash(mic, 11);
      this.arrive[mic] = this.riseStart[mic] + 1.3;
    });
    this.waveTime = this.mics.map((p) => T.wave + (1.35 - p.dot(this.u)) / 1.1);

    this.buildStars();
    this.buildTerrain();
    this.buildArray();
    this.buildDynamics();
    this.buildDome();
    this.buildDrone();

    const target = new THREE.WebGLRenderTarget(W, H, { type: THREE.HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(renderer, target);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(new UnrealBloomPass(new THREE.Vector2(W, H), 0.85, 0.6, 0.3));
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

  heightAt(x, frame) {
    const band = 1.5 + (Math.abs(x) / X_HALF) * 56;
    const envelope = 0.12 + 0.88 * Math.exp(-((x / 1.7) ** 2));
    return AMP * envelope * this.mag(frame, band);
  }

  /** Rows scrolled toward the camera: integral of a smooth-start speed. */
  scroll(t) {
    const rate = 14; const t0 = T.terrain; const ramp = 1.6;
    if (t < t0) return 0;
    const x = Math.min(1, (t - t0) / ramp);
    const eased = ramp * (x ** 3 - (x ** 4) / 2);
    return rate * (eased + Math.max(0, t - t0 - ramp));
  }

  // ── Construction ─────────────────────────────────────────────────────────
  buildStars() {
    const positions = [];
    for (let i = 0; i < 900; i++) {
      const theta = hash(i, 1) * Math.PI * 2; const phi = Math.acos(2 * hash(i, 2) - 1);
      const r = 7 + 9 * hash(i, 3);
      positions.push(r * Math.sin(phi) * Math.cos(theta), r * Math.cos(phi) * 0.7, r * Math.sin(phi) * Math.sin(theta));
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    this.stars = new THREE.Points(geometry, new THREE.PointsMaterial({
      color: new THREE.Color(0.55, 0.62, 0.75), size: 1.6, sizeAttenuation: false,
      transparent: true, opacity: 0.0, blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    this.scene.add(this.stars);
  }

  buildTerrain() {
    const vertices = ROWS * (COLS + 1) * 2;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(vertices * 3), 3));
    const index = [];
    for (let r = 0; r < ROWS; r++) {
      const base = r * (COLS + 1) * 2;
      for (let c = 0; c < COLS; c++) {
        const a = base + c * 2; const b = a + 1; const d = a + 2; const e = a + 3;
        index.push(a, b, d, d, b, e);
      }
    }
    geometry.setIndex(index);
    // Black "curtains" under each ridge hide the ridges behind it.
    this.curtains = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      color: 0x000000, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    }));
    this.curtains.frustumCulled = false;
    this.curtains.renderOrder = 0;
    this.scene.add(this.curtains);
    this.ridges = new Wires(this.scene, 1.25);
  }

  buildArray() {
    const plateShape = new THREE.Shape(hexPoints(PLATE_R, 0).map((p) => new THREE.Vector2(p.x, p.y)));
    this.plate = new THREE.Mesh(new THREE.ShapeGeometry(plateShape), new THREE.MeshBasicMaterial({ color: 0x000000, side: THREE.DoubleSide }));
    this.plate.position.z = -0.004;
    this.plate.renderOrder = 0;
    this.scene.add(this.plate);
    this.frameWires = new Wires(this.scene, 1.7);
    this.detailWires = new Wires(this.scene, 1.0);
    this.ringWires = new Wires(this.scene, 1.35);
    this.micGlows = this.mics.map((p) => {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glow, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
      sprite.position.copy(p).setZ(0.004);
      sprite.scale.setScalar(0.05);
      sprite.renderOrder = 3;
      this.scene.add(sprite);
      return sprite;
    });
    // Octilinear signal routes from the hub to every port.
    this.routes = this.mics.map((mic) => {
      const angle = Math.atan2(mic.y, mic.x);
      const start = new THREE.Vector3(Math.cos(angle) * 0.053, Math.sin(angle) * 0.053, 0);
      const dx = mic.x - start.x; const dy = mic.y - start.y;
      const bend = Math.abs(dx) > Math.abs(dy)
        ? new THREE.Vector3(start.x + Math.sign(dx) * Math.abs(dy), mic.y, 0)
        : new THREE.Vector3(mic.x, start.y + Math.sign(dy) * Math.abs(dx), 0);
      const toPort = mic.clone().sub(bend);
      const stop = toPort.length() > 0.012 ? mic.clone().sub(toPort.normalize().multiplyScalar(0.0115)) : bend;
      return [stop, bend, start];
    });
  }

  buildDynamics() {
    this.particleWires = new Wires(this.scene, 1.5, 3);
    this.heads = this.mics.map(() => {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glow, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
      sprite.scale.setScalar(0.05);
      sprite.renderOrder = 4;
      this.scene.add(sprite);
      return sprite;
    });
    this.waveWires = new Wires(this.scene, 1.0);
    this.waveHit = new Wires(this.scene, 2.6, 3);
    this.convergeWires = new Wires(this.scene, 1.1, 3);
    this.beamWires = new Wires(this.scene, 3.4, 3);
    this.beamCore = new Wires(this.scene, 1.2, 4);
    this.focusGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glow, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    this.focusGlow.position.copy(this.F);
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
        const p = direction(lerp(this.azEdges[col], this.azEdges[col + 1], i / n), lerp(this.elEdges[row], this.elEdges[row + 1], j / n)).multiplyScalar(DOME_R);
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
      color: 0x000000, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide, toneMapped: false,
    }));
    this.lockFill.renderOrder = 1;
    this.scene.add(this.lockFill);
  }

  sectorOutline(k, radius) {
    const row = Math.floor(k / this.array.columns); const col = k % this.array.columns;
    const a0 = this.azEdges[col]; const a1 = this.azEdges[col + 1]; const e0 = this.elEdges[row]; const e1 = this.elEdges[row + 1];
    const points = [];
    for (const [fa, fe, ta, te] of [[a0, e0, a1, e0], [a1, e0, a1, e1], [a1, e1, a0, e1], [a0, e1, a0, e0]]) {
      for (let s = 0; s < 8; s++) points.push(direction(lerp(fa, ta, s / 8), lerp(fe, te, s / 8)).multiplyScalar(radius));
    }
    return points;
  }

  buildDrone() {
    this.drone = new THREE.Group();
    this.drone.position.copy(this.D);
    this.drone.rotation.set(0.12, 0.5, -0.06);
    this.scene.add(this.drone);
    this.droneBody = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.02, 0.06), new THREE.MeshBasicMaterial({ color: 0x000000 }));
    this.drone.add(this.droneBody);
    this.discs = [];
    for (const [sx, sz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
      const disc = new THREE.Mesh(new THREE.CircleGeometry(0.055, 40).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({
        color: 0x000000, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
      }));
      disc.position.set(sx * 0.085, 0.012, sz * 0.085);
      this.drone.add(disc);
      this.discs.push(disc);
    }
    this.droneWires = new Wires(this.scene, 1.4, 3);
    this.printWires = new Wires(this.scene, 1.8, 3);
  }

  // ── Camera ───────────────────────────────────────────────────────────────
  keys() {
    if (this._keys) return this._keys;
    const v = (x, y, z) => new THREE.Vector3(x, y, z);
    const D = this.D; const u = this.u;
    const side = new THREE.Vector3().crossVectors(u, v(0, 1, 0)).normalize();
    this._keys = [
      { t: 0.0, p: v(0, -0.925, 4.0), l: v(0, -0.93, 0), fov: 30 },
      { t: 2.8, p: v(0, -0.9, 3.95), l: v(0, -0.93, 0), fov: 30 },
      { t: 5.2, p: v(0, 0.08, 4.4), l: v(0, -0.95, -0.3), fov: 34 },
      { t: 7.4, p: v(0.1, 0.28, 3.3), l: v(0, -0.35, -0.2), fov: 34 },
      { t: 9.0, p: v(0.25, 0.18, 2.1), l: v(0, -0.06, 0), fov: 32 },
      { t: 10.4, p: v(0.5, 0.12, 1.5), l: v(0, 0, 0), fov: 32 },
      { t: 12.0, p: v(1.95, 0.55, 1.4), l: v(0.05, 0.08, 0.3), fov: 38 },
      { t: 13.9, p: v(0.45, 0.3, -1.0), l: u.clone().multiplyScalar(2.4), fov: 42 },
      { t: 15.5, p: D.clone().sub(u.clone().multiplyScalar(1.9)).add(side.clone().multiplyScalar(-0.3)).add(v(0, -0.12, 0)), l: D.clone().add(side.clone().multiplyScalar(0.12)), fov: 34 },
      { t: 17.0, p: D.clone().sub(u.clone().multiplyScalar(1.7)).add(side.clone().multiplyScalar(-0.24)).add(v(0, -0.1, 0)), l: D.clone().add(side.clone().multiplyScalar(0.12)), fov: 34 },
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
    return { p: interp('p'), l: interp('l'), fov: lerp(k1.fov, k2.fov, smooth(s)), offset: 0 };
  }

  titleCamera(t) {
    const k = easeOut(range01(t, T.title, T.end));
    return { p: new THREE.Vector3(lerp(0.12, 0.04, k), lerp(0.05, 0.02, k), lerp(2.45, 2.15, k)), l: new THREE.Vector3(0, 0, 0), fov: 30, offset: -560 };
  }

  setCamera(pose) {
    this.camera.position.copy(pose.p);
    this.camera.fov = pose.fov;
    this.camera.aspect = W / H;
    if (pose.offset) this.camera.setViewOffset(W, H, pose.offset, 0, W, H); else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
    this.camera.lookAt(pose.l);
    this.camera.updateMatrixWorld();
  }

  // ── Frame update ─────────────────────────────────────────────────────────
  update(t) {
    this.stars.material.opacity = 0.5 * smooth(range01(t, 3.5, 7.0));
    this.updateTerrain(t);
    this.updateArray(t);
    this.updateParticles(t);
    this.updateWave(t);
    this.updateBeam(t);
    this.updateDome(t);
    this.updateDrone(t);
  }

  updateTerrain(t) {
    const fadeOut = 1 - smooth(range01(t, 8.4, 10.6));
    const visible = t >= T.lineIn && fadeOut > 0.001;
    this.curtains.visible = visible;
    const ridges = this.ridges.begin();
    if (!visible) { ridges.end(); return; }
    const lineIn = smooth(range01(t, T.lineIn, 1.2));
    const tremble = smooth(range01(t, 1.3, 2.8));
    const spread = smooth(range01(t, T.spread, 4.3));
    const s = this.scroll(t);
    const base = Math.floor(s); const frac = s - base;
    const positions = this.curtains.geometry.attributes.position;
    let v = 0;
    for (let r = 0; r < ROWS; r++) {
      const j = base + r;
      const z = Z_NEAR - (r - frac) * DZ;
      // Before the terrain spreads out only the front ridge exists: silence, then a tremble.
      const front = r === 0 && t < T.spread + 1.4;
      const frame = front && t < T.spread ? t * 30 : j * 2;
      const amp = front ? lerp(0.08 * tremble, 1, spread) : spread;
      let bright = (front ? lineIn : spread) * fadeOut * Math.pow(1 - r / ROWS, 1.4);
      if (r === 0 && t >= T.spread) bright *= 1 - frac;
      const points = [];
      for (let c = 0; c <= COLS; c++) {
        const x = -X_HALF + (2 * X_HALF * c) / COLS;
        const h = amp * this.heightAt(x, frame);
        const y = BASE_Y + h;
        points.push([x, y, h]);
        positions.setXYZ(v++, x, y, z);
        positions.setXYZ(v++, x, BASE_Y - 0.03, z);
      }
      if (bright < 0.004) continue;
      for (let c = 0; c < COLS; c++) {
        const [x0, y0, h0] = points[c]; const [x1, y1, h1] = points[c + 1];
        const color = (h) => mix(scale(WIRE, 0.62), scale(SIGNAL, 2.6), smooth(range01(h / AMP, 0.16, 0.62))).map((q) => q * bright);
        ridges.seg(new THREE.Vector3(x0, y0, z), new THREE.Vector3(x1, y1, z), color(h0), color(h1));
      }
    }
    positions.needsUpdate = true;
    ridges.end();
  }

  micState(m, t) {
    // Returns [color, glow] for microphone m at time t.
    const arrive = this.arrive[m];
    if (t < arrive) return [[0, 0, 0], 0];
    let color = scale(WIRE, 0.75);
    let glow = 0.25;
    const pop = Math.exp(-(t - arrive) / 0.14);
    color = mix(color, scale(SIGNAL, 2.0), pop);
    glow += 0.6 * pop;
    const hit = this.waveTime[m];
    if (t > hit - 0.1) {
      const flash = Math.exp(-(((t - hit) / 0.05) ** 2));
      const armed = t < 16.6 ? smooth(range01(t, hit, hit + 0.12)) : 0;
      color = mix(color, scale(SIGNAL, 1.4), armed);
      color = mix(color, [1.8, 1.55, 1.3], flash);
      glow += 0.7 * flash + 0.3 * armed;
    }
    if (t > T.title) {
      const r = this.mics[m].length();
      const k = smooth(range01(t, T.title, T.titleFull));
      color = mix(color, scale(WIRE, 0.95 + 0.35 * Math.sin(t * 3.2 - r * 22)), k);
      glow = lerp(glow, 0.45 + 0.25 * Math.sin(t * 3.2 - r * 22), k);
    }
    return [color, glow];
  }

  updateArray(t) {
    const visible = t >= T.frameDraw - 0.2;
    this.plate.visible = visible;
    const frame = this.frameWires.begin(); const detail = this.detailWires.begin(); const rings = this.ringWires.begin();
    if (visible) {
      const draw = easeInOut(range01(t, T.frameDraw, 9.7));
      const titleDim = 1;
      const frameColor = () => scale(WIRE, 0.85 * titleDim);
      for (const z of [0.003, -0.01]) {
        frame.polyline(hexPoints(PLATE_R, z), frameColor, true, draw);
        frame.polyline(hexPoints(PLATE_R - 0.02, z), () => scale(WIRE, 0.35), true, draw);
      }
      detail.polyline(hexPoints(0.058, 0.003, Math.PI / 6), () => scale(WIRE, 0.8), true, draw);
      detail.polyline(hexPoints(0.032, 0.003, Math.PI / 6), () => scale(SIGNAL, 1.2 * draw), true, draw);
      this.mics.forEach((mic, m) => {
        const [color, glow] = this.micState(m, t);
        const arrive = this.arrive[m];
        const sprite = this.micGlows[m];
        sprite.visible = t >= arrive;
        sprite.material.color.setRGB(color[0] * glow, color[1] * glow, color[2] * glow);
        sprite.scale.setScalar(0.03 + 0.03 * Math.min(1.2, glow));
        if (t < arrive) return;
        const grow = easeOut(range01(t, arrive, arrive + 0.18));
        const radius = 0.0095 * (grow + 0.5 * Math.exp(-(t - arrive) / 0.1));
        const ring = Array.from({ length: 20 }, (_, i) => {
          const a = (i / 20) * Math.PI * 2;
          return new THREE.Vector3(mic.x + radius * Math.cos(a), mic.y + radius * Math.sin(a), 0.003);
        });
        rings.polyline(ring, () => color, true);
        const route = this.routes[m];
        const reach = easeOut(range01(t, arrive, arrive + 0.4));
        detail.polyline(route.map((p) => p.clone().setZ(0.002)), () => scale(WIRE, 0.32), false, reach);
      });
    } else {
      this.micGlows.forEach((sprite) => { sprite.visible = false; });
    }
    frame.end(); detail.end(); rings.end();
  }

  bezier(m, s) {
    const end = this.mics[m];
    const start = new THREE.Vector3(lerp(-1.7, 1.7, hash(m, 3)), BASE_Y + 0.05 + 0.2 * hash(m, 5), lerp(-1.6, 1.3, hash(m, 7)));
    const c1 = start.clone().add(new THREE.Vector3(0, 0.95, 0));
    const c2 = end.clone().add(new THREE.Vector3(0, 0.3, 0.9));
    const a = 1 - s;
    return start.multiplyScalar(a * a * a).add(c1.multiplyScalar(3 * a * a * s)).add(c2.multiplyScalar(3 * a * s * s)).add(end.clone().multiplyScalar(s * s * s));
  }

  updateParticles(t) {
    const wires = this.particleWires.begin();
    this.mics.forEach((_, m) => {
      const t0 = this.riseStart[m];
      const head = this.heads[m];
      const s = easeInOut(range01(t, t0, t0 + 1.3));
      const active = t >= t0 && t <= t0 + 1.32;
      head.visible = active;
      if (!active) return;
      const intensity = smooth(range01(t, t0, t0 + 0.15));
      const trail = [];
      for (let i = 0; i <= 14; i++) trail.push(this.bezier(m, Math.max(0, s - 0.22 * (1 - i / 14))));
      wires.polyline(trail, (k) => mix(scale(SIGNAL, 0.0), scale(SIGNAL, 2.4 * intensity), k * k));
      head.position.copy(trail.at(-1));
      head.material.color.setRGB(1.3 * intensity, 0.62 * intensity, 0.25 * intensity);
    });
    wires.end();
  }

  updateWave(t) {
    const wave = this.waveWires.begin(); const hit = this.waveHit.begin();
    const vis = window01(t, T.wave, T.wave + 0.35, T.wave + 1.55, T.wave + 1.95);
    if (vis > 0.002) {
      const u = this.u;
      const e1 = new THREE.Vector3().crossVectors(u, new THREE.Vector3(0, 1, 0)).normalize();
      const e2 = new THREE.Vector3().crossVectors(e1, u).normalize();
      const d = 1.35 - 1.1 * (t - T.wave);
      const center = u.clone().multiplyScalar(d);
      const n = 18; const half = 1.05; const pieces = 24;
      const at = (a, b) => center.clone().add(e1.clone().multiplyScalar(a)).add(e2.clone().multiplyScalar(b));
      const fade = (a, b) => 0.55 * vis * Math.exp(-(((a * a + b * b) / (0.9 * 0.9)) ** 2));
      for (let i = 0; i <= n; i++) {
        const fixed = -half + (2 * half * i) / n;
        for (let p = 0; p < pieces; p++) {
          const q0 = -half + (2 * half * p) / pieces; const q1 = -half + (2 * half * (p + 1)) / pieces;
          wave.seg(at(fixed, q0), at(fixed, q1), scale(WIRE, fade(fixed, q0)), scale(WIRE, fade(fixed, q1)));
          wave.seg(at(q0, fixed), at(q1, fixed), scale(WIRE, fade(q0, fixed)), scale(WIRE, fade(q1, fixed)));
        }
      }
      // Where the wavefront currently cuts the board.
      const ux = u.x; const uy = u.y; const n2 = ux * ux + uy * uy;
      const p0 = new THREE.Vector3((d * ux) / n2, (d * uy) / n2, 0.005);
      const r0 = p0.length(); const reach = PLATE_R - 0.004;
      if (r0 < reach) {
        const along = new THREE.Vector3(-uy, ux, 0).normalize().multiplyScalar(Math.sqrt(reach * reach - r0 * r0));
        hit.seg(p0.clone().sub(along), p0.clone().add(along), scale(SIGNAL, 3.2 * vis));
      }
    }
    wave.end(); hit.end();
  }

  updateBeam(t) {
    const lines = this.convergeWires.begin(); const beam = this.beamWires.begin(); const core = this.beamCore.begin();
    const fade = 1 - smooth(range01(t, 16.4, 17.2));
    if (t >= T.converge && fade > 0.001) {
      this.mics.forEach((mic, m) => {
        const start = this.waveTime[m] + 0.12;
        const q = easeOut(range01(t, start, start + 0.45));
        if (q <= 0) return;
        const end = mic.clone().lerp(this.F, q);
        const points = Array.from({ length: 7 }, (_, i) => mic.clone().lerp(end, i / 6));
        const settle = 1 - 0.65 * smooth(range01(t, 13.6, 14.4));
        lines.polyline(points, (k) => mix(scale(WIRE, 0.4 * fade * settle), scale(SIGNAL, 1.5 * fade * settle), k * q));
      });
      const reach = easeInOut(range01(t, T.beam, 13.75));
      if (reach > 0) {
        const tip = this.F.clone().lerp(this.D, reach);
        const points = Array.from({ length: 24 }, (_, i) => this.F.clone().lerp(tip, i / 23));
        beam.polyline(points, (k) => scale(SIGNAL, (3.4 - 1.4 * k) * fade));
        core.polyline(points, () => [2.4 * fade, 2.1 * fade, 1.8 * fade]);
      }
    }
    const focus = smooth(range01(t, 12.5, 13.0)) * fade;
    this.focusGlow.visible = focus > 0.001;
    this.focusGlow.material.color.setRGB(2.6 * focus, 1.2 * focus, 0.4 * focus);
    this.focusGlow.scale.setScalar(0.22);
    lines.end(); beam.end(); core.end();
  }

  updateDome(t) {
    const dome = this.domeWires.begin(); const cursor = this.cursorWires.begin();
    const vis = window01(t, T.dome, 13.4, 15.3, 16.2);
    if (vis > 0.002) {
      const sectors = this.array.rows * this.array.columns;
      for (let k = 0; k < sectors; k++) dome.polyline(this.sectorOutline(k, DOME_R), () => scale(WIRE, 0.26 * vis), true);
      let k = this.lockSector;
      if (t < T.lock) k = Math.floor((t - 13.0) / 0.0125) % sectors;
      if (t >= 13.0) {
        const locked = t >= T.lock;
        cursor.polyline(this.sectorOutline(Math.max(0, k), DOME_R * 0.998), () => (locked ? scale(SIGNAL, 3.0 * vis) : scale(WIRE, 1.6 * vis)), true);
      }
    }
    const fill = t >= T.lock ? vis * (0.28 + 0.9 * Math.exp(-(t - T.lock) / 0.25)) : 0;
    this.lockFill.visible = fill > 0.001;
    this.lockFill.material.color.setRGB(SIGNAL[0] * fill, SIGNAL[1] * fill, SIGNAL[2] * fill);
    dome.end(); cursor.end();
  }

  updateDrone(t) {
    const wires = this.droneWires.begin(); const print = this.printWires.begin();
    const vis = smooth(range01(t, T.drone, T.drone + 0.6)) * (1 - smooth(range01(t, 16.5, 17.3)));
    this.drone.visible = vis > 0.002;
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
        const spin = t * 37 + sx * 1.3 + sz * 0.7;
        wires.seg(local(sx * 0.085 + 0.05 * Math.cos(spin), 0.012, sz * 0.085 + 0.05 * Math.sin(spin)),
          local(sx * 0.085 - 0.05 * Math.cos(spin), 0.012, sz * 0.085 - 0.05 * Math.sin(spin)), scale(WIRE, 0.6 * vis));
      }
      this.discs.forEach((disc) => disc.material.color.setRGB(0.05 * vis, 0.06 * vis, 0.08 * vis));
      // Spectral fingerprint: the drone's rotor spectrum wrapped into a ring facing the camera.
      const draw = easeOut(range01(t, T.print, T.print + 0.9));
      if (draw > 0) {
        const toCamera = this.camera.position.clone().sub(this.D).normalize();
        const a1 = new THREE.Vector3().crossVectors(toCamera, new THREE.Vector3(0, 1, 0)).normalize();
        const a2 = new THREE.Vector3().crossVectors(a1, toCamera).normalize();
        // Circular spectrum analyser: one spoke per band, mirrored left/right.
        const frame = t * 25;
        const at = (radius, theta) => this.D.clone()
          .add(a1.clone().multiplyScalar(radius * Math.cos(theta)))
          .add(a2.clone().multiplyScalar(radius * Math.sin(theta)));
        const spokes = 128; const r0 = 0.25;
        const inner = [];
        for (let i = 0; i < spokes; i++) {
          const theta = Math.PI / 2 + (i / spokes) * Math.PI * 2 + t * 0.08;
          inner.push(at(r0 - 0.012, theta));
          if (i / spokes > draw) continue;
          const mirrored = 1 - Math.abs(1 - (2 * i) / spokes);
          const m = this.mag(frame, 1.5 + mirrored * 44);
          const length = 0.012 + 0.2 * m;
          print.seg(at(r0, theta), at(r0 + length, theta), scale(SIGNAL, 0.5 * vis), scale(SIGNAL, (0.8 + 2.4 * m) * vis));
        }
        print.polyline(inner, () => scale(WIRE, 0.3 * vis), true, draw);
      }
    }
    wires.end(); print.end();
  }

  render(pose) {
    this.setCamera(pose);
    this.composer.render();
  }

  project(point) {
    const v = point.clone().project(this.camera);
    return { x: ((v.x + 1) / 2) * W, y: ((1 - v.y) / 2) * H, visible: v.z < 1 };
  }
}
