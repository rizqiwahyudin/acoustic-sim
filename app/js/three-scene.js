/**
 * three-scene.js — Three.js renderer, scene, camera, controls, groups, and render helpers.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// ── Renderer + Scene ──
export const canvas = document.getElementById('canvas3d');
export const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x0a0a0f);

export const scene = new THREE.Scene();
export const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.05, 200);
camera.position.set(3, 3, 3);

export const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dLight = new THREE.DirectionalLight(0xffffff, 0.8);
dLight.position.set(5, 8, 5);
scene.add(dLight);

const axesHelper = new THREE.AxesHelper(0.4);
scene.add(axesHelper);

// ── Groups ──
export const micGroup = new THREE.Group();
scene.add(micGroup);

export const steerGroup = new THREE.Group();
scene.add(steerGroup);

export const trueDirGroup = new THREE.Group();
scene.add(trueDirGroup);

export const beamGroup = new THREE.Group();
scene.add(beamGroup);

export const roomGroup = new THREE.Group();
scene.add(roomGroup);

export const sourceGroup = new THREE.Group();
scene.add(sourceGroup);

export const hwRingGroup = new THREE.Group();
scene.add(hwRingGroup);

export const hwArrayMarkerGroup = new THREE.Group();
scene.add(hwArrayMarkerGroup);

// ── Render helpers ──
export function renderMics(mics, selectedIdx = -1) {
  micGroup.clear();
  const geo = new THREE.SphereGeometry(0.02, 8, 8);
  const mat = new THREE.MeshPhongMaterial({ color: 0x7ecfff, emissive: 0x112233 });
  const selectedMat = new THREE.MeshPhongMaterial({ color: 0xffdd44, emissive: 0x443300 });
  for (let i = 0; i < mics.length; i++) {
    const p = mics[i];
    const material = (i === selectedIdx) ? selectedMat : mat;
    const m = new THREE.Mesh(geo, material);
    m.position.set(p[0], p[2], p[1]);
    micGroup.add(m);
  }
}

export function renderSteerDir(azRad, elRad, radius) {
  steerGroup.clear();
  const colat = Math.PI / 2 - elRad;
  const x = radius * Math.sin(colat) * Math.cos(azRad);
  const z = radius * Math.sin(colat) * Math.sin(azRad);
  const y = radius * Math.cos(colat);
  const end = new THREE.Vector3(x, y, z);

  const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), end]);
  const lineMat = new THREE.LineBasicMaterial({ color: 0xffcc44, linewidth: 2 });
  steerGroup.add(new THREE.Line(lineGeo, lineMat));

  const cone = new THREE.Mesh(
    new THREE.ConeGeometry(0.06, 0.18, 8),
    new THREE.MeshPhongMaterial({ color: 0xffcc44, emissive: 0x332200 })
  );
  cone.position.copy(end);
  cone.lookAt(0, 0, 0);
  cone.rotateX(Math.PI / 2);
  steerGroup.add(cone);
}

export function renderTrueDir(azRad, elRad, radius) {
  trueDirGroup.clear();
  const colat = Math.PI / 2 - elRad;
  const x = radius * Math.sin(colat) * Math.cos(azRad);
  const z = radius * Math.sin(colat) * Math.sin(azRad);
  const y = radius * Math.cos(colat);
  const end = new THREE.Vector3(x, y, z);

  const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), end]);
  const lineMat = new THREE.LineBasicMaterial({ color: 0x44cc44, linewidth: 2 });
  trueDirGroup.add(new THREE.Line(lineGeo, lineMat));

  const cone = new THREE.Mesh(
    new THREE.ConeGeometry(0.06, 0.18, 8),
    new THREE.MeshPhongMaterial({ color: 0x44cc44, emissive: 0x113311 })
  );
  cone.position.copy(end);
  cone.lookAt(0, 0, 0);
  cone.rotateX(Math.PI / 2);
  trueDirGroup.add(cone);
}

function toScene(pos, center, s) {
  return new THREE.Vector3(
    (pos[0] - center[0]) * s,
    (pos[2] - center[2]) * s,
    (pos[1] - center[1]) * s
  );
}

export function renderRoom(roomDim, arrayCenter) {
  roomGroup.clear();
  const s = 6.0 / Math.max(roomDim[0], roomDim[1], roomDim[2]);
  const w = roomDim[0] * s, d = roomDim[1] * s, h = roomDim[2] * s;
  const box = new THREE.BoxGeometry(w, h, d);
  const edges = new THREE.EdgesGeometry(box);
  const mat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.18 });
  const wireframe = new THREE.LineSegments(edges, mat);
  wireframe.position.set(
    (roomDim[0] / 2 - arrayCenter[0]) * s,
    (roomDim[2] / 2 - arrayCenter[2]) * s,
    (roomDim[1] / 2 - arrayCenter[1]) * s
  );
  roomGroup.add(wireframe);

  const floorGeo = new THREE.PlaneGeometry(w, d);
  const floorMat = new THREE.MeshBasicMaterial({ color: 0x1a1a2e, transparent: true, opacity: 0.15, side: THREE.DoubleSide });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(
    (roomDim[0] / 2 - arrayCenter[0]) * s,
    (0 - arrayCenter[2]) * s,
    (roomDim[1] / 2 - arrayCenter[1]) * s
  );
  roomGroup.add(floor);

  return s;
}

export function renderSources(srcPos, crowdPos, paPos, imgPos, center, s, trajectory) {
  sourceGroup.clear();
  const traj = (trajectory && trajectory.length > 1) ? trajectory : null;

  if (traj) {
    const pts = traj.map(p => toScene(p, center, s));
    const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
    const lineMat = new THREE.LineBasicMaterial({ color: 0xffdd77, transparent: true, opacity: 0.85 });
    sourceGroup.add(new THREE.Line(lineGeo, lineMat));

    const waypointGeo = new THREE.SphereGeometry(0.035, 8, 8);
    const waypointMat = new THREE.MeshPhongMaterial({ color: 0xffdd77, emissive: 0x332200 });
    for (const p of pts) {
      const m = new THREE.Mesh(waypointGeo, waypointMat);
      m.position.copy(p);
      sourceGroup.add(m);
    }
    const startGeo = new THREE.SphereGeometry(0.09, 12, 12);
    const startMat = new THREE.MeshBasicMaterial({ color: 0xff4444, wireframe: true });
    const startMesh = new THREE.Mesh(startGeo, startMat);
    startMesh.position.copy(pts[0]);
    sourceGroup.add(startMesh);

    const endMat = new THREE.MeshPhongMaterial({ color: 0xff4444, emissive: 0x331111 });
    const endMesh = new THREE.Mesh(new THREE.SphereGeometry(0.08, 12, 12), endMat);
    endMesh.position.copy(pts[pts.length - 1]);
    sourceGroup.add(endMesh);
  } else {
    const droneMat = new THREE.MeshPhongMaterial({ color: 0xff4444, emissive: 0x331111 });
    const drone = new THREE.Mesh(new THREE.SphereGeometry(0.08, 12, 12), droneMat);
    drone.position.copy(toScene(srcPos, center, s));
    sourceGroup.add(drone);
  }

  const crowdGeo = new THREE.SphereGeometry(0.03, 6, 6);
  const crowdMat = new THREE.MeshPhongMaterial({ color: 0xff8833, emissive: 0x221100 });
  for (const p of crowdPos) {
    const m = new THREE.Mesh(crowdGeo, crowdMat);
    m.position.copy(toScene(p, center, s));
    sourceGroup.add(m);
  }

  const paGeo = new THREE.SphereGeometry(0.04, 6, 6);
  const paMat = new THREE.MeshPhongMaterial({ color: 0xaa44ff, emissive: 0x110022 });
  for (const p of paPos) {
    const m = new THREE.Mesh(paGeo, paMat);
    m.position.copy(toScene(p, center, s));
    sourceGroup.add(m);
  }

  const imgGeo = new THREE.SphereGeometry(0.05, 8, 8);
  const imgMat = new THREE.MeshPhongMaterial({ color: 0xff4444, transparent: true, opacity: 0.25, emissive: 0x220000 });
  for (const p of imgPos) {
    const m = new THREE.Mesh(imgGeo, imgMat);
    m.position.copy(toScene(p, center, s));
    sourceGroup.add(m);
  }
}

export function renderRIR(rirData) {
  const wrap = document.getElementById('rirWrap');
  if (!rirData || rirData.length === 0) { wrap.classList.remove('visible'); return; }
  wrap.classList.add('visible');
  const rirCanvas = document.getElementById('rirCanvas');
  const ctx = rirCanvas.getContext('2d');
  const W = rirCanvas.width, H = rirCanvas.height;
  ctx.fillStyle = 'rgba(10,10,20,0.95)';
  ctx.fillRect(0, 0, W, H);

  let peak = 0;
  for (let i = 0; i < rirData.length; i++) peak = Math.max(peak, Math.abs(rirData[i]));
  if (peak === 0) peak = 1;

  const step = Math.max(1, Math.floor(rirData.length / W));
  const mid = H / 2;

  ctx.beginPath();
  ctx.strokeStyle = '#7ecfff';
  ctx.lineWidth = 1;
  for (let x = 0; x < W; x++) {
    const idx = Math.min(x * step, rirData.length - 1);
    const y = mid - (rirData[idx] / peak) * (mid - 4);
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();

  ctx.fillStyle = '#666';
  ctx.font = '8px monospace';
  const ms = (rirData.length / 16000 * 1000).toFixed(0);
  ctx.fillText(ms + ' ms', W - 32, H - 4);
}

// ── Animate loop ──
let _onFrameCallback = null;

export function setOnFrameCallback(cb) {
  _onFrameCallback = cb;
}

function animate() {
  requestAnimationFrame(animate);
  if (_onFrameCallback) _onFrameCallback();
  controls.update();
  renderer.render(scene, camera);
}
animate();

// ── Resize handler ──
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
