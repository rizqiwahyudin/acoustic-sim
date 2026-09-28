/**
 * Three.js stage: a placeholder HEIMDALL array built from the real microphone
 * coordinates, its rear electronics pod, a hex-grid floor and the 6x6 sector dome.
 *
 * World axes: +x = the array's right, +y = up, the array faces -z (three.js is
 * right-handed, the contract's z-forward frame is not).
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { T, shotAt } from './story.js';
import { W, H, clamp, lerp, range01, smooth, easeInOut, easeOut, heat, hash } from './util.js';

const BOARD_R = 0.275;
const FRONT_Z = -0.006;
const HUB_R = 0.058;
const DOME_R = 1.35;
const ORANGE = new THREE.Color(1.0, 0.42, 0.04);

function hexShape(radius) {
  const shape = new THREE.Shape();
  for (let i = 0; i < 6; i++) {
    const a = (i * Math.PI) / 3;
    const x = radius * Math.cos(a); const y = radius * Math.sin(a);
    if (i === 0) shape.moveTo(x, y); else shape.lineTo(x, y);
  }
  shape.closePath();
  return shape;
}

function hexPrism(radius, depth) {
  // Cylinder with six segments, first vertex on +x, axis along z.
  const geometry = new THREE.CylinderGeometry(radius, radius, depth, 6, 1, false, Math.PI / 2);
  geometry.rotateX(Math.PI / 2);
  return geometry;
}

function glowLines(geometry, color, intensity, opacity = 1) {
  const material = new THREE.LineBasicMaterial({ transparent: true, opacity, toneMapped: false });
  material.color.copy(color).multiplyScalar(intensity);
  return new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 25), material);
}

function canvasTexture(size, draw) {
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  draw(canvas.getContext('2d'), size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

export class Stage {
  constructor(canvas, story) {
    this.story = story;
    this.canvas = canvas;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
    renderer.setPixelRatio(1);
    renderer.setSize(W, H, false);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    this.renderer = renderer;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);
    const pmrem = new THREE.PMREMGenerator(renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.22;
    this.camera = new THREE.PerspectiveCamera(30, W / H, 0.01, 60);

    this.mics = story.array.microphones_mm.map(([x, y]) => new THREE.Vector3(x / 1000, y / 1000, FRONT_Z));
    this.buildLights();
    this.buildArray();
    this.buildElectronics();
    this.buildFloor();
    this.buildDome();
    this.buildAtmosphere();

    const target = new THREE.WebGLRenderTarget(W, H, { type: THREE.HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(renderer, target);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(W, H), 0.85, 0.55, 0.72);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.xray = false;
  }

  // ── Construction ─────────────────────────────────────────────────────────
  buildLights() {
    this.scene.add(new THREE.AmbientLight(0x1a1410, 0.35));
    this.key = new THREE.DirectionalLight(0xfff1e0, 1.4);
    this.key.position.set(-1.2, 1.4, -1.6);
    this.rim = new THREE.DirectionalLight(0xff7a1a, 3.0);
    this.rim.position.set(1.0, 0.8, 1.6);
    this.fill = new THREE.DirectionalLight(0x4a6aff, 0.35);
    this.fill.position.set(1.5, -0.4, -0.8);
    this.sweepLight = new THREE.PointLight(0xffa040, 0, 0.9, 1.6);
    this.scene.add(this.key, this.rim, this.fill, this.sweepLight);
  }

  buildArray() {
    const group = new THREE.Group();
    this.arrayGroup = group;
    this.scene.add(group);
    this.fadeMaterials = [];

    // PCB front face: a transparent-outside hexagon texture on a plane.
    const boardTexture = canvasTexture(2048, (ctx, size) => this.drawBoardTexture(ctx, size));
    const front = new THREE.Mesh(
      new THREE.PlaneGeometry(2 * BOARD_R, 2 * BOARD_R).rotateY(Math.PI),
      new THREE.MeshStandardMaterial({ map: boardTexture, roughness: 0.62, metalness: 0.25, alphaTest: 0.5 }),
    );
    front.position.z = FRONT_Z - 0.0002;
    group.add(front);
    this.boardFront = front;

    const coreGeometry = new THREE.ExtrudeGeometry(hexShape(BOARD_R), { depth: 0.012, bevelEnabled: false });
    coreGeometry.translate(0, 0, -0.006);
    this.boardCore = new THREE.Mesh(coreGeometry, new THREE.MeshStandardMaterial({ color: 0x0b0c0e, roughness: 0.7, metalness: 0.3 }));
    group.add(this.boardCore);

    // Machined hex frame with a lip proud of the PCB.
    const ring = hexShape(0.297);
    ring.holes.push(hexShape(BOARD_R - 0.001));
    const frameGeometry = new THREE.ExtrudeGeometry(ring, {
      depth: 0.046, bevelEnabled: true, bevelThickness: 0.003, bevelSize: 0.003, bevelSegments: 2,
    });
    frameGeometry.translate(0, 0, -0.018);
    this.frame = new THREE.Mesh(frameGeometry, new THREE.MeshStandardMaterial({ color: 0x16181c, roughness: 0.32, metalness: 0.85 }));
    group.add(this.frame);
    this.frameEdges = glowLines(frameGeometry, ORANGE, 1.6, 0.85);
    group.add(this.frameEdges);

    // Microphone ports.
    const ringGeometry = new THREE.TorusGeometry(0.0085, 0.0016, 8, 32);
    const coreDisc = new THREE.CircleGeometry(0.0062, 28).rotateY(Math.PI);
    this.ports = this.mics.map((position) => {
      const ringMesh = new THREE.Mesh(ringGeometry, new THREE.MeshStandardMaterial({
        color: 0x2b2b2b, roughness: 0.28, metalness: 0.9, emissive: 0xff6a00, emissiveIntensity: 0,
      }));
      ringMesh.position.set(position.x, position.y, FRONT_Z - 0.0016);
      const disc = new THREE.Mesh(coreDisc, new THREE.MeshBasicMaterial({ color: 0x000000, toneMapped: false }));
      disc.position.set(position.x, position.y, FRONT_Z - 0.0009);
      group.add(ringMesh, disc);
      return { ring: ringMesh, disc };
    });

    // Glowing signal traces, octilinear routes from the hub to every port.
    const positions = []; const colors = []; this.traceMeta = [];
    this.mics.forEach((mic, index) => {
      const route = this.traceRoute(mic);
      let total = 0;
      for (let i = 1; i < route.length; i++) total += route[i].distanceTo(route[i - 1]);
      if (total < 0.004) return;
      let travelled = 0;
      for (let i = 1; i < route.length; i++) {
        const a = route[i - 1]; const b = route[i];
        const length = a.distanceTo(b);
        const pieces = Math.max(2, Math.ceil(length / 0.008));
        for (let p = 0; p < pieces; p++) {
          const p0 = a.clone().lerp(b, p / pieces); const p1 = a.clone().lerp(b, (p + 1) / pieces);
          positions.push(p0.x, p0.y, FRONT_Z - 0.0006, p1.x, p1.y, FRONT_Z - 0.0006);
          colors.push(0, 0, 0, 0, 0, 0);
          const s0 = (travelled + (length * p) / pieces) / total;
          const s1 = (travelled + (length * (p + 1)) / pieces) / total;
          this.traceMeta.push([index, s0], [index, s1]);
        }
        travelled += length;
      }
    });
    const traceGeometry = new THREE.BufferGeometry();
    traceGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    traceGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    this.traces = new THREE.LineSegments(traceGeometry, new THREE.LineBasicMaterial({ vertexColors: true, toneMapped: false, transparent: true }));
    group.add(this.traces);

    // Front hub with the HEIMDALL eye emblem.
    const hubGeometry = hexPrism(HUB_R, 0.012);
    this.hub = new THREE.Mesh(hubGeometry, new THREE.MeshStandardMaterial({ color: 0x121418, roughness: 0.3, metalness: 0.85 }));
    this.hub.position.z = FRONT_Z - 0.006;
    this.hub.rotation.z = Math.PI / 6;   // flats face the inner microphone ring
    group.add(this.hub);
    this.hubEdges = glowLines(hubGeometry, ORANGE, 1.4, 0.9);
    this.hubEdges.position.copy(this.hub.position);
    this.hubEdges.rotation.z = Math.PI / 6;
    group.add(this.hubEdges);
    const emblemTexture = canvasTexture(512, (ctx, size) => this.drawEmblem(ctx, size));
    this.emblem = new THREE.Mesh(
      new THREE.PlaneGeometry(HUB_R * 1.9, HUB_R * 1.9).rotateY(Math.PI),
      new THREE.MeshBasicMaterial({ map: emblemTexture, transparent: true, toneMapped: false, depthWrite: false }),
    );
    this.emblem.position.z = FRONT_Z - 0.0122;
    group.add(this.emblem);

    // Mast and yoke.
    const metal = new THREE.MeshStandardMaterial({ color: 0x1b1d21, roughness: 0.35, metalness: 0.85 });
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.017, 0.017, 0.9, 20), metal);
    mast.position.set(0, -0.575, 0.045);
    const collar = new THREE.Mesh(new THREE.CylinderGeometry(0.026, 0.026, 0.04, 6), metal);
    collar.position.set(0, -0.14, 0.045);
    group.add(mast, collar);
    this.metalParts = [metal];
  }

  traceRoute(mic) {
    const angle = Math.atan2(mic.y, mic.x);
    const start = new THREE.Vector2(Math.cos(angle) * 0.053, Math.sin(angle) * 0.053);
    const dx = mic.x - start.x; const dy = mic.y - start.y;
    let bend;
    if (Math.abs(dx) > Math.abs(dy)) bend = new THREE.Vector2(start.x + Math.sign(dx) * Math.abs(dy), mic.y);
    else bend = new THREE.Vector2(mic.x, start.y + Math.sign(dy) * Math.abs(dx));
    const port = new THREE.Vector2(mic.x, mic.y);
    const toPort = port.clone().sub(bend);
    const stop = toPort.length() > 0.011 ? port.clone().sub(toPort.normalize().multiplyScalar(0.0105)) : bend;
    return [start, bend, stop].map((p) => new THREE.Vector3(p.x, p.y, 0));
  }

  drawBoardTexture(ctx, size) {
    const scale = size / (2 * BOARD_R);
    const px = (x, y) => [(0.5 - x / (2 * BOARD_R)) * size, (0.5 - y / (2 * BOARD_R)) * size];
    ctx.clearRect(0, 0, size, size);
    ctx.save();
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = (i * Math.PI) / 3;
      const [x, y] = px(BOARD_R * Math.cos(a), BOARD_R * Math.sin(a));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = '#0c0d0f';
    ctx.fill();
    ctx.clip();
    // Faint hex ground-pour texture.
    ctx.strokeStyle = 'rgba(255,255,255,0.035)';
    ctx.lineWidth = 2;
    const cell = 0.018 * scale;
    for (let row = -2; row < size / (cell * 1.5) + 2; row++) {
      for (let col = -2; col < size / (cell * 1.732) + 2; col++) {
        const cx = col * cell * 1.732 + (row % 2) * cell * 0.866;
        const cy = row * cell * 1.5;
        ctx.beginPath();
        for (let i = 0; i < 6; i++) {
          const a = Math.PI / 6 + (i * Math.PI) / 3;
          ctx.lineTo(cx + cell * 0.95 * Math.cos(a), cy + cell * 0.95 * Math.sin(a));
        }
        ctx.closePath();
        ctx.stroke();
      }
    }
    // Copper routes (the 3D lines add the glow on top).
    ctx.strokeStyle = '#3b2412';
    ctx.lineWidth = 0.0024 * scale;
    ctx.lineJoin = 'round';
    for (const mic of this.mics) {
      const route = this.traceRoute(mic);
      ctx.beginPath();
      route.forEach((p, i) => { const [x, y] = px(p.x, p.y); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.stroke();
    }
    // Pads and silkscreen designators.
    this.mics.forEach((mic, index) => {
      const [x, y] = px(mic.x, mic.y);
      ctx.fillStyle = '#6e4a22';
      ctx.beginPath(); ctx.arc(x, y, 0.0115 * scale, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#050505';
      ctx.beginPath(); ctx.arc(x, y, 0.0072 * scale, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = 'rgba(225,220,210,0.75)';
      ctx.font = `${Math.round(0.0062 * scale)}px "Share Tech Mono"`;
      ctx.textAlign = 'center';
      ctx.fillText(`M${String(index + 1).padStart(2, '0')}`, x, y + 0.0185 * scale);
    });
    // Mounting holes at the six vertices, fiducials and board legend.
    for (let i = 0; i < 6; i++) {
      const a = (i * Math.PI) / 3;
      const [x, y] = px((BOARD_R - 0.02) * Math.cos(a), (BOARD_R - 0.02) * Math.sin(a));
      ctx.fillStyle = '#8a6a3a'; ctx.beginPath(); ctx.arc(x, y, 0.0065 * scale, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#000'; ctx.beginPath(); ctx.arc(x, y, 0.0038 * scale, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = 'rgba(230,225,215,0.8)';
    ctx.textAlign = 'center';
    ctx.font = `${Math.round(0.0105 * scale)}px "Barlow Condensed"`;
    const [lx, ly] = px(0, -0.228);
    ctx.fillText('HEIMDALL  ·  44-CH ACOUSTIC ARRAY  ·  REV C', lx, ly);
    const [rx, ry] = px(0, 0.236);
    ctx.font = `${Math.round(0.0075 * scale)}px "Share Tech Mono"`;
    ctx.fillText('48 kHz  ·  HEX APERTURE 471 mm  ·  UP ^', rx, ry);
    ctx.restore();
  }

  drawEmblem(ctx, size) {
    const c = size / 2;
    ctx.clearRect(0, 0, size, size);
    ctx.strokeStyle = '#ff7a00';
    ctx.lineWidth = size * 0.02;
    const hex = (r) => {
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const a = Math.PI / 6 + (i * Math.PI) / 3;
        ctx.lineTo(c + r * Math.cos(a), c + r * Math.sin(a));
      }
      ctx.closePath();
    };
    hex(size * 0.44); ctx.stroke();
    ctx.lineWidth = size * 0.008; hex(size * 0.38); ctx.stroke();
    // Watchman's eye.
    ctx.lineWidth = size * 0.022;
    ctx.beginPath();
    ctx.moveTo(c - size * 0.3, c);
    ctx.quadraticCurveTo(c, c - size * 0.26, c + size * 0.3, c);
    ctx.quadraticCurveTo(c, c + size * 0.26, c - size * 0.3, c);
    ctx.stroke();
    ctx.fillStyle = '#ffd08a';
    hex(size * 0.075); ctx.fill();
    ctx.fillStyle = '#ff7a00';
    ctx.font = `700 ${Math.round(size * 0.07)}px "Barlow Condensed"`;
    ctx.textAlign = 'center';
    ctx.letterSpacing = `${Math.round(size * 0.012)}px`;
    ctx.fillText('HEIMDALL', c, c + size * 0.3);
  }

  buildElectronics() {
    const group = new THREE.Group();
    this.electronics = group;
    this.arrayGroup.add(group);
    const housingGeometry = hexPrism(0.122, 0.052);
    this.housing = new THREE.Mesh(housingGeometry, new THREE.MeshStandardMaterial({ color: 0x15171b, roughness: 0.36, metalness: 0.85 }));
    this.housing.position.z = 0.032;
    this.housingEdges = glowLines(housingGeometry, ORANGE, 1.2, 0.7);
    this.housingEdges.position.copy(this.housing.position);
    group.add(this.housing, this.housingEdges);

    const finMaterial = new THREE.MeshStandardMaterial({ color: 0x1a1c20, roughness: 0.4, metalness: 0.8 });
    this.fins = [];
    for (let i = -4; i <= 4; i++) {
      const fin = new THREE.Mesh(new THREE.BoxGeometry(0.004, 0.15 - Math.abs(i) * 0.012, 0.018), finMaterial);
      fin.position.set(i * 0.017, 0, 0.067);
      this.fins.push(fin);
      group.add(fin);
    }
    this.finMaterial = finMaterial;

    const pcbTexture = canvasTexture(1024, (ctx, size) => this.drawCorePcb(ctx, size));
    const pcbTop = new THREE.MeshStandardMaterial({
      map: pcbTexture, emissiveMap: pcbTexture, emissive: 0xffffff, emissiveIntensity: 0, roughness: 0.6, metalness: 0.2,
    });
    const pcbSide = new THREE.MeshStandardMaterial({ color: 0x0d0f10, roughness: 0.7, metalness: 0.2 });
    const inner = new THREE.Mesh(new THREE.BoxGeometry(0.17, 0.12, 0.0016), [pcbSide, pcbSide, pcbSide, pcbSide, pcbTop, pcbSide]);
    this.pcbTop = pcbTop;
    inner.position.set(0, 0, 0.026);
    group.add(inner);
    this.innerBoard = inner;

    const chip = (label, sub, width, depth, x, pins) => {
      const labelTexture = canvasTexture(512, (ctx, size) => {
        ctx.fillStyle = '#16161a'; ctx.fillRect(0, 0, size, size);
        ctx.fillStyle = 'rgba(210,205,195,0.9)';
        ctx.textAlign = 'center';
        ctx.font = `600 ${Math.round(size * 0.14)}px "Barlow Condensed"`;
        ctx.fillText(label, size / 2, size * 0.46);
        ctx.font = `${Math.round(size * 0.08)}px "Share Tech Mono"`;
        ctx.fillText(sub, size / 2, size * 0.62);
        ctx.beginPath(); ctx.arc(size * 0.12, size * 0.12, size * 0.035, 0, Math.PI * 2); ctx.fill();
      });
      const side = new THREE.MeshStandardMaterial({ color: 0x141418, roughness: 0.5, metalness: 0.3, emissive: 0xff6a00, emissiveIntensity: 0 });
      const top = new THREE.MeshStandardMaterial({ map: labelTexture, roughness: 0.45, metalness: 0.2, emissive: 0xff6a00, emissiveIntensity: 0 });
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, width, depth), [side, side, side, side, top, side]);
      mesh.position.set(x, 0.012, 0.0268 + depth / 2);
      group.add(mesh);
      if (pins) {
        const pinGeometry = new THREE.BoxGeometry(0.0006, 0.0024, 0.0008);
        const pinMaterial = new THREE.MeshStandardMaterial({ color: 0xb8b0a0, metalness: 1.0, roughness: 0.25 });
        for (let side = 0; side < 4; side++) {
          for (let p = 0; p < 14; p++) {
            const offset = -width / 2 + width * ((p + 0.5) / 14);
            const pin = new THREE.Mesh(pinGeometry, pinMaterial);
            const edge = width / 2 + 0.0012;
            if (side === 0) pin.position.set(x + offset, 0.012 + edge, 0.0272);
            if (side === 1) pin.position.set(x + offset, 0.012 - edge, 0.0272);
            if (side === 2) { pin.position.set(x + edge, 0.012 + offset, 0.0272); pin.rotation.z = Math.PI / 2; }
            if (side === 3) { pin.position.set(x - edge, 0.012 + offset, 0.0272); pin.rotation.z = Math.PI / 2; }
            group.add(pin);
          }
        }
      }
      const edges = glowLines(mesh.geometry, ORANGE, 2.2, 0);
      edges.position.copy(mesh.position);
      group.add(edges);
      return { mesh, top, side, edges, anchor: mesh.position.clone().add(new THREE.Vector3(0, 0, depth / 2)) };
    };
    this.mcu = chip('MAX78002', 'AI MCU · CNN', 0.03, 0.004, -0.042, false);
    this.dsp = chip('ADAU1467', 'SIGMADSP', 0.028, 0.003, 0.044, true);

    const connector = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.03, 20), this.metalParts[0]);
    connector.position.set(0, -0.112, 0.032);
    group.add(connector);
  }

  drawCorePcb(ctx, size) {
    // Square canvas stretched over the 170 x 120 mm core board (+z face, viewed from behind).
    const px = (x, y) => [((x + 0.085) / 0.17) * size, ((0.06 - y) / 0.12) * size];
    ctx.fillStyle = '#0b0d0e'; ctx.fillRect(0, 0, size, size);
    ctx.strokeStyle = '#ff7a00'; ctx.lineCap = 'round';
    const mcu = [-0.042, 0.012]; const dsp = [0.044, 0.012];
    for (let i = 0; i < 9; i++) {
      const y0 = 0.012 - 0.012 + i * 0.003;
      ctx.lineWidth = 3; ctx.globalAlpha = 0.85;
      ctx.beginPath();
      let [x, y] = px(mcu[0] + 0.017, y0); ctx.moveTo(x, y);
      [x, y] = px(-0.004, y0); ctx.lineTo(x, y);
      [x, y] = px(0.004, y0 + 0.004); ctx.lineTo(x, y);
      [x, y] = px(dsp[0] - 0.017, y0 + 0.004); ctx.lineTo(x, y);
      ctx.stroke();
    }
    ctx.globalAlpha = 0.5; ctx.lineWidth = 2;
    for (let i = 0; i < 14; i++) {
      const x0 = -0.075 + i * 0.0115;
      ctx.beginPath();
      let [x, y] = px(x0, -0.055); ctx.moveTo(x, y);
      [x, y] = px(x0, -0.03); ctx.lineTo(x, y);
      [x, y] = px(x0 * 0.5 + (i < 7 ? mcu[0] : dsp[0]) * 0.5, -0.01); ctx.lineTo(x, y);
      ctx.stroke();
      const [vx, vy] = px(x0, -0.055);
      ctx.beginPath(); ctx.arc(vx, vy, 6, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.globalAlpha = 0.8;
    ctx.fillStyle = 'rgba(230,220,205,0.8)';
    ctx.font = `${Math.round(size * 0.03)}px "Share Tech Mono"`;
    const [tx, ty] = px(-0.08, 0.052);
    ctx.fillText('HEIMDALL CORE  ·  MAX78002 + ADAU1467  ·  44 CH TDM', tx, ty);
    ctx.globalAlpha = 1;
  }

  buildFloor() {
    const material = new THREE.ShaderMaterial({
      uniforms: { uFade: { value: 0 }, uPulseR: { value: 0 }, uPulseA: { value: 0 }, uGlow: { value: 0 } },
      vertexShader: `varying vec3 vPos; void main(){ vec4 w = modelMatrix * vec4(position,1.0); vPos = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
      fragmentShader: `
        varying vec3 vPos;
        uniform float uFade, uPulseR, uPulseA, uGlow;
        float hexDist(vec2 p){ p = abs(p); return max(dot(p, normalize(vec2(1.0, 1.7320508))), p.x); }
        void main(){
          vec2 uv = vPos.xz * 5.0;
          vec2 r = vec2(1.0, 1.7320508); vec2 h = r * 0.5;
          vec2 a = mod(uv, r) - h; vec2 b = mod(uv - h, r) - h;
          vec2 g = dot(a, a) < dot(b, b) ? a : b;
          float line = smoothstep(0.045, 0.0, abs(hexDist(g) - 0.5));
          float dist = length(vPos.xz - vec2(0.0, -0.2));
          float fade = exp(-dist * dist * 0.22) * uFade;
          float ring = uPulseA * smoothstep(0.08, 0.0, abs(dist - uPulseR)) * exp(-dist * 0.35);
          vec3 orange = vec3(1.0, 0.42, 0.05);
          vec3 col = orange * (line * 0.55 * fade + ring * 2.2 * (0.3 + line)) + vec3(0.35, 0.09, 0.0) * uGlow * exp(-dist * dist * 2.0);
          gl_FragColor = vec4(col, 1.0);
        }`,
    });
    this.floor = new THREE.Mesh(new THREE.PlaneGeometry(18, 18).rotateX(-Math.PI / 2), material);
    this.floor.position.y = -1.0;
    this.scene.add(this.floor);
  }

  domeDirection(azDeg, elDeg) {
    const az = (azDeg * Math.PI) / 180; const el = (elDeg * Math.PI) / 180;
    return new THREE.Vector3(Math.cos(el) * Math.sin(az), Math.sin(el), -Math.cos(el) * Math.cos(az));
  }

  buildDome() {
    const story = this.story;
    const edgesOf = (centers) => {
      const step = centers[1] - centers[0];
      return [...centers.map((c) => c - step / 2), centers.at(-1) + step / 2];
    };
    this.azEdges = edgesOf(story.array.azimuth_deg);
    this.elEdges = edgesOf(story.array.elevation_deg);
    const group = new THREE.Group();
    this.dome = group;
    this.scene.add(group);
    this.tiles = [];
    const segments = 6;
    const gridPositions = [];
    for (let k = 0; k < story.sectors; k++) {
      const [row, col] = story.rowCol(k);
      const az0 = this.azEdges[col]; const az1 = this.azEdges[col + 1];
      const el0 = this.elEdges[row]; const el1 = this.elEdges[row + 1];
      const positions = []; const index = [];
      const inset = 0.35;
      for (let j = 0; j <= segments; j++) {
        for (let i = 0; i <= segments; i++) {
          const az = lerp(az0 + inset, az1 - inset, i / segments);
          const el = lerp(el0 + inset, el1 - inset, j / segments);
          const p = this.domeDirection(az, el).multiplyScalar(DOME_R);
          positions.push(p.x, p.y, p.z);
        }
      }
      for (let j = 0; j < segments; j++) {
        for (let i = 0; i < segments; i++) {
          const a = j * (segments + 1) + i; const b = a + 1; const c = a + segments + 1; const d = c + 1;
          index.push(a, c, b, b, c, d);
        }
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geometry.setIndex(index);
      const material = new THREE.MeshBasicMaterial({
        color: 0x000000, transparent: true, opacity: 1, side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false,
      });
      const mesh = new THREE.Mesh(geometry, material);
      group.add(mesh);
      this.tiles.push(mesh);
      const arc = (fromAz, fromEl, toAz, toEl) => {
        for (let s = 0; s < 8; s++) {
          const p0 = this.domeDirection(lerp(fromAz, toAz, s / 8), lerp(fromEl, toEl, s / 8)).multiplyScalar(DOME_R);
          const p1 = this.domeDirection(lerp(fromAz, toAz, (s + 1) / 8), lerp(fromEl, toEl, (s + 1) / 8)).multiplyScalar(DOME_R);
          gridPositions.push(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);
        }
      };
      arc(az0, el0, az1, el0); arc(az1, el0, az1, el1); arc(az1, el1, az0, el1); arc(az0, el1, az0, el0);
    }
    const gridGeometry = new THREE.BufferGeometry();
    gridGeometry.setAttribute('position', new THREE.Float32BufferAttribute(gridPositions, 3));
    const gridMaterial = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.55, toneMapped: false });
    gridMaterial.color.copy(ORANGE).multiplyScalar(0.9);
    group.add(new THREE.LineSegments(gridGeometry, gridMaterial));

    this.cursor = new THREE.LineLoop(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: new THREE.Color(4, 3.2, 2.2), toneMapped: false }));
    this.cursor.geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(33 * 3), 3));
    group.add(this.cursor);

    const coneMaterial = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.16, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide, toneMapped: false });
    coneMaterial.color.copy(ORANGE).multiplyScalar(1.6);
    const coneGeometry = new THREE.CylinderGeometry(0.15, 0.012, 1, 32, 1, true);
    coneGeometry.translate(0, 0.5, 0);
    this.cone = new THREE.Mesh(coneGeometry, coneMaterial);
    const coreMaterial = coneMaterial.clone(); coreMaterial.opacity = 0.35;
    this.coneCore = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.004, 1, 16, 1, true).translate(0, 0.5, 0), coreMaterial);
    group.add(this.cone, this.coneCore);
  }

  buildAtmosphere() {
    const haloTexture = canvasTexture(256, (ctx, size) => {
      const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
      g.addColorStop(0, 'rgba(255,140,40,1)'); g.addColorStop(0.35, 'rgba(255,90,0,0.35)'); g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g; ctx.fillRect(0, 0, size, size);
    });
    this.halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTexture, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0.0 }));
    this.halo.position.set(0, 0.05, 1.3);
    this.halo.scale.set(2.8, 2.8, 1);
    this.scene.add(this.halo);

    const lineMaterial = new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false, opacity: 0 });
    lineMaterial.color.copy(ORANGE).multiplyScalar(6);
    this.scanLine = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.0035).rotateY(Math.PI), lineMaterial);
    this.scanLine.position.z = FRONT_Z - 0.025;
    this.scene.add(this.scanLine);
  }

  /** Swap the procedural board for a real enclosure model (see README). */
  async useStl(url) {
    const geometry = await new STLLoader().loadAsync(url);
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const size = new THREE.Vector3(); box.getSize(size);
    const center = new THREE.Vector3(); box.getCenter(center);
    geometry.translate(-center.x, -center.y, -center.z);
    const scale = (2 * 0.297) / Math.max(size.x, size.y);
    geometry.scale(scale, scale, scale);
    geometry.rotateY(Math.PI);
    geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0x15171b, roughness: 0.4, metalness: 0.8 }));
    mesh.position.z = 0.012 + (size.z * scale) / 2 - 0.018;
    this.arrayGroup.add(mesh, glowLines(geometry, ORANGE, 1.2, 0.6));
    this.frame.visible = false; this.frameEdges.visible = false; this.boardCore.visible = false;
    this.stl = mesh;
  }

  // ── Per-frame state ──────────────────────────────────────────────────────
  setXray(on) {
    if (this.xray === on) return;
    this.xray = on;
    const ghost = [this.boardFront.material, this.boardCore.material, this.frame.material, this.housing.material,
      this.finMaterial, this.hub.material, this.metalParts[0]];
    for (const material of ghost) {
      material.transparent = on;
      material.opacity = on ? 0.13 : 1;
      material.depthWrite = !on;
      if (material === this.boardFront.material) material.alphaTest = on ? 0 : 0.5;
      material.needsUpdate = true;
    }
  }

  cameraFor(t, shot) {
    const orbit = (target, r, phiDeg, elDeg) => {
      const phi = (phiDeg * Math.PI) / 180; const el = (elDeg * Math.PI) / 180;
      return target.clone().add(new THREE.Vector3(Math.sin(phi) * Math.cos(el), Math.sin(el), -Math.cos(phi) * Math.cos(el)).multiplyScalar(r));
    };
    let position; let target; let fov = 30;
    if (shot === 'reveal') {
      const k = easeInOut(range01(t, T.reveal, T.xray));
      target = new THREE.Vector3(0, lerp(-0.02, 0.0, k), 0);
      position = orbit(target, lerp(0.7, 1.28, k), lerp(38, -16, k), lerp(-7, 9, k));
    } else if (shot === 'xray') {
      const k = easeInOut(range01(t, T.xray, T.dome));
      target = new THREE.Vector3(0, 0.006, 0.03);
      position = orbit(target, lerp(0.43, 0.34, k), lerp(152, 130, k), lerp(17, 9, k));
      fov = 32;
    } else if (shot === 'dome') {
      // Over-the-shoulder: the array in the lower-right foreground, the dome beyond.
      const k = smooth(range01(t, T.dome, T.console));
      position = new THREE.Vector3(lerp(0.8, 0.66, k), lerp(0.34, 0.27, k), lerp(0.88, 0.74, k));
      target = new THREE.Vector3(-0.2, 0.04, -1.3);
      fov = 50;
    } else {
      const k = easeOut(range01(t, T.end, T.total));
      target = new THREE.Vector3(0, 0.0, 0);
      position = orbit(target, lerp(2.25, 2.05, k), lerp(17, 8, k), lerp(3, 6, k));
    }
    position.x += 0.003 * Math.sin(t * 1.7); position.y += 0.002 * Math.sin(t * 2.3 + 1);
    this.camera.position.copy(position);
    this.camera.fov = fov;
    this.camera.aspect = W / H;
    // End card: lens-shift the array to the right third, clear of the wordmark.
    if (shot === 'end') this.camera.setViewOffset(W, H, -520, 0, W, H);
    else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
    this.camera.lookAt(target);
    this.camera.updateMatrixWorld();
  }

  update(t) {
    const shot = shotAt(t).id;
    const story = this.story;
    this.setXray(shot === 'xray');
    this.cameraFor(t, shot);
    this.dome.visible = shot === 'dome';
    this.halo.visible = shot === 'reveal' || shot === 'end';
    this.halo.material.opacity = shot === 'reveal' ? 0.55 * smooth(range01(t, 2.3, 3.6)) : 0.6;

    // Lighting: the array is revealed by the sweeping scan line, then the key light.
    const revealK = shot === 'reveal' ? smooth(range01(t, 2.35, 3.4)) : 1;
    this.key.intensity = 1.4 * revealK;
    this.rim.intensity = shot === 'reveal' ? 0.6 + 2.6 * smooth(range01(t, 2.0, 2.7)) : 3.0;
    const sweepK = range01(t, 2.02, T.sweepEnd);
    const sweeping = shot === 'reveal' && t < T.sweepEnd + 0.05;
    const sweepY = lerp(0.34, -0.34, sweepK);
    this.scanLine.visible = sweeping;
    this.scanLine.position.y = sweepY;
    this.scanLine.material.opacity = sweeping ? Math.sin(Math.PI * clamp(sweepK * 1.05)) : 0;
    this.sweepLight.position.set(0, sweepY, -0.12);
    this.sweepLight.intensity = sweeping ? 3.5 * Math.sin(Math.PI * sweepK) : 0;

    // Microphone boot flashes and glow.
    const allOn = t >= T.igniteEnd;
    this.ports.forEach((port, i) => {
      const ignite = story.igniteTime(i);
      let intensity = t < ignite ? 0.015 : 1.1 + 3.2 * Math.exp(-(t - ignite) / 0.11);
      if (shot === 'end') {
        const r = this.mics[i].length();
        intensity = 1.25 + 0.55 * Math.sin(t * 5.2 - r * 26);
      } else if (shot === 'xray') intensity *= 0.55;
      if (allOn && shot === 'reveal') intensity += 1.4 * Math.exp(-(t - T.igniteEnd) / 0.18);
      port.disc.material.color.setRGB(1.0 * intensity, 0.43 * intensity, 0.05 * intensity);
      port.ring.visible = shot !== 'xray';
      port.ring.material.emissiveIntensity = 0.35 * intensity;
    });

    // Signal pulses racing along the traces into each port, then data flowing back.
    const colors = this.traces.geometry.attributes.color;
    for (let v = 0; v < this.traceMeta.length; v++) {
      const [mic, s] = this.traceMeta[v];
      const ignite = story.igniteTime(mic);
      let level = t < ignite ? 0.03 : 0.32;
      const p = (t - (ignite - 0.28)) / 0.28;
      if (p > -0.2 && p < 1.3) level += 3.0 * Math.exp(-(((s - p) / 0.12) ** 2));
      if (shot === 'end' || shot === 'xray' || (shot === 'reveal' && t > T.igniteEnd)) {
        const flow = (t * 0.9 + hash(mic) ) % 1;
        level += 1.6 * Math.exp(-(((s - (1 - flow)) / 0.07) ** 2));
      }
      colors.setXYZ(v, 1.0 * level, 0.42 * level, 0.05 * level);
    }
    colors.needsUpdate = true;
    const hubK = shot === 'reveal' ? smooth(range01(t, 2.5, 3.1)) + 1.6 * Math.exp(-Math.max(0, t - T.igniteEnd) / 0.25) * (allOn ? 1 : 0) : 1.2;
    this.emblem.material.color.setScalar(0.35 + 1.6 * hubK);
    this.frameEdges.material.opacity = shot === 'reveal' ? 0.85 * smooth(range01(t, 2.1, 2.9)) : shot === 'xray' ? 0.3 : 0.8;
    this.traces.material.opacity = shot === 'xray' ? 0.35 : 1;
    this.hubEdges.material.opacity = this.frameEdges.material.opacity;
    this.housingEdges.material.opacity = shot === 'xray' ? 0.95 : 0.55;

    // X-ray chip highlights.
    for (const [chip, at] of [[this.dsp, T.dspCallout], [this.mcu, T.mcuCallout]]) {
      const on = shot === 'xray' && t >= at;
      const glow = on ? 0.35 + 1.8 * Math.exp(-(t - at) / 0.25) : 0;
      chip.top.emissiveIntensity = glow * 0.5;
      chip.side.emissiveIntensity = glow;
      chip.edges.material.opacity = on ? 0.9 : shot === 'xray' ? 0.25 : 0;
    }

    this.pcbTop.emissiveIntensity = shot === 'xray' ? 0.55 + 0.25 * Math.sin(t * 9) : 0;

    // Floor.
    const floorU = this.floor.material.uniforms;
    floorU.uFade.value = shot === 'reveal' ? smooth(range01(t, 2.4, 3.8)) : 1;
    floorU.uGlow.value = shot === 'end' ? 1 : 0.5;
    const pulsePeriod = shot === 'dome' ? 0.6 : 1.6;
    const pulsePhase = ((t - (shot === 'dome' ? T.dome : 0)) / pulsePeriod) % 1;
    floorU.uPulseR.value = pulsePhase * 3.2;
    floorU.uPulseA.value = shot === 'dome' || shot === 'end' ? (1 - pulsePhase) : shot === 'reveal' && t > T.igniteEnd ? (1 - pulsePhase) * 0.6 : 0;

    if (shot === 'dome') this.updateDome(t);
  }

  updateDome(t) {
    const story = this.story;
    for (let k = 0; k < story.sectors; k++) {
      const cell = story.cell(k, t);
      const material = this.tiles[k].material;
      if (!cell) { material.color.setRGB(0.02, 0.008, 0); continue; }
      const v = clamp((cell.db + 32) / 28);
      const [r, g, b] = heat(v);
      const gain = 0.25 + 2.2 * v * v;
      material.color.setRGB((r / 255) * gain, (g / 255) * gain, (b / 255) * gain);
    }
    const cursor = story.sweepCursor(t);
    this.cursor.visible = Boolean(cursor);
    this.cone.visible = this.coneCore.visible = Boolean(cursor);
    if (!cursor) return;
    const [row, col] = story.rowCol(cursor.sector);
    const az0 = this.azEdges[col]; const az1 = this.azEdges[col + 1];
    const el0 = this.elEdges[row]; const el1 = this.elEdges[row + 1];
    const positions = this.cursor.geometry.attributes.position;
    const corners = [[az0, el0, az1, el0], [az1, el0, az1, el1], [az1, el1, az0, el1], [az0, el1, az0, el0]];
    let n = 0;
    for (const [fa, fe, ta, te] of corners) {
      for (let s = 0; s < 8; s++) {
        const p = this.domeDirection(lerp(fa, ta, s / 8), lerp(fe, te, s / 8)).multiplyScalar(DOME_R * 0.995);
        positions.setXYZ(n++, p.x, p.y, p.z);
      }
    }
    const first = this.domeDirection(az0, el0).multiplyScalar(DOME_R * 0.995);
    positions.setXYZ(n, first.x, first.y, first.z);
    positions.needsUpdate = true;
    const direction = this.domeDirection(story.sectorAz(cursor.sector), story.sectorEl(cursor.sector));
    for (const cone of [this.cone, this.coneCore]) {
      cone.position.set(0, 0, FRONT_Z - 0.01);
      cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
      cone.scale.set(1, DOME_R * 0.99, 1);
    }
  }

  render() { this.composer.render(); }

  project(vector) {
    const v = vector.clone().project(this.camera);
    return { x: ((v.x + 1) / 2) * W, y: ((1 - v.y) / 2) * H, visible: v.z < 1 };
  }

  anchors() {
    return {
      mic: new THREE.Vector3(this.mics[13].x, this.mics[13].y, FRONT_Z - 0.002),
      dsp: this.dsp.anchor.clone(),
      mcu: this.mcu.anchor.clone(),
    };
  }
}
