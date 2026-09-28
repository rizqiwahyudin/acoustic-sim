/**
 * 2D layer: title cards, shot overlays, the HEIMDALL console, alert cards and
 * film post-processing. Everything is drawn from time t only.
 */
import { T, CLASSES, TRACK_RATE_HZ, FULL_SCAN_HZ } from './story.js';
import {
  W, H, C, FONT, clamp, lerp, range01, smooth, easeOut, easeInOut, blink, hash, heat, heatCss, deg, seeded,
} from './util.js';

const DB_FLOOR = -32;
const DB_SPAN = 28;
const levelNorm = (db) => clamp((db - DB_FLOOR) / DB_SPAN);
const GLITCH_CUTS = [T.reveal, T.xray, T.dome, T.console, T.lock1, T.lock2, T.verdictCard, T.end];
const FLASHES = [[T.reveal, 0.55], [T.verdictCard, 0.8], [T.end, 0.45], [T.lock2, 0.25]];

function makeCanvas(w, h) {
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  return canvas;
}

export class Hud {
  constructor(ctx, story, spectra, stage) {
    this.ctx = ctx;
    this.story = story;
    this.stage = stage;
    this.spec = { drone: this.spectrogram(spectra.drone), voice: this.spectrogram(spectra.voice) };
    this.grain = [11, 23, 37, 51].map((seed) => this.noise(256, seed));
    this.scanlines = this.makeScanlines();
    this.vignette = this.makeVignette();
    this.hexWatermark = this.makeHexWatermark();
    this.scratch = makeCanvas(W, H);
  }

  // ── Primitives ───────────────────────────────────────────────────────────
  text(str, x, y, o = {}) {
    const ctx = this.ctx;
    ctx.save();
    ctx.font = `${o.weight ?? 500} ${o.size ?? 20}px ${o.font ?? FONT.cond}`;
    ctx.fillStyle = o.color ?? C.white;
    ctx.textAlign = o.align ?? 'left';
    ctx.textBaseline = o.baseline ?? 'alphabetic';
    ctx.letterSpacing = `${o.spacing ?? 0}px`;
    ctx.globalAlpha *= o.alpha ?? 1;
    if (o.glow) { ctx.shadowColor = o.glowColor ?? ctx.fillStyle; ctx.shadowBlur = o.glow; }
    if (o.scaleX && o.scaleX !== 1) {
      ctx.translate(x, y); ctx.scale(o.scaleX, o.scaleY ?? 1); ctx.fillText(str, 0, 0);
    } else ctx.fillText(str, x, y);
    ctx.restore();
  }

  width(str, o = {}) {
    const ctx = this.ctx;
    ctx.save();
    ctx.font = `${o.weight ?? 500} ${o.size ?? 20}px ${o.font ?? FONT.cond}`;
    ctx.letterSpacing = `${o.spacing ?? 0}px`;
    const w = ctx.measureText(str).width * (o.scaleX ?? 1);
    ctx.restore();
    return w;
  }

  jp(str, x, y, o = {}) { this.text(str, x, y, { font: FONT.jp, weight: 700, ...o }); }

  line(points, color, width = 1.5, alpha = 1, dash = null) {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.globalAlpha *= alpha;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
    ctx.restore();
  }

  /** Polyline drawn up to fraction k of its length. */
  partialLine(points, k, color, width = 2) {
    const lengths = [];
    let total = 0;
    for (let i = 1; i < points.length; i++) {
      const l = Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
      lengths.push(l); total += l;
    }
    let remaining = total * clamp(k);
    const out = [points[0]];
    for (let i = 1; i < points.length && remaining > 0; i++) {
      const f = Math.min(1, remaining / lengths[i - 1]);
      out.push([lerp(points[i - 1][0], points[i][0], f), lerp(points[i - 1][1], points[i][1], f)]);
      remaining -= lengths[i - 1];
    }
    this.line(out, color, width);
  }

  brackets(x, y, w, h, len, color, width = 2, alpha = 1) {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.globalAlpha *= alpha;
    ctx.beginPath();
    ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y);
    ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len);
    ctx.moveTo(x + w, y + h - len); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - len, y + h);
    ctx.moveTo(x + len, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - len);
    ctx.stroke();
    ctx.restore();
  }

  hexPath(cx, cy, r, rotation = 0) {
    const ctx = this.ctx;
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = rotation + (i * Math.PI) / 3;
      ctx.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a));
    }
    ctx.closePath();
  }

  hazard(x, y, w, h, offset, a = C.red, b = '#000', stripe = 44) {
    const ctx = this.ctx;
    ctx.save();
    ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
    ctx.fillStyle = a; ctx.fillRect(x, y, w, h);
    ctx.fillStyle = b;
    const period = stripe * 2;
    for (let s = -h - period + (((offset % period) + period) % period); s < w + h; s += period) {
      ctx.beginPath();
      ctx.moveTo(x + s, y + h); ctx.lineTo(x + s + stripe, y + h);
      ctx.lineTo(x + s + stripe + h, y); ctx.lineTo(x + s + h, y);
      ctx.closePath(); ctx.fill();
    }
    ctx.restore();
  }

  panel(x, y, w, h, title, jp, color = C.orange, alpha = 1) {
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha *= alpha;
    const notch = 18;
    ctx.beginPath();
    ctx.moveTo(x, y); ctx.lineTo(x + w - notch, y); ctx.lineTo(x + w, y + notch);
    ctx.lineTo(x + w, y + h); ctx.lineTo(x, y + h); ctx.closePath();
    ctx.fillStyle = 'rgba(14,9,4,0.88)'; ctx.fill();
    ctx.strokeStyle = color; ctx.globalAlpha *= 0.6; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.globalAlpha /= 0.6;
    const titleWidth = this.width(title, { size: 20, weight: 700, spacing: 3 }) + 28;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x, y); ctx.lineTo(x + titleWidth + 14, y); ctx.lineTo(x + titleWidth, y + 28); ctx.lineTo(x, y + 28);
    ctx.closePath(); ctx.fill();
    ctx.restore();
    this.text(title, x + 12, y + 21, { size: 20, weight: 700, spacing: 3, color: '#000', alpha });
    if (jp) this.jp(jp, x + titleWidth + 26, y + 21, { size: 17, color, alpha: alpha * 0.9 });
    for (let i = 0; i < 6; i++) this.line([[x + 10 + i * 9, y + h - 8], [x + 16 + i * 9, y + h - 8]], color, 3, alpha * 0.5);
  }

  vertical(str, x, y, size, color, alpha = 1) {
    [...str].forEach((ch, i) => this.jp(ch, x, y + i * size * 1.12, { size, color, align: 'center', alpha }));
  }

  typed(str, k) { return str.slice(0, Math.round(str.length * clamp(k))); }

  // ── Offscreen resources ──────────────────────────────────────────────────
  spectrogram(spec) {
    const canvas = makeCanvas(spec.frames, spec.bands);
    const ctx = canvas.getContext('2d');
    const image = ctx.createImageData(spec.frames, spec.bands);
    for (let band = 0; band < spec.bands; band++) {
      for (let x = 0; x < spec.frames; x++) {
        const v = spec.data[band * spec.frames + x] / 255;
        const [r, g, b] = heat(Math.pow(v, 1.35));
        const i = ((spec.bands - 1 - band) * spec.frames + x) * 4;
        image.data[i] = r; image.data[i + 1] = g; image.data[i + 2] = b; image.data[i + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
    return canvas;
  }

  noise(size, seed) {
    const canvas = makeCanvas(size, size);
    const ctx = canvas.getContext('2d');
    const image = ctx.createImageData(size, size);
    const rand = seeded(seed);
    for (let i = 0; i < size * size; i++) {
      const v = Math.round(rand() * 255);
      image.data[i * 4] = v; image.data[i * 4 + 1] = v; image.data[i * 4 + 2] = v; image.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    return canvas;
  }

  makeScanlines() {
    const canvas = makeCanvas(W, H);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    for (let y = 0; y < H; y += 3) ctx.fillRect(0, y, W, 1);
    return canvas;
  }

  makeVignette() {
    const canvas = makeCanvas(W, H);
    const ctx = canvas.getContext('2d');
    const g = ctx.createRadialGradient(W / 2, H / 2, H * 0.35, W / 2, H / 2, H * 1.05);
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(0,0,0,0.72)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    return canvas;
  }

  makeHexWatermark() {
    const canvas = makeCanvas(W, H);
    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = 'rgba(255,122,0,0.05)';
    ctx.lineWidth = 1;
    const r = 34;
    for (let row = -1; row < H / (r * 1.5) + 1; row++) {
      for (let col = -1; col < W / (r * 1.732) + 1; col++) {
        const cx = col * r * 1.732 + (row % 2) * r * 0.866;
        const cy = row * r * 1.5;
        ctx.beginPath();
        for (let i = 0; i < 6; i++) {
          const a = Math.PI / 6 + (i * Math.PI) / 3;
          ctx.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a));
        }
        ctx.closePath(); ctx.stroke();
      }
    }
    return canvas;
  }

  // ── Dispatch ─────────────────────────────────────────────────────────────
  draw(t, shot) {
    if (shot === 'cards') this.cards(t);
    else if (shot === 'reveal' || shot === 'xray') this.revealHud(t, shot);
    else if (shot === 'dome') this.domeHud(t);
    else if (shot === 'console') this.console(t);
    else if (shot === 'verdict') this.verdict(t);
    else if (shot === 'end') this.endCard(t);
  }

  timecode(t) {
    const frames = Math.round(t * 30);
    const ff = String(frames % 30).padStart(2, '0');
    const ss = String(Math.floor(frames / 30) % 60).padStart(2, '0');
    return `00:00:${ss}:${ff}`;
  }

  clock(t) {
    const seconds = 14 * 3600 + 32 * 60 + 7 + Math.floor(t);
    const hh = String(Math.floor(seconds / 3600)).padStart(2, '0');
    const mm = String(Math.floor(seconds / 60) % 60).padStart(2, '0');
    const ss = String(seconds % 60).padStart(2, '0');
    return `${hh}:${mm}:${ss}`;
  }

  // ── 0-2 s: title cards ───────────────────────────────────────────────────
  cards(t) {
    const ctx = this.ctx;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
    if (t < T.cardA) return;
    const card = t < T.cardB ? 'A' : t < T.cardC ? 'B' : 'C';
    const start = { A: T.cardA, B: T.cardB, C: T.cardC }[card];
    const local = t - start;
    const inverted = local < 1 / 30 + 1e-6;
    if (inverted) { ctx.fillStyle = C.white; ctx.fillRect(0, 0, W, H); }
    const fg = inverted ? '#000' : C.white;
    const zoom = 1 + 0.045 * local;
    ctx.save();
    ctx.translate(W / 2, H / 2); ctx.scale(zoom, zoom); ctx.translate(-W / 2, -H / 2);
    const serif = (str, x, y, size, o = {}) => this.text(str, x, y, { font: FONT.mincho, weight: 800, size, color: fg, ...o });
    if (card === 'A') {
      serif('音響', 120, 800, 560, { scaleX: 0.9 });
      serif('ACOUSTIC', 1210, 470, 104, { scaleX: 0.86 });
      serif('SURVEILLANCE', 1210, 590, 104, { scaleX: 0.72 });
      this.line([[1214, 640], [1800, 640]], fg, 3);
      this.text('SYSTEM 44  //  DEMONSTRATION', 1214, 690, { size: 30, weight: 600, spacing: 8, color: fg });
    } else if (card === 'B') {
      serif('EARLY', 96, 420, 330, { scaleX: 0.84 });
      serif('WARNING', 96, 790, 330, { scaleX: 0.84 });
      serif('早期警戒', 1830, 1000, 132, { align: 'right', scaleX: 0.9 });
      this.line([[100, 870], [1100, 870]], fg, 3);
    } else {
      this.text('DEMONSTRATION : 01', 128, 150, { size: 32, weight: 600, spacing: 10, color: fg });
      serif('四十四の耳', 110, 560, 258, { scaleX: 0.9 });
      serif('FORTY-FOUR EARS.', 124, 730, 96, { scaleX: 0.84 });
      serif('ONE TARGET.', 124, 850, 96, { scaleX: 0.84 });
      this.line([[1400, 150], [1820, 150]], fg, 3);
      this.text('音響早期警戒システム', 1820, 200, { font: FONT.mincho, weight: 800, size: 40, color: fg, align: 'right' });
    }
    ctx.restore();
  }

  // ── 2-6.2 s: hardware reveal overlays ────────────────────────────────────
  frameChrome(t, label, sub) {
    this.brackets(40, 40, W - 80, H - 80, 60, C.orange, 2.5, 0.85);
    this.text(label, 90, 96, { size: 26, weight: 700, spacing: 5, color: C.orange });
    this.text(sub, 90, 126, { font: FONT.mono, size: 18, color: C.grey });
    const recOn = blink(t, 1.5, 0.6);
    const ctx = this.ctx;
    if (recOn) { ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(W - 330, 88, 8, 0, Math.PI * 2); ctx.fill(); }
    this.text('REC', W - 312, 96, { font: FONT.mono, size: 22, color: C.red });
    this.text(this.timecode(t), W - 90, 96, { font: FONT.mono, size: 22, color: C.white, align: 'right' });
    // Side ruler.
    for (let i = 0; i < 24; i++) {
      const y = 200 + i * 28 - ((t * 60) % 28);
      if (y < 190 || y > 880) continue;
      const major = (i + Math.floor(t * 60 / 28)) % 4 === 0;
      this.line([[46, y], [major ? 70 : 58, y]], C.orange, 1.5, 0.6);
    }
  }

  callout(anchor, box, title, lines, jp, t0, t, color = C.orange) {
    if (t < t0 || !anchor) return;
    const drawK = easeOut(range01(t, t0, t0 + 0.22));
    const typeK = range01(t, t0 + 0.12, t0 + 0.5);
    const onLeft = box.x + box.w / 2 < anchor.x;
    const attach = [onLeft ? box.x + box.w : box.x, box.y + 17];
    const elbow = [anchor.x + (onLeft ? -1 : 1) * 60, attach[1]];
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    const pulse = range01(t, t0, t0 + 0.6);
    ctx.globalAlpha = 1 - pulse;
    ctx.beginPath(); ctx.arc(anchor.x, anchor.y, 10 + 40 * pulse, 0, Math.PI * 2); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.strokeRect(anchor.x - 7, anchor.y - 7, 14, 14);
    ctx.restore();
    this.partialLine([[anchor.x, anchor.y], elbow, attach], drawK, color, 2);
    if (drawK < 0.9) return;
    ctx.save();
    ctx.fillStyle = 'rgba(8,5,2,0.82)';
    ctx.fillRect(box.x, box.y, box.w, 44 + lines.length * 32 + 36);
    ctx.fillStyle = color; ctx.fillRect(box.x, box.y, box.w, 34);
    ctx.restore();
    this.text(this.typed(title, typeK * 1.4), box.x + 14, box.y + 26, { size: 26, weight: 700, spacing: 3, color: '#000' });
    lines.forEach((str, i) => this.text(this.typed(str, typeK * 1.2 - i * 0.15), box.x + 14, box.y + 70 + i * 32, { size: 25, weight: 500, spacing: 1, color: C.white }));
    this.jp(this.typed(jp, typeK), box.x + 14, box.y + 70 + lines.length * 32 + 2, { size: 22, color, weight: 500 });
    this.line([[box.x, box.y + 44 + lines.length * 32 + 36], [box.x + box.w, box.y + 44 + lines.length * 32 + 36]], color, 2, 0.8);
  }

  revealHud(t, shot) {
    const story = this.story;
    const stage = this.stage;
    const anchors = stage.anchors();
    if (shot === 'reveal') {
      this.frameChrome(t, 'HEIMDALL  //  44-CH ACOUSTIC ARRAY', 'CAM-01  ORBIT  ·  HARDWARE REVEAL');
      const online = story.array.ignite_order.filter((mic) => story.igniteTime(mic) <= t).length;
      const x = 90; const y = 820;
      this.text('MICROPHONE ARRAY', x, y, { size: 22, weight: 700, spacing: 4, color: C.orange });
      this.jp('マイクロホン起動', x + 250, y, { size: 20, color: C.orange, weight: 500 });
      const done = online === 44;
      this.text(`${String(online).padStart(2, '0')} / 44`, x, y + 84, { size: 88, weight: 700, color: done ? C.green : C.white, spacing: 2 });
      this.text(done ? 'ONLINE' : 'BOOTING', x + 300, y + 84, { size: 34, weight: 700, spacing: 6, color: done ? C.green : C.amber, alpha: done || blink(t, 4) ? 1 : 0.4 });
      for (let i = 0; i < 44; i++) {
        const cx = x + (i % 22) * 20; const cy = y + 108 + Math.floor(i / 22) * 20;
        this.ctx.fillStyle = i < online ? (done ? C.green : C.orange) : 'rgba(255,122,0,0.15)';
        this.ctx.fillRect(cx, cy, 14, 14);
      }
      const boot = clamp(range01(t, T.reveal, T.xray));
      this.text('SYSTEM BOOT  起動', W - 520, y, { size: 22, weight: 700, spacing: 4, color: C.orange });
      this.ctx.strokeStyle = C.orange; this.ctx.lineWidth = 2; this.ctx.strokeRect(W - 520, y + 20, 430, 22);
      this.ctx.fillStyle = C.orange; this.ctx.fillRect(W - 516, y + 24, 422 * boot, 14);
      this.text(`${Math.round(boot * 100)}%`, W - 90, y + 84, { size: 60, weight: 700, color: C.white, align: 'right' });
      const mic = stage.project(anchors.mic);
      const box = mic.x < W / 2 ? { x: 110, y: 250, w: 560 } : { x: W - 670, y: 250, w: 560 };
      this.callout(mic, box, '44 × MEMS MICROPHONES', ['HEXAGONAL APERTURE · 471 mm', '48 kHz SYNCHRONOUS SAMPLING'], 'マイクロホン44基', T.micCallout, t);
    } else {
      this.frameChrome(t, 'HEIMDALL  //  PROCESSING CORE', 'CAM-02  X-RAY  ·  REAR ELECTRONICS POD');
      const dsp = stage.project(anchors.dsp);
      const mcu = stage.project(anchors.mcu);
      const dspBox = dsp.x > W / 2 ? { x: W - 700, y: 190, w: 590 } : { x: 110, y: 190, w: 590 };
      const mcuBox = mcu.x > W / 2 ? { x: W - 700, y: 640, w: 590 } : { x: 110, y: 640, w: 590 };
      this.callout(dsp, dspBox, 'ADAU1467 SIGMADSP', ['REAL-TIME DELAY-AND-SUM BEAMFORMER', '36 STEERED SECTORS · 1–4 kHz DETECTOR'], 'ビームフォーミング処理', T.dspCallout, t);
      this.callout(mcu, mcuBox, 'MAX78002 AI MICROCONTROLLER', ['CNN ACOUSTIC CLASSIFIER · TRACK CONTROL', 'LOG-MEL 64 × 298 · INT8 INFERENCE'], '人工知能による音響識別', T.mcuCallout, t);
    }
  }

  // ── 6.2-7.4 s: 3D sector dome ────────────────────────────────────────────
  domeHud(t) {
    const story = this.story;
    this.frameChrome(t, 'SECTOR SCAN  //  扇区走査', 'CAM-03  OPERATOR VIEW  ·  6 × 6 GRID  ·  AZ ±40°  EL ±40°');
    const cursor = story.sweepCursor(t);
    if (cursor) {
      const k = cursor.sector;
      this.text(`BEAM ${String(k + 1).padStart(2, '0')} / 36`, W - 90, 200, { size: 54, weight: 700, color: C.white, align: 'right' });
      this.text(`AZ ${deg(story.sectorAz(k))}   EL ${deg(story.sectorEl(k))}`, W - 90, 240, { font: FONT.mono, size: 24, color: C.amber, align: 'right' });
      const pass = (cursor.index % 36) / 36;
      this.ctx.strokeStyle = C.orange; this.ctx.lineWidth = 2; this.ctx.strokeRect(W - 450, 262, 360, 14);
      this.ctx.fillStyle = C.orange; this.ctx.fillRect(W - 447, 265, 354 * pass, 8);
    }
    this.text('探索中', W - 90, 930, { font: FONT.mincho, weight: 800, size: 110, align: 'right', color: C.white, alpha: blink(t, 2) ? 1 : 0.55 });
    this.text('SEARCHING  ·  FULL SECTOR SWEEP', W - 90, 985, { size: 30, weight: 700, spacing: 10, align: 'right', color: C.orange });
  }

  // ── 7.4-15.6 s: HEIMDALL console ─────────────────────────────────────────
  console(t) {
    const ctx = this.ctx;
    ctx.fillStyle = C.ink; ctx.fillRect(0, 0, W, H);
    ctx.drawImage(this.hexWatermark, 0, 0);
    const story = this.story;
    const mode = story.mode(t);
    const target = story.targetAt(t);
    this.header(t, mode);
    this.heatGrid(t, mode, target);
    this.solutionPanel(t, mode, target);
    this.classifierPanel(t, target);
    this.steeringPanel(t, target);
    this.logPanel(t);
    this.traceStrip(t);
    this.vertical('音響早期警戒システム稼働中', 1890, 130, 26, C.orange, 0.55);
    this.lockOverlay(t);
    if (t >= T.verdict1 && t < T.rescan) this.negativeStamp(t);
    if (t >= T.rescan && t < T.lock2) {
      this.text('再探索  RESCAN', 760, 168, { font: FONT.mincho, weight: 800, size: 30, align: 'right', color: C.white, alpha: blink(t, 5) ? 1 : 0.4 });
    }
  }

  header(t, mode) {
    const ctx = this.ctx;
    ctx.fillStyle = C.orange;
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(380, 0); ctx.lineTo(346, 64); ctx.lineTo(0, 64); ctx.closePath(); ctx.fill();
    this.text('HEIMDALL', 40, 46, { size: 42, weight: 700, spacing: 10, color: '#000' });
    this.text('ACOUSTIC SURVEILLANCE CONSOLE', 390, 30, { size: 22, weight: 600, spacing: 4, color: C.white });
    this.jp('音響監視システム', 390, 55, { size: 17, color: C.orange, weight: 500 });
    const chips = [];
    chips.push({ label: 'LINK', value: 'ONLINE', color: C.green, dot: true });
    chips.push({ label: 'ARRAY', value: '44 CH · 6×6', color: C.white });
    const modeInfo = {
      search: { value: 'SEARCH 探索中', color: C.amber },
      track1: { value: 'TRACK 追尾中', color: C.amber },
      release: { value: 'RELEASE 追尾解除', color: C.green },
      track2: { value: 'TRACK 追尾中', color: C.red },
    }[mode] || { value: 'STANDBY', color: C.grey };
    chips.push({ label: 'MODE', value: modeInfo.value, color: modeInfo.color, blink: mode === 'search' || mode === 'track2' });
    const cursor = this.story.sweepCursor(t);
    const passHz = cursor ? cursor.rate / 36 : FULL_SCAN_HZ;
    chips.push({ label: 'FULL SCAN', value: `${passHz.toFixed(1)} Hz`, color: C.white });
    let x = 820;
    for (const chip of chips) {
      this.text(chip.label, x, 26, { size: 16, weight: 600, spacing: 3, color: C.grey });
      const on = !chip.blink || blink(t, 2.2, 0.7);
      if (chip.dot) { ctx.fillStyle = chip.color; ctx.beginPath(); ctx.arc(x + 6, 48, 6, 0, Math.PI * 2); ctx.fill(); }
      const valueX = x + (chip.dot ? 18 : 0);
      if (chip.value.match(/[^\x00-\x7f]/)) {
        const [en, ...rest] = chip.value.split(' ');
        this.text(en, valueX, 54, { size: 28, weight: 700, spacing: 2, color: chip.color, alpha: on ? 1 : 0.45 });
        this.jp(rest.join(' '), valueX + this.width(en, { size: 28, weight: 700, spacing: 2 }) + 10, 54, { size: 22, color: chip.color, alpha: on ? 1 : 0.45 });
      } else this.text(chip.value, valueX, 54, { size: 28, weight: 700, spacing: 2, color: chip.color, alpha: on ? 1 : 0.45 });
      x += 230;
    }
    this.text(`UTC ${this.clock(t)}`, W - 40, 50, { font: FONT.mono, size: 26, color: C.white, align: 'right' });
    this.line([[0, 68], [W, 68]], C.orange, 2, 0.8);
    for (let i = 0; i < 96; i++) this.line([[i * 20, 68], [i * 20, i % 5 ? 74 : 80]], C.orange, 1, 0.5);
  }

  heatGrid(t, mode, target) {
    const story = this.story;
    const ctx = this.ctx;
    const gx = 130; const gy = 190; const gs = 690; const cell = gs / 6;
    this.text('SECTOR POWER MAP', gx, gy - 22, { size: 24, weight: 700, spacing: 4, color: C.orange });
    this.jp('扇区音響強度', gx + 250, gy - 22, { size: 18, color: C.orange, weight: 500 });
    this.text('dBFS', gx + gs, gy - 22, { font: FONT.mono, size: 18, color: C.grey, align: 'right' });
    const centerOf = (k) => { const [row, col] = story.rowCol(k); return [gx + (col + 0.5) * cell, gy + (5 - row + 0.5) * cell]; };
    for (let k = 0; k < story.sectors; k++) {
      const [row, col] = story.rowCol(k);
      const x = gx + col * cell; const y = gy + (5 - row) * cell;
      const c = story.cell(k, t);
      if (!c) {
        ctx.fillStyle = '#0c0805'; ctx.fillRect(x + 2, y + 2, cell - 4, cell - 4);
        this.text('NO DATA', x + cell / 2, y + cell / 2 + 6, { font: FONT.mono, size: 14, color: '#3a2a1a', align: 'center' });
        continue;
      }
      const v = levelNorm(c.db);
      ctx.fillStyle = heatCss(v); ctx.fillRect(x + 2, y + 2, cell - 4, cell - 4);
      const stale = c.age > 0.7;
      if (stale) {
        ctx.save();
        ctx.beginPath(); ctx.rect(x + 2, y + 2, cell - 4, cell - 4); ctx.clip();
        ctx.strokeStyle = 'rgba(0,0,0,0.45)'; ctx.lineWidth = 3;
        for (let o = -cell; o < cell * 2; o += 14) { ctx.beginPath(); ctx.moveTo(x + o, y + cell); ctx.lineTo(x + o + cell, y); ctx.stroke(); }
        ctx.restore();
      }
      this.text(c.db.toFixed(1), x + cell / 2, y + cell / 2 + 10, { font: FONT.mono, size: 27, align: 'center', color: v > 0.72 ? '#140800' : C.white, alpha: stale ? 0.6 : 1 });
      if (story.isIgnored(k, t)) {
        ctx.save(); ctx.strokeStyle = C.green; ctx.lineWidth = 3; ctx.setLineDash([8, 6]);
        ctx.strokeRect(x + 6, y + 6, cell - 12, cell - 12); ctx.restore();
        this.text('VOICE', x + cell / 2, y + 26, { size: 18, weight: 700, spacing: 3, color: C.green, align: 'center' });
        this.text('IGNORED', x + cell / 2, y + cell - 12, { size: 16, weight: 700, spacing: 3, color: C.green, align: 'center' });
      }
      ctx.strokeStyle = 'rgba(255,122,0,0.25)'; ctx.lineWidth = 1; ctx.strokeRect(x + 2, y + 2, cell - 4, cell - 4);
    }
    // Sweep cursor with a motion trail at high pass rates.
    const cursor = story.sweepCursor(t);
    if (cursor) {
      const trail = Math.min(14, Math.max(3, Math.round(cursor.rate / 30)));
      for (let i = trail; i >= 0; i--) {
        const k = ((Math.floor(cursor.index) - i) % 36 + 36) % 36;
        const [cx, cy] = centerOf(k);
        ctx.strokeStyle = i === 0 ? '#ffffff' : `rgba(255,220,160,${0.7 * (1 - i / (trail + 1))})`;
        ctx.lineWidth = i === 0 ? 4 : 2;
        ctx.strokeRect(cx - cell / 2 + 3, cy - cell / 2 + 3, cell - 6, cell - 6);
      }
    }
    // Lock brackets converge on the tracked sector.
    if (target !== null && (mode === 'track1' || mode === 'track2')) {
      const [cx, cy] = centerOf(target);
      const lockStart = mode === 'track1' ? T.lock1 : T.lock2;
      const k = easeOut(range01(t, lockStart, lockStart + 0.3));
      const half = lerp(cell * 1.6, cell * 0.56, k);
      const color = mode === 'track1' ? C.amber : C.red;
      this.brackets(cx - half, cy - half, half * 2, half * 2, 22, color, 5);
      this.line([[gx, cy], [cx - half - 6, cy]], color, 1.5, 0.6, [6, 6]);
      this.line([[cx + half + 6, cy], [gx + gs, cy]], color, 1.5, 0.6, [6, 6]);
      this.line([[cx, gy], [cx, cy - half - 6]], color, 1.5, 0.6, [6, 6]);
      this.line([[cx, cy + half + 6], [cx, gy + gs]], color, 1.5, 0.6, [6, 6]);
      this.text(mode === 'track1' ? 'TGT 01' : 'TGT 02', cx + half + 8, cy - half + 18, { size: 20, weight: 700, spacing: 2, color });
    }
    // Axes.
    story.array.azimuth_deg.forEach((az, col) => this.text(deg(az), gx + (col + 0.5) * cell, gy + gs + 32, { font: FONT.mono, size: 20, color: C.white, align: 'center' }));
    story.array.elevation_deg.forEach((el, row) => this.text(deg(el), gx - 12, gy + (5 - row + 0.5) * cell + 7, { font: FONT.mono, size: 20, color: C.white, align: 'right' }));
    this.text('AZIMUTH  方位', gx + gs / 2, gy + gs + 64, { size: 20, weight: 700, spacing: 6, color: C.orange, align: 'center' });
    ctx.save(); ctx.translate(30, gy + gs / 2); ctx.rotate(-Math.PI / 2);
    this.text('ELEVATION  仰角', 0, 0, { size: 20, weight: 700, spacing: 6, color: C.orange, align: 'center' });
    ctx.restore();
    const bx = gx + gs + 22;
    const gradient = ctx.createLinearGradient(0, gy + gs, 0, gy);
    for (let i = 0; i <= 10; i++) gradient.addColorStop(i / 10, heatCss(i / 10));
    ctx.fillStyle = gradient; ctx.fillRect(bx, gy, 16, gs);
    ctx.strokeStyle = C.orangeDim; ctx.strokeRect(bx, gy, 16, gs);
    for (const [db, frac] of [[DB_FLOOR + DB_SPAN, 0], [DB_FLOOR + DB_SPAN / 2, 0.5], [DB_FLOOR, 1]]) {
      this.text(`${db}`, bx + 24, gy + frac * gs + 6, { font: FONT.mono, size: 16, color: C.grey });
    }
  }

  solutionPanel(t, mode, target) {
    const story = this.story;
    const x = 930; const y = 100; const w = 910; const h = 262;
    const color = mode === 'track2' ? C.red : mode === 'track1' ? C.amber : mode === 'release' ? C.green : C.orange;
    this.panel(x, y, w, h, 'TARGET SOLUTION', '目標諸元', color);
    let state = 'NO TARGET'; let stateJp = '目標なし'; let stateColor = C.grey;
    if (mode === 'track1') {
      const classified = t >= T.verdict1;
      state = classified ? 'TGT 01 · HUMAN VOICE — NON-THREAT' : 'TGT 01 · TRACKING'; stateJp = classified ? '脅威なし' : '追尾中';
      stateColor = classified ? C.green : C.amber;
    } else if (mode === 'release') { state = 'TGT 01 RELEASED'; stateJp = '追尾解除'; stateColor = C.green; }
    else if (mode === 'track2') { state = 'TGT 02 · TRACKING'; stateJp = '追尾中'; stateColor = C.red; }
    this.text(state, x + 24, y + 70, { size: 30, weight: 700, spacing: 3, color: stateColor });
    this.jp(stateJp, x + 30 + this.width(state, { size: 30, weight: 700, spacing: 3 }) + 10, y + 70, { size: 22, color: stateColor, weight: 500 });
    const tracking = target !== null && (mode === 'track1' || mode === 'track2');
    const angle = tracking ? { size: 86, weight: 700, color: C.white } : { font: FONT.mono, size: 64, color: C.grey };
    this.text('AZ', x + 24, y + 112, { size: 20, weight: 700, spacing: 3, color: C.grey });
    this.text(tracking ? deg(story.sectorAz(target)) : '--.-', x + 24, y + 186, angle);
    this.text('EL', x + 300, y + 112, { size: 20, weight: 700, spacing: 3, color: C.grey });
    this.text(tracking ? deg(story.sectorEl(target)) : '--.-', x + 300, y + 186, angle);
    if (tracking) {
      const [row, col] = story.rowCol(target);
      const level = story.dbAt(target, t);
      this.text(`SECTOR ${target}  ·  ROW ${row}  COL ${col}  ·  ${level.toFixed(1)} dBFS`, x + 24, y + 230, { font: FONT.mono, size: 20, color: C.white });
    } else {
      this.text('AWAITING DETECTION ABOVE BACKGROUND × 1.5', x + 24, y + 230, { font: FONT.mono, size: 20, color: C.grey });
    }
    // Refresh-rate comparison.
    const bx = x + 590; const by = y + 90;
    this.text('TRACK UPDATE', bx, by, { size: 18, weight: 700, spacing: 3, color: C.grey });
    this.text(tracking ? `${TRACK_RATE_HZ.toFixed(1)} Hz` : '--', bx, by + 48, { size: 46, weight: 700, color: tracking ? color : C.grey });
    this.text('FULL SCAN', bx + 170, by, { size: 18, weight: 700, spacing: 3, color: C.grey });
    const cursor = story.sweepCursor(t);
    const passHz = cursor ? cursor.rate / 36 : FULL_SCAN_HZ;
    this.text(`${passHz.toFixed(1)} Hz`, bx + 170, by + 48, { size: 46, weight: 700, color: C.white });
    if (tracking) {
      this.text(`${(TRACK_RATE_HZ / FULL_SCAN_HZ).toFixed(1)}× FASTER SOLUTION REFRESH`, bx, by + 96, { size: 22, weight: 700, spacing: 2, color });
      this.text('LOCAL 5-SECTOR CROSS TRACK', bx, by + 124, { font: FONT.mono, size: 17, color: C.grey });
    } else {
      this.text('GLOBAL SEARCH IN PROGRESS', bx, by + 96, { size: 22, weight: 700, spacing: 2, color: C.amber, alpha: blink(t, 2) ? 1 : 0.5 });
    }
  }

  classifierPanel(t, target) {
    const ctx = this.ctx;
    const x = 930; const y = 382; const w = 910; const h = 318;
    const cls = this.story.classifier(t);
    const accent = cls.verdict === 'positive' ? C.red : cls.verdict === 'negative' ? C.green : C.orange;
    this.panel(x, y, w, h, 'ACOUSTIC CLASSIFIER', '識別', accent);
    this.text('MAX78002 CNN  ·  64-BAND LOG-MEL  ·  BEAMFORMED AUDIO', x + w - 24, y + 22, { font: FONT.mono, size: 16, color: C.grey, align: 'right' });
    const sx = x + 24; const sy = y + 48; const sw = 520; const sh = 208;
    ctx.fillStyle = '#080503'; ctx.fillRect(sx, sy, sw, sh);
    const clip = cls.clip || null;
    if (clip) {
      const image = this.spec[clip];
      const progress = cls.progress;
      const cols = Math.max(1, Math.round(image.width * progress));
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(image, 0, 0, cols, image.height, sx, sy, (sw * cols) / image.width, sh);
      if (progress < 1) {
        const hx = sx + (sw * cols) / image.width;
        this.line([[hx, sy], [hx, sy + sh]], '#ffffff', 2);
      }
    } else {
      for (let i = 0; i < 40; i++) {
        const yy = sy + hash(i, Math.floor(t * 12)) * sh;
        this.line([[sx, yy], [sx + sw, yy]], C.orange, 1, 0.08);
      }
      this.text('STANDBY', sx + sw / 2, sy + sh / 2 + 6, { size: 36, weight: 700, spacing: 10, align: 'center', color: C.grey });
      this.jp('待機', sx + sw / 2, sy + sh / 2 + 44, { size: 22, align: 'center', color: C.grey, weight: 500 });
    }
    ctx.strokeStyle = C.orangeDim; ctx.lineWidth = 1; ctx.strokeRect(sx, sy, sw, sh);
    this.text('0', sx + 4, sy + sh + 18, { font: FONT.mono, size: 14, color: C.grey });
    this.text('3.0 s', sx + sw, sy + sh + 18, { font: FONT.mono, size: 14, color: C.grey, align: 'right' });
    this.text('8 kHz', sx + sw - 6, sy + 18, { font: FONT.mono, size: 14, color: C.white, align: 'right', alpha: 0.7 });
    // Class probabilities.
    const bx = x + 580; const bw = 250;
    const probs = cls.probs || CLASSES.map(() => 0);
    const winner = cls.state === 'result' ? probs.indexOf(Math.max(...probs)) : -1;
    CLASSES.forEach((cl, i) => {
      const yy = sy + 16 + i * 42;
      const p = probs[i];
      const win = i === winner;
      const barColor = win ? accent : 'rgba(255,122,0,0.75)';
      this.text(cl.en, bx, yy + 4, { size: 20, weight: 700, spacing: 2, color: win ? accent : C.white });
      this.jp(cl.jp, bx + bw, yy + 4, { size: 15, color: C.grey, align: 'right', weight: 500 });
      ctx.fillStyle = 'rgba(255,122,0,0.12)'; ctx.fillRect(bx, yy + 12, bw, 12);
      ctx.fillStyle = barColor; ctx.fillRect(bx, yy + 12, bw * p, 12);
      this.text(`${(p * 100).toFixed(1)}%`, bx + bw + 8, yy + 24, { font: FONT.mono, size: 16, color: win ? accent : C.grey });
    });
    let status; let statusColor = C.grey;
    if (cls.state === 'analysing') {
      status = `ANALYZING 解析中 · SECTOR ${target} · ${Math.round(cls.progress * 100)}%`; statusColor = C.amber;
    } else if (cls.state === 'result' && cls.verdict === 'negative') {
      status = 'RESULT: HUMAN VOICE (93.4%) — NON-THREAT'; statusColor = C.green;
    } else if (cls.state === 'result') {
      status = 'RESULT: DRONE (97.8%) — THREAT'; statusColor = C.red;
    } else status = 'AWAITING TRACKED TARGET';
    const [en, jpPart] = status.includes('解析中') ? status.split('解析中') : [status, null];
    this.text(en, sx, y + h - 22, { size: 24, weight: 700, spacing: 2, color: statusColor, alpha: cls.state === 'analysing' && !blink(t, 3) ? 0.55 : 1 });
    if (jpPart !== null) {
      const ex = sx + this.width(en, { size: 24, weight: 700, spacing: 2 });
      this.jp('解析中', ex, y + h - 22, { size: 20, color: statusColor });
      this.text(jpPart, ex + 66, y + h - 22, { size: 24, weight: 700, spacing: 2, color: statusColor });
    }
  }

  steeringPanel(t, target) {
    const story = this.story;
    const ctx = this.ctx;
    const x = 930; const y = 720; const w = 360; const h = 208;
    this.panel(x, y, w, h, 'BEAM STEERING', 'ビーム', C.orange);
    const cursor = story.sweepCursor(t);
    const sector = target ?? (cursor ? cursor.sector : 0);
    const delays = story.array.delay_samples[sector];
    const maxDelay = Math.max(...delays);
    const cx = x + 118; const cy = y + 122; const s = 0.33;
    ctx.strokeStyle = C.orangeDim; ctx.lineWidth = 1.5;
    this.hexPath(cx, cy, 0.275 * 1000 * s, 0); ctx.stroke();
    story.array.microphones_mm.forEach(([mx, my], m) => {
      const v = maxDelay > 0 ? delays[m] / maxDelay : 0;
      ctx.fillStyle = heatCss(0.25 + 0.75 * v);
      ctx.beginPath(); ctx.arc(cx - mx * s, cy - my * s, 5, 0, Math.PI * 2); ctx.fill();
    });
    this.text(`SECTOR ${String(sector).padStart(2, '0')}`, x + 222, y + 74, { size: 26, weight: 700, color: C.white });
    this.text('DELAY', x + 222, y + 110, { size: 16, weight: 700, spacing: 3, color: C.grey });
    this.text(`0–${maxDelay.toFixed(1)}`, x + 222, y + 140, { font: FONT.mono, size: 22, color: C.amber });
    this.text('SAMPLES', x + 222, y + 162, { font: FONT.mono, size: 14, color: C.grey });
    this.text('@ 48 kHz', x + 222, y + 180, { font: FONT.mono, size: 14, color: C.grey });
  }

  logPanel(t) {
    const x = 1310; const y = 720; const w = 530; const h = 208;
    this.panel(x, y, w, h, 'PROTOCOL', '通信記録', C.orange);
    const lines = this.story.logLines(t, 8);
    const tone = { lock: C.amber, alert: C.red, ok: C.green, cls: C.white, mode: C.orange };
    lines.forEach((entry, i) => {
      const alpha = 0.35 + 0.65 * ((i + 1) / lines.length);
      let str = entry.text;
      if (str.length > 46) str = `${str.slice(0, 45)}…`;
      this.text(str, x + 18, y + 54 + i * 19, { font: FONT.mono, size: 16, color: tone[entry.tone] || C.green, alpha });
    });
  }

  traceStrip(t) {
    const story = this.story;
    const x = 130; const y = 950; const w = 1710; const h = 100;
    const ctx = this.ctx;
    ctx.fillStyle = 'rgba(14,9,4,0.9)'; ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = C.orangeDim; ctx.lineWidth = 1; ctx.strokeRect(x, y, w, h);
    const window = 8.4; const t0 = t - window;
    const toX = (tau) => x + ((tau - t0) / window) * w;
    for (let g = Math.ceil(t0); g <= t; g++) this.line([[toX(g), y], [toX(g), y + h]], C.orange, 1, 0.12);
    const level = []; const az = []; const el = [];
    for (let tau = Math.max(T.dome, t0); tau <= t; tau += 1 / 30) {
      const target = story.targetAt(tau);
      let k = target;
      let db;
      if (k === null) {
        let best = -Infinity;
        for (let s = 0; s < story.sectors; s++) {
          if (story.isIgnored(s, tau)) continue;
          const c = story.cell(s, tau);
          if (c && c.db > best) { best = c.db; k = s; }
        }
        db = best;
      } else db = story.dbAt(k, tau);
      if (k === null || !Number.isFinite(db)) continue;
      const px = toX(tau);
      level.push([px, y + h - 8 - levelNorm(db) * (h - 16)]);
      az.push([px, y + h / 2 - (story.sectorAz(k) / 40) * (h / 2 - 8)]);
      el.push([px, y + h / 2 - (story.sectorEl(k) / 40) * (h / 2 - 8)]);
    }
    if (az.length > 1) this.line(az, C.orange, 2, 0.9);
    if (el.length > 1) this.line(el, C.green, 2, 0.9);
    if (level.length > 1) this.line(level, C.amber, 2.5, 1);
    for (const [lockT, label, color] of [[T.lock1, 'LOCK 01', C.amber], [T.lock2, 'LOCK 02', C.red]]) {
      if (lockT <= t && lockT >= t0) {
        this.line([[toX(lockT), y], [toX(lockT), y + h]], color, 2, 0.9, [5, 4]);
        this.text(label, toX(lockT) + 6, y + 18, { size: 16, weight: 700, spacing: 2, color });
      }
    }
    this.text('AZ', x + 10, y + 20, { size: 16, weight: 700, color: C.orange });
    this.text('EL', x + 40, y + 20, { size: 16, weight: 700, color: C.green });
    this.text('LEVEL', x + 68, y + 20, { size: 16, weight: 700, color: C.amber });
    this.text('SOLUTION TRACE · 8 s', x + w - 12, y + 20, { font: FONT.mono, size: 15, color: C.grey, align: 'right' });
  }

  lockOverlay(t) {
    for (const [start, color, sectorTrack, label] of [[T.lock1, C.amber, this.story.track1, 'TGT 01'], [T.lock2, C.red, this.story.track2, 'TGT 02']]) {
      const local = t - start;
      if (local < 0 || local > 0.8) continue;
      const scale = lerp(1.18, 1, easeOut(range01(local, 0, 0.12)));
      const alpha = local > 0.65 ? 1 - range01(local, 0.65, 0.8) : 1;
      const ctx = this.ctx;
      const cx = 475; const cy = 520; const w = 640; const h = 230;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.translate(cx, cy); ctx.scale(scale, scale); ctx.translate(-cx, -cy);
      ctx.fillStyle = 'rgba(0,0,0,0.88)'; ctx.fillRect(cx - w / 2, cy - h / 2, w, h);
      this.hazard(cx - w / 2, cy - h / 2, w, 22, local * 300, color, '#000', 22);
      this.hazard(cx - w / 2, cy + h / 2 - 22, w, 22, -local * 300, color, '#000', 22);
      ctx.strokeStyle = color; ctx.lineWidth = 4;
      if (blink(local, 2.5, 0.7)) ctx.strokeRect(cx - w / 2, cy - h / 2, w, h);
      this.text('目標捕捉', cx, cy + 36, { font: FONT.mincho, weight: 800, size: 104, align: 'center', color: C.white, scaleX: 0.92 });
      this.text(`TARGET ACQUIRED  ·  ${label}  ·  SECTOR ${sectorTrack.start}`, cx, cy + 88, { size: 28, weight: 700, spacing: 4, align: 'center', color });
      ctx.restore();
    }
  }

  negativeStamp(t) {
    const local = t - T.verdict1;
    const x = 954; const y = 440; const w = 520; const h = 150;
    const k = easeOut(range01(local, 0, 0.12));
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = k;
    ctx.translate(x + w / 2, y + h / 2); ctx.rotate(-0.05); ctx.scale(lerp(1.3, 1, k), lerp(1.3, 1, k)); ctx.translate(-(x + w / 2), -(y + h / 2));
    ctx.fillStyle = 'rgba(0,20,6,0.9)'; ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = C.green; ctx.lineWidth = 5; ctx.strokeRect(x, y, w, h);
    ctx.lineWidth = 1.5; ctx.strokeRect(x + 10, y + 10, w - 20, h - 20);
    this.text('NEGATIVE', x + 30, y + 86, { size: 78, weight: 700, spacing: 8, color: C.green });
    this.jp('脅威なし', x + w - 30, y + 80, { size: 40, color: C.green, align: 'right' });
    this.text('HUMAN VOICE  ·  NO THREAT  ·  TRACK RELEASED', x + 30, y + 124, { size: 22, weight: 700, spacing: 2, color: C.green });
    ctx.restore();
  }

  // ── 15.6-17 s: drone confirmed ───────────────────────────────────────────
  verdict(t) {
    const ctx = this.ctx;
    const local = t - T.verdictCard;
    const solution = this.story.finalDroneSolution();
    const shake = Math.exp(-local / 0.12) * 10;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
    ctx.save();
    ctx.translate((hash(Math.round(t * 30), 7) - 0.5) * shake, (hash(Math.round(t * 30), 9) - 0.5) * shake);
    for (let row = 0; row < 12; row++) {
      const yy = 190 + row * 64;
      const offset = ((row % 2 ? 1 : -1) * local * 160) % 520;
      for (let i = -1; i < 6; i++) {
        this.text('WARNING 警告', offset + i * 520, yy, { font: FONT.jp, weight: 700, size: 44, color: C.redDeep, alpha: 0.22 });
      }
    }
    this.hazard(0, 0, W, 150, local * 420, C.red, '#000', 60);
    this.hazard(0, H - 150, W, 150, -local * 420, C.red, '#000', 60);
    const inverted = (local > 0.62 && local < 0.62 + 2 / 30);
    const main = inverted ? C.red : C.white;
    const accent = inverted ? C.white : C.red;
    if (inverted) { ctx.fillStyle = C.white; ctx.fillRect(0, 150, W, H - 300); }
    const popIn = easeOut(range01(local, 0, 0.1));
    ctx.save();
    ctx.translate(W / 2, 470); ctx.scale(lerp(1.25, 1, popIn), lerp(1.25, 1, popIn)); ctx.translate(-W / 2, -470);
    this.text('無人機確認', W / 2, 520, { font: FONT.mincho, weight: 800, size: 250, align: 'center', color: main, scaleX: 0.9 });
    ctx.restore();
    this.text('DRONE CONFIRMED', W / 2, 700, { size: 150, weight: 700, spacing: 14, align: 'center', color: accent });
    const details = `CLASS: MULTIROTOR UAS   ·   CONFIDENCE ${(solution.confidence * 100).toFixed(1)}%   ·   AZ ${deg(solution.az)}  EL ${deg(solution.el)}   ·   SECTOR ${solution.sector}`;
    this.text(this.typed(details, range01(local, 0.15, 0.6)), W / 2, 790, { size: 36, weight: 600, spacing: 3, align: 'center', color: inverted ? '#000' : C.white });
    for (const cx of [210, W - 210]) {
      const on = blink(local, 2, 0.6);
      ctx.fillStyle = on ? C.red : C.redDeep;
      this.hexPath(cx, 470, 120, Math.PI / 6); ctx.fill();
      ctx.strokeStyle = C.white; ctx.lineWidth = 4; this.hexPath(cx, 470, 104, Math.PI / 6); ctx.stroke();
      this.jp('警告', cx, 498, { size: 76, align: 'center', color: C.white });
    }
    this.text(`TRACKING ${TRACK_RATE_HZ.toFixed(1)} Hz  ·  ALERT ISSUED ${this.clock(t)} UTC`, W / 2, 870, { font: FONT.mono, size: 26, align: 'center', color: inverted ? '#000' : C.amber });
    ctx.restore();
  }

  // ── 17-20 s: end card ────────────────────────────────────────────────────
  endCard(t) {
    const local = t - T.end;
    const ctx = this.ctx;
    const shade = ctx.createLinearGradient(0, 0, W * 0.62, 0);
    shade.addColorStop(0, 'rgba(0,0,0,0.85)'); shade.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = shade; ctx.fillRect(0, 0, W, H);
    this.brackets(40, 40, W - 80, H - 80, 60, C.orange, 2.5, 0.7);
    const reveal = easeOut(range01(local, 0.2, 0.9));
    ctx.save();
    ctx.beginPath(); ctx.rect(0, 0, 150 + 1000 * reveal, H); ctx.clip();
    this.text('HEIMDALL', 146, 500, { font: FONT.mincho, weight: 800, size: 140, spacing: 12, color: C.white, glow: 18, glowColor: 'rgba(255,122,0,0.55)' });
    ctx.restore();
    const rule = easeOut(range01(local, 0.5, 1.1));
    this.line([[150, 540], [150 + 900 * rule, 540]], C.orange, 3);
    this.text(this.typed('44-CHANNEL ACOUSTIC DRONE DETECTION', range01(local, 0.7, 1.3)), 150, 600, { size: 46, weight: 600, spacing: 6, color: C.white });
    this.text(this.typed('AI MCU  ·  DSP BEAMFORMING  ·  CNN CLASSIFICATION', range01(local, 1.0, 1.6)), 150, 652, { size: 32, weight: 600, spacing: 5, color: C.orange });
    this.jp('音響ドローン探知システム', 150, 712, { size: 36, color: C.white, alpha: smooth(range01(local, 1.3, 1.8)) * 0.9, weight: 700 });
    this.text('INTERNAL DEMONSTRATION  ·  SIMULATED SCENARIO  ·  社内限定', 150, 1000, { font: FONT.mono, size: 18, color: C.grey, alpha: smooth(range01(local, 1.4, 2.0)) });
  }

  // ── Film post-processing ─────────────────────────────────────────────────
  post(t) {
    const ctx = this.ctx;
    const frame = Math.round(t * 30);
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'overlay';
    ctx.globalAlpha = 0.14;
    const ox = Math.floor(hash(frame, 1) * 256); const oy = Math.floor(hash(frame, 2) * 256);
    ctx.translate(-ox, -oy);
    ctx.fillStyle = ctx.createPattern(this.grain[frame % 4], 'repeat');
    ctx.fillRect(ox, oy, W, H);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 0.28;
    ctx.drawImage(this.scanlines, 0, 0);
    ctx.globalAlpha = 1;
    ctx.drawImage(this.vignette, 0, 0);
    ctx.restore();

    let glitch = 0;
    for (const cut of GLITCH_CUTS) if (t >= cut && t < cut + 0.1) glitch = Math.max(glitch, 1 - ((t - cut) / 0.1) * 0.6);
    if (glitch > 0) this.glitch(frame, glitch);

    let flash = 0;
    for (const [at, peak] of FLASHES) if (t >= at && t < at + 0.15) flash = Math.max(flash, peak * Math.exp(-(t - at) / 0.05));
    if (flash > 0.01) { ctx.fillStyle = `rgba(255,255,255,${flash})`; ctx.fillRect(0, 0, W, H); }

    const fade = range01(t, T.fadeOut, T.total);
    if (fade > 0) { ctx.fillStyle = `rgba(0,0,0,${smooth(fade)})`; ctx.fillRect(0, 0, W, H); }
  }

  glitch(frame, strength) {
    const ctx = this.ctx;
    const scratch = this.scratch.getContext('2d');
    scratch.drawImage(ctx.canvas, 0, 0);
    for (let i = 0; i < 12; i++) {
      const y = Math.floor(hash(frame, i, 3) * H);
      const h = 6 + Math.floor(hash(frame, i, 4) * 70);
      const dx = (hash(frame, i, 5) - 0.5) * 180 * strength;
      ctx.drawImage(this.scratch, 0, y, W, h, dx, y, W, h);
    }
    const image = ctx.getImageData(0, 0, W, H);
    const src = new Uint8ClampedArray(image.data);
    const d = image.data;
    const shift = Math.max(1, Math.round(9 * strength)) * 4;
    for (let i = 0; i < d.length; i += 4) {
      d[i] = src[Math.min(d.length - 4, i + shift)];
      d[i + 2] = src[Math.max(0, i - shift) + 2];
    }
    ctx.putImageData(image, 0, 0);
  }
}
