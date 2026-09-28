/** Sparse typography, the drone reticle and film finish for the teaser. */
import { T, SOURCE, direction } from './scene.js';
import { W, H, FONT, clamp, lerp, range01, smooth, easeOut, window01, hash } from './util.js';

export class Overlay {
  constructor(ctx, teaser) {
    this.ctx = ctx;
    this.teaser = teaser;
    this.grain = [3, 5, 7, 9].map((seed) => this.noise(256, seed));
    this.vignette = this.makeVignette();
  }

  text(str, x, y, o = {}) {
    const ctx = this.ctx;
    ctx.save();
    ctx.font = `${o.weight ?? 300} ${o.size ?? 20}px ${o.font ?? FONT.mono}`;
    ctx.fillStyle = o.color ?? 'rgba(226,236,255,1)';
    ctx.textAlign = o.align ?? 'left';
    ctx.letterSpacing = `${o.spacing ?? 0}px`;
    ctx.globalAlpha = clamp(o.alpha ?? 1);
    if (o.glow) { ctx.shadowColor = o.glowColor ?? 'rgba(255,120,40,0.6)'; ctx.shadowBlur = o.glow; }
    ctx.fillText(str, x, y);
    ctx.restore();
  }

  line(x0, y0, x1, y1, color, alpha = 1, width = 1) {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = color; ctx.globalAlpha = clamp(alpha); ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.restore();
  }

  noise(size, seed) {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext('2d');
    const image = ctx.createImageData(size, size);
    for (let i = 0; i < size * size; i++) {
      const v = Math.round(hash(seed, i) * 255);
      image.data.set([v, v, v, 255], i * 4);
    }
    ctx.putImageData(image, 0, 0);
    return canvas;
  }

  makeVignette() {
    const canvas = document.createElement('canvas');
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    const g = ctx.createRadialGradient(W / 2, H / 2, H * 0.3, W / 2, H / 2, H * 1.0);
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(0,0,0,0.78)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    return canvas;
  }

  draw(t) {
    this.hud(t);
    // "every sound has a direction."
    const a = window01(t, T.textA, 1.9, 3.1, 3.7);
    if (a > 0) {
      this.text('every sound has a direction.', W / 2, H * 0.74, { size: 30, spacing: 9, align: 'center', alpha: a * 0.9 });
      this.text('すべての音には、方向がある。', W / 2, H * 0.74 + 48, { font: FONT.serif, weight: 400, size: 21, spacing: 8, align: 'center', alpha: a * 0.5 });
    }
    const caption = window01(t, 4.6, 5.2, 6.6, 7.1);
    if (caption > 0) {
      this.text('SIGNAL 01  ·  ROTOR HARMONICS  ·  1–4 kHz', 120, H - 120, { size: 15, spacing: 6, alpha: caption * 0.55 });
    }
    // "44"
    const b = window01(t, T.textB, 10.2, 11.0, 11.5);
    if (b > 0) {
      this.text('44', 150, 540, { size: 190, weight: 300, spacing: 6, alpha: b * 0.92 });
      this.text('ears. one direction.', 158, 600, { size: 24, spacing: 8, alpha: b * 0.75 });
      this.text('四十四の耳', 158, 646, { font: FONT.serif, weight: 400, size: 22, spacing: 10, alpha: b * 0.45 });
    }
    const delay = window01(t, 11.2, 11.6, 12.4, 12.8);
    if (delay > 0) {
      this.text('ARRIVAL DELAY  0 – 0.97 ms', 120, H - 120, { size: 15, spacing: 6, alpha: delay * 0.6 });
      this.text('44 CHANNELS  ·  DELAY-AND-SUM', 120, H - 96, { size: 15, spacing: 6, alpha: delay * 0.4 });
    }
    const sector = window01(t, T.lock + 0.05, T.lock + 0.35, 14.6, 15.0);
    if (sector > 0) {
      const teaser = this.teaser;
      const row = Math.floor(teaser.lockSector / 6); const col = teaser.lockSector % 6;
      const center = direction(teaser.array.azimuth_deg[col], teaser.array.elevation_deg[row]).multiplyScalar(2.5);
      const p = teaser.project(center);
      if (p.visible) {
        this.line(p.x, p.y, p.x + 90, p.y - 60, 'rgba(255,110,40,1)', sector * 0.8);
        this.line(p.x + 90, p.y - 60, p.x + 330, p.y - 60, 'rgba(255,110,40,1)', sector * 0.8);
        this.text(`SECTOR ${teaser.lockSector}  ·  AZ +${teaser.array.azimuth_deg[col].toFixed(1)}°  EL +${teaser.array.elevation_deg[row].toFixed(1)}°`, p.x + 96, p.y - 70, { size: 14, spacing: 4, alpha: sector * 0.8 });
      }
    }
    this.reticle(t);
    this.title(t);
  }

  hud(t) {
    const alpha = 0.22 * smooth(range01(t, 0.5, 2.0));
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = `rgba(210,225,255,${alpha})`;
    ctx.lineWidth = 1;
    const m = 56; const l = 26;
    for (const [x, y, dx, dy] of [[m, m, 1, 1], [W - m, m, -1, 1], [m, H - m, 1, -1], [W - m, H - m, -1, -1]]) {
      ctx.beginPath(); ctx.moveTo(x, y + dy * l); ctx.lineTo(x, y); ctx.lineTo(x + dx * l, y); ctx.stroke();
    }
    ctx.restore();
    const frames = Math.round(t * 30);
    const tc = `${String(Math.floor(frames / 30)).padStart(2, '0')}:${String(frames % 30).padStart(2, '0')}`;
    this.text(tc, W - 72, H - 72, { size: 13, spacing: 4, align: 'right', alpha: alpha * 2.2 });
    this.text('44CH  /  48 kHz', 72, 84, { size: 13, spacing: 4, alpha: alpha * 2.2 });
  }

  reticle(t) {
    const vis = window01(t, T.print, T.print + 0.4, 16.5, 17.0);
    if (vis <= 0) return;
    const teaser = this.teaser;
    const p = teaser.project(teaser.D);
    if (!p.visible) return;
    const size = lerp(150, 118, easeOut(range01(t, T.print, T.print + 0.5)));
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,112,44,1)'; ctx.globalAlpha = vis * 0.85; ctx.lineWidth = 1.5;
    const l = 20;
    for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const x = p.x + sx * size; const y = p.y + sy * size;
      ctx.beginPath(); ctx.moveTo(x - sx * l, y); ctx.lineTo(x, y); ctx.lineTo(x, y - sy * l); ctx.stroke();
    }
    ctx.restore();
    const label = window01(t, 15.3, 15.7, 16.5, 17.0);
    if (label > 0) {
      // Sit the label outside the fingerprint ring.
      const edge = teaser.project(teaser.D.clone().add(teaser.camera.up.clone().cross(teaser.camera.getWorldDirection(teaser.D.clone())).normalize().multiplyScalar(-0.5)));
      const x = Math.max(p.x + size + 40, edge.x + 30);
      this.line(p.x + size + 8, p.y - size, x + 250, p.y - size, 'rgba(255,112,44,1)', label * 0.6);
      this.text('無人機', x, p.y - size + 58, { font: FONT.serif, weight: 500, size: 52, spacing: 10, alpha: label, glow: 16 });
      this.text('DRONE  ·  97.8%', x + 2, p.y - size + 94, { size: 17, spacing: 8, alpha: label * 0.8 });
    }
  }

  title(t) {
    const k = smooth(range01(t, 17.5, 18.4));
    if (k <= 0) return;
    const x = 150;
    const ctx = this.ctx;
    ctx.save();
    ctx.beginPath(); ctx.rect(0, 0, x + 1100 * easeOut(range01(t, 17.5, 18.6)), H); ctx.clip();
    this.text('HEIMDALL', x, 520, { font: FONT.serif, weight: 500, size: 108, spacing: 38, alpha: 0.96, glow: 22, glowColor: 'rgba(160,190,255,0.35)' });
    ctx.restore();
    const rule = easeOut(range01(t, 18.0, 18.9));
    this.line(x + 2, 566, x + 2 + 760 * rule, 566, 'rgba(255,112,44,1)', 0.9, 1.5);
    const sub = smooth(range01(t, 18.4, 19.0));
    this.text('ACOUSTIC DRONE DETECTION', x + 2, 616, { size: 22, spacing: 12, alpha: sub * 0.85 });
    this.text('音響ドローン探知', x + 2, 662, { font: FONT.serif, weight: 400, size: 24, spacing: 12, alpha: sub * 0.5 });
  }

  post(t) {
    const ctx = this.ctx;
    const frame = Math.round(t * 30);
    ctx.save();
    ctx.globalCompositeOperation = 'overlay';
    ctx.globalAlpha = 0.09;
    const ox = Math.floor(hash(frame, 1) * 256); const oy = Math.floor(hash(frame, 2) * 256);
    ctx.translate(-ox, -oy);
    ctx.fillStyle = ctx.createPattern(this.grain[frame % 4], 'repeat');
    ctx.fillRect(ox, oy, W, H);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.drawImage(this.vignette, 0, 0);
    const black = Math.max(1 - smooth(range01(t, 0, 0.5)), smooth(range01(t, 19.3, 20)));
    if (black > 0) { ctx.fillStyle = `rgba(0,0,0,${black})`; ctx.fillRect(0, 0, W, H); }
    ctx.restore();
  }
}
