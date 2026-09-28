/** Epigraph, letterbox, section cards, the drone reticle, the title and film finish. */
import { T, DURATION } from './scene.js';
import { W, H, FONT, clamp, lerp, range01, smooth, easeOut, window01, hash } from './util.js';

const BAR = Math.round((H - W / 2.39) / 2);   // 2.39:1 letterbox
const ICE = 'rgba(214,236,255,1)';
const CYAN = 'rgba(128,218,255,1)';
const EMBER = 'rgba(255,112,44,1)';

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
    ctx.textAlign = o.align ?? 'left';
    ctx.letterSpacing = `${o.spacing ?? 0}px`;
    ctx.globalAlpha = clamp(o.alpha ?? 1);
    if (o.glow) { ctx.shadowColor = o.glowColor ?? 'rgba(120,210,255,0.5)'; ctx.shadowBlur = o.glow; }
    if (o.stroke) {
      ctx.strokeStyle = o.color ?? CYAN; ctx.lineWidth = o.stroke; ctx.strokeText(str, x, y);
    } else {
      ctx.fillStyle = o.color ?? ICE; ctx.fillText(str, x, y);
    }
    ctx.restore();
  }

  width(str, o) {
    const ctx = this.ctx;
    ctx.save();
    ctx.font = `${o.weight ?? 300} ${o.size ?? 20}px ${o.font ?? FONT.mono}`;
    ctx.letterSpacing = `${o.spacing ?? 0}px`;
    const w = ctx.measureText(str).width;
    ctx.restore();
    return w;
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
    const g = ctx.createRadialGradient(W / 2, H / 2, H * 0.28, W / 2, H / 2, H * 0.98);
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(0,0,0,0.8)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    return canvas;
  }

  draw(t) {
    this.card(t, 'LISTEN', T.listen, T.listenOut, W - 250, 'right');
    this.annotations(t);
    this.card(t, 'LOCK', T.capture, T.captureOut, 150, 'left');
    this.reticle(t);
  }

  epigraph(t) {
    const out = 1 - smooth(range01(t, T.epigraphOut - 0.6, T.epigraphOut));
    if (t > T.epigraphOut) return;
    const serif = { font: FONT.serif, weight: 400, size: 40, spacing: 3, align: 'center' };
    this.text('He hears the grass grow on the earth,', W / 2, 470, { ...serif, alpha: smooth(range01(t, 0.5, 1.5)) * out });
    this.text('and the wool on the sheep.', W / 2, 530, { ...serif, alpha: smooth(range01(t, 1.4, 2.4)) * out });
    this.text('— after the Prose Edda', W / 2, 618, { size: 15, spacing: 7, align: 'center', alpha: 0.55 * smooth(range01(t, 2.6, 3.2)) * out });
  }

  subtitle(t) {
    const a = window01(t, T.subtitle, T.subtitle + 0.8, T.subtitleOut - 0.8, T.subtitleOut);
    if (a <= 0) return;
    this.text('every sound has a direction.', W / 2, H - BAR + 76, { size: 22, spacing: 9, align: 'center', alpha: a * 0.85 });
  }

  /** A small section card under the top bar. */
  card(t, gloss, t0, t1, x, align) {
    const a = window01(t, t0, t0 + 0.7, t1 - 0.7, t1);
    if (a <= 0) return;
    const gx = align === 'left' ? x + 8 : x - 8;
    this.text(gloss, gx, BAR + 70, { size: 14, spacing: 10, align, alpha: a * 0.6 });
    this.line(align === 'left' ? gx : gx - 150, BAR + 84, align === 'left' ? gx + 150 : gx, BAR + 84, CYAN, a * 0.4);
  }

  annotations(t) {
    const wave = window01(t, 13.2, 13.7, 15.3, 15.8);
    if (wave > 0) {
      this.text('WAVEFRONT  ·  ARRIVAL ORDER', 150, H - BAR - 64, { size: 14, spacing: 7, alpha: wave * 0.6 });
      this.text('Δt  0 – 0.97 ms  ·  44 CHANNELS', 150, H - BAR - 40, { size: 14, spacing: 7, alpha: wave * 0.42 });
    }
    const sector = window01(t, T.lock + 0.05, T.lock + 0.4, 18.2, 18.8);
    if (sector > 0) {
      const teaser = this.teaser;
      const p = teaser.project(teaser.sectorCenter(teaser.lockSector));
      if (p.visible) {
        const row = Math.floor(teaser.lockSector / 6); const col = teaser.lockSector % 6;
        this.line(p.x, p.y, p.x + 70, p.y - 60, EMBER, sector * 0.7);
        this.line(p.x + 70, p.y - 60, p.x + 330, p.y - 60, EMBER, sector * 0.7);
        this.text(`SECTOR ${teaser.lockSector}  ·  AZ +${teaser.array.azimuth_deg[col].toFixed(1)}°  EL +${teaser.array.elevation_deg[row].toFixed(1)}°`, p.x + 76, p.y - 70, { size: 13, spacing: 4, alpha: sector * 0.8 });
      }
    }
  }

  reticle(t) {
    const vis = window01(t, T.print, T.print + 0.5, T.fadeOut - 0.2, T.fadeOut + 0.4);
    if (vis <= 0) return;
    const teaser = this.teaser;
    const p = teaser.project(teaser.D);
    if (!p.visible) return;
    // Seen from above the drone is wide and flat, so the reticle is too.
    const size = teaser.screenRadius(teaser.D, 0.27) * lerp(1.3, 1, easeOut(range01(t, T.print, T.print + 0.6)));
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = EMBER; ctx.globalAlpha = vis * 0.85; ctx.lineWidth = 1.5;
    const l = 20;
    for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const x = p.x + sx * size; const y = p.y + sy * size * 0.62;
      ctx.beginPath(); ctx.moveTo(x - sx * l, y); ctx.lineTo(x, y); ctx.lineTo(x, y - sy * l); ctx.stroke();
    }
    ctx.restore();
    const label = window01(t, T.label, T.label + 0.5, T.fadeOut - 0.2, T.fadeOut + 0.4);
    if (label > 0) {
      // Left of the drone, above its wake.
      const x = p.x - size - 50;
      const top = p.y - size * 0.62;
      this.line(x + 20, top, x - 280, top, EMBER, label * 0.6);
      const right = { align: 'right' };
      this.text('DRONE', x, top + 58, { ...right, font: FONT.serif, weight: 500, size: 52, spacing: 14, alpha: label, glow: 16, glowColor: 'rgba(255,120,50,0.55)' });
      this.text('CONFIDENCE  97.8%', x - 2, top + 94, { ...right, size: 17, spacing: 8, alpha: label * 0.8 });
      this.text('ROTOR SIGNATURE MATCH', x - 2, top + 120, { ...right, size: 13, spacing: 6, alpha: label * 0.45 });
    }
  }

  title(t) {
    if (t < T.title) return;
    const out = 1 - smooth(range01(t, DURATION - 0.7, DURATION));
    const word = 'HEIMDALL';
    const style = { font: FONT.serif, weight: 500, size: 116, spacing: 0 };
    const gap = 54;
    const widths = [...word].map((ch) => this.width(ch, style));
    const total = widths.reduce((a, b) => a + b, 0) + gap * (word.length - 1);
    let x = W / 2 - total / 2;
    [...word].forEach((ch, i) => {
      const a = smooth(range01(t, T.title + 0.3 + i * 0.16, T.title + 0.9 + i * 0.16)) * out;
      this.text(ch, x, 540, { ...style, alpha: a * 0.96, glow: 20 });
      x += widths[i] + gap;
    });
    const strand = easeOut(range01(t, T.title + 1.4, T.title + 2.4));
    this.line(W / 2 - 460 * strand, 586, W / 2 + 460 * strand, 586, CYAN, 0.55 * out);
    this.line(W / 2 - 14, 586, W / 2 + 14, 586, EMBER, strand * out, 2);
    const sub = smooth(range01(t, T.title + 2.0, T.title + 2.8)) * out;
    this.text('ACOUSTIC DRONE DETECTION', W / 2, 640, { size: 22, spacing: 14, align: 'center', alpha: sub * 0.85 });
    const tag = smooth(range01(t, T.title + 3.4, T.title + 4.2)) * out;
    this.text('every sound has a direction.', W / 2, H - BAR + 64, { size: 15, spacing: 9, align: 'center', alpha: tag * 0.5 });
  }

  post(t) {
    const ctx = this.ctx;
    const frame = Math.round(t * 30);
    ctx.save();
    ctx.globalCompositeOperation = 'overlay';
    ctx.globalAlpha = 0.1;
    const ox = Math.floor(hash(frame, 1) * 256); const oy = Math.floor(hash(frame, 2) * 256);
    ctx.translate(-ox, -oy);
    ctx.fillStyle = ctx.createPattern(this.grain[frame % 4], 'repeat');
    ctx.fillRect(ox, oy, W, H);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.drawImage(this.vignette, 0, 0);
    // Fade the 3D world in and out of black between the cards.
    const black = Math.max(1 - smooth(range01(t, T.fadeIn, T.fadeIn + 1.2)), smooth(range01(t, T.fadeOut, T.black)));
    if (black > 0) {
      ctx.fillStyle = `rgba(0,0,0,${black})`;
      ctx.fillRect(0, 0, W, H);
    }
    ctx.restore();
    // Letterbox, then the type that lives on black.
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, W, BAR); ctx.fillRect(0, H - BAR, W, BAR);
    this.epigraph(t);
    this.subtitle(t);
    this.title(t);
    const hud = 0.24 * window01(t, T.fadeIn + 0.5, T.fadeIn + 1.5, T.fadeOut, T.black);
    if (hud > 0) {
      const frames = Math.round(t * 30);
      this.text(`${String(Math.floor(frames / 30)).padStart(2, '0')}:${String(frames % 30).padStart(2, '0')}`, W - 90, H - BAR / 2 + 5, { size: 12, spacing: 5, align: 'right', alpha: hud * 1.6 });
      this.text('44CH  ·  48 kHz  ·  1–4 kHz', 90, H - BAR / 2 + 5, { size: 12, spacing: 5, alpha: hud * 1.6 });
    }
  }
}
