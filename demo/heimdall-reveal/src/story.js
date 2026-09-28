/**
 * The scripted scenario behind the video.
 *
 * Everything is a pure function of time so any frame can be rendered in any
 * order. Sector levels come from the real 44-microphone geometry: each sector's
 * power is the broadband (1-4 kHz) delay-and-sum response of the deployed 6x6
 * steering directions to two sources -- a person talking and a drone.
 */
import { clamp, hash, lerp, range01, smooth, easeInOut } from './util.js';

export const T = {
  cardA: 0.12, cardB: 0.70, cardC: 1.30,
  reveal: 2.0, sweepEnd: 2.8, igniteStart: 2.55, igniteEnd: 3.95,
  micCallout: 4.05, xray: 4.7, dspCallout: 4.85, mcuCallout: 5.35,
  dome: 6.2, console: 7.4,
  lock1: 10.0, analyze1: 10.55, analyze1End: 11.95, verdict1: 12.0, release1: 12.75,
  rescan: 12.9, lock2: 13.6, analyze2: 14.05, analyze2End: 15.45,
  verdictCard: 15.6, end: 17.0, fadeOut: 19.35, total: 20.0,
};

export const SHOTS = [
  { id: 'cards', start: 0, end: T.reveal },
  { id: 'reveal', start: T.reveal, end: T.xray },
  { id: 'xray', start: T.xray, end: T.dome },
  { id: 'dome', start: T.dome, end: T.console },
  { id: 'console', start: T.console, end: T.verdictCard },
  { id: 'verdict', start: T.verdictCard, end: T.end },
  { id: 'end', start: T.end, end: T.total + 1 },
];
export const shotAt = (t) => SHOTS.find((shot) => t >= shot.start && t < shot.end) || SHOTS.at(-1);

// Full-search passes. Early passes are slowed down so the eye can follow the
// sequential sector scan; the last ones run at the measured 9.3 Hz pass rate.
const SWEEPS = [
  { t0: T.dome, t1: T.console, rate: 30 },
  { t0: T.console, t1: 8.4, rate: 90 },
  { t0: 8.4, t1: T.lock1, rate: 335 },
  { t0: T.rescan, t1: T.lock2, rate: 335, restart: true },
];

export const TRACK_RATE_HZ = 81.6;     // measured host-side TARGET_UPDATED cadence (acoustic emulator)
export const FULL_SCAN_HZ = 9.3;       // measured TIMING full-pass rate
const DB_CAL = -9.0;
const FLOOR_POWER = Math.pow(10, (-34 - DB_CAL) / 10);
const LUT_N = 241;
const LUT_L = 1.5;

export const CLASSES = [
  { en: 'DRONE', jp: '無人機' },
  { en: 'HUMAN VOICE', jp: '人の声' },
  { en: 'VEHICLE', jp: '車両' },
  { en: 'BIRD', jp: '鳥' },
  { en: 'WIND / NOISE', jp: '風・雑音' },
];
const FINAL_VOICE = [0.031, 0.934, 0.018, 0.009, 0.008];
const FINAL_DRONE = [0.978, 0.006, 0.011, 0.002, 0.003];

export class Story {
  constructor(array) {
    this.array = array;
    this.rows = array.rows;
    this.cols = array.columns;
    this.sectors = this.rows * this.cols;
    const mm = array.microphones_mm;
    const cx = mm.reduce((sum, p) => sum + p[0], 0) / mm.length;
    const cy = mm.reduce((sum, p) => sum + p[1], 0) / mm.length;
    this.mics = mm.map(([x, y]) => [(x - cx) / 1000, (y - cy) / 1000]);
    this.sectorUv = [];
    for (let k = 0; k < this.sectors; k++) {
      this.sectorUv.push(directionUv(this.sectorAz(k), this.sectorEl(k)));
    }
    this.buildLut();
    this.buildSweeps();
    this.ignoredFrom = Infinity;
    const lock1Sector = this.strongestDisplayed(T.lock1, new Set());
    this.track1 = this.simulateTrack(T.lock1, T.release1, lock1Sector);
    this.ignored = lock1Sector;
    this.ignoredFrom = T.verdict1;
    const lock2Sector = this.strongestDisplayed(T.lock2, new Set([lock1Sector]));
    this.track2 = this.simulateTrack(T.lock2, T.total, lock2Sector);
    this.log = this.buildLog();
  }

  // ── Geometry ─────────────────────────────────────────────────────────────
  rowCol(k) { return [Math.floor(k / this.cols), k % this.cols]; }
  sectorAz(k) { return this.array.azimuth_deg[k % this.cols]; }
  sectorEl(k) { return this.array.elevation_deg[Math.floor(k / this.cols)]; }
  cross(k) {
    const [row, col] = this.rowCol(k);
    const out = [k];
    if (col > 0) out.push(k - 1);
    if (col + 1 < this.cols) out.push(k + 1);
    if (row > 0) out.push(k - this.cols);
    if (row + 1 < this.rows) out.push(k + this.cols);
    return out;
  }

  /** Broadband (1-4 kHz) array power pattern as a function of the steering error. */
  buildLut() {
    const c = this.array.speed_of_sound_m_s;
    const freqs = [];
    for (let f = 1000; f <= 4000; f += 250) freqs.push(f);
    const lut = new Float32Array(LUT_N * LUT_N);
    const M = this.mics.length;
    for (let iy = 0; iy < LUT_N; iy++) {
      const dy = -LUT_L + (2 * LUT_L * iy) / (LUT_N - 1);
      for (let ix = 0; ix < LUT_N; ix++) {
        const dx = -LUT_L + (2 * LUT_L * ix) / (LUT_N - 1);
        let acc = 0;
        for (const f of freqs) {
          const k = (2 * Math.PI * f) / c;
          let re = 0; let im = 0;
          for (let m = 0; m < M; m++) {
            const phase = k * (this.mics[m][0] * dx + this.mics[m][1] * dy);
            re += Math.cos(phase); im += Math.sin(phase);
          }
          acc += (re * re + im * im) / (M * M);
        }
        lut[iy * LUT_N + ix] = acc / freqs.length;
      }
    }
    this.lut = lut;
  }

  beam(dx, dy) {
    const fx = clamp((dx + LUT_L) / (2 * LUT_L), 0, 1) * (LUT_N - 1);
    const fy = clamp((dy + LUT_L) / (2 * LUT_L), 0, 1) * (LUT_N - 1);
    const x0 = Math.min(LUT_N - 2, Math.floor(fx));
    const y0 = Math.min(LUT_N - 2, Math.floor(fy));
    const ax = fx - x0; const ay = fy - y0;
    const l = this.lut;
    const top = l[y0 * LUT_N + x0] * (1 - ax) + l[y0 * LUT_N + x0 + 1] * ax;
    const bottom = l[(y0 + 1) * LUT_N + x0] * (1 - ax) + l[(y0 + 1) * LUT_N + x0 + 1] * ax;
    return top * (1 - ay) + bottom * ay;
  }

  // ── Scene: a talker on the right, a drone closing from upper left ─────────
  sources(t) {
    const syllables = 0.5 + 0.5 * Math.sin(t * 9.1) * Math.sin(t * 2.3 + 1.0);
    const voice = {
      id: 'voice', az: 23 + 1.0 * Math.sin(t * 0.7), el: 3 + 0.5 * Math.sin(t * 1.3),
      gainDb: -1.5 + 2.5 * syllables,
    };
    let az; let el; let gainDb;
    if (t < 13) {
      const k = range01(t, T.dome, 13);
      az = lerp(-31, -22, k);
      el = lerp(27, 21, k);
      gainDb = t < T.lock1
        ? lerp(-20, -12, range01(t, T.dome, T.lock1))
        : lerp(-12, 2.5, smooth(range01(t, T.lock1, 12.7)));
    } else {
      az = -22 + (t - 13) * 5.0;
      el = 21 - (t - 13) * 0.6;
      gainDb = 2.5;
    }
    gainDb += 0.6 * Math.sin(t * 23.0);
    return [voice, { id: 'drone', az, el, gainDb }];
  }

  dbAt(k, t) {
    const [ux, uy] = this.sectorUv[k];
    let power = FLOOR_POWER * (0.8 + 0.4 * hash(k, Math.floor(t * 40)));
    for (const source of this.sources(t)) {
      const [sx, sy] = directionUv(source.az, source.el);
      power += Math.pow(10, source.gainDb / 10) * this.beam(ux - sx, uy - sy);
    }
    return 10 * Math.log10(power) + DB_CAL;
  }

  /** Continuous (not sector-quantised) response, for the dense underlay. */
  fieldDb(az, el, t) {
    const [ux, uy] = directionUv(az, el);
    let power = FLOOR_POWER;
    for (const source of this.sources(t)) {
      const [sx, sy] = directionUv(source.az, source.el);
      power += Math.pow(10, source.gainDb / 10) * this.beam(ux - sx, uy - sy);
    }
    return 10 * Math.log10(power) + DB_CAL;
  }

  // ── Firmware-style scheduling ───────────────────────────────────────────
  buildSweeps() {
    let index = 0;
    this.sweeps = SWEEPS.map((sweep) => {
      if (sweep.restart) index = Math.ceil(index / this.sectors) * this.sectors;
      const segment = { ...sweep, i0: index, i1: index + (sweep.t1 - sweep.t0) * sweep.rate };
      index = segment.i1;
      return segment;
    });
  }

  sweepIndex(t) {
    let current = null;
    for (const segment of this.sweeps) if (t >= segment.t0) current = segment;
    if (!current) return null;
    return { segment: current, index: current.i0 + (Math.min(t, current.t1) - current.t0) * current.rate };
  }

  timeOfIndex(j) {
    for (const segment of this.sweeps) {
      if (j >= segment.i0 && j < segment.i1) return segment.t0 + (j - segment.i0) / segment.rate;
    }
    return null;
  }

  lastSweepVisit(k, t) {
    const position = this.sweepIndex(t);
    if (!position) return null;
    const n = Math.floor(position.index - 1e-9);
    let j = n - ((((n - k) % this.sectors) + this.sectors) % this.sectors);
    while (j >= 0) {
      const visit = this.timeOfIndex(j);
      if (visit !== null && visit <= t) return visit;
      j -= this.sectors;
    }
    return null;
  }

  /** Current sweep cursor, or null outside full-search passes. */
  sweepCursor(t) {
    for (const segment of this.sweeps) {
      if (t >= segment.t0 && t < segment.t1) {
        const index = segment.i0 + (t - segment.t0) * segment.rate;
        return { sector: Math.floor(index) % this.sectors, index, rate: segment.rate, segment };
      }
    }
    return null;
  }

  simulateTrack(t0, t1, start) {
    const dt = 1 / TRACK_RATE_HZ;
    const steps = [];
    const moves = [];
    let target = start; let pending = null; let confirmations = 0;
    for (let tau = t0; tau <= t1 + 1e-9; tau += dt) {
      let best = target; let bestDb = this.dbAt(target, tau);
      const targetDb = bestDb;
      for (const k of this.cross(target)) {
        const level = this.dbAt(k, tau);
        if (level > bestDb) { bestDb = level; best = k; }
      }
      if (best === target || bestDb < targetDb + 0.3) confirmations = 0;
      else if (best === pending) confirmations++;
      else { pending = best; confirmations = 1; }
      if (confirmations >= 2) {
        moves.push({ t: tau, from: target, to: best });
        target = best; confirmations = 0;
      }
      steps.push({ t: tau, target });
    }
    const spans = [];
    for (const step of steps) {
      if (!spans.length || spans.at(-1).target !== step.target) spans.push({ t0: step.t, t1: step.t, target: step.target });
      spans.at(-1).t1 = step.t;
    }
    return { t0, t1, start, steps, moves, spans };
  }

  activeTrack(t) {
    if (t >= this.track1.t0 && t < this.track1.t1) return this.track1;
    if (t >= this.track2.t0) return this.track2;
    return null;
  }

  targetAt(t) {
    const track = this.activeTrack(t);
    if (!track) return null;
    const index = clamp(Math.floor((t - track.t0) * TRACK_RATE_HZ), 0, track.steps.length - 1);
    return track.steps[index].target;
  }

  lastTrackVisit(k, t) {
    let best = null;
    for (const track of [this.track1, this.track2]) {
      if (!track || t < track.t0) continue;
      const end = Math.min(t, track.t1);
      for (let i = track.spans.length - 1; i >= 0; i--) {
        const span = track.spans[i];
        if (span.t0 > end) continue;
        if (this.cross(span.target).includes(k)) { best = Math.max(best ?? -Infinity, Math.min(end, span.t1)); break; }
      }
    }
    return best;
  }

  lastVisit(k, t) {
    const sweep = this.lastSweepVisit(k, t);
    const track = this.lastTrackVisit(k, t);
    if (sweep === null) return track;
    if (track === null) return sweep;
    return Math.max(sweep, track);
  }

  cell(k, t) {
    const visit = this.lastVisit(k, t);
    if (visit === null) return null;
    return { db: this.dbAt(k, visit), age: t - visit, visit };
  }

  strongestDisplayed(t, excluded) {
    let best = null;
    for (let k = 0; k < this.sectors; k++) {
      if (excluded.has(k)) continue;
      const cell = this.cell(k, t);
      if (cell && (!best || cell.db > best.db)) best = { k, db: cell.db };
    }
    return best ? best.k : 0;
  }

  isIgnored(k, t) { return t >= this.ignoredFrom && k === this.ignored; }

  // ── Operating mode and classifier ───────────────────────────────────────
  mode(t) {
    if (t < T.dome) return 'boot';
    if (t < T.lock1) return 'search';
    if (t < T.release1) return 'track1';
    if (t < T.rescan) return 'release';
    if (t < T.lock2) return 'search';
    return 'track2';
  }

  classifier(t) {
    const analysing = (clip, t0, t1, final) => {
      const progress = range01(t, t0, t1);
      const k = easeInOut(progress);
      let probs = final.map((value, i) => {
        const jitter = (hash(i, Math.floor(t * 14)) - 0.5) * 0.28 * (1 - k);
        return Math.max(0.002, lerp(0.2, value, k) + jitter);
      });
      const total = probs.reduce((a, b) => a + b, 0);
      probs = probs.map((value) => value / total);
      return { state: 'analysing', clip, progress, probs };
    };
    if (t < T.analyze1) return { state: 'standby' };
    if (t < T.analyze1End) return analysing('voice', T.analyze1, T.analyze1End, FINAL_VOICE);
    if (t < T.rescan) return { state: 'result', clip: 'voice', progress: 1, probs: FINAL_VOICE, verdict: 'negative' };
    if (t < T.analyze2) return { state: 'standby', previous: 'voice' };
    if (t < T.analyze2End) return analysing('drone', T.analyze2, T.analyze2End, FINAL_DRONE);
    return { state: 'result', clip: 'drone', progress: 1, probs: FINAL_DRONE, verdict: 'positive' };
  }

  finalDroneSolution() {
    const k = this.targetAt(T.verdictCard);
    return { sector: k, az: this.sectorAz(k), el: this.sectorEl(k), confidence: FINAL_DRONE[0] };
  }

  // ── Protocol log (real Heimdall record formats + host classifier lines) ─
  rawLevel(db) { return Math.max(1, Math.round(Math.pow(10, db / 20) * 16777216)); }

  sectorRecord(type, k, t, withLevel = true) {
    const [row, col] = this.rowCol(k);
    const fields = [type, k, row, col, this.sectorAz(k).toFixed(4), this.sectorEl(k).toFixed(4)];
    if (withLevel) fields.push(this.rawLevel(this.dbAt(k, t)));
    return fields.join(',');
  }

  buildLog() {
    const log = [];
    const push = (t, text, tone = '') => log.push({ t, text, tone });
    push(T.dome - 0.4, 'READY,2D,6,6,36,44');
    push(T.dome - 0.35, `AZIMUTH,6,${this.array.azimuth_deg.map((v) => v.toFixed(4)).join(',')}`);
    push(T.dome - 0.3, `ELEVATION,6,${this.array.elevation_deg.map((v) => v.toFixed(4)).join(',')}`);
    for (const segment of this.sweeps) {
      if (segment.t0 === T.dome || segment.restart) push(segment.t0, 'SCAN_STARTED,G', 'mode');
      const stride = segment.rate <= 30 ? 1 : segment.rate <= 90 ? 3 : 11;
      for (let j = Math.ceil(segment.i0); j < segment.i1; j++) {
        const t = this.timeOfIndex(j);
        const k = j % this.sectors;
        if (j % stride === 0) {
          const [row, col] = this.rowCol(k);
          push(t, `P,${row},${col},${this.sectorAz(k).toFixed(4)},${this.sectorEl(k).toFixed(4)},${this.rawLevel(this.dbAt(k, t))}`);
        }
        if (k === this.sectors - 1 && (segment.rate <= 90 || Math.floor(j / this.sectors) % 3 === 0)) {
          const us = Math.round((this.sectors / segment.rate) * 1e6);
          push(t + 1e-4, 'SCAN_DONE');
          push(t + 2e-4, `TIMING,G,36,${us},${us},${Math.round(us / 36)},${Math.round(us / 36)}`);
        }
      }
    }
    for (const [track, n] of [[this.track1, 1], [this.track2, 2]]) {
      push(track.t0, this.sectorRecord('TARGET_ACQUIRED', track.start, track.t0), n === 1 ? 'lock' : 'alert');
      track.steps.forEach((step, i) => {
        if (i > 0 && i % 4 === 0) push(step.t, this.sectorRecord('TARGET_UPDATED', step.target, step.t));
      });
      for (const move of track.moves) push(move.t, this.sectorRecord('TARGET_UPDATED', move.to, move.t), n === 1 ? 'lock' : 'alert');
    }
    const s1 = this.track1.start;
    push(T.analyze1, `CLS,START,SECTOR=${s1},MEL=64x298,INT8`, 'cls');
    push(T.analyze1End, 'CLS,RESULT,HUMAN_VOICE,P=0.934', 'ok');
    push(T.verdict1, `HOST,IGNORE,${s1},NON_THREAT`, 'ok');
    push(T.release1, this.sectorRecord('TARGET_LOST', this.targetAt(T.release1 - 0.01) ?? s1, T.release1), 'ok');
    push(T.analyze2, `CLS,START,SECTOR=${this.targetAt(T.analyze2)},MEL=64x298,INT8`, 'cls');
    push(T.analyze2End, 'CLS,RESULT,DRONE,P=0.978', 'alert');
    push(T.verdictCard - 0.05, `ALERT,UAS,SECTOR=${this.targetAt(T.verdictCard)}`, 'alert');
    log.sort((a, b) => a.t - b.t);
    return log;
  }

  logLines(t, count) {
    let end = 0;
    while (end < this.log.length && this.log[end].t <= t) end++;
    return this.log.slice(Math.max(0, end - count), end);
  }

  // ── Mic boot order and sound cues ───────────────────────────────────────
  igniteTime(mic) {
    const order = this.array.ignite_order.indexOf(mic);
    return T.igniteStart + (order / (this.array.ignite_order.length - 1)) * (T.igniteEnd - T.igniteStart);
  }

  audioEvents() {
    const events = [];
    const add = (type, t, extra = {}) => events.push({ type, t: +t.toFixed(4), ...extra });
    add('hit', T.cardA, { strength: 0.8 });
    add('hit', T.cardB, { strength: 0.9 });
    add('hit', T.cardC, { strength: 1.0 });
    add('boom', T.reveal);
    add('riser', T.reveal, { t1: T.sweepEnd });
    add('pad', T.reveal, { t1: T.dome });
    this.array.ignite_order.forEach((mic) => {
      add('mic', this.igniteTime(mic), { pan: clamp(this.mics[mic][0] / 0.24, -1, 1), index: this.array.ignite_order.indexOf(mic) });
    });
    for (const t of [T.micCallout, T.dspCallout, T.mcuCallout]) add('type', t, { t1: t + 0.35 });
    for (const t of [T.xray, T.dome, T.console, T.end]) add('cut', t);
    for (const segment of this.sweeps) {
      if (segment.rate <= 90) {
        for (let j = Math.ceil(segment.i0); j < segment.i1; j++) {
          const k = j % this.sectors;
          add('tick', this.timeOfIndex(j), { row: this.rowCol(k)[0], strong: k === 0 });
        }
      } else {
        add('scanhum', segment.t0, { t1: segment.t1 });
      }
    }
    add('lock', T.lock1, { tone: 'amber' });
    add('data', T.analyze1, { t1: T.analyze1End });
    add('clear', T.verdict1);
    add('release', T.release1);
    add('lock', T.lock2, { tone: 'red' });
    for (const move of this.track2.moves) if (move.t < T.verdictCard) add('move', move.t);
    add('data', T.analyze2, { t1: T.analyze2End });
    add('alarm', T.verdictCard, { t1: T.end });
    add('boom', T.verdictCard);
    add('end', T.end);
    add('bed_console', T.dome, { t1: T.verdictCard });
    add('bed_voice', T.lock1 - 0.2, { t1: T.rescan });
    add('bed_drone', T.rescan, { t1: T.end });
    return events.sort((a, b) => a.t - b.t);
  }
}

export function directionUv(azDeg, elDeg) {
  const az = (azDeg * Math.PI) / 180;
  const el = (elDeg * Math.PI) / 180;
  return [Math.cos(el) * Math.sin(az), Math.sin(el)];
}
