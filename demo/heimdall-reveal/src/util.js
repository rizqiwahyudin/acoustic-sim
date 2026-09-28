export const W = 1920;
export const H = 1080;

export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const range01 = (t, a, b) => clamp((t - a) / (b - a));
export const smooth = (t) => { t = clamp(t); return t * t * (3 - 2 * t); };
export const easeOut = (t) => 1 - Math.pow(1 - clamp(t), 3);
export const easeIn = (t) => Math.pow(clamp(t), 3);
export const easeInOut = (t) => {
  t = clamp(t);
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
};

/** Deterministic hash of any number of numeric keys to [0, 1). */
export function hash(...keys) {
  let h = 2166136261;
  for (const key of keys) {
    h ^= Math.floor(key * 1000003) | 0;
    h = Math.imul(h, 16777619);
    h ^= h >>> 13;
  }
  h = Math.imul(h ^ (h >>> 15), 0x5bd1e995);
  h ^= h >>> 12;
  return (h >>> 0) / 4294967296;
}

export function seeded(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let v = state;
    v = Math.imul(v ^ (v >>> 15), v | 1);
    v ^= v + Math.imul(v ^ (v >>> 7), v | 61);
    return ((v ^ (v >>> 14)) >>> 0) / 4294967296;
  };
}

/** True when a square wave of the given frequency is in its "on" half. */
export const blink = (t, hz, duty = 0.5) => ((t * hz) % 1) < duty;

export const deg = (v, digits = 1) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(digits)}°`;

// NGE-inspired palette: orange/amber UI, red alerts, terminal green.
export const C = {
  black: '#000000',
  ink: '#050505',
  panel: '#0b0906',
  orange: '#ff7a00',
  orangeDim: 'rgba(255,122,0,0.35)',
  orangeFaint: 'rgba(255,122,0,0.12)',
  amber: '#ffb000',
  red: '#ff1f1f',
  redDeep: '#b3000c',
  green: '#39ff6a',
  greenDim: 'rgba(57,255,106,0.35)',
  white: '#f4f1ea',
  grey: '#8a8378',
};

/** Heat palette for sector power: black -> blood red -> orange -> amber -> white. */
const HEAT_STOPS = [
  [0.0, [8, 3, 0]],
  [0.3, [96, 14, 0]],
  [0.58, [255, 84, 0]],
  [0.82, [255, 176, 0]],
  [1.0, [255, 246, 214]],
];
export function heat(v) {
  v = clamp(v);
  for (let i = 1; i < HEAT_STOPS.length; i++) {
    const [p1, c1] = HEAT_STOPS[i];
    const [p0, c0] = HEAT_STOPS[i - 1];
    if (v <= p1) {
      const k = (v - p0) / (p1 - p0);
      return [0, 1, 2].map((j) => Math.round(lerp(c0[j], c1[j], k)));
    }
  }
  return HEAT_STOPS.at(-1)[1];
}
export const heatCss = (v, a = 1) => { const [r, g, b] = heat(v); return `rgba(${r},${g},${b},${a})`; };

export const FONT = {
  mincho: '"Shippori Mincho B1"',
  cond: '"Barlow Condensed", "Noto Sans JP"',
  mono: '"Share Tech Mono", "Noto Sans JP"',
  jp: '"Noto Sans JP"',
};
