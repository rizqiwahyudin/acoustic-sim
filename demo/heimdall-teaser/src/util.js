export const W = 1920;
export const H = 1080;
export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const range01 = (t, a, b) => clamp((t - a) / (b - a));
export const smooth = (t) => { t = clamp(t); return t * t * (3 - 2 * t); };
export const easeInOut = (t) => { t = clamp(t); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
export const easeOut = (t) => 1 - Math.pow(1 - clamp(t), 3);
/** Rises over [a, b] and falls over [c, d]. */
export const window01 = (t, a, b, c, d) => smooth(range01(t, a, b)) * (1 - smooth(range01(t, c, d)));

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

// One cold wire colour, one signal colour.
export const WIRE = [0.78, 0.88, 1.0];
export const SIGNAL = [1.0, 0.36, 0.07];
export const FONT = { serif: '"Shippori Mincho B1"', mono: '"IBM Plex Mono"' };
