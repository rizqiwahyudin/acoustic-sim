/**
 * pattern.js — delay-and-sum array factor and the figures shown next to it.
 * Tool axes: right-handed, z up; azimuth counter-clockwise from +x seen from
 * above, elevation up from the horizontal plane. Pure functions.
 */

export const SOUND = 343;
const RAD = Math.PI / 180;

export function direction(azDeg, elDeg) {
  const a = azDeg * RAD;
  const e = elDeg * RAD;
  return [Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e)];
}

/** Returns g(u): normalised array gain (0..1) towards unit vector u. */
export function makeGain(mics, freq, steerAz, steerEl) {
  const k = 2 * Math.PI * freq / SOUND;
  const u0 = direction(steerAz, steerEl);
  const n = Math.max(1, mics.length);
  return (u) => {
    let re = 0;
    let im = 0;
    for (const m of mics) {
      const phase = k * (m[0] * (u[0] - u0[0]) + m[1] * (u[1] - u0[1]) + m[2] * (u[2] - u0[2]));
      re += Math.cos(phase);
      im += Math.sin(phase);
    }
    return Math.hypot(re, im) / n;
  };
}

export function toDb(gain) {
  return 20 * Math.log10(Math.max(1e-6, gain));
}

/** Gain on an azimuth × elevation grid (degrees). */
export function patternGrid(gain, step = 5) {
  const az = [];
  const el = [];
  for (let a = 0; a <= 360; a += step) az.push(a);
  for (let e = -90; e <= 90; e += step) el.push(e);
  const values = az.map((a) => el.map((e) => gain(direction(a, e))));
  return {az, el, values};
}

/** Horizontal cut at the steered elevation and vertical cut through the steered azimuth. */
export function cuts(gain, steerAz, steerEl, step = 2) {
  const azCut = [];
  const elCut = [];
  const h0 = [Math.cos(steerAz * RAD), Math.sin(steerAz * RAD), 0];
  for (let a = 0; a < 360; a += step) {
    azCut.push({angle: a, db: toDb(gain(direction(a, steerEl)))});
    const p = a * RAD;
    elCut.push({angle: a, db: toDb(gain([Math.cos(p) * h0[0], Math.cos(p) * h0[1], Math.sin(p)]))});
  }
  const index = (deg) => Math.round((((deg % 360) + 360) % 360) / step) % Math.round(360 / step);
  return {
    azimuth: {samples: azCut, center: index(steerAz), step},
    elevation: {samples: elCut, center: index(steerEl), step},
  };
}

/** Width of the −3 dB main lobe around `center`, in degrees (null if no edge). */
export function width3(cut) {
  const {samples, center, step} = cut;
  const n = samples.length;
  if (samples.every((s) => s.db >= -3)) return null;
  let lo = 0;
  let hi = 0;
  while (lo < n && samples[(center - lo - 1 + n) % n].db >= -3) lo++;
  while (hi < n && samples[(center + hi + 1) % n].db >= -3) hi++;
  return (lo + hi + 1) * step;
}

export function figures(mics, freq, gain) {
  let aperture = 0;
  let nearest = Infinity;
  for (let i = 0; i < mics.length; i++) {
    for (let j = i + 1; j < mics.length; j++) {
      const d = Math.hypot(mics[i][0] - mics[j][0], mics[i][1] - mics[j][1], mics[i][2] - mics[j][2]);
      aperture = Math.max(aperture, d);
      if (d > 1e-6) nearest = Math.min(nearest, d);
    }
  }
  let solid = 0;
  for (let a = 0; a < 360; a += 5) {
    for (let e = -87.5; e < 90; e += 5) {
      const g = gain(direction(a, e));
      solid += g * g * Math.cos(e * RAD) * 25;
    }
  }
  return {
    count: mics.length,
    aperture,
    nearest: Number.isFinite(nearest) ? nearest : null,
    wavelength: SOUND / freq,
    aliasHz: Number.isFinite(nearest) ? SOUND / (2 * nearest) : null,
    solidAngle: solid,
  };
}
