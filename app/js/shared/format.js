/**
 * format.js — number and time formatting with real minus signs.
 */

export const MINUS = '−';
export const DASH = '—';

export function num(value, digits = 1) {
  if (!Number.isFinite(value)) return DASH;
  const text = Math.abs(value).toFixed(digits);
  const negative = value < 0 && Number(text) !== 0;
  return (negative ? MINUS : '') + text;
}

export function signed(value, digits = 1) {
  if (!Number.isFinite(value)) return DASH;
  return (value > 0 ? '+' : '') + num(value, digits);
}

export function deg(value, digits = 1) {
  return Number.isFinite(value) ? num(value, digits) + '°' : DASH;
}

export function db(value, digits = 1) {
  return Number.isFinite(value) ? num(value, digits) + ' dB' : DASH;
}

export function dbfs(value, digits = 1) {
  return Number.isFinite(value) ? num(value, digits) + ' dBFS' : DASH;
}

export function hz(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) + ' Hz' : DASH;
}

export function formatAge(ms) {
  if (!Number.isFinite(ms)) return DASH;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`;
  return `${Math.floor(ms / 60000)} min ${Math.floor((ms % 60000) / 1000)} s`;
}

export function clockTime(date = new Date()) {
  return date.toLocaleTimeString([], {hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'});
}

export function plural(count, one, many = one + 's') {
  return `${count} ${count === 1 ? one : many}`;
}

export function fileStamp(date = new Date()) {
  return date.toISOString().replace(/[:.]/g, '-').slice(0, 19);
}
