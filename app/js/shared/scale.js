/**
 * scale.js — fits the whole app to the window like a control panel.
 *
 * The layout is designed for a 1600 × 900 canvas. The app element is zoomed
 * uniformly so that this canvas fits the window: text, controls, charts and
 * views scale together. Wider or taller windows keep the same scale and give
 * the layout more room (ultrawide screens get extra columns), so nothing has
 * to be set with the browser's own zoom.
 */

const BASE_WIDTH = 1600;
const BASE_HEIGHT = 900;
const MIN_ZOOM = 0.6;
const MAX_ZOOM = 2.4;

let zoom = 1;
const listeners = new Set();

export function currentZoom() {
  return zoom;
}

/** Call `fn(zoom)` whenever the scale changes. Returns an unsubscribe function. */
export function onZoom(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function apply() {
  const app = document.getElementById('app');
  if (!app) return;
  const width = window.innerWidth;
  const height = window.innerHeight;
  const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, Math.min(width / BASE_WIDTH, height / BASE_HEIGHT)));
  const changed = Math.abs(next - zoom) > 1e-4;
  zoom = next;
  // Sizes are set in design pixels; the zoom brings them back to the window size.
  app.style.zoom = String(zoom);
  app.style.width = `${width / zoom}px`;
  app.style.height = `${height / zoom}px`;
  document.documentElement.style.setProperty('--app-w', `${width / zoom}px`);
  document.documentElement.style.setProperty('--app-h', `${height / zoom}px`);
  document.documentElement.dataset.wide = width / height >= 2.3 ? 'true' : 'false';
  if (changed) for (const fn of listeners) fn(zoom);
}

window.addEventListener('resize', apply);
apply();
