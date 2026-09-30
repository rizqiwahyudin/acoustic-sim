/**
 * shell.js — entry point: fonts, styles, hash routing and the global header.
 * Screens are loaded on first visit and keep their state while hidden.
 */

import '@fontsource/instrument-sans/latin-400.css';
import '@fontsource/instrument-sans/latin-500.css';
import '@fontsource/instrument-sans/latin-600.css';
import '@fontsource/ibm-plex-mono/latin-400.css';
import '@fontsource/ibm-plex-mono/latin-500.css';
import '@fontsource/archivo/latin-400.css';
import '@fontsource/archivo/latin-600.css';
import '@fontsource/archivo/latin-800.css';
import '../styles/app.css';

import { session } from './hardware/session.js';
import { $, h, setText } from './shared/dom.js';

const ROUTES = {
  beam: () => import('./beam/view.js'),
  sim: () => import('./sim/view.js'),
  hardware: () => import('./hardware/view.js'),
  study: () => import('./study/view.js'),
  exhibit: () => import('./exhibit/view.js'),
};

const LIVE_ROUTES = new Set(['hardware', 'exhibit', 'study']);
const views = new Map();
let current = null;

function parseHash() {
  const hash = window.location.hash || '';
  if (hash.startsWith('#preset=')) return {route: 'sim', preset: hash.slice('#preset='.length)};
  const route = hash.replace(/^#\/?/, '').split(/[/?&]/)[0];
  return {route: ROUTES[route] ? route : 'beam'};
}

export function navigate(route) {
  if (window.location.hash !== '#' + route) window.location.hash = route;
  else show(parseHash());
}

async function show({route, preset}) {
  const container = $('#views');
  if (!views.has(route)) {
    const section = h('section', {class: `view view--${route}`, id: `view-${route}`, 'aria-label': route});
    container.append(section);
    const module = await ROUTES[route]();
    views.set(route, {section, view: module.createView(section, {session, navigate, preset}), presetSeen: preset});
  }
  if (current && current !== route) {
    const previous = views.get(current);
    previous.section.hidden = true;
    previous.view.hide?.();
  }
  current = route;
  const entry = views.get(route);
  entry.section.hidden = false;
  if (preset && entry.view.applyPreset && views.has(route) && entry.presetSeen !== preset) {
    entry.presetSeen = preset;
    entry.view.applyPreset(preset);
  }
  document.body.classList.toggle('is-exhibit', route === 'exhibit');
  for (const link of document.querySelectorAll('#appNav a')) {
    if (link.dataset.route === route) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  if (!['hardware', 'exhibit'].includes(route)) session.stopMonitor();
  if (LIVE_ROUTES.has(route) && !session.open) session.discover();
  entry.view.show?.();
}

// ── Global link summary in the header ────────────────────────────────────

function renderLink() {
  const state = session.link;
  const cfg = session.configuration;
  $('#linkState').dataset.state = state;
  let text;
  if (state === 'online') text = `Connected to ${session.transportLabel}`;
  else if (state === 'connecting') text = 'Connecting…';
  else if (state === 'fault') text = 'Link fault';
  else text = 'Not connected';
  setText($('#linkText'), text);
  $('#linkState').title = session.linkDetail || '';
  const grid = $('#linkGrid');
  grid.hidden = !(state === 'online' && cfg);
  if (cfg) setText(grid, `${cfg.rows} × ${cfg.columns} grid · ${cfg.microphones} microphones`);
  $('#btnDisconnect').hidden = !(session.open || state === 'online');
}

session.addEventListener('link', renderLink);
session.addEventListener('init', renderLink);
session.addEventListener('frame', renderLink);
$('#btnDisconnect').addEventListener('click', () => session.disconnect());

window.addEventListener('hashchange', () => show(parseHash()));
renderLink();
show(parseHash());
