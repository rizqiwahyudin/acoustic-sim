/**
 * view.js — the Study screen: a bar with the device state and the explicit
 * "Enable writes" switch, then Parameters or Measurements (#study/parameters,
 * #study/measurements).
 */

import { h, setText } from '../shared/dom.js';
import { dsp } from './dsp.js';
import { parametersScreen } from './params.js';
import { measurementsScreen } from './measure.js';

export function createView(container, {session}) {
  const state = {writes: false, visible: false, sub: 'parameters'};

  const subLinks = [['parameters', 'Parameters'], ['measurements', 'Measurements']].map(([key, label]) =>
    h('a', {href: `#study/${key}`, 'data-sub': key}, label));
  const devDot = h('span', {class: 'link-state__dot', 'aria-hidden': 'true'});
  const devText = h('span', {});
  const devState = h('span', {class: 'link-state', role: 'status'}, devDot, devText);
  const switchKnob = h('span', {class: 'switch__knob', 'aria-hidden': 'true'});
  const writesSwitch = h('button', {type: 'button', class: 'switch', role: 'switch', 'aria-checked': 'false', 'aria-labelledby': 'st-writes-label',
    onClick: () => { state.writes = !state.writes; render(); }}, switchKnob);
  const bar = h('div', {class: 'st-bar'},
    h('div', {class: 'st-bar__left'}, h('nav', {class: 'sub-nav', 'aria-label': 'Study'}, ...subLinks), devState),
    h('div', {class: 'st-bar__right'},
      h('span', {class: 'note'}, 'Writes change the running DSP only. They are lost when it restarts.'),
      writesSwitch, h('span', {id: 'st-writes-label', class: 'st-bar__label'}, 'Enable writes')),
  );

  const missing = h('div', {class: 'placeholder', hidden: true});
  const params = parametersScreen({canWrite: () => state.writes});
  const measure = measurementsScreen({session});
  const footer = h('footer', {class: 'app-footer'}, h('span', {id: 'st-footer-left'}), h('span', {}, 'Proposed device commands: docs/study-protocol.md'));
  container.append(bar, missing, params.el, measure.el, footer);

  function subFromHash() {
    const match = /^#study\/(\w+)/.exec(window.location.hash || '');
    return match && ['parameters', 'measurements'].includes(match[1]) ? match[1] : 'parameters';
  }

  function render() {
    state.sub = subFromHash();
    for (const link of subLinks) {
      if (link.dataset.sub === state.sub) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
    const status = dsp.status;
    let text = 'Checking the device…';
    let tone = 'connecting';
    if (dsp.error) { text = `No register map: ${dsp.error}`; tone = 'fault'; }
    else if (!session.open) { text = 'Not connected. Connect on the Hardware screen to write or measure.'; tone = 'disconnected'; }
    else if (status && !status.supported) { text = status.reason; tone = 'fault'; }
    else if (status && !status.idle) { text = 'Scan running · writes wait until it stops'; tone = 'connecting'; }
    else if (status) { text = `Device idle · register map ${dsp.registry?.export?.sha256?.slice(0, 8) || ''}… from ${dsp.registry?.export?.file || 'the export'}`; tone = 'online'; }
    devState.dataset.state = tone;
    setText(devText, text);
    const locked = !status?.supported;
    if (locked) state.writes = false;
    writesSwitch.disabled = locked;
    writesSwitch.setAttribute('aria-checked', String(state.writes));
    missing.hidden = !dsp.error;
    if (dsp.error) missing.replaceChildren(h('h1', {}, 'The register map could not be loaded'),
      h('p', {}, `${dsp.error}. The backend reads the SigmaStudio export from ../MAX78002/SigmaStudioExport, or from the path in HEIMDALL_DSP_EXPORT.`));
    params.el.hidden = state.sub !== 'parameters' || Boolean(dsp.error);
    measure.el.hidden = state.sub !== 'measurements';
    const modified = status?.modified_words ?? 0;
    setText(footer.querySelector('#st-footer-left'), dsp.registry
      ? `${dsp.registry.parameters.length} parameters · ${dsp.registry.writable_words.toLocaleString()} writable words · ${modified} words differ from the flashed program`
      : '');
    if (!params.el.hidden) params.render();
  }

  let poll = null;
  dsp.addEventListener('change', () => { if (state.visible) render(); });
  session.addEventListener('link', () => { if (state.visible) { dsp.refresh(); measure.render(); } });
  session.addEventListener('init', () => { if (state.visible) { dsp.refresh(); measure.render(); } });
  window.addEventListener('hashchange', () => { if (state.visible) render(); });

  return {
    show() {
      state.visible = true;
      dsp.load().then(() => dsp.refresh()).catch(() => render());
      poll = setInterval(() => dsp.refresh(), 3000);
      render();
      measure.render();
    },
    hide() {
      state.visible = false;
      clearInterval(poll);
    },
  };
}
