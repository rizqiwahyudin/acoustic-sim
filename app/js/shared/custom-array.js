/**
 * custom-array.js — the user's custom microphone coordinates, shared by the
 * Beam pattern and Simulator screens (metres, offsets from the array centre).
 */

import { h, setText } from './dom.js';
import { buildGeometry } from '../geometry.js';
import { heimdallToolPositions } from './heimdall-array.js';

function validTriple(p) {
  return Array.isArray(p) && p.length === 3 && p.every((v) => Number.isFinite(Number(v)));
}

function fmt(value) {
  return Number(value).toFixed(4).replace(/\.?0+$/, '') || '0';
}

/** Accepts [x,y,z], [[x,y,z],...], comma lists or one "x y z" per line. */
export function parseCoordinates(text) {
  const trimmed = String(text || '').trim();
  if (!trimmed) return [];
  const normalise = (value) => {
    if (validTriple(value)) return [value.map(Number)];
    if (Array.isArray(value) && value.every(validTriple)) return value.map((p) => p.map(Number));
    throw new Error('Expected [x, y, z] or [[x, y, z], ...].');
  };
  try { return normalise(JSON.parse(trimmed)); } catch (_) { /* try other shapes */ }
  try { return normalise(JSON.parse(`[${trimmed}]`)); } catch (_) { /* try lines */ }
  const rows = trimmed.split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'))
    .map((line) => line.replace(/[[\],]/g, ' ').trim().split(/\s+/).map(Number));
  return normalise(rows);
}

class CustomArray extends EventTarget {
  constructor() {
    super();
    this.positions = [];
    this.selected = -1;
  }

  changed() { this.dispatchEvent(new Event('change')); }

  set(positions) {
    this.positions = positions.filter(validTriple).map((p) => p.map(Number));
    this.selected = this.positions.length ? 0 : -1;
    this.changed();
  }

  add(x, y, z) {
    const p = [Number(x), Number(y), Number(z)];
    if (!validTriple(p) || [x, y, z].some((v) => String(v).trim() === '')) throw new Error('Enter numeric x, y and z offsets in metres.');
    if (this.positions.some((q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]) < 1e-6)) {
      throw new Error('That microphone position already exists.');
    }
    this.positions.push(p);
    this.selected = this.positions.length - 1;
    this.changed();
  }

  remove(index) {
    this.positions.splice(index, 1);
    if (this.selected === index) this.selected = -1;
    else if (this.selected > index) this.selected -= 1;
    this.changed();
  }

  clear() {
    this.positions = [];
    this.selected = -1;
    this.changed();
  }

  seed(geometry, count, radius, separation) {
    if (geometry === 'HEIMDALL') this.set(heimdallToolPositions());
    else this.set(buildGeometry(geometry, count, radius, separation));
  }

  select(index) {
    this.selected = index;
    this.changed();
  }
}

export const customArray = new CustomArray();

/**
 * Editor UI: add a microphone, list with remove, seed from a geometry,
 * clear, import a file. `getSeed()` returns {geometry, count, radius, separation}.
 */
export function customArrayEditor({getSeed}) {
  const x = h('input', {class: 'input input--mono input--sm', type: 'number', step: '0.001', placeholder: 'x', 'aria-label': 'x in metres'});
  const y = h('input', {class: 'input input--mono input--sm', type: 'number', step: '0.001', placeholder: 'y', 'aria-label': 'y in metres'});
  const z = h('input', {class: 'input input--mono input--sm', type: 'number', step: '0.001', placeholder: 'z', 'aria-label': 'z in metres'});
  const message = h('p', {class: 'field__hint', role: 'status'});
  const count = h('span', {class: 'mono'});
  const list = h('ol', {class: 'mic-list'});
  const file = h('input', {type: 'file', accept: '.json,.txt,.csv', hidden: true});
  const seedGeometry = h('select', {class: 'select select--sm', 'aria-label': 'Geometry to start from'},
    h('option', {value: 'HEIMDALL'}, 'Heimdall, 44 microphones'),
    h('option', {value: 'UCA'}, 'Circular ring'),
    h('option', {value: 'CROSS'}, 'Standing cross'),
    h('option', {value: 'ULA'}, 'Linear'),
    h('option', {value: 'CYLINDER'}, 'Stacked rings'),
  );

  const add = () => {
    try {
      customArray.add(x.value, y.value, z.value);
      x.value = ''; y.value = ''; z.value = '';
      setText(message, '');
      x.focus();
    } catch (error) {
      setText(message, error.message);
    }
  };
  for (const input of [x, y, z]) {
    input.addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); add(); } });
  }
  file.addEventListener('change', () => {
    const chosen = file.files?.[0];
    if (!chosen) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        customArray.set(parseCoordinates(reader.result));
        setText(message, `Imported ${customArray.positions.length} microphones.`);
      } catch (error) {
        setText(message, `Could not read that file: ${error.message}`);
      }
    };
    reader.readAsText(chosen);
    file.value = '';
  });

  const el = h('div', {class: 'custom-array'},
    h('div', {class: 'field'},
      h('span', {class: 'field__label'}, 'Add a microphone (metres from the centre)'),
      h('div', {class: 'custom-array__add'}, x, y, z,
        h('button', {type: 'button', class: 'btn btn--sm', onClick: add}, 'Add')),
    ),
    message,
    h('div', {class: 'section__row'}, h('span', {class: 'field__label'}, 'Microphones'), count),
    list,
    h('div', {class: 'custom-array__actions'},
      h('div', {class: 'inline-field'}, seedGeometry,
        h('button', {type: 'button', class: 'btn btn--sm', onClick: () => {
          const seed = getSeed();
          customArray.seed(seedGeometry.value, seed.count, seed.radius, seed.separation);
        }}, 'Start from this')),
      h('div', {class: 'inline-field'},
        h('button', {type: 'button', class: 'btn btn--sm btn--ghost', onClick: () => file.click()}, 'Import file…'),
        h('button', {type: 'button', class: 'btn btn--sm btn--ghost', onClick: () => customArray.clear()}, 'Clear all')),
    ),
    file,
  );

  const render = () => {
    setText(count, String(customArray.positions.length));
    if (!customArray.positions.length) {
      list.replaceChildren(h('li', {class: 'mic-list__empty'}, 'No microphones yet.'));
      return;
    }
    list.replaceChildren(...customArray.positions.map((p, index) => h('li', {
      class: `mic-list__item ${index === customArray.selected ? 'is-selected' : ''}`.trim(),
    },
    h('button', {type: 'button', class: 'mic-list__pick', onClick: () => customArray.select(index)},
      h('span', {class: 'mic-list__index'}, String(index + 1)),
      h('span', {class: 'mono'}, `${fmt(p[0])}, ${fmt(p[1])}, ${fmt(p[2])}`)),
    h('button', {type: 'button', class: 'btn btn--sm btn--quiet', 'aria-label': `Remove microphone ${index + 1}`,
      onClick: () => customArray.remove(index)}, '×'),
    )));
  };
  customArray.addEventListener('change', render);
  render();
  return el;
}
