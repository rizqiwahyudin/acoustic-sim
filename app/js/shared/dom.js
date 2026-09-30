/**
 * dom.js — tiny element builders. `h('button', {class: 'btn', onClick}, 'Stop')`.
 * Props starting with "on" become listeners; `style` may be a string or object;
 * everything else is set as an attribute (false/null skip it, true sets it empty).
 */

const SVG_NS = 'http://www.w3.org/2000/svg';

function applyProps(el, props) {
  for (const [key, value] of Object.entries(props || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith('on') && typeof value === 'function') {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === 'style' && typeof value === 'object') {
      Object.assign(el.style, value);
    } else if (key === 'text') {
      el.textContent = value;
    } else if (key === 'dataset') {
      Object.assign(el.dataset, value);
    } else if (key === 'value' && 'value' in el) {
      el.value = value;
    } else if (key === 'checked' && 'checked' in el) {
      el.checked = Boolean(value);
    } else {
      el.setAttribute(key === 'className' ? 'class' : key, value === true ? '' : String(value));
    }
  }
}

function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  applyProps(el, props);
  append(el, children);
  return el;
}

export function s(tag, props = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  applyProps(el, props);
  append(el, children);
  return el;
}

export function $(selector, root = document) {
  return root.querySelector(selector);
}

export function setText(el, text) {
  const next = String(text ?? '');
  if (el.textContent !== next) el.textContent = next;
}

/** Options for a <select>: [[value, label], ...]. */
export function options(pairs, selected) {
  return pairs.map(([value, label]) => h('option', {value, selected: String(value) === String(selected)}, label));
}

/**
 * Segmented control. `items` is [[key, label], ...]; `onPick(key)` fires on click.
 * Returns {el, set(key), setDisabled(bool)}.
 */
export function segmented(items, {label, onPick, size = '', role = 'group'} = {}) {
  const buttons = new Map();
  const el = h('div', {class: `seg ${size ? 'seg--' + size : ''}`.trim(), role, 'aria-label': label});
  for (const [key, text] of items) {
    const button = h('button', {
      type: 'button',
      'aria-pressed': 'false',
      onClick: () => onPick?.(key),
    }, text);
    buttons.set(key, button);
    el.append(button);
  }
  return {
    el,
    buttons,
    set(key) {
      for (const [k, button] of buttons) button.setAttribute('aria-pressed', String(k === key));
    },
    setDisabled(disabled) {
      for (const button of buttons.values()) button.disabled = disabled;
    },
  };
}

/** Download a Blob or URL under a file name. */
export function download(blobOrUrl, fileName) {
  const url = typeof blobOrUrl === 'string' ? blobOrUrl : URL.createObjectURL(blobOrUrl);
  const link = h('a', {href: url, download: fileName});
  document.body.append(link);
  link.click();
  link.remove();
  if (typeof blobOrUrl !== 'string') setTimeout(() => URL.revokeObjectURL(url), 5000);
}
