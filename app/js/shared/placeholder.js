/**
 * placeholder.js — shown for screens that are still being rebuilt on this branch.
 */

import { h } from './dom.js';

export function placeholderView(container, title, text) {
  container.append(h('div', {class: 'placeholder'},
    h('h1', {}, title),
    h('p', {}, text),
  ));
  return {show() {}, hide() {}};
}
