import { placeholderView } from '../shared/placeholder.js';

export function createView(container) {
  return placeholderView(container, 'Simulator', 'This screen is being rebuilt in the new style on this branch.');
}
