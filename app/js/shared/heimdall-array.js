/**
 * heimdall-array.js — the deployed Heimdall microphone layout (DSP00–DSP43),
 * copied from data/heimdall_acoustic_contract.json. Millimetres, contract
 * axes: x = right, y = up, as seen from behind the array.
 */

export const HEIMDALL_POSITIONS_MM = [
  [31.177, 54.0],
  [55.426, 96.0],
  [117.779, 96.0],
  [148.956, 150.0],
  [86.603, 150.0],
  [117.779, 204.0],
  [55.426, 204.0],
  [24.249, 150.0],
  [62.354, 0.0],
  [110.851, 0.0],
  [142.028, -54.0],
  [204.382, -54.0],
  [173.205, 0.0],
  [235.559, 0.0],
  [204.382, 54.0],
  [142.028, 54.0],
  [86.603, -150.0],
  [117.779, -204.0],
  [148.956, -150.0],
  [117.779, -96.0],
  [-31.177, -54.0],
  [-55.426, -96.0],
  [-117.779, -96.0],
  [-148.956, -150.0],
  [-86.603, -150.0],
  [-117.779, -204.0],
  [-55.426, -204.0],
  [-24.249, -150.0],
  [-62.354, 0.0],
  [-110.851, 0.0],
  [-142.028, 54.0],
  [-204.382, 54.0],
  [-173.205, 0.0],
  [-235.559, 0.0],
  [-204.382, -54.0],
  [-142.028, -54.0],
  [-31.177, 54.0],
  [-55.426, 96.0],
  [-24.249, 150.0],
  [-55.426, 204.0],
  [-86.603, 150.0],
  [-117.779, 204.0],
  [-148.956, 150.0],
  [-117.779, 96.0],
];

/**
 * Positions in metres for the Beam pattern and Simulator tools. Their axes
 * are right-handed with z up and azimuth counter-clockwise seen from above
 * (x = 0°, y = 90°). The array faces 0° azimuth, so Heimdall's "right" is
 * the tool's negative y (negative azimuth).
 */
export function heimdallToolPositions() {
  return HEIMDALL_POSITIONS_MM.map(([right, up]) => [0, -right / 1000, up / 1000]);
}
