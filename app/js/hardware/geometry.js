/**
 * geometry.js — sector angles and array coordinates for the Hardware views.
 *
 * The acoustic contract uses x = right, y = up, z = forward (seen from behind
 * the array); positive azimuth points right and positive elevation up.
 */

/** Sector edges from sector centres (degrees). Single sectors get ±halfSpan. */
export function angularEdges(centers, fallbackHalfSpan = 10) {
  if (!centers || centers.length === 0) return [];
  if (centers.length === 1) return [centers[0] - fallbackHalfSpan, centers[0] + fallbackHalfSpan];
  const edges = [centers[0] - (centers[1] - centers[0]) / 2];
  for (let i = 0; i < centers.length - 1; i++) edges.push((centers[i] + centers[i + 1]) / 2);
  edges.push(centers.at(-1) + (centers.at(-1) - centers.at(-2)) / 2);
  return edges;
}

/** Unit direction for azimuth/elevation in contract coordinates [right, up, forward]. */
export function direction(azimuthDeg, elevationDeg) {
  const a = azimuthDeg * Math.PI / 180;
  const e = elevationDeg * Math.PI / 180;
  return [Math.cos(e) * Math.sin(a), Math.sin(e), Math.cos(e) * Math.cos(a)];
}

/** Field of view text such as "±40° azimuth and elevation". */
export function fovText(frame) {
  const az = angularEdges(frame?.azimuth_deg || []);
  const el = angularEdges(frame?.elevation_deg || []);
  if (!az.length || !el.length) return '';
  const span = (edges) => {
    const lo = Math.round(edges[0] * 10) / 10;
    const hi = Math.round(edges.at(-1) * 10) / 10;
    return Math.abs(lo + hi) < 0.05 ? `±${Math.abs(hi)}°` : `${lo}° to ${hi}°`.replace(/-/g, '−');
  };
  const a = span(az);
  const e = span(el);
  return a === e ? `${a} azimuth and elevation` : `${a} azimuth, ${e} elevation`;
}
