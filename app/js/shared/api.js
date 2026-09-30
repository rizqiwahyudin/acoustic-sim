/**
 * api.js — the local sim_server backend (FastAPI on 127.0.0.1:8766).
 */

export const API_BASE = 'http://127.0.0.1:8766';
export const WS_BASE = 'ws://127.0.0.1:8766';

/** Fetch JSON; throws Error(detail) on non-2xx responses. */
export async function apiJson(path, {method = 'GET', body} = {}) {
  const response = await fetch(API_BASE + path, {
    method,
    headers: body === undefined ? undefined : {'Content-Type': 'application/json'},
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let data = null;
  try { data = await response.json(); } catch (_) { /* empty or non-JSON body */ }
  if (!response.ok) {
    const detail = data && data.detail ? data.detail : `${response.status} ${response.statusText}`;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return data;
}

export function base64ToBlobUrl(b64, type = 'audio/wav') {
  const raw = atob(b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], {type}));
}
