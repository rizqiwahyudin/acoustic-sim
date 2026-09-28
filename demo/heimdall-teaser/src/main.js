import { Teaser, T } from './scene.js';
import { Overlay } from './overlay.js';
import { W, H, smooth, range01 } from './util.js';

const [array, spectra] = await Promise.all([
  fetch('../heimdall-reveal/assets/array.json').then((r) => r.json()),
  fetch('../heimdall-reveal/assets/spectra.json').then((r) => r.json()),
]);
await Promise.all([
  document.fonts.load('400 20px "Shippori Mincho B1"', '方向'),
  document.fonts.load('500 20px "Shippori Mincho B1"', 'HEIMDALL無人機'),
  document.fonts.load('300 20px "IBM Plex Mono"'),
  document.fonts.load('400 20px "IBM Plex Mono"'),
]);

const teaser = new Teaser(document.getElementById('gl'), array, spectra);
window.teaser = teaser;   // handy for debugging from the console
const ctx = document.getElementById('out').getContext('2d');
const overlay = new Overlay(ctx, teaser);

window.renderFrame = (t) => {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  const path = teaser.pathCamera(t);
  const title = teaser.titleCamera(t);
  const blend = smooth(range01(t, T.title, T.titleFull));
  teaser.setCamera(blend < 1 ? path : title);
  teaser.update(t);
  if (blend < 1) {
    teaser.render(path);
    ctx.drawImage(teaser.canvas, 0, 0, W, H);
  }
  if (blend > 0) {
    teaser.render(title);
    ctx.globalAlpha = blend;
    ctx.drawImage(teaser.canvas, 0, 0, W, H);
    ctx.globalAlpha = 1;
  }
  overlay.draw(t);
  overlay.post(t);
  return 'teaser';
};
window.revealReady = true;

const params = new URLSearchParams(location.search);
if (params.has('t')) window.renderFrame(parseFloat(params.get('t')));
