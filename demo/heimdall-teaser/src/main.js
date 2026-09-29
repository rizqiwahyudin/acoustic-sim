import { Teaser, T, DURATION, storyTime, wallTime } from './scene.js';
import { Overlay } from './overlay.js';
import { W, H } from './util.js';

const [array, spectra] = await Promise.all([
  fetch('../heimdall-reveal/assets/array.json').then((r) => r.json()),
  fetch('../heimdall-reveal/assets/spectra.json').then((r) => r.json()),
]);
await Promise.all([
  document.fonts.load('400 20px "Shippori Mincho B1"', 'He hears'),
  document.fonts.load('500 20px "Shippori Mincho B1"', 'HEIMDALL DRONE'),
  document.fonts.load('300 20px "IBM Plex Mono"'),
  document.fonts.load('400 20px "IBM Plex Mono"'),
]);

const teaser = new Teaser(document.getElementById('gl'), array, spectra);
window.teaser = teaser;   // handy for debugging from the console
const ctx = document.getElementById('out').getContext('2d');
const overlay = new Overlay(ctx, teaser);

// Frames and sound cues are in wall-clock time; the scene is written in story time.
window.renderFrame = (wall) => {
  const t = storyTime(wall);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  if (t >= T.fadeIn - 0.05 && t < T.black + 0.05) {
    teaser.setCamera(teaser.pathCamera(t));
    teaser.update(t, wall);
    teaser.render();
    ctx.drawImage(teaser.canvas, 0, 0, W, H);
  } else {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, W, H);
  }
  overlay.draw(t);
  overlay.post(t, wall);
  return 'teaser';
};
window.audioEvents = () => teaser.audioEvents().map((e) => ({
  ...e, t: +wallTime(e.t).toFixed(4), ...(e.t1 !== undefined && { t1: +wallTime(e.t1).toFixed(4) }),
}));
window.DURATION = DURATION;
window.revealReady = true;

const params = new URLSearchParams(location.search);
if (params.has('t')) window.renderFrame(parseFloat(params.get('t')));
