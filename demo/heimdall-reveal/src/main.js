import { Story, shotAt } from './story.js';
import { Stage } from './stage.js';
import { Hud } from './hud.js';
import { W, H } from './util.js';

const [array, spectra] = await Promise.all([
  fetch('assets/array.json').then((r) => r.json()),
  fetch('assets/spectra.json').then((r) => r.json()),
]);

await Promise.all([
  document.fonts.load('800 100px "Shippori Mincho B1"', '音響ABC'),
  document.fonts.load('500 20px "Barlow Condensed"'),
  document.fonts.load('600 20px "Barlow Condensed"'),
  document.fonts.load('700 20px "Barlow Condensed"'),
  document.fonts.load('400 20px "Share Tech Mono"'),
  document.fonts.load('500 20px "Noto Sans JP"', '方位'),
  document.fonts.load('700 20px "Noto Sans JP"', '方位'),
]);

const story = new Story(array);
const stage = new Stage(document.getElementById('gl'), story);
const params = new URLSearchParams(location.search);
const stl = params.get('stl') ?? 'assets/array.stl';
if ((await fetch(stl, { method: 'HEAD' }).catch(() => null))?.ok) await stage.useStl(stl);

const out = document.getElementById('out');
const ctx = out.getContext('2d', { willReadFrequently: true });
const hud = new Hud(ctx, story, spectra, stage);
const THREE_D = new Set(['reveal', 'xray', 'dome', 'end']);

window.renderFrame = (t) => {
  const shot = shotAt(t).id;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  if (THREE_D.has(shot)) {
    stage.update(t);
    stage.render();
    ctx.drawImage(stage.canvas, 0, 0, W, H);
  } else {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, W, H);
  }
  hud.draw(t, shot);
  hud.post(t);
  return shot;
};
window.audioEvents = () => story.audioEvents();
window.revealReady = true;

// ?t=5.2 renders one still; ?play plays in (approximate) real time.
if (params.has('t')) window.renderFrame(parseFloat(params.get('t')));
if (params.has('play')) {
  const start = performance.now();
  const tick = () => {
    const t = (performance.now() - start) / 1000;
    window.renderFrame(t % 20);
    requestAnimationFrame(tick);
  };
  tick();
}
