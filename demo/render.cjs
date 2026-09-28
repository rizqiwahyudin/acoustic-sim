#!/usr/bin/env node
/**
 * Frame-accurate renderer for the HEIMDALL reveal.
 *
 *   node demo/render.cjs                                        # heimdall-reveal MP4 with soundtrack
 *   node demo/render.cjs --scene demo/heimdall-teaser           # another scene directory
 *   node demo/render.cjs --stills 2.4,9.5                       # review stills only
 *   node demo/render.cjs --fps 10 --out preview.mp4
 *
 * Each frame is rendered from a virtual clock, so a slow (software) GPU only
 * makes the render take longer; it never drops or stutters frames.
 */
const fs = require('fs');
const http = require('http');
const path = require('path');
const { execSync, spawn, spawnSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const WIDTH = 1920;
const HEIGHT = 1080;
const DURATION = 20;

function parseArgs(argv) {
  const args = { fps: 30, gpu: false, from: 0, to: DURATION, scene: path.join(__dirname, 'heimdall-reveal') };
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    const next = () => argv[++i];
    if (key === '--stills') args.stills = next().split(',').map(Number);
    else if (key === '--fps') args.fps = Number(next());
    else if (key === '--out') args.out = path.resolve(next());
    else if (key === '--from') args.from = Number(next());
    else if (key === '--to') args.to = Number(next());
    else if (key === '--gpu') args.gpu = true;
    else if (key === '--scene') args.scene = path.resolve(next());
    else if (key === '--python') args.python = next();
    else if (key === '--ffmpeg') args.ffmpeg = next();
    else throw new Error(`unknown argument ${key}`);
  }
  args.outDir = path.join(args.scene, 'out');
  args.out ??= path.join(args.outDir, `${path.basename(args.scene)}.mp4`);
  return args;
}

function loadPlaywright() {
  try { return require('playwright'); } catch {
    const globalRoot = execSync('npm root -g').toString().trim();
    return require(path.join(globalRoot, 'playwright'));
  }
}

function findPython(args) {
  if (args.python) return args.python;
  for (const candidate of [path.join(ROOT, '.venv', 'bin', 'python'), path.join(ROOT, '.venv', 'Scripts', 'python.exe')]) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return process.platform === 'win32' ? 'python' : 'python3';
}

function findFfmpeg(args, python) {
  if (args.ffmpeg) return args.ffmpeg;
  const probe = spawnSync(python, ['-c', 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'], { encoding: 'utf8' });
  if (probe.status === 0 && probe.stdout.trim()) return probe.stdout.trim();
  return 'ffmpeg';
}

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.css': 'text/css',
  '.woff2': 'font/woff2', '.png': 'image/png', '.stl': 'model/stl',
};

function serve() {
  const server = http.createServer((request, response) => {
    const url = decodeURIComponent(new URL(request.url, 'http://x').pathname);
    const file = path.normalize(path.join(ROOT, url));
    if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      response.writeHead(404); response.end(); return;
    }
    response.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    if (request.method === 'HEAD') { response.end(); return; }
    fs.createReadStream(file).pipe(response);
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server)));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const OUT_DIR = args.outDir;
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const server = await serve();
  const { chromium } = loadPlaywright();
  const launchArgs = args.gpu ? [] : ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];
  const browser = await chromium.launch({ args: launchArgs });
  const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });
  page.on('pageerror', (error) => console.error('[page]', error.message));
  // The optional assets/array.stl probe logs a 404 when no STL is present; that is expected.
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.text().includes('404')) console.error('[console]', message.text());
  });
  const scenePath = path.relative(ROOT, args.scene).split(path.sep).join('/');
  const url = `http://127.0.0.1:${server.address().port}/${scenePath}/index.html`;
  await page.goto(url);
  await page.waitForFunction(() => window.revealReady === true, null, { timeout: 180000 });
  const clip = { x: 0, y: 0, width: WIDTH, height: HEIGHT };

  if (args.stills) {
    const stillDir = path.join(OUT_DIR, 'stills');
    fs.mkdirSync(stillDir, { recursive: true });
    for (const t of args.stills) {
      const started = Date.now();
      const shot = await page.evaluate((time) => window.renderFrame(time), t);
      const file = path.join(stillDir, `t${t.toFixed(2).padStart(5, '0')}.png`);
      await page.screenshot({ path: file, clip });
      console.log(`${file}  (${shot}, ${Date.now() - started} ms)`);
    }
  } else {
    const python = findPython(args);
    const events = await page.evaluate(() => (window.audioEvents ? window.audioEvents() : []));
    const eventsFile = path.join(OUT_DIR, 'events.json');
    const wavFile = path.join(OUT_DIR, 'soundtrack.wav');
    fs.writeFileSync(eventsFile, JSON.stringify(events, null, 1));
    const soundtrack = path.join(args.scene, 'soundtrack.py');
    const hasAudio = fs.existsSync(soundtrack);
    if (hasAudio) {
      const audio = spawnSync(python, [soundtrack, eventsFile, wavFile], { stdio: 'inherit', cwd: ROOT });
      if (audio.status !== 0) throw new Error('soundtrack synthesis failed');
    }
    const audioArgs = hasAudio ? ['-ss', String(args.from), '-i', wavFile] : [];
    const audioCodec = hasAudio ? ['-c:a', 'aac', '-b:a', '192k', '-shortest'] : [];
    const ffmpeg = spawn(findFfmpeg(args, python), [
      '-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(args.fps), '-c:v', 'png', '-i', '-',
      ...audioArgs,
      '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
      ...audioCodec, args.out,
    ], { stdio: ['pipe', 'inherit', 'inherit'] });
    const done = new Promise((resolve, reject) => ffmpeg.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg exited ${code}`)))));
    const first = Math.round(args.from * args.fps);
    const last = Math.round(args.to * args.fps);
    const started = Date.now();
    for (let frame = first; frame < last; frame++) {
      await page.evaluate((time) => window.renderFrame(time), frame / args.fps);
      const png = await page.screenshot({ type: 'png', clip });
      if (!ffmpeg.stdin.write(png)) await new Promise((resolve) => ffmpeg.stdin.once('drain', resolve));
      if ((frame - first + 1) % 30 === 0) {
        const doneFrames = frame - first + 1;
        const eta = ((Date.now() - started) / doneFrames) * (last - frame - 1) / 1000;
        console.log(`frame ${doneFrames}/${last - first}  ETA ${eta.toFixed(0)} s`);
      }
    }
    ffmpeg.stdin.end();
    await done;
    console.log(`wrote ${args.out}`);
  }
  await browser.close();
  server.close();
}

main().catch((error) => { console.error(error); process.exit(1); });
