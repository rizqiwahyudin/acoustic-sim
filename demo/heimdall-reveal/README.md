# HEIMDALL reveal video

A 20-second, 1080p30 product-reveal clip for the Hardware mode: a
44-microphone acoustic array scans a 6x6 sector grid, locks onto a talker,
classifies it as a non-threat, reacquires a drone and confirms it.

The visuals are rendered from code, one frame at a time, so the output is
identical on any machine and a slow GPU only makes rendering take longer.

## Render

Prerequisites: Node.js with Playwright (`npm i -g playwright` or a local
install), and the repository Python environment plus `imageio-ffmpeg`
(which ships an H.264-capable ffmpeg).

```powershell
.\.venv\Scripts\python.exe -m pip install imageio-ffmpeg
cd app; npm.cmd ci; cd ..          # three.js is loaded from app/node_modules
node demo\heimdall-reveal\render.cjs
```

Output: `demo/heimdall-reveal/out/heimdall-reveal.mp4` (plus the generated
`soundtrack.wav` and `events.json`). Other modes:

```powershell
node demo\heimdall-reveal\render.cjs --stills 2.4,9.6,16   # PNG stills in out/stills
node demo\heimdall-reveal\render.cjs --fps 10 --out preview.mp4
node demo\heimdall-reveal\render.cjs --gpu                  # use the real GPU instead of SwiftShader
```

To scrub interactively, serve the repository root (for example
`python -m http.server 8000`) and open
`http://localhost:8000/demo/heimdall-reveal/index.html?t=12.4` for one frame or
`?play` for approximate real time.

## Using the real array STL

Drop the model at `demo/heimdall-reveal/assets/array.stl` (or pass
`index.html?stl=<path>`). The loader centres it, scales its largest in-plane
dimension to the 594 mm frame, turns it to face the camera and hides the
procedural frame and board. The 44 glowing ports and traces stay at the
contract coordinates, so check they line up with the STL's port holes; adjust
`useStl()` in `src/stage.js` if the model's axes differ (it assumes the array
face is the STL's +Z side).

## What is real and what is illustrative

| Element | Source |
| --- | --- |
| Microphone positions, hexagonal aperture, 6x6 sector centres, steering delays | `data/heimdall_acoustic_contract.json` (deployed table) |
| Sector levels and the 3D dome | Broadband 1-4 kHz delay-and-sum response of the real geometry to two scripted sources |
| Tracker behaviour | Same policy as the firmware emulator: full search, 5-sector cross tracking, two-pass move hysteresis |
| Protocol log lines | Real `READY`/`P`/`SCAN_DONE`/`TIMING`/`TARGET_*` record formats |
| 81.6 Hz track update, 9.3 Hz full scan | Measured from the acoustic emulator in this repository |
| Drone and voice audio and spectrograms | `audio/drone.wav`, `audio/crowd.wav` |
| CNN classifier panel, class probabilities, `CLS`/`HOST`/`ALERT` log lines | **Illustrative.** The current firmware protocol and Hardware UI have no classification step |
| Enclosure, rear pod, mast | Placeholder model until the STL is dropped in |

The end card labels the clip "simulated scenario".

## Files

- `build_assets.py` – exports `assets/array.json` and `assets/spectra.json`
- `fetch_fonts.py` – downloads the OFL Google Fonts, subset to the text in `src/`
  (re-run after adding new Japanese text)
- `src/story.js` – timeline, scripted sources, sweep/track scheduling, log, sound cues
- `src/stage.js` – three.js array, electronics pod, floor, sector dome, cameras
- `src/hud.js` – title cards, overlays, console, alert cards, film post-processing
- `soundtrack.py` – synthesises the soundtrack from the scene's cue list
- `render.cjs` – static server + headless Chromium frame capture + ffmpeg encode

## Timeline

| Time | Shot |
| --- | --- |
| 0.0-2.0 s | Title cards: 音響 / EARLY WARNING / 四十四の耳 |
| 2.0-4.7 s | Hardware reveal: light sweep, 44 ports boot, microphone callout |
| 4.7-6.2 s | X-ray of the rear pod: ADAU1467 and MAX78002 callouts |
| 6.2-7.4 s | Over-the-shoulder 3D sector dome, first full search pass |
| 7.4-10.0 s | Console: accelerating full search |
| 10.0-12.9 s | Lock on a talker, classifier returns HUMAN VOICE, target released |
| 12.9-15.6 s | Rescan, lock on the drone, tracker follows it one sector, classifier runs |
| 15.6-17.0 s | DRONE CONFIRMED alert |
| 17.0-20.0 s | End card |
