# HEIMDALL teaser

A 30-second, 1080p30 teaser: slow, wireframe and mostly dark, framed 2.39:1
inside 16:9. Rendered from code one frame at a time, like the reveal video.

## Render

Same prerequisites as `../heimdall-reveal` (Playwright, the repository Python
environment with `imageio-ffmpeg`, and `app/node_modules` for three.js).

```powershell
node demo\render.cjs --scene demo\heimdall-teaser                  # MP4 with soundtrack
node demo\render.cjs --scene demo\heimdall-teaser --stills 5.6,22.8 # stills in out/stills
node demo\render.cjs --scene demo\heimdall-teaser --audio           # out/soundtrack.wav only
```

Output: `demo/heimdall-teaser/out/heimdall-teaser.mp4`. For a single frame in a
browser, serve the repository root and open
`/demo/heimdall-teaser/index.html?t=20.6`.

## Shots

| Time | Shot |
| --- | --- |
| 0.0-4.6 s | Black. Rain. Epigraph from the Prose Edda |
| 4.6-9.0 s | A valley of ridgelines, a watchman, the array on its mast. "every sound has a direction." |
| 9.0-12.6 s | Sparks rise from the valley into the microphones, centre first. LISTEN |
| 12.6-15.6 s | A plane wavefront falls from the sky and crosses the board in delay order; a hex ripple |
| 15.6-18.4 s | The channels converge into one beam; through the array into its 6x6 sector dome; a sector locks. LOCK |
| 18.4-23.8 s | Up the beam to the drone. Its spectrum unrolls behind it as a second valley. DRONE, 97.8% |
| 24.2-30.0 s | HEIMDALL / acoustic drone detection |

## What is real and what is illustrative

| Element | Source |
| --- | --- |
| Microphone positions, hexagonal aperture, 6x6 sector centres | `data/heimdall_acoustic_contract.json` |
| Ridgelines of the valley and of the drone's wake | Log-mel spectrogram of `audio/drone.wav` (`../heimdall-reveal/assets/spectra.json`) |
| Order in which the wavefront reaches the microphones | Plane-wave delays for the real geometry |
| Distant rotor noise in the score | `audio/drone.wav` |
| Classification label and confidence | **Illustrative.** The firmware protocol has no classification step |
| Enclosure, mast, watchman, drone model | Placeholders |

## Files

- `src/scene.js` – timeline, camera path, the three.js world and the sound cue list
- `src/overlay.js` – epigraph, letterbox, section cards, reticle, title, grain and vignette
- `soundtrack.py` – ambient score synthesised from the cue list
- `fonts.json` – families for `python demo/fetch_fonts.py demo/heimdall-teaser`
