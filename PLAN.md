# Live Simulation Server -- Implementation Plan

## Overview

Add a FastAPI backend that runs single-trial pyroomacoustics simulations on demand, and wire it into the existing `results/array_explorer.html` as a third "Live Simulation" mode. Includes realistic exhibition hall parameters, real audio samples, diffuse noise controls, mic imperfections, and beamformed audio playback.

## Execution Phases

The build is split into phases with verification checkpoints to catch issues early.

### Phase 1 + 2: Foundation + Backend

| # | Task | Owner | Status |
|---|------|-------|--------|
| 1.1 | Fix corrupted `run_comparison.py` (restore constants FS/NFFT/HOP/FMIN/FMAX/C + matplotlib import) | Agent | Pending |
| 1.2 | Create `requirements.txt` | Agent | Pending |
| 1.3 | Convert `soundclips/*.mp3` to 16 kHz mono WAV in `audio/` (drone.wav + crowd.wav) | Agent | Pending |
| 1.4 | Verify audio files (check duration, sample rate, mono) | Agent | Pending |
| 2.1 | Build `sim_server.py` (FastAPI backend with `/simulate` endpoint) | Agent | Pending |
| 2.2 | Test server standalone (curl/Python request, confirm valid JSON + audio) | Agent | Pending |

**Checkpoint**: Server returns correct power grid + beamformed audio for a test request.

### Phase 3: Frontend

| # | Task | Owner | Status |
|---|------|-------|--------|
| 3.1 | Add "Live Sim" mode to `array_explorer.html` (UI controls, Run button, spinner, audio player) | Agent | Pending |
| 3.2 | Wire fetch to `/simulate`, render power grid + mic positions + DOA arrows + audio | Agent | Pending |
| 3.3 | Verify end-to-end in browser (all geometries, diffuse toggle, SNR extremes, audio playback) | Agent | Pending |

**Checkpoint**: Full interactive simulation works in browser.

### Audio source files (provided by you)

Raw MP3s in `soundclips/`:
- `soundclips/drone_noise.mp3` -- drone recording
- `soundclips/ambientnoiseconvention1.mp3` -- crowd ambience clip 1
- `soundclips/ambientnoiseconvention2.mp3` -- crowd ambience clip 2 (concatenated with clip 1 for ~3 min total)

> The server falls back to synthetic signals if audio files are missing, so audio conversion and server build are independent.

## Architecture

```
Browser (array_explorer.html)              Python Server (sim_server.py)
┌──────────────────────────────┐          ┌─────────────────────────────────┐
│ "Live Sim" tab               │          │ FastAPI on localhost:8766       │
│  - Room / source / noise     │  POST    │                                │
│    sliders                   │ ──────►  │ 1. Build mic array geometry    │
│  - "Run Simulation" button   │  /sim    │ 2. Create pyroomacoustics room │
│  - Loading spinner           │          │ 3. Add drone + noise sources   │
│                              │  ◄────── │ 4. Simulate + SRP-PHAT        │
│ Three.js renders SRP sphere  │  JSON    │ 5. Beamform toward est. DOA    │
│ <audio> plays beamformed WAV │          │ 6. Return power grid + audio   │
└──────────────────────────────┘          └─────────────────────────────────┘
```

---

## 1. Fix `run_comparison.py`

Line 24 currently reads `iis it` -- a corruption that wiped out the constants block and matplotlib import. Restore:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 16_000
NFFT = 1024
HOP = 512
FMIN = 200.0
FMAX = 2000.0
C = 343.0
```

## 2. `requirements.txt`

```
numpy
pyroomacoustics
matplotlib
fastapi
uvicorn
```

## 3. Audio Files

Raw source files provided in `soundclips/`:
- `soundclips/drone_noise.mp3`
- `soundclips/ambientnoiseconvention1.mp3`
- `soundclips/ambientnoiseconvention2.mp3`

Converted by agent (step 1.3) to `audio/` folder:
- **`audio/drone.wav`** -- 16 kHz, mono (from drone_noise.mp3)
- **`audio/crowd.wav`** -- 16 kHz, mono (ambientnoiseconvention1 + 2 concatenated)

The server normalizes both to unit RMS on load, so original recording levels don't matter.

### How the server uses them

- **Drone**: loaded, normalized, truncated/looped to match integration window, fed to `room.add_source(drone_position, signal=...)`
- **Crowd**: chopped into non-overlapping 1-second segments. Each segment goes to a different noise source position. If more sources than segments, reuse with random circular shift to decorrelate.
- **Fallback**: if files not found, uses synthetic signals (4 sine waves + white noise for drone, Gaussian noise for crowd) -- same as current simulation.

## 4. `sim_server.py` -- FastAPI Backend

Single Python file in project root. Has its own parameterized geometry builders (no modification needed to `run_comparison.py`'s existing builders).

### Endpoint: `POST /simulate`

**Request body (JSON):**

| Parameter | Type | Range | Default | Notes |
|-----------|------|-------|---------|-------|
| `geometry` | string | UCA / CROSS / ULA / CYLINDER | UCA | |
| `mic_count` | int | 4-32 | 12 | |
| `radius` | float | 0.03-1.0 m | 0.15 | Half-length for CROSS, total length = 2x for ULA |
| `ring_separation` | float | 0.02-0.50 m | 0.12 | Cylinder only |
| `room_length` | float | 10-100 m | 50 | |
| `room_width` | float | 10-80 m | 40 | |
| `room_height` | float | 4-15 m | 12 | |
| `rt60` | float | 0.0-2.5 s | 1.5 | 0 = anechoic |
| `source_az_deg` | float | 0-355 | 60 | |
| `source_el_deg` | float | -90 to 90 | 30 | |
| `source_distance` | float | 2-20 m | 6 | |
| `snr_db` | float | -20 to 30 | 0 | |
| `diffuse` | bool | | true | Enable crowd + PA noise |
| `crowd_count` | int | 0-50 | 30 | |
| `pa_count` | int | 0-12 | 8 | |
| `mic_mismatch` | bool | | false | +/-1 dB gain, +/-2 deg phase per channel |
| `integration_ms` | int | 50-1000 | 1000 | Signal duration in ms |

**Processing logic:**

1. Build mic array from geometry params
2. Compute array center as centroid of mic positions
3. Create room: `pra.AnechoicRoom(3)` if rt60=0, else `pra.ShoeBox` with `inverse_sabine` (max_order capped at 6 for large rooms)
4. Place drone source at (az, el, distance) from array center
5. If diffuse: place crowd sources (random perimeter positions at z=1.5m) + PA sources (random wall-adjacent positions at z=3.0m)
6. Load drone.wav / crowd.wav (or fall back to synthetic), normalize to unit RMS
7. Chop crowd audio into non-overlapping segments, assign one per noise source
8. Scale noise source amplitudes to achieve target SNR at the array
9. If mic_mismatch: apply random per-channel gain (+/-1 dB) and phase (+/-2 deg) after simulation
10. Compute STFT, run SRP-PHAT on 72x19 spherical grid (200-2000 Hz)
11. Beamformed audio: delay-and-sum toward estimated DOA, sum channels, normalize, encode as 16-bit WAV -> base64
12. Return JSON response

**Response body (JSON):**

```json
{
  "power": [[...], ...],          // 19x72 (colatitude x azimuth)
  "est_az_deg": 62.5,
  "est_el_deg": 28.0,
  "true_az_deg": 60.0,
  "true_el_deg": 30.0,
  "mic_positions": [[x,y,z], ...],  // relative to array center
  "audio_b64": "UklGR...",          // base64 WAV
  "elapsed_s": 4.2,
  "params": { ... }                 // echo of input
}
```

**CORS**: enabled for all localhost origins.

### Performance considerations

- Large room (50x40x12) at RT60=2.0s will generate many image sources. `max_order` capped at 6 keeps single-trial runtime under ~10 seconds.
- The server auto-computes a feasible max_order: `min(inverse_sabine_order, 6)`.
- If `inverse_sabine` fails (room too large for requested RT60), cap at max_order=3 with best-effort absorption.

## 5. `array_explorer.html` -- Live Sim Mode

### New UI elements

- **Third toggle button**: "Live Sim" alongside existing "Beam Pattern" and "SRP Simulation"
- **`#liveControls` panel** (visible only in Live Sim mode):
  - Geometry selector + mic count + radius + ring separation (same sliders as beam mode, shared)
  - Room: length, width, height sliders
  - RT60 slider (0.0-2.5s)
  - Source: azimuth, elevation, distance sliders
  - SNR slider (-20 to 30 dB)
  - Diffuse noise: toggle checkbox + crowd count + PA count sliders
  - Mic mismatch: toggle checkbox
  - Integration window: slider (50-1000ms)
- **"Run Simulation" button** -- does NOT auto-fire on slider change
- **Loading spinner** overlay while waiting for server
- **Audio player**: `<audio>` element with play/pause, appears after simulation completes

### Rendering on response

- Display returned `power` grid on 3D sphere using existing `buildPatternMesh`
- Place mic markers from returned `mic_positions`
- Show steer arrow toward estimated DOA (yellow) and true DOA (green, second arrow)
- Info panel: true vs estimated angles, azimuth/elevation errors, computation time, room dimensions, RT60, SNR

### Audio playback

- Decode base64 WAV string -> Blob -> object URL
- Set as `src` on `<audio>` element
- Show play button below info panel

## 6. Verification

- Start sim_server.py on port 8766
- Serve results/ on port 8765 (existing HTTP server)
- Open array_explorer.html, switch to Live Sim tab
- Run simulation with default params (UCA, 50x40x12 room, RT60=1.5, SNR=0, diffuse on)
- Confirm: SRP sphere renders, DOA arrow appears, info panel shows errors, audio plays
- Test each geometry, toggle diffuse off, adjust SNR extremes
