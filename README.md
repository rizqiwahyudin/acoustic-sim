# Acoustic Visualiser

The project has one Vite frontend source under `app/` and one FastAPI backend in
`sim_server.py`. Do not maintain a separate HTML implementation under
`results/`.

## Setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
cd app
npm.cmd ci
```

Validated prerequisites are 64-bit Python 3.14, Node.js 24, and npm 11. The
repository contains the required audio files and `app/data/srp_spectra.json`.
PyRoomAcoustics currently installs from a Windows wheel; a C++ build toolchain
is needed only if pip cannot obtain a compatible wheel.

## Browser Development

Start the backend:

```powershell
.\.venv\Scripts\python.exe sim_server.py
```

Start the frontend in another terminal:

```powershell
cd app
npm.cmd run dev -- --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/`.

## Hardware-Free Heimdall Demo

The firmware emulator speaks the same parsed protocol and uses the same
WebSocket path as serial hardware:

```powershell
.\.venv\Scripts\python.exe sim_server.py --emulator
```

Open the Hardware tab and use `Full Sweep`, `Continuous`, `Track`, `Stop`, direct
sector steering, or `Monitor Beam`. The emulator supplies a moving target over
the current 7x7 grid and is suitable for GUI, protocol, reconnect, and command
testing.

Adaptive `Track` follows the firmware policy: one complete search, two local
cross confirmations, five-sector tracking with two-pass movement hysteresis,
three failed passes before a local 3x3 reacquisition, and a periodic global
search after 50 tracking passes. The source trajectory only generates detector
levels; it is not used directly to choose the tracked sector.

The target panel compares `TRACK UPDATE` with `FULL SCAN`. Track update is the
host-observed cadence of `TARGET_UPDATED` records; full scan comes from the most
recent firmware `TIMING` record. The displayed multiplier shows the practical
solution-refresh advantage of local tracking. Full-scan rate is expected to
remain nearly constant while tracking.

### Emulator flight profiles

Flight motion is generated in 3D Cartesian metres and then converted to
azimuth, elevation, and range. The tracker receives only simulated sector
levels. It does not receive the known trajectory coordinates.

| Profile | Nominal 1x behavior |
| --- | --- |
| Crossing pass | Smooth lateral pass, approximately 14 m/s and 36-60 m range. |
| Approach and retreat | Closing/receding path, approximately 13 m/s and 16-83 m range. |
| Orbit | Smooth offset orbit, approximately 9 m/s and 27-52 m range. |
| Patrol | Slower multi-axis route, approximately 8 m/s and 31-68 m range. |
| Evasive stress | Multi-frequency maneuvers up to approximately 27 m/s, wide angular motion, and a 0.9 s acoustic dropout every 17 profile seconds. |
| Legacy angular sweep | Original fixed-range sinusoidal motion for regression comparison. |

`Flight speed` scales trajectory time. At 2x, speeds double and evasive dropout
intervals/durations halve in wall-clock time. Received pressure uses a simplified
$1/r$ amplitude law referenced to 30 m and capped at close range. This is a
tracking stress model, not a room-acoustics or drone-signature simulation.

### Cached acoustic emulator

Hardware -> Advanced controls -> Emulator model also offers `Acoustic room
(cached)`. It replaces the fast Gaussian level provider with cached detector
levels generated from `drone.wav`, `crowd.wav`, a 48 kHz room model, the exact
44-microphone DSP-order geometry, the deployed `beam_table_2d.h` delays, and the
exported SigmaStudio 1 kHz high-pass / 4 kHz low-pass FIR stages. It still uses the same firmware-equivalent
commands and adaptive state machine.

The detector filter uses the exported order-10 FIR coefficient prototypes.
Stage 1 is applied with the High Pass spectral-inversion topology visible in
SigmaStudio; Stage 2 is applied directly as Low Pass. The unity-gain output uses
the 44-channel sum before the exported 10,000 dB/s RMS envelope, matching the
current DSP graph. Because the filters are only order 10, the effective exported
cascade is gradual rather than an ideal 1-4 kHz brick-wall response (about
-4.8 dB at DC, -5.0 dB at 1 kHz, -7.2 dB at 4 kHz, and -17.2 dB at 8 kHz).
Absolute ADC/microphone scaling remains uncalibrated.

Available scenarios:

| Scenario | Contents |
| --- | --- |
| Conference room · evasive 2x | 20 x 40 x 10 m, vertical array at `[10, 0.5, 1]` m, RT60 1.2 s, 30-second room-bounded evasive loop, 12-direction diffuse crowd, and an off-axis speech source at +25 degrees / 12 m / 1.5 m high. |
| Handheld 2 m · drone only | Drone speaker follows a slow 12-second XZ figure-eight at fixed global Y=2.5 m: 2 m broadside depth, +/-0.75 m lateral travel, 1.2-2.0 m height, no crowd or speech. |
| Handheld 2 m · crowd + speech | Identical fixed-depth handheld motion with the conference crowd field and directional speech source enabled. |
| Stationary speaker · broadside 2 m | Drone recording only, broadside at 2 m, using its time-averaged spectrum for repeatable sequential sector comparison with the physical speaker setup. |

Select a scenario and click `Prepare Scene`. The first conference preparation
took 57-88 seconds on the development machine; stationary preparation took
9-14 seconds. The 16-waypoint handheld caches took 60 seconds drone-only and
94 seconds mixed. Completed caches are content-addressed under
`.cache/heimdall_acoustic/`, are approximately 1.18 MB each, and reload in tens
of milliseconds. Preparation reports stage, overall progress, and RIR ETA and
can be canceled between RIR calculations.

Rows, columns, and azimuth/elevation FOV are adjustable from 1x1 through 20x20
for both simulator models. The default 7x7 acoustic grid uses the exact deployed
delay table and is labeled `Deployment parity`. Other acoustic grids generate
quantized simulation-only delays and are labeled `Exploratory acoustic`; they
do not modify firmware or the serial Connect path. Room propagation is cached
separately from beam projection, so changing only grid/FOV reuses the expensive
RIR result. Measured stationary reprojection took about 0.4 s for 9x11 and 1.0 s
for 20x20 after a 3.0 s room build.

The fixed-depth handheld diagnostic preserves the real sequential sector scan.
With the 2.8 ms emulator sector time and audited exported-filter/summed-array
path, completed 49-sector sweeps localized within one grid cell of scenario
truth in 65.1% of drone-only sweeps and 49.5% of mixed crowd/speech sweeps.
Coherent all-sector reference frames scored 96.9% and 86.2%, respectively. The
source remained at 2 m broadside depth; its true
slant range varied from 2.05 to 2.30 m as it moved laterally and vertically.

After status becomes `Ready`, click `Acoustic`, then use the normal
`Full Sweep`, `Continuous`, `Track`, `Stop`, `Steer`, and `Monitor Beam`
controls. Diagnostics identify `acoustic-emulator`, scenario, measured RT60,
and uncalibrated status.

The Hardware view keeps grid ownership explicit. The fast kinematic emulator
can use an arbitrary 1-20 row/column grid and FOV. Acoustic mode is read-only
because its 7x7 grid, sector angles, delays, and cached responses come from the
generated deployment contract. Serial mode is also read-only because the
connected firmware reports the grid compiled into its beam table; changing it
requires regenerating the table and reflashing the device.

Acoustic scenarios also show a separate `SCENARIO TRUTH` readout and green
crosshair/sphere. It follows the same cache clock and interpolated room-response
waypoints as the detector levels, and reports true azimuth/elevation, range,
room position, loop time, acoustic dropout state, and angular error from the
current measured solution. This truth is emulator-only and is never fed to the
firmware-equivalent tracker.

`AUDIO AUDITION` renders a jointly normalized 3-second comparison at the
current scene time. Available outputs are the generated filtered mix, one room
microphone, the unsteered 44-microphone sum, a truth-steered beam, and the
currently selected/tracked beam. Rendering uses the cached complex microphone
transfer data and is a simulator diagnostic; it does not claim sample-exact
ADAU1467 output. The shared gain preserves relative loudness between clips.

The room model uses native 48 kHz audio, polyphase resampling, order-2 geometric
early reflections, and a deterministic diffuse tail calibrated to approximately
RT60 1.2 s. It evaluates all sectors from the same acoustic scene; sectors are
not separate virtual sources. The conference cache measured RT60 1.18 s and the
stationary cache 1.20 s in the final build.

Absolute SPL-to-ADAU1467 digital scaling remains nominal. The acoustic emulator
is useful for relative sector ranking, reflection/interference stress, and
tracking behavior, but it is not calibrated for detection range or probability.
Its moving-source response interpolates cached waypoint power responses and does
not reproduce continuous Doppler phase.

The Hardware tab can also start or replace the emulator at runtime without
restarting the backend.

Enable `Advanced controls` before clicking `Emulator` to set 1-20 rows and
columns plus azimuth/elevation limits. These use the same cell-center convention
as `beamforming_mathematics/hemisphere_scan_visualizer.py`, so they are useful
for testing candidate table layouts before generating firmware coefficients.

`Monitor Beam` steers once and repeatedly issues `M` at 10, 20, or 50 Hz. Only
the selected sector updates because the beam remains fixed; the yellow timeline
trace shows its level as the emulated source moves. Starting a sweep, tracking,
manual steering, or disconnecting stops monitoring automatically.

The Hardware console distinguishes a tracked target, fixed beam, and strongest
cached observation. Hatched cells are more than five seconds old; an old cell is
not a current detection. `Peak-to-next` is the difference between the two
strongest cached cells, not a statistical confidence value. The system estimates
direction only and does not estimate range.

The `dB Floor` control is an absolute dBFS color floor with 0 dBFS as the hot
end of the scale. The display does not normalize every frame to its own peak;
uniformly weak measurements and evasive-profile dropouts therefore remain dark
instead of appearing as a false full-grid detection.

`Freeze Display` pauses WebSocket presentation only. Firmware or emulator work
continues; use `Stop` to send the firmware `X` command. Disconnect and transport
replacement also attempt a best-effort `X` before closing the link.

## Serial Hardware

Start the backend normally, open the Hardware tab, enter the COM port, select
921600 or 115200 baud to match the firmware build, and click `Connect`. No other
serial terminal may hold that COM port. Serial transports can be replaced at
runtime through `/hw_connect`; the simulation backend does not restart.

The host sends `I` on connection so a late GUI receives:

```text
READY,2D,<rows>,<columns>,<sectors>,<microphones>
AZIMUTH,<count>,...
ELEVATION,<count>,...
```

The UI supports generated grids up to 20x20 and forwards `S/F/C/G/X/I/M` firmware
commands over the existing `/realtime_hw` WebSocket.

The acoustic emulator is intentionally tied to the deployed 7x7 machine-readable
contract under `data/heimdall_acoustic_contract.json`. It refuses to start if
the contract's recorded C-header hash differs from the actual firmware
`beam_table_2d.h`. Regenerate and copy both C and JSON artifacts after changing
the deployment table.

To test a layout on real hardware, generate `beam_table_2d.h` in the sibling
`beamforming_mathematics` visualizer, replace the MAX78002 firmware copy, perform
a clean build, flash it, and reconnect. The firmware `I` response makes the GUI
adopt the new rows, columns, and center angles automatically.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q `
  tests\test_heimdall_protocol.py tests\test_heimdall_api.py

cd app
npm.cmd run build
```

The backend tests cover strict protocol parsing, dynamic dimensions, emulator
commands, FastAPI lifecycle, and WebSocket frames.

Browser mode is the supported development and hardware bring-up path because
backend logs and failures are directly visible.

If `app/dist` is already current, Node.js is not required at runtime. Serve the
built frontend in a second terminal:

```powershell
.\.venv\Scripts\python.exe -m http.server 8080 --directory app\dist
```
