# Acoustic Visualiser

The project has one Vite frontend source under `app/` and one FastAPI backend in
`sim_server.py`. Do not maintain a separate HTML implementation under
`results/`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt
cd app
npm install
```

## Browser Development

Start the backend:

```powershell
.venv\Scripts\python.exe sim_server.py
```

Start the frontend in another terminal:

```powershell
cd app
npm run dev -- --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/`.

## Hardware-Free Heimdall Demo

The firmware emulator speaks the same parsed protocol and uses the same
WebSocket path as serial hardware:

```powershell
.venv\Scripts\python.exe sim_server.py --emulator
```

Open the Hardware tab and use `Full Sweep`, `Continuous`, `Track`, `Stop`, or
direct sector steering. `Steer` fixes the beam and reports one level; `Measure`
samples that fixed beam again without rewriting its delays. The emulator
supplies a moving target over the current 7x7 grid and is suitable for GUI,
protocol, reconnect, and command testing.

The Hardware tab can also start or replace the emulator at runtime without
restarting the backend.

Expand `Emulator Grid` before clicking `Start Emulator` to set 1-20 rows and
columns plus azimuth/elevation limits. These use the same cell-center convention
as `beamforming_mathematics/hemisphere_scan_visualizer.py`, so they are useful
for testing candidate table layouts before generating firmware coefficients.

`Monitor Beam` steers once and repeatedly issues `M` at 10, 20, or 50 Hz. Only
the selected sector updates because the beam remains fixed; the yellow timeline
trace shows its level as the emulated source moves. Starting a sweep, tracking,
manual steering, or disconnecting stops monitoring automatically.

## Serial Hardware

Start the backend normally, open the Hardware tab, enter the COM port, select
921600 or 115200 baud, and click `Connect`. Serial transports can be replaced at
runtime through `/hw_connect`; the simulation backend does not restart.

The host sends `I` on connection so a late GUI receives:

```text
READY,2D,<rows>,<columns>,<sectors>,<microphones>
AZIMUTH,<count>,...
ELEVATION,<count>,...
```

The UI supports generated grids up to 20x20 and forwards `S/F/C/G/X/I/M` firmware
commands over the existing `/realtime_hw` WebSocket.

To test a layout on real hardware, generate `beam_table_2d.h` in the sibling
`beamforming_mathematics` visualizer, replace the MAX78002 firmware copy, perform
a clean build, flash it, and reconnect. The firmware `I` response makes the GUI
adopt the new rows, columns, and center angles automatically.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q `
  tests\test_heimdall_protocol.py tests\test_heimdall_api.py

cd app
npm run build
```

The backend tests cover strict protocol parsing, dynamic dimensions, emulator
commands, FastAPI lifecycle, and WebSocket frames.

Browser mode is the supported development and hardware bring-up path because
backend logs and failures are directly visible.
