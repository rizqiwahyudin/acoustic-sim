# Acoustic Beamformer Simulation -- Technical Reference

> **Phase 1 audio model update.** This guide describes the pre-Phase-1 simulation. As of the SPL overhaul:
>
> - Source levels are specified in **dB SPL at 1 m** (drone 78, crowd 67, PA 80 defaults) via `acoustic_utils.spl_to_amplitude()`; the old `snr_db` knob is gone.
> - The mic noise floor is a **dB SPL** slider (default 30) that maps to `sigma2_awgn`.
> - Air absorption (ISO 9613-1, 20 °C / 50 % RH) is enabled whenever `rt60 > 0`.
> - The first 100 ms of every simulation is trimmed to remove RIR-onset artefacts.
> - Mic mismatch uses **FIR-based fractional delays + gain ripple + DC offset** (`apply_mic_mismatch_v2`), replacing the old Hilbert hack.
> - The live server emits **three audio clips** (raw mic / unsteered average / beamformed) with a single joint peak scale so relative loudness is preserved.
> - `run_comparison.py` renamed its `snr_db` CSV column to `drone_spl_db`.
>
> See **[LIMITATIONS.md](LIMITATIONS.md)** for what the simulation still *cannot* capture and how to calibrate results against reality. A full rewrite of this guide is deferred to Phase 2.

> **Phase 2a realism + QoL update.** On top of the Phase 1 audio model:
>
> - **Per-wall materials.** Room absorption can be specified per surface (floor / ceiling / four walls) using pyroomacoustics' material database instead of a single RT60 knob. The curated "Exhibition Hall" preset (hairy carpet, fissured-tile ceiling, plasterboard walls, heavy cotton curtains on the stage side) is available via both the live toggle and `run_comparison.py --materials-profile exhibition_hall`. The response carries the **measured** `rt60_actual` back to the frontend.
> - **Hardware impairments.** The mic signal path now supports optional microphone **crosstalk** (`apply_crosstalk`, configurable coupling in dB) and **codec / ADC quantization** (`apply_codec_quantization`, 8/12/16/24-bit). A one-click "Hardware-realistic" preset enables mismatch + crosstalk@-40 dB + 16-bit together.
> - **SRP-PHAT heatmap + top-N peaks.** The live response includes the full 2-D (azimuth × colatitude) power grid (rendered as a Canvas2D heatmap) and the **top-3 DOA candidates** with 15° non-maximum suppression, so UCA mirror ambiguity and side-lobes are visible directly instead of inferred.
> - **Save / Load / URL-share presets.** The entire sim configuration can be exported to JSON, loaded back from JSON, or copied as a shareable URL hash. Presets are applied after the async `/materials` fetch so dropdowns restore correctly.
> - **WAV downloads.** Raw-mic / unsteered-average / beamformed audio are all downloadable next to the existing playback toggle, sharing the object URLs that already power playback.
> - **Live-vs-batch parity test.** `pytest tests/test_parity.py` runs the same clean trial through `sim_server.run_live_trial()` and `run_comparison.run_single_trial()` and asserts the DOA estimates agree to within 1°. This is the guard against future refactors silently desynchronising the two pipelines.
>
> The existing RT60 slider and the legacy `run_comparison.py` sweep still work unchanged; Phase 2a extends them, it does not replace them.

> **Phase 2b realism update.** On top of Phase 2a, the simulation now models a **moving drone source** and **first-order atmospheric realism**:
>
> - **Moving drone source (live sim only).** A new checkbox on the Live Sim panel renders the drone along a **straight-line** or **circular-arc** trajectory instead of pinning it to one point. The audio is split into `K` chunks (slider, 4 – 16) joined by **Hann-squared-half crossfades** that sum exactly to 1.0, and each chunk is added as its own static source at the interpolated drone position inside a single `room.simulate()` call. This captures both in-window Doppler shift and gradual DOA drift without any changes to the SRP-PHAT core. The 3D viz now draws the trajectory as a polyline with a wireframe start sphere and a solid end sphere, and reports speed (m/s) and heading (deg, for straight-line trajectories) in the info panel. `run_comparison.py` intentionally does **not** implement moving sources -- batch is a static-source sweep by design.
> - **Explicit temperature / humidity.** Both live and batch now thread the configured temperature (°C) and relative humidity (%) directly into pyroomacoustics' ISO 9613-1 air absorption model via `acoustic_utils.air_absorption_kwargs(T, RH)`. The "Exhibition Hall" preset bumps to 22 °C / 55 % RH. The batch script exposes `--temperature` and `--humidity` CLI flags so full sweeps can be rerun under a different atmosphere.
> - **Vertical temperature gradient.** A new slider (°C/m) applies a **first-order analytic beam-bending bias** to the source z-coordinate used by the room simulation (`atmospheric_z_bias`) and separately reports the resulting apparent elevation bias in degrees (`atmospheric_elevation_bias_deg`) in the info panel. At typical indoor-venue gradients (±1-2 °C/m over ~8 m of horizontal distance) this is a fraction of a degree, so you'll usually see it numerically before it shifts the argmax of the 10°-stepped SRP grid. Batch exposes this as `--temp-gradient`.
> - **New parity & invariant tests.** `tests/test_parity.py` gains three cases: live-vs-batch parity with a non-default atmosphere, chunk-crossfade reconstruction to float epsilon, and `speed_mps=0` moving-source equivalence to the static case.
>
> See **[LIMITATIONS.md §2b](LIMITATIONS.md)** for the exact list of what Phase 2b models, what it still simplifies, and where the model should be calibrated against real recordings.

> **Phase 3 realism update.** On top of Phase 2b, the simulator adds three optional models that default to the Phase 2b baseline so existing sweeps and tests stay byte-compatible:
>
> - **MAX78000 fixed-point ML-path preview.** New Live Sim checkbox "MAX78000 ML preview (Phase 3)" and `run_comparison.py --ml-preview` flag. Re-quantizes the beamformed audio to **int8 or int16**, runs a proxy **log-mel-dB feature extractor** (configurable `n_mels`, defaults to 64 over the 200-2000 Hz band), optionally re-quantizes the feature tensor, and reports `ml_path_snr_db` (audio quantization SNR) + `feature_snr_db` (feature quantization SNR) in the response. The live UI additionally shows an inline spectrogram PNG and a fourth "ML .wav" download next to Raw / Unsteered / Beam. The batch script writes both SNRs as new CSV columns when `--ml-preview` is active. See [LIMITATIONS.md §2c](LIMITATIONS.md) for the log-mel-proxy caveats.
> - **Isotropic-diffuse plane-wave crowd model.** `crowd_model="plane_wave"` (dropdown in the Live Sim diffuse block, or `--crowd-model plane_wave` in batch) replaces the old discrete-point-source crowd with a sum of `n_plane_waves` uncorrelated plane waves (default 64) applied directly to the mic signals via frequency-domain fractional delays. PA speakers stay as localised point sources. This reproduces the textbook `sinc(k·d)` isotropic-diffuse inter-mic coherence curve.
> - **FIR capacitive-coupling crosstalk.** `crosstalk_model="fir_capacitive"` (new dropdown inside the Crosstalk details panel, or `--crosstalk-model fir_capacitive`) swaps the Phase 2a flat neighbour-leakage model for a **1-pole high-pass** leakage path (corner Hz configurable). Setting the corner frequency to near zero collapses back to the flat model, so the FIR path is a strict superset. A `--crosstalk-fir-path` hook lets future measured-pair FIRs drop in without a code change.
>
> `tests/test_parity.py` gains five Phase 3 cases (default-parity, plane-wave coherence, FIR HPF behaviour, FIR-collapses-to-flat, int8-vs-int16 SNR). `tests/_smoke_live.py` gains three HTTP check blocks `[i]/[j]/[k]`. The "Exhibition Hall" preset is unchanged -- it keeps the defaults (`point_source` crowd, `simple` crosstalk, ML preview off) so it remains the known-baseline button.
>
> ### How to preview the MAX78000 ML path
>
> - **UI:** on the Live Sim panel, tick "MAX78000 ML preview (Phase 3)" under the codec-quantization row. Pick `Audio bit depth` (8 or 16), `Feature bit depth` (8 or 16), and `Mel bands` (32 / 48 / 64 / 96). Run a simulation -- you'll see an extra "ML .wav" download, an inline log-mel spectrogram PNG, and two new info-panel lines (`ML path SNR` and `ML feature SNR`).
> - **CLI:** `python run_comparison.py --ml-preview --ml-bit-depth 8 --ml-feature-bit-depth 8 --ml-n-mels 64 --test` -- the sweep CSV (`results/metrics.csv` or `results/metrics_materials.csv`) gains `ml_path_snr_db` and `feature_snr_db` columns.
>
> ### How to switch crowd-noise correlation model
>
> - **UI:** in the Live Sim diffuse block, change the "Crowd model (Phase 3)" dropdown from "Point sources (2b)" to "Plane-wave diffuse (3B)". A sub-slider appears for the plane-wave count.
> - **CLI:** `python run_comparison.py --crowd-model plane_wave --n-plane-waves 64 --test`.
>
> ### How to switch crosstalk model
>
> - **UI:** enable "Crosstalk", then inside the details panel change "Model (Phase 3)" from "Simple (flat)" to "FIR capacitive". The corner-frequency slider appears only in FIR mode.
> - **CLI:** `python run_comparison.py --crosstalk-model fir_capacitive --crosstalk-corner-hz 500 --crosstalk-coupling-db -40 --test`.
>
> See **[LIMITATIONS.md §2c](LIMITATIONS.md)** for the remaining simplifications (log-mel is a proxy, finite plane-wave count, analytic 1-pole FIR).

> **Phase 3+ DOA-band update.** The SRP-PHAT frequency band is no longer hardcoded at 200-2000 Hz. Both live and batch now expose three new knobs that default to the Phase 3 baseline so existing sweeps and URL presets stay byte-compatible:
>
> - **`fmin_hz` / `fmax_hz` sliders and CLI flags.** Narrow the DOA-integration band to the drone-relevant slice of spectrum. Default 200-2000 Hz; typical tuning for small drones is 300-2500 Hz, which drops HVAC rumble and most codec-artefact-band energy.
> - **Harmonic-comb weighting.** A checkbox + "Drone fundamental f0" slider that restricts SRP-PHAT to +/- 10 Hz windows around n*f0 (n = 1..20) inside the band. Imitates a matched-filter front-end tuned to a known drone's blade-pass fundamental and its rotor harmonics. Empty comb falls back silently to the flat band; the response dict reports the actual `n_freq_bins` used so users can confirm the comb actually narrowed the set.
>
> Both routes share `acoustic_utils.build_freq_bin_mask` so live and batch pick identical bins for matched parameters; `tests/test_parity.py` has four new cases (defaults-parity, narrowing, comb-subset, end-to-end harmonic lock) plus one new HTTP smoke-test block `[l]`. The Exhibition Hall preset resets these knobs to 200-2000 Hz flat so it remains the known baseline.
>
> ### How to tune the DOA band
>
> - **UI:** on the Live Sim panel, the "DOA band (Phase 3+)" block sits right below Atmosphere. Drag `DOA fmin` and `DOA fmax` to the drone-relevant slice (try 300 / 2500 Hz for small drones). Tick "Harmonic comb weighting" to engage matched-filter mode; the "Drone fundamental f0" slider appears below (80-400 Hz). The info panel reports `DOA band: <fmin>-<fmax> Hz` and, when the comb is on, the active `f0` and final bin count.
> - **CLI:** `python run_comparison.py --fmin 300 --fmax 2500 --harmonic-comb --drone-fundamental 200 --test` for a batch sweep under the tuned band.
>
> See **[LIMITATIONS.md §2d](LIMITATIONS.md)** for the remaining simplifications (rectangular +/- 10 Hz weighting, user-supplied f0, no sub-band voting, no MVDR adaptation).

This document describes the simulation system used to evaluate microphone array geometries for acoustic direction-of-arrival (DOA) estimation. It covers the physical models, algorithms, design decisions, known limitations, and the scope of valid conclusions.

---

## 1. Overview

This system simulates a drone emitting sound in a reverberant room with background noise, captured by a microphone array. It then runs the SRP-PHAT algorithm to estimate the drone's direction and compares the estimate against ground truth.

The primary purpose is to **compare array geometries** (circular, standing cross, linear, cylinder) under controlled acoustic conditions -- varying reverberation, noise, and source position -- to determine which geometry is best suited for a convention/exhibition hall deployment.

There are two simulation paths:

- **Batch sweep** (`run_comparison.py`): systematically tests all geometries across a matrix of RT60, SNR, azimuth, and elevation values. Produces CSV results and static plots.
- **Interactive live simulation** (`sim_server.py` + `results/array_explorer.html`): runs a single configurable trial on demand via a browser interface with 3D visualization.

---

## 2. System Architecture

The interactive system has two processes:

| Process | Role | Port |
|---------|------|------|
| `sim_server.py` (FastAPI + Uvicorn) | Runs the acoustic simulation on demand | 8766 |
| Python HTTP server | Serves the static HTML/JS frontend | 8080 |

The frontend (`array_explorer.html`) has three modes:

1. **Beam Pattern** -- computes and displays the theoretical array response in the browser using narrowband delay-and-sum math. No server call required.
2. **SRP Simulation** -- displays pre-computed batch simulation results embedded as JSON in the HTML file. No server call required.
3. **Live Sim** -- sends a JSON POST to `http://127.0.0.1:8766/simulate` with all configurable parameters. The server returns a JSON response containing the SRP-PHAT power grid, DOA estimates, mic positions, room geometry, source positions, image source positions, the room impulse response, and base64-encoded beamformed audio.

---

## 3. Room and Environment Model

### Room geometry

Rooms are modeled as rectangular boxes using pyroomacoustics' `ShoeBox` class, which implements the **Image Source Method (ISM)**. ISM computes reflections by mirroring the sound source across each wall surface, creating "image sources" that represent the reflected sound paths. The room is axis-aligned with corner at the origin.

Default room dimensions:

| Mode | Length (x) | Width (y) | Height (z) |
|------|-----------|-----------|------------|
| Live | 50 m | 40 m | 12 m |
| Batch | 20 m | 15 m | 10 m |

The live defaults are sized to approximate a large exhibition hall (e.g., Embedded World scale). All dimensions are adjustable via the frontend sliders.

### Reverberation

Reverberation time (RT60) is the time for sound energy to decay by 60 dB after the source stops. It is mapped to wall absorption coefficients via pyroomacoustics' `inverse_sabine` function, which also returns the required ISM reflection order.

The reflection order is capped:

- **Live**: `max_order = min(max_order, 6)`
- **Batch**: `max_order = min(max_order, 10)`

This cap limits computation time at the cost of truncating late-arriving reflections. For RT60 values above ~1.5s in the live server, the truncation means the simulated reverb tail may be shorter than the specified RT60.

When RT60 is set to 0, an anechoic room is used instead (`pra.AnechoicRoom`), which models free-field propagation with no reflections.

If `inverse_sabine` fails (can happen for extreme RT60/room-size combinations), the live server falls back to `e_absorption = 0.5, max_order = 3`; the batch script skips the trial entirely.

### Array placement

The microphone array is placed at the horizontal center of the room at a fixed height of 1.0 m:

```python
array_center = room_dim / 2
array_center[2] = 1.0
```

### Additive white Gaussian noise

Background noise power is set via:

$$\sigma^2 = \frac{10^{-\text{SNR}_{\text{dB}} / 10}}{(4\pi \cdot d)^2}$$

where $d$ is the source distance. This formula incorporates geometric spreading loss $(4\pi d)^2$, meaning the SNR parameter represents the signal-to-noise ratio referenced to the source power at distance $d$, not at the microphone. This is passed to pyroomacoustics as `sigma2_awgn`, which adds spatially white noise independently to each microphone channel.

---

## 4. Array Geometries

Four geometries are implemented. All are constructed relative to `array_center` (room center at z=1.0m).

### UCA (Uniform Circular Array)

A flat ring of microphones in the x-y plane at `z = array_center[2]`. Uses `pra.circular_2D_array` with `phi0 = 0`.

- Default: 12 mics, radius 0.15 m, adjacent spacing ~7.8 cm
- Provides 360-degree azimuth coverage
- **Cannot resolve elevation** -- the array has no vertical aperture, so a source at elevation +30 degrees produces the same time-delay pattern as one at -30 degrees (mirror ambiguity)

### Standing Cross

Two perpendicular arms: one along the **x-axis** (horizontal) and one along the **z-axis** (vertical). Both pass through the array center.

```python
# Horizontal arm
pts.append([center[0] + offset, center[1], center[2]])
# Vertical arm
pts.append([center[0], center[1], center[2] + offset])
```

- `mics_per_arm = mic_count // 2`
- Offsets are linearly spaced from `-half_length` to `+half_length` with the center point excluded (`abs(offset) > 1e-9`)
- `half_length` equals the `radius` parameter (default 0.15 m), so total arm span is 0.30 m
- The vertical arm provides elevation discrimination

### ULA (Uniform Linear Array)

A single line of microphones along the x-axis.

- Total length = `2 * radius` (0.30 m by default), evenly spaced
- Can resolve azimuth only along its axis; has a cone-of-confusion ambiguity for sources at equal angles above/below or in front/behind the array
- Included as a baseline reference

### Cylinder (Stacked Rings)

Two horizontally-oriented circular rings stacked vertically, with the top ring angularly staggered.

```python
mics_per_ring = mic_count // 2
bot_z = center[2] - separation / 2
top_z = center[2] + separation / 2
# Top ring rotated by pi/mics_per_ring
```

- Default: 12 mics total (6 per ring), radius 0.15 m, ring separation 0.12 m
- The angular stagger avoids redundant microphone positions between rings
- The vertical separation provides elevation resolution, similar to the standing cross but with full 360-degree azimuth coverage on both rings

### Parameter mapping

The `radius` slider in the frontend maps to different physical quantities depending on geometry:

| Geometry | `radius` parameter becomes |
|----------|---------------------------|
| UCA | Circle radius |
| Cross | Half-length of each arm |
| ULA | Half the total array length (`length = radius * 2`) |
| Cylinder | Circle radius of each ring |

---

## 5. Signal Sources

### Drone (primary source)

Positioned in 3D space relative to the array center using spherical coordinates:

$$x = x_c + d \cos(\theta_{el}) \cos(\theta_{az})$$
$$y = y_c + d \cos(\theta_{el}) \sin(\theta_{az})$$
$$z = z_c + d \sin(\theta_{el})$$

The position is then clipped to stay at least 0.3 m (the `MARGIN` constant) from any room wall.

**Audio content**: If `audio/drone.wav` exists, a random segment of the required duration is extracted. The WAV file is loaded at startup, converted to 16 kHz mono, and RMS-normalized. If the file is shorter than the requested duration, it is tiled (repeated). If the file does not exist, a synthetic signal is used: sum of four sinusoids at 150, 300, 600, and 900 Hz (with random initial phases) plus Gaussian noise at 30% amplitude, peak-normalized.

### Crowd noise (diffuse field)

Crowd noise sources are distributed along the floor perimeter of the room at z = 1.5 m (live) or z = 1.2 m (batch). The positions are equally spaced along the inner rectangular perimeter (inset by `MARGIN` from walls).

Each source gets an independent audio segment chopped from `audio/crowd.wav`. When more sources are needed than available contiguous segments, the excess segments are created by circular-shifting (`np.roll`) an existing segment by a random amount, providing approximate decorrelation. If no WAV file is available, each source gets independent Gaussian white noise.

Crowd source signals are scaled by **0.3x** before being added to the room.

### PA speakers (diffuse field)

PA sources are placed adjacent to walls:

- **Live**: at z = 3.0 m, randomly distributed across the four walls (fixed seed 42 for reproducibility)
- **Batch**: at z = 9.0 m (near ceiling), at four fixed corner positions

PA source signals are scaled by **0.2x** before being added to the room.

### Audio normalization pipeline

All WAV files go through the same preprocessing:

1. Read via `scipy.io.wavfile`
2. Cast to `float64`
3. Average channels to mono (if stereo)
4. Resample to 16 kHz via linear interpolation (`np.interp`)
5. Normalize to unit RMS

---

## 6. The SRP-PHAT Algorithm

### Concept

Steered Response Power with Phase Transform (SRP-PHAT) is a DOA estimation algorithm. For each candidate direction on a spherical grid, it computes the total power that would be received by a delay-and-sum beamformer steered in that direction. The direction with maximum power is the DOA estimate.

The "PHAT" weighting normalizes the cross-spectral density by its magnitude, retaining only phase information. This makes the algorithm robust to reverberation and spectral coloring.

### Implementation details

The signal from each microphone is transformed to the frequency domain using a Short-Time Fourier Transform:

- **Sample rate**: 16,000 Hz
- **FFT size**: 1024 samples (64 ms)
- **Hop size**: 512 samples (32 ms)
- **Frequency range**: 200 -- 2000 Hz (only these STFT bins are used)

The 200 Hz lower bound excludes frequencies where the small array has negligible spatial resolution. The 2000 Hz upper bound stays below the spatial aliasing frequency for typical microphone spacings (~7.8 cm adjacent spacing aliases at ~2200 Hz).

### Search grid

The algorithm searches over a spherical grid:

- **72 azimuth values** uniformly spaced on $[0, 2\pi)$ (5-degree steps)
- **19 colatitude values** uniformly spaced on $[0, \pi]$ (10-degree steps)
- **Total**: 1,368 candidate look-directions

Colatitude is measured from the positive z-axis (north pole), so the conversion to elevation is:

$$\theta_{el} = 90° - \theta_{colat}$$

where $\theta_{colat} = 0°$ corresponds to straight up (elevation = 90°) and $\theta_{colat} = 90°$ corresponds to the horizon (elevation = 0°).

### Output

The algorithm produces:

1. **Estimated DOA**: azimuth and elevation of the peak power direction (rounded to 1 decimal place)
2. **Power grid**: a 19 x 72 matrix of SRP power values, sent to the frontend for sphere visualization

---

## 7. Beamformed Audio Output

After DOA estimation, a delay-and-sum beamformer produces an audio signal steered toward the **estimated** direction (not the true direction). This lets the user hear what the array "locks onto."

### Delay computation

For a unit direction vector $\hat{d}$ derived from the estimated azimuth and elevation:

$$\tau_m = \frac{(\mathbf{r}_m - \bar{\mathbf{r}}) \cdot \hat{d}}{c}$$

where $\mathbf{r}_m$ is mic $m$'s position, $\bar{\mathbf{r}}$ is the array centroid, and $c = 343$ m/s. Delays are shifted to be non-negative: $\tau_m \leftarrow \tau_m - \min(\tau)$.

Each mic's signal is shifted by `int(round(tau_m * fs))` samples and summed. The output is averaged over all microphones.

### Encoding

The output is peak-normalized to 0.9 full-scale, quantized to 16-bit signed integers, written as a WAV file to an in-memory buffer, then base64-encoded for JSON transport. The frontend decodes this back into a Blob URL for the `<audio>` player.

The audio duration equals the integration window (default 1 second).

---

## 8. Microphone Mismatch Model

When enabled, each microphone receives a random gain and phase perturbation before the SRP-PHAT analysis:

- **Gain**: drawn uniformly from $[-1, +1]$ dB, converted to linear: $g = 10^{G_{\text{dB}}/20}$
- **Phase**: drawn uniformly from $[-2°, +2°]$, applied via the Hilbert transform

The phase is applied by computing the analytic signal (via `scipy.signal.hilbert`), then reconstructing with the shifted phase:

$$s'_m(t) = g_m \cdot |a_m(t)| \cdot \cos(\angle a_m(t) + \Delta\phi_m)$$

where $a_m(t)$ is the analytic signal of mic $m$.

These tolerances (±1 dB gain, ±2° phase) are representative of uncalibrated MEMS microphone arrays. In practice, gain mismatch degrades beamforming SNR, while phase mismatch shifts the effective beam direction.

---

## 9. Theoretical Beam Pattern Mode

This mode runs entirely in the browser with no server call. It computes the narrowband delay-and-sum array response for a given geometry, frequency, and steering direction.

### Formula

For each look-direction $\hat{u}_{\text{look}}$ on the spherical grid, the array response is:

$$P(\hat{u}_{\text{look}}) = \frac{1}{M^2} \left| \sum_{m=1}^{M} e^{jk \, \mathbf{r}_m \cdot (\hat{u}_{\text{look}} - \hat{u}_{\text{steer}})} \right|^2$$

where $k = 2\pi f / c$ is the wavenumber, $\mathbf{r}_m$ is the mic position, and $M$ is the total mic count. The result is a normalized power between 0 and 1.

This is computed as a phasor sum: for each mic, accumulate `cos(phase)` and `sin(phase)`, then take the squared magnitude divided by $M^2$.

### Spatial aliasing indicator

The info panel shows the spatial aliasing frequency:

$$f_{\text{alias}} = \frac{c}{2 \cdot d_{\text{adj}}}$$

where $d_{\text{adj}}$ is the distance between adjacent microphones. Above this frequency, grating lobes appear and the beam pattern becomes ambiguous.

---

## 10. Visualization

### Power sphere

Both the SRP simulation results and the live sim results are displayed on a deformed sphere in Three.js. The 19 x 72 power grid maps to a spherical mesh where:

- **Radius** at each vertex is modulated by power: $r = r_{\text{disp}} \cdot (0.25 + 0.75 \cdot p_{\text{norm}})$, where $p_{\text{norm}}$ is the power value divided by the grid maximum. Low-power regions contract to 25% of the display radius; the peak extends to 100%.
- **Color** follows a three-stop ramp:
  - 0.0 → dark navy (`#0a1628`)
  - 0.5 → blue (`#1a6baa`)
  - 1.0 → red (`#ff3344`)

  Linear interpolation between stops at the 0.5 threshold.

### DOA arrows

- **Yellow arrow** (cone + line): estimated DOA from SRP-PHAT
- **Green arrow** (cone + line): true source direction (Live Sim mode only)

### Room wireframe (Live Sim only)

The room box is drawn as a white wireframe with a faint floor plane. Since rooms can be very large (50 m) while the SRP sphere is only ~2 m radius, a scale factor maps the room to the scene:

$$s = \frac{6.0}{\max(L, W, H)}$$

All room dimensions and source positions are multiplied by this scale. The SRP sphere and mic markers remain at their native scale at the origin (which represents the array center).

### Source markers (Live Sim only)

| Marker | Color | Size | Represents |
|--------|-------|------|-----------|
| Red sphere | `#ff4444` | 0.08 | Drone source |
| Orange dots | `#ff8833` | 0.03 | Crowd noise sources |
| Purple dots | `#aa44ff` | 0.04 | PA speakers |
| Faded red spheres | `#ff4444` @ 25% opacity | 0.05 | Image sources (reflections) |

### Coordinate mapping

Pyroomacoustics uses $[x, y, z]$ where $z$ is vertical. Three.js convention uses $Y$ as vertical. The mapping is:

$$\text{Three.js}(X, Y, Z) = \text{Python}(x, z, y)$$

This swap is applied consistently in mic positions, source positions, direction arrows, and room geometry.

### RIR inset (Live Sim only)

A 250 x 80 pixel canvas displays the Room Impulse Response for the drone source at microphone 0. The waveform is peak-normalized and downsampled to fit the canvas width. It shows the time-domain structure of the room's acoustic response -- the direct path appears as an initial spike, followed by early reflections and a decaying reverberant tail. The duration label is computed assuming 16 kHz sample rate. The RIR is truncated to 2000 samples (125 ms) on the server side.

---

## 11. Known Limitations and Assumptions

**Physical model:**

- Microphones are modeled as **omnidirectional point receivers**. Real MEMS microphones on a PCB exhibit directivity at high frequencies due to baffle/diffraction effects, but this is negligible below ~4 kHz for typical package sizes. The simulation's frequency range (200--2000 Hz) stays within this regime.
- The room is a **rectangular shoebox**. Real venues have irregular geometry, columns, furniture, and people -- none of which are modeled. The ShoeBox model captures the gross reverberation behavior but not fine spatial detail.
- **Image source method** is truncated to order 6 (live) or 10 (batch). This limits the accuracy of late reverb, particularly for high RT60 values where many reflection orders contribute.
- The source is **stationary by default**. In Phase 2b, the live simulator can optionally render the drone along a straight-line or circular-arc trajectory via `K`-chunk phantom rendering (see [LIMITATIONS.md §2b](LIMITATIONS.md)); position is piecewise constant inside each chunk. Batch (`run_comparison.py`) remains static-only by design.
- **Atmospheric realism is first-order.** Temperature and humidity are spatially uniform knobs feeding ISO 9613-1 air absorption; a single vertical temperature-gradient slider applies a closed-form analytic beam-bending bias to the source z. No wind, turbulence, or horizontal gradient is modelled.

**Signal model:**

- The integration window equals the audio duration (default 1 second). SRP-PHAT averages over this window -- shorter windows give faster but noisier DOA estimates.
- Random number generators use **fixed seeds** (0 for the simulation path, 42 for PA positions). Results are deterministic and reproducible, but represent single noise/signal realizations. For statistical confidence, multiple seeds should be tested (the batch script uses seeds 0 and 1).
- Crowd noise decorrelation has two modes. The default `point_source` model is **approximate** -- real crowd noise has spatial correlation structure that segment-rotation cannot fully reproduce. Phase 3 adds a `plane_wave` mode (`synthesize_diffuse_crowd_plane_waves`) that produces a proper isotropic-diffuse field via an N-plane-wave sum with fractional-sample delays, approaching the ideal `sinc(k·d)` inter-mic coherence as `n_plane_waves` grows.
- The SNR formula incorporates geometric spreading, so the configured SNR value is **referenced to source power at the source distance**, not a simple receiver-side noise floor.
- **MAX78000 ML path is previewed, not executed.** Phase 3 optionally re-quantizes the beamformed audio to int8 / int16 and runs a **proxy log-mel feature extractor**, reporting audio-domain and feature-domain quantization SNR. This is a stand-in for what a typical audio-classifier CNN would see; actual MAX78000 network inference (INL, accumulator-width activations, real classifier weights) is out of scope.
- **Crosstalk model is either flat or 1-pole.** With the Phase 3 FIR model active, the neighbour-leakage path is a single-pole high-pass (capacitive-coupling physics). Real PCB coupling has multiple poles and layout-dependent non-nearest-neighbour paths; these drop in via `--crosstalk-fir-path` once measured traces are available.
- **DOA band is user-tunable, but the harmonic-comb weighting is deliberately simple.** Phase 3+ lets the user set `fmin_hz` / `fmax_hz` and optionally enable a harmonic comb that restricts SRP-PHAT integration to +/- 10 Hz around each n*f0 (n = 1..20). No shaped Gaussian lobes, no auto-tracking of f0, no sub-band voting, and no MVDR adaptive null-forming. On real hardware the DSP should recalibrate the band from recorded drone spectra and consider harmonic tracking if the drone's RPM varies significantly.

---

## 12. Differences Between Batch and Live Simulation

The batch script (`run_comparison.py`) and live server (`sim_server.py`) share the same SRP-PHAT pipeline and array construction logic, but differ in several environmental defaults and implementation details:

| Aspect | Batch (`run_comparison.py`) | Live (`sim_server.py`) |
|--------|----------------------------|------------------------|
| Room dimensions | Fixed 20 x 15 x 10 m | Default 50 x 40 x 12 m (configurable) |
| ISM max_order cap | 10 | 6 |
| Sabine failure | Trial skipped | Fallback: absorption=0.5, order=3 |
| Drone signal | Always synthetic (multi-tone + noise) | Real audio from WAV (synthetic fallback) |
| Crowd sources | 12, at z = 1.2 m, 0.5 m wall margin | Configurable (default 30), at z = 1.5 m, 0.3 m wall margin |
| PA sources | 4 fixed ceiling corners at z = 9 m | Configurable (default 8), random wall positions at z = 3 m |
| Mic mismatch | Not available | Optional (gain ±1 dB, phase ±2°) |
| RNG seeds | Per-trial (0, 1) | Fixed seed 0 per request |
| Source distance | Fixed 4 m | Default 6 m (configurable) |

Because of these differences, a live sim run with default parameters does **not** reproduce a batch result. To compare apples-to-apples, you would need to configure the live server parameters to match the batch defaults (room 20x15x10, distance 4 m, etc.).

---

## 13. What You Can and Cannot Conclude

### Valid conclusions from this simulation

- **Relative geometry ranking**: the simulation reliably shows which array geometry performs better under matched conditions. Cylinder and standing cross consistently resolve elevation; UCA and ULA do not.
- **UCA elevation mirror ambiguity**: a flat circular array produces identical time-delay patterns for sources at $+\theta_{el}$ and $-\theta_{el}$, so SRP-PHAT picks one arbitrarily. This is a fundamental physical limitation, not a software issue.
- **Reverberation impact**: increasing RT60 degrades DOA accuracy for all geometries, with UCA and ULA degrading faster in elevation.
- **SNR impact**: DOA accuracy degrades as SNR decreases, with predictable behavior across geometries.
- **Diffuse noise field**: crowd and PA noise sources representative of an exhibition hall scenario degrade performance, particularly for geometries with limited aperture.
- **Design parameter sensitivity**: the effect of mic count, array radius, and ring separation on beam width and spatial aliasing frequency.

### Not valid to conclude

- **Absolute detection range or dB levels**: the simulation does not model source power in physical units. SNR is parametric, not derived from a specific drone at a specific distance with a calibrated microphone.
- **Venue-specific performance**: the ShoeBox room does not capture the acoustics of a specific venue. For that, measured impulse responses would be needed.
- **Outdoor / battlefield performance**: no ground reflection, atmospheric absorption, wind noise, or long-range propagation is modeled.
- **Real-time latency**: the simulation runs offline. Processing latency on embedded hardware (SigmaDSP, MAX78000) is a separate engineering concern.
- **Multi-source scenarios**: only single-source DOA is estimated. Tracking multiple simultaneous sources would require a different algorithm (e.g., MUSIC, TOPS) and is not covered here.
- **Absolute accuracy guarantees**: fixed seeds mean each result is a single realization. Statistical confidence requires sweeping over multiple seeds, which the batch script does (seeds 0 and 1) but the live server does not.
