# Simulation Limitations -- Phase 1 + 2a + 2b + 3

This document inventories what our `pyroomacoustics`-based simulation
**can** and **cannot** tell us, so that conclusions drawn from the
comparison and live-sim tools are correctly scoped for our real
drone-detection build.

> **What's new in Phase 3.** The pipeline now includes a fixed-point
> **MAX78000 ML-path preview** (beamformed audio is re-quantized to
> int8 or int16, passed through a proxy **log-mel feature extractor**
> that mirrors what a typical audio-classifier CNN would see, and
> both stages report a quantization SNR in the response); a rigorous
> **isotropic-diffuse-field plane-wave crowd generator** (sum of N
> uncorrelated plane waves with fractional-sample delays in the
> frequency domain) as an optional `crowd_model="plane_wave"`; and an
> **FIR capacitive-coupling crosstalk model**
> (`crosstalk_model="fir_capacitive"`) that replaces the flat
> neighbour-leakage model with a 1-pole high-pass leakage path, with a
> hook for loading measured FIR traces via `--crosstalk-fir-path`. All
> three knobs default to the Phase 2b baseline so existing sweeps and
> parity tests stay reproducible. Batch (`run_comparison.py`) gains
> CSV columns `ml_path_snr_db` and `feature_snr_db` when
> `--ml-preview` is passed. These items have moved out of §2 into a
> new §3 below, and the phase roadmap is updated.

> **What's new in Phase 2b.** The live simulator now supports a
> **moving drone source** (straight-line or circular-arc trajectory)
> rendered as K time-gated, Hann-crossfaded phantom sources inside
> a single `room.simulate()` call -- this naturally captures
> Doppler shift and changing arrival angle within the integration
> window. The live and batch pipelines also expose explicit
> **temperature** and **relative humidity** knobs (feeding ISO 9613-1
> air absorption) and a **vertical temperature gradient** knob that
> applies a first-order analytic beam-bending bias to the reported
> elevation. The "Exhibition Hall" preset sets realistic atmospheric
> defaults. These items have moved out of §2 into a new §2b below.
> Batch (`run_comparison.py`) gains `--temperature / --humidity /
> --temp-gradient` CLI flags, but intentionally does **not** support
> moving sources -- batch remains a static-source sweep.

> **What's new in Phase 2a.** Per-wall materials (carpet / plasterboard /
> ceiling tile / curtains) replace the single Sabine coefficient as the
> default absorption model, with an "Exhibition Hall" preset that matches
> the target venue. Microphone crosstalk and codec quantization are now
> simulated as part of the analog front-end chain. An SRP-PHAT power
> heatmap plus top-3 DOA candidates are returned alongside the single
> argmax, and a pytest parity test keeps the live and batch pipelines
> numerically in sync. These items have moved out of §2 below.

> **TL;DR.** The simulation is reliable for *relative* ranking of
> microphone array geometries and for seeing how performance *changes*
> with RT60, drone SPL, and noise level. It is **not** a trustworthy
> predictor of absolute accuracy or absolute detection range at an
> Embedded-World-style venue. Plan to calibrate against real recordings
> as early as possible.

---

## 1. What the simulation reliably gives us

- **Geometry ranking.** "UCA vs standing-cross vs ULA vs cylinder" in a
  given acoustic environment. Differences of 10° or more in the mean
  total angular error are real and should hold up in hardware.
- **Qualitative trends.** "More reverb → worse", "lower drone SPL →
  worse", "a stacked-ring gives elevation discrimination that a flat
  ring doesn't". These trends are physics-driven (ISM + delay-and-sum
  SRP-PHAT) and are robust.
- **Spatial aliasing behaviour.** The simulation respects inter-mic
  spacing vs wavelength, so sidelobe / grating-lobe structure and the
  aliasing frequency scale accurately with array radius.
- **Integration-time / beamforming-gain trends.** Doubling the
  integration window, or beamforming vs unsteered averaging, gives
  realistic *relative* SNR improvements.

## 2. Known physical simplifications

| Area | What we assume | Why it matters in reality | Mitigation / hardware check |
|---|---|---|---|
| Mic directivity | Omnidirectional point receivers | Real MEMS / condenser mics are omni below ~2-3 kHz (our 200-2000 Hz band), so **error here is small** for drone-band work. Becomes significant if you push up to propeller-blade harmonics above 3 kHz. | Confirm mic directivity plot from datasheet below 2 kHz is within ±2 dB. |
| Near field | Free-space 1/distance attenuation (far-field) | Breaks down below ~λ/2π ≈ 0.3 m at 200 Hz; mildly breaks down below ~2 m. | Ignore any simulation result with `source_distance < 2 m`. |
| Room geometry | Empty shoebox, flat walls (per-wall materials supported in Phase 2a; see §2a) | Real Embedded-World halls have booths, trusses, people, curtains, PA rigs that scatter and absorb. The sim will be more reverberant-looking in the direct-path band but *less* diffuse in general. | Plan to test in a booth-like environment (lunchroom with people / partitions) before venue day. |
| Air absorption | ISO 9613-1, temperature + humidity now explicit knobs in both live and batch (see §2b) | Correct to first order. Reality deviates with local HVAC pockets, crowd humidity, and time-varying conditions. | Sweep the humidity/temperature sliders for sensitivity analysis. |
| Drone source | Single omni point emitting a recorded drone clip | Real drones radiate a tonal pattern from each rotor plus broadband blade-passing, and the radiation pattern is *not* omni. Signal also changes with thrust. | Record your own drone at a fixed RPM / distance in Month 2 and feed the recording back through the sim. |
| Drone motion | Live sim supports straight-line / arc trajectories as chunked phantom sources (see §2b). Batch (`run_comparison.py`) remains static. | Real drones move -> Doppler, fluctuating arrival angle within integration window. Our K-chunk model is piecewise-static in position within each chunk. | Integrate < 200 ms for moving drones; expect additional elevation-error jitter in hardware. Cross-check live-sim moving results against a real flypast in Month 3. |
| Speed of sound | Fixed `c = 343 m/s` baseline, with a first-order analytic **vertical temp gradient** bias applied to source z (see §2b) | Real temperature gradients (stage lights, HVAC plumes) bend rays by a few degrees over 10-20 m. Horizontal gradients and turbulent eddies are not modelled. | Budget an extra ±1-2° systematic elevation error on top of the sim's modelled bias in hardware. |
| Mic mismatch | Phase-1 FIR-based per-channel gain (0.5 dB σ) + fractional delay (0.2 samples σ) + DC offset | Good first-order model of preamp tolerance + TDM/PDM clock skew. Per-channel crosstalk and ADC quantization are modelled separately in Phase 2a; time-varying jitter is still missing. | Measure your own preamp block with white-noise test; re-inject numbers into `apply_mic_mismatch_v2`. |
| Crowd audio | Two clips from one YouTube source, 30 s each; by default re-used across discrete point sources with rotation, optionally replaced in Phase 3 with a plane-wave-sum isotropic-diffuse synthesiser (`crowd_model="plane_wave"`; see §2c) | Real crowd is spatially decorrelated at every position, with specific speech tonality. The default point-source model is mildly over-correlated; the plane-wave model approaches the ideal `sinc(k·d)` coherence, with residual error from the finite plane-wave count. | With `crowd_model="plane_wave"`, bump `n_plane_waves` to 64-128 for the closest-to-ideal coherence; still treat "diffuse" results as a *lower bound* on real-world crowd-induced error. |
| Noise floor | Gaussian AWGN at `sigma2_awgn = P_mic_floor²` | Models MEMS self-noise well but ignores PDM clock noise, 1/f drift, and EMI. | If the MAX78000 eval kit adds detectable EMI pickup, raise the mic-floor slider to compensate. |
| Numerical precision | Double-precision float throughout until the codec-quantization stage (Phase 2a) and the optional ML-path preview (Phase 3, see §2c) | MAX78000 will run in int16/int8 activations with quantization noise. Codec quantization (8/12/16/24 bit) covers the analog front-end; the Phase 3 ML preview additionally models a post-beamforming int8/16 capture + log-mel quantization. | Use `ml_preview=True` to see the expected quantization SNR hit before hardware; budget an additional few dB beyond that for the real CNN. |

## 2a. Newly modelled in Phase 2a

These items used to live in §2 and are now part of the simulation.
Where relevant they are still *simplified* models -- see the notes.

- **Per-wall materials.** The live simulator and
  `run_comparison.py --materials-profile exhibition_hall` both build
  the room with six independent `pyroomacoustics` materials (floor,
  ceiling, and four walls), defaulting to a curated
  "Exhibition Hall" preset (hairy carpet, fissured-tile ceiling,
  plasterboard walls, heavy cotton curtains on one side). The RT60
  slider is still available as a fallback. The response now reports
  `rt60_actual` measured from the simulated room impulse response, so
  you can see the effective Sabine time without having to pick one up
  front. **Simplification that remains:** still frequency-flat within
  each octave (we use the first catalogued coefficient), and
  directional scattering is not modelled.
- **Microphone crosstalk.** `apply_crosstalk(signals, coupling_db)`
  adds a linear nearest-neighbour leak (wrap-around on circular
  geometries, clipped ends on linear ones) at a configurable coupling
  level. Defaults to -40 dB when enabled. Good first-order model of
  board-level PCB capacitive coupling and shared-ground leakage. **Not
  modelled:** frequency-dependent coupling, power-supply-rejection
  artefacts, or long-tail EMI leakage.
- **Codec / ADC quantization.** `apply_codec_quantization(signals,
  bit_depth)` mid-tread-quantizes each channel to 8 / 12 / 16 / 24
  bits with a joint full-scale so that relative channel levels are
  preserved. This gives a realistic quantization-noise floor
  (~6 dB/bit) and lets you preview the MAX78000's int-audio path.
  **Not modelled:** dithering behaviour, codec pre-filter shape,
  companding.
- **SRP-PHAT heatmap + top-N candidates.** The live response now
  carries the full 2-D (az × colat) SRP grid and the top-3 DOA peaks
  with 15° non-maximum suppression, so you can see the UCA mirror
  ambiguity and side-lobe structure directly instead of inferring
  them from the argmax.
- **Live-vs-batch parity test.** `tests/test_parity.py` forces
  synthetic drone audio and runs the same clean trial through
  `sim_server.run_live_trial()` and
  `run_comparison.run_single_trial()`, asserting that the two DOA
  estimates agree to within 1°. This is the guard against future
  refactors silently desynchronising the two code paths.

## 2b. Newly modelled in Phase 2b

These items used to live in §2 and are now part of the simulation.
They are still *simplified* models -- see the per-item notes.

- **Moving drone source (live sim only).** The live simulator can
  render the drone along a **straight-line** or **circular-arc**
  trajectory instead of pinning it to a single point. Implementation
  is **segmented phantom-source rendering**: we split the drone's
  audio into `K` chunks (UI: 4 - 16), apply overlapping
  Hann-squared-half crossfades that sum exactly to 1.0 across the
  full duration, and add each chunk as a separate static source at
  the interpolated drone position in the same `room.simulate()`
  call. Because each phantom has a different position and its
  signal energy is concentrated in a different time window, the
  resulting microphone signals pick up the correct per-chunk TDOA
  shift, which is the dominant acoustic consequence of motion -- so
  both **Doppler frequency shift** within the integration window
  and **DOA drift** across the window appear naturally in the
  SRP-PHAT output. The trajectory is drawn in the 3D viz as a
  polyline with a wireframe start sphere and a solid end sphere.
  **What remains simplified:** the drone's position is *piecewise
  constant* within each chunk rather than truly continuous, so
  very high speeds with few chunks can produce staircase-like
  Doppler artefacts; raise `n_trajectory_chunks` if you see them.
  There is also no rotor-direction cardioid radiation pattern --
  each phantom still radiates omnidirectionally. Batch
  (`run_comparison.py`) intentionally does **not** implement this,
  and remains a static-source sweep by design.
- **Explicit temperature and humidity knobs.** Both live and batch
  paths now pass the configured temperature (°C) and relative
  humidity (%) directly into pyroomacoustics' ISO 9613-1 air
  absorption model via `acoustic_utils.air_absorption_kwargs(...)`.
  The Phase 1 defaults (20 °C / 50 % RH) are preserved as
  module-level constants, and the "Exhibition Hall" UI preset sets
  more realistic indoor-venue values (22 °C / 55 % RH). The batch
  script exposes `--temperature`, `--humidity` and `--temp-gradient`
  CLI flags so sweeps can be re-run under a different atmosphere
  without editing the source. **What remains simplified:** air
  conditions are **spatially uniform** across the room. Local
  HVAC plumes, hot spots under stage lights, and condensation on
  curtains are not modelled.
- **Vertical temperature gradient beam-bending.** The user can
  dial a vertical temperature gradient in °C/m. Instead of
  running a full ray-tracing solver, we apply a closed-form
  first-order analytic bias: sound launched from height `z_s`
  toward a receiver at height `z_r` is deflected by an angle
  proportional to the gradient and to the horizontal distance;
  we fold that into a small **z-shift** on the source position
  used by the room simulation (`atmospheric_z_bias`), and
  separately report the resulting apparent-elevation bias in
  degrees (`atmospheric_elevation_bias_deg`) so the frontend
  can display it. With a typical lab/hall gradient of
  ~1-2 °C/m and a ~8 m source distance this is a fraction of
  a degree, which is well below the SRP grid's 10° colatitude
  step -- so you typically see the bias numerically in the
  info panel before it shifts the DOA argmax. **What remains
  simplified:** only vertical (z) gradients are supported; the
  model assumes a ray-optics regime; curvature due to crosswinds
  and horizontal temperature structure is ignored.
- **Phase 2b tests.** `tests/test_parity.py` gains three new cases:
  `test_live_vs_batch_parity_cylinder_with_atmosphere` re-runs
  the existing parity guard with a non-default temperature,
  humidity, and vertical gradient;
  `test_moving_source_chunks_sum_to_original` proves that the
  chunk-crossfade helper reconstructs the original signal to
  float epsilon (guarding Doppler / level correctness); and
  `test_moving_source_static_equivalence` verifies that
  `moving_source=True` with `speed_mps=0` collapses back to the
  original single-source result.

## 2c. Newly modelled in Phase 3

These items used to live in §2 (e.g. the "numerical precision" row
for the MAX78000 and the "crowd audio over-correlation" row) and are
now explicit knobs in the simulator. As with §2a and §2b, each model
is *still simplified* in specific ways; those caveats are called out
below. All three knobs default to the Phase 2b baseline, so existing
sweeps and parity tests stay numerically identical when Phase 3 is
off.

- **MAX78000 fixed-point ML-path preview.** When `ml_preview=True`,
  the live simulator re-quantizes the beamformed audio to int8 or
  int16 (`ml_bit_depth`), runs a proxy **log-mel-dB feature
  extractor** (64 bands by default, Hann-windowed 512-point STFT with
  a 256-sample hop over the 200-2000 Hz SRP-PHAT band) on both the
  float and the quantized audio, optionally re-quantizes the feature
  tensor to `ml_feature_bit_depth` bits, and reports two SNR numbers
  in the response: `ml_path_snr_db` (audio-domain quantization SNR)
  and `feature_snr_db` (log-mel domain quantization SNR). A fourth
  downloadable WAV and an inline log-mel PNG are exposed in the UI,
  and `run_comparison.py --ml-preview` adds two CSV columns so sweeps
  can compare geometry-ranking under MAX78000-style quantization.
  **What remains simplified:** this is a **log-mel proxy** for what a
  MAX78000 audio-classifier CNN sees, not the actual ADI kws20-style
  network, and it cannot tell you classification accuracy. It also
  models quantization only: ADC INL, clocking jitter, and the
  MAX78000's accumulator-width activations are still out of scope.
- **Isotropic-diffuse-field plane-wave crowd model.** When
  `crowd_model="plane_wave"` and `diffuse=True`, the crowd-noise
  component of the diffuse bed is replaced with an explicit
  plane-wave-sum synthesis (`synthesize_diffuse_crowd_plane_waves`):
  `n_plane_waves` random unit-vector directions, each fed an
  independent window of the real crowd corpus, are applied directly
  to the mic signals as fractional-sample delays in the frequency
  domain via FFT phase multiplication. This produces the textbook
  isotropic-diffuse inter-mic coherence curve (`sinc(k·d)` at wide
  spacings) that our old `np.roll`-based point-source crowd could
  not, and is physically closer to what a convention-hall crowd
  actually sounds like at a stationary array. PA speakers stay as
  discrete point sources because they *are* localised. **What
  remains simplified:** the plane-wave count is finite
  (default 64) so the coherence curve has a finite-N-sample noise
  floor; the crowd source signals still come from the same 3-minute
  corpus, so long recordings will repeat; and the field is assumed
  **time-stationary** over the integration window.
- **FIR capacitive-coupling crosstalk.** When
  `crosstalk_model="fir_capacitive"`, the Phase 2a flat-neighbour
  crosstalk is replaced with a 1-pole high-pass leakage path in
  `apply_crosstalk_fir` (corner frequency `crosstalk_corner_hz`
  configurable; coupling dB reused from the Phase 2a knob). This
  mirrors the physics of PCB parasitic-C coupling, which rolls off
  at ~6 dB/oct below the corner. With `corner_hz` driven to near
  zero the model converges exactly on the flat
  `apply_crosstalk`, so **the FIR model is a strict superset** of
  the Phase 2a model. A stub `load_crosstalk_fir(path)` can be wired
  to a `--crosstalk-fir-path=<file.json|.npz>` hook so measured
  coupled-pair FIRs from the real PCB (when we have them) drop in
  without a code edit. **What remains simplified:** the default
  analytic model is only a 1-pole HPF; real PCB coupling has
  multiple poles (layout-dependent), and non-nearest-neighbour
  coupling on fine-pitch routing is not modelled.
- **Phase 3 tests.** `tests/test_parity.py` gains five new cases:
  `test_crowd_model_default_is_point_source_parity` (explicit
  no-regression guard for the Phase 2b default);
  `test_plane_wave_lower_inter_mic_coherence` (proves the
  plane-wave field has lower coherence between opposite mics than a
  single-source reference, i.e. the diffuse synthesis actually
  decorrelates); `test_crosstalk_fir_lowpass_attenuation` (confirms
  the capacitive HPF character -- less low-frequency leakage than
  high-frequency);
  `test_crosstalk_fir_limit_matches_simple` (the superset property);
  and `test_ml_preview_int8_worse_than_int16` (quantization SNR
  behaves as expected). `tests/_smoke_live.py` adds three HTTP
  check blocks `[i] [j] [k]` covering ML preview, plane-wave crowd,
  and FIR crosstalk end-to-end through the FastAPI server.

## 3. Expected calibration offsets to reality

Rule-of-thumb numbers based on published literature and the ways the
simulation simplifies reality:

- **Absolute DOA error.** Real-world RMS DOA error is typically
  **2-3× larger** than simulation for the same geometry and RT60. A
  simulated 3° mean azimuth error often becomes 6-10° at the venue.
- **Beamforming SNR gain.** Simulated beamforming SNR gain over raw
  mic is often **10-15 dB** in diffuse noise. Real-world tends to land
  in **3-8 dB** because of mic self-noise, mismatch, and crosstalk.
- **Maximum usable distance.** Simulated "detection range" (distance at
  which DOA error stays below 5°) is usually **2×** the realistic range
  because the sim has no booth scattering and a milder crowd model.
  Expect roughly **40-60 %** of simulated range in an actual hall.
- **Elevation estimation.** Works perfectly in sim on planar arrays (up
  to the fundamental up/down ambiguity). In hardware, you also lose
  accuracy to mic backplane reflections and near-field effects. Budget
  an extra **5-10°** elevation error on UCA/ULA; CYLINDER is the most
  robust.

Apply these offsets as **additive conservatism** to any quantitative
claim derived from the simulation.

## 4. Recommended real-world validation workflow

A small, cheap, staged validation plan that closes the sim-to-reality
gap before venue day:

1. **Month 2** -- Breadboard a **4-mic UCA** with whatever eval MEMS
   mics you picked. Record your team's drone at 1 m / 3 m / 5 m / 10 m
   in a quiet office. Log SPL with a cheap reference meter.
2. **Month 3** -- Import those drone recordings as the sim source
   signal (replace `audio/drone.wav`). Re-run `run_comparison.py`. Are
   the geometry rankings the same? If not, Phase 2 the sim.
3. **Month 4** -- Run the full prototype in a reverberant room
   (atrium, warehouse, lunchroom when closed). Compare live-sim
   predictions for that approximate RT60 to the measured DOA error.
   Expect the ×2-3 calibration offset from §3.
4. **Month 5** -- Trade-show-like **dress rehearsal** in a large,
   crowded space (trade-show setup, conference hall during setup day).
   Record audio; re-inject into sim; lock down any last-minute
   changes.
5. **Venue week** -- Reserve **2-3 working days** at the target venue
   specifically for integration debugging: mic pickup of PA, EMI from
   nearby equipment, mounting acoustics. Do not treat the sim as a
   substitute for this.

---

## 5. Phase roadmap for this simulation

- **Phase 1 (done)** -- dB SPL source model, air absorption, FIR mic
  mismatch, warm-up trim, joint-normalized audio playback, shared
  `acoustic_utils.py`, randomized crowd placement, this document.
- **Phase 2a (done)** -- Per-wall materials with "Exhibition Hall"
  preset + measured RT60; microphone crosstalk; codec / ADC
  quantization (8/12/16/24 bit); SRP-PHAT heatmap + top-N DOA
  candidates; "Hardware-realistic" impairment preset; save / load /
  URL-share of sim presets; Raw / Unsteered / Beam .wav download;
  live-vs-batch parity test in `tests/`.
- **Phase 2b (done)** -- Moving-drone source in the live sim
  (straight-line + circular-arc trajectories) via K-chunk phantom
  rendering with exact-reconstruction Hann crossfades, yielding
  in-window Doppler + DOA drift; explicit temperature / humidity
  knobs feeding ISO 9613-1 in both live and batch; first-order
  vertical-gradient beam-bending (analytic source-z shift + apparent
  elevation bias reporting); `--temperature / --humidity /
  --temp-gradient` CLI flags on `run_comparison.py`; three new
  pytest cases guarding atmosphere parity, chunk-sum reconstruction,
  and static-speed-zero equivalence.
- **Phase 3 (done)** -- MAX78000 fixed-point ML-path preview (int8 /
  int16 audio + log-mel feature quantization, with SNR metrics and
  an inline spectrogram PNG in the UI); rigorous isotropic-diffuse
  plane-wave crowd generator as an optional `crowd_model="plane_wave"`
  toggle; FIR capacitive-coupling crosstalk as an optional
  `crosstalk_model="fir_capacitive"` toggle with a measured-FIR
  loader hook; `--ml-preview / --crowd-model / --crosstalk-model`
  CLI flags on `run_comparison.py`; two new CSV columns
  (`ml_path_snr_db`, `feature_snr_db`) populated when
  `--ml-preview` is passed; five new pytest cases guarding the
  Phase 3 invariants; three new HTTP smoke-test blocks.

The explicit roadmap is now **closed**. Any further realism work
(real MAX78000 CNN inference, measured FIR traces from the actual
PCB, non-nearest-neighbour crosstalk topologies, measured RIRs
replacing ShoeBox ISM) is out of the documented Phase-1-through-3
scope and should be tracked separately against hardware milestones.
