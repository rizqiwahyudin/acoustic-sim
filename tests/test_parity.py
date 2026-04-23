"""Live-vs-batch parity test.

Runs a clean (no diffuse noise, no hardware impairments, RT60 mode)
simulation through both the FastAPI live path (``run_live_trial``) and
the batch comparison path (``run_single_trial``), with the same seed
and matched parameters, and asserts that the SRP-PHAT DOA estimates
agree to within 1 degree in azimuth and elevation.

If this test ever regresses it almost always means one of the two code
paths has changed its internal order of RNG consumption, STFT config,
array geometry defaults, or room absorption model. Keep them aligned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sim_server
import run_comparison
from sim_server import SimRequest, run_live_trial


def _wrap_deg(a: float) -> float:
    """Wrap an angle to (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def _force_synthetic_audio(monkeypatch):
    """Both code paths must use the same synthetic drone signal for
    parity (same frequencies, same RNG consumption). The live path
    picks real audio if ``DRONE_AUDIO`` is not ``None``; we null both
    audio globals so ``get_drone_signal`` falls back to the synthetic
    generator that matches ``run_comparison.drone_like_signal``.
    """
    monkeypatch.setattr(sim_server, "DRONE_AUDIO", None)
    monkeypatch.setattr(sim_server, "CROWD_AUDIO", None)


def _make_req(seed: int, geometry: str = "CYLINDER") -> SimRequest:
    """Build a SimRequest that mirrors run_single_trial defaults for
    a clean, non-diffuse trial.

    Default geometry is CYLINDER because it is 3D-resolving and free
    from the UCA mirror ambiguity that would otherwise make elevation
    signs non-deterministic across RT60 seeds.
    """
    return SimRequest(
        geometry=geometry,
        mic_count=12,
        radius=0.15,
        ring_separation=0.12,
        room_length=float(run_comparison.ROOM_DIM[0]),
        room_width=float(run_comparison.ROOM_DIM[1]),
        room_height=float(run_comparison.ROOM_DIM[2]),
        rt60=1.0,
        source_az_deg=run_comparison.DEFAULT_AZ,
        source_el_deg=run_comparison.DEFAULT_EL,
        source_distance=float(run_comparison.SOURCE_DISTANCE),
        drone_spl_db=run_comparison.DEFAULT_DRONE_SPL,
        crowd_spl_db=float(run_comparison.CROWD_SPL_DB),
        pa_spl_db=float(run_comparison.PA_SPL_DB),
        mic_noise_floor_db=float(run_comparison.MIC_NOISE_FLOOR_DB),
        seed=seed,
        diffuse=False,
        crowd_count=30,
        pa_count=8,
        mic_mismatch=False,
        crosstalk=False,
        quantization=False,
        absorption_mode="rt60",
        integration_ms=int(run_comparison.SIGNAL_SECONDS * 1000),
    )


def _assert_doa_match(live, batch_az, batch_el, *, tol_deg=1.0, context=""):
    """Assert that live and batch DOA estimates agree to within ``tol_deg``."""
    est_az_live = float(live["est_az_deg"])
    est_el_live = float(live["est_el_deg"])

    daz = abs(_wrap_deg(est_az_live - float(batch_az)))
    del_ = abs(est_el_live - float(batch_el))

    prefix = f"{context}: " if context else ""
    assert daz < tol_deg, (
        f"{prefix}azimuth parity broken -- live={est_az_live:.2f} "
        f"batch={float(batch_az):.2f} (|Δ|={daz:.3f}° > {tol_deg}°)"
    )
    assert del_ < tol_deg, (
        f"{prefix}elevation parity broken -- live={est_el_live:.2f} "
        f"batch={float(batch_el):.2f} (|Δ|={del_:.3f}° > {tol_deg}°)"
    )


def test_live_vs_batch_parity_clean_rt60(monkeypatch):
    """Cylinder array, clean, RT60=1s: live and batch DOA must agree within 1 deg."""
    _force_synthetic_audio(monkeypatch)

    seed = 0
    req = _make_req(seed, geometry="CYLINDER")

    live = run_live_trial(req)

    array_R = run_comparison.build_geometry("CYLINDER", run_comparison.ARRAY_CENTER)
    est_az_batch, est_el_batch, _gaz, _gcolat, _gvals, _ml, _feat = (
        run_comparison.run_single_trial(
            array_R,
            run_comparison.DEFAULT_AZ,
            run_comparison.DEFAULT_EL,
            1.0,
            run_comparison.DEFAULT_DRONE_SPL,
            run_comparison.ROOM_DIM,
            diffuse=False,
            seed=seed,
        )
    )
    assert est_az_batch is not None, "batch trial failed"

    _assert_doa_match(live, est_az_batch, est_el_batch, context="RT60=1.0s")


def test_live_vs_batch_parity_anechoic(monkeypatch):
    """Anechoic path (rt60=0) must also match, because run_live_trial
    branches on rt60<=0 into ``pra.AnechoicRoom`` -- different code
    path from the rt60>0 branch, so we cover both."""
    _force_synthetic_audio(monkeypatch)

    seed = 7
    req = _make_req(seed, geometry="CYLINDER")
    req.rt60 = 0.0

    live = run_live_trial(req)

    array_R = run_comparison.build_geometry("CYLINDER", run_comparison.ARRAY_CENTER)
    est_az_batch, est_el_batch, _gaz, _gcolat, _gvals, _ml, _feat = (
        run_comparison.run_single_trial(
            array_R,
            run_comparison.DEFAULT_AZ,
            run_comparison.DEFAULT_EL,
            0.0,
            run_comparison.DEFAULT_DRONE_SPL,
            run_comparison.ROOM_DIM,
            diffuse=False,
            seed=seed,
        )
    )
    assert est_az_batch is not None, "batch anechoic trial failed"

    _assert_doa_match(live, est_az_batch, est_el_batch, context="anechoic")


# ── Phase 2b ──────────────────────────────────────────────────────────────────

def test_live_vs_batch_parity_cylinder_with_atmosphere(monkeypatch):
    """Phase 2b atmosphere parity: when the live and batch paths are both
    given the same T / RH / dT/dz, the DOA estimates must still agree to
    within 1 deg. Guards against divergence in how the two paths apply the
    gradient beam-bending shift and air-absorption kwargs.

    Both calls seed numpy's global RNG immediately before invoking the
    simulator because ``pra.ShoeBox(... sigma2_awgn=...)`` draws its
    per-mic AWGN from the *global* numpy state, so otherwise inter-test
    ordering can push the estimate +/- one SRP grid bin (10 deg in
    colatitude) independently on the two sides.
    """
    _force_synthetic_audio(monkeypatch)

    seed = 11
    req = _make_req(seed, geometry="CYLINDER")
    req.temperature_c = 25.0
    req.humidity_pct = 60.0
    req.temp_gradient_c_per_m = 1.0

    np.random.seed(seed)
    live = run_live_trial(req)

    array_R = run_comparison.build_geometry("CYLINDER", run_comparison.ARRAY_CENTER)
    np.random.seed(seed)
    est_az_batch, est_el_batch, _gaz, _gcolat, _gvals, _ml, _feat = (
        run_comparison.run_single_trial(
            array_R,
            run_comparison.DEFAULT_AZ,
            run_comparison.DEFAULT_EL,
            1.0,
            run_comparison.DEFAULT_DRONE_SPL,
            run_comparison.ROOM_DIM,
            diffuse=False,
            seed=seed,
            temperature_c=25.0,
            humidity_pct=60.0,
            temp_gradient_c_per_m=1.0,
        )
    )
    assert est_az_batch is not None, "batch atmosphere trial failed"

    _assert_doa_match(live, est_az_batch, est_el_batch,
                      context="T=25 RH=60 dT/dz=1.0")


def test_moving_source_chunks_sum_to_original():
    """``chunk_signal_with_crossfade`` must reconstruct the original signal
    to float epsilon for various (K, crossfade) combinations. This is the
    invariant that makes ``moving_source`` with speed=0 identical to the
    static path, and the one the Hann crossfade weights must satisfy.
    """
    from acoustic_utils import chunk_signal_with_crossfade

    rng = np.random.default_rng(123)
    fs = 16000
    sig = rng.standard_normal(fs).astype(np.float64)

    for n_k, xf_ms in [(8, 10.0), (4, 20.0), (16, 5.0), (8, 0.0), (2, 12.0)]:
        chunks = chunk_signal_with_crossfade(sig, n_k, fs, crossfade_ms=xf_ms)
        assert len(chunks) == n_k, f"got {len(chunks)} chunks, expected {n_k}"
        assert all(c.shape == sig.shape for c in chunks), (
            "all chunks should be the same length as the original signal"
        )
        summed = np.sum(chunks, axis=0)
        err = float(np.max(np.abs(summed - sig)))
        assert err < 1e-10, (
            f"chunk sum regressed: n_k={n_k} xf={xf_ms}ms max_err={err:.2e}"
        )

    # Edge case: a single chunk must return the signal unchanged.
    chunks = chunk_signal_with_crossfade(sig, 1, fs, crossfade_ms=10.0)
    assert len(chunks) == 1
    assert np.allclose(chunks[0], sig, atol=0.0, rtol=0.0)


def test_moving_source_static_equivalence(monkeypatch):
    """``moving_source=True, speed_mps=0`` must give the *same* DOA as
    ``moving_source=False`` for an otherwise identical trial.

    The moving-source code path renders K phantom static sources along the
    trajectory. At speed=0 every phantom collapses to the same start
    position, and the Hann-crossfaded chunk signals sum back to the
    original drone signal, so the resulting mic recordings should be
    identical to the single-source path up to floating-point determinism
    of the pyroomacoustics ISM render. A 1° DOA tolerance is plenty of
    slack for any stochastic reorder of internal sums.
    """
    _force_synthetic_audio(monkeypatch)

    seed = 19
    req_static = _make_req(seed, geometry="CYLINDER")
    req_moving = _make_req(seed, geometry="CYLINDER")
    req_moving.moving_source = True
    req_moving.speed_mps = 0.0
    req_moving.n_trajectory_chunks = 8
    req_moving.trajectory_type = "straight"

    r_static = run_live_trial(req_static)
    r_moving = run_live_trial(req_moving)

    # Trajectory length reflects the chunks; positions must all equal the
    # static src_pos (within float epsilon) because speed is zero.
    assert len(r_moving["trajectory"]) == 8
    src0 = r_moving["source_pos"]
    for pos in r_moving["trajectory"]:
        for a, b in zip(pos, src0):
            assert abs(a - b) < 1e-6, (
                f"moving@speed=0 phantom {pos} drifted from static {src0}"
            )

    _assert_doa_match(
        r_moving,
        r_static["est_az_deg"], r_static["est_el_deg"],
        context="moving@speed=0 vs static",
    )


# ── Phase 3 ───────────────────────────────────────────────────────────────────

def test_crowd_model_default_is_point_source_parity(monkeypatch):
    """Phase 3B safety net: ``crowd_model="point_source"`` (the default)
    must produce bit-for-bit identical mic signals as not setting the
    knob at all, so existing sweeps and parity tests stay unaffected.
    """
    _force_synthetic_audio(monkeypatch)
    seed = 41

    req_default = _make_req(seed, geometry="CYLINDER")
    req_explicit = _make_req(seed, geometry="CYLINDER")
    req_explicit.crowd_model = "point_source"

    r0 = run_live_trial(req_default)
    r1 = run_live_trial(req_explicit)
    _assert_doa_match(
        r1, r0["est_az_deg"], r0["est_el_deg"],
        tol_deg=0.01, context="crowd_model default vs explicit point_source",
    )


def test_plane_wave_lower_inter_mic_coherence():
    """Phase 3B physicality: the plane-wave diffuse synthesizer should
    produce a *less* coherent mic field between opposite microphones
    than a single point source would, because a proper diffuse field
    has coherence ~sinc(k*d) at wide spacings while a single source has
    near-unit coherence (modulo delay) at all frequencies.
    """
    from acoustic_utils import synthesize_diffuse_crowd_plane_waves
    from scipy.signal import csd as _csd

    fs = 16000
    duration_s = 1.0
    rng = np.random.default_rng(0)

    # 16-element UCA of radius 0.3 m -- wide spacing so the sinc(k*d)
    # decorrelation at mid-band frequencies is pronounced.
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    mic_R = np.stack([
        0.3 * np.cos(angles),
        0.3 * np.sin(angles),
        np.zeros_like(angles),
    ], axis=0)

    # --- plane-wave diffuse field ---
    sources = [rng.standard_normal(int(fs * duration_s)) for _ in range(64)]
    diffuse = synthesize_diffuse_crowd_plane_waves(
        mic_R, duration_s=duration_s, fs=fs, n_planes=64,
        source_signals=sources, rng=rng,
    )
    m0 = diffuse[0]
    m_opp = diffuse[8]  # diametrically opposite mic
    f, P00 = _csd(m0, m0, fs=fs, nperseg=1024)
    _, P11 = _csd(m_opp, m_opp, fs=fs, nperseg=1024)
    _, P01 = _csd(m0, m_opp, fs=fs, nperseg=1024)
    coh_pw = np.abs(P01) ** 2 / (np.abs(P00) * np.abs(P11) + 1e-20)
    band = (f >= 300.0) & (f <= 2000.0)
    mean_coh_pw = float(np.mean(coh_pw[band]))

    # --- reference: same noise at both mics (coherence ~ 1 across band) ---
    same = rng.standard_normal(int(fs * duration_s))
    s0 = same
    s1 = same.copy()
    _, Q00 = _csd(s0, s0, fs=fs, nperseg=1024)
    _, Q11 = _csd(s1, s1, fs=fs, nperseg=1024)
    _, Q01 = _csd(s0, s1, fs=fs, nperseg=1024)
    coh_same = np.abs(Q01) ** 2 / (np.abs(Q00) * np.abs(Q11) + 1e-20)
    mean_coh_same = float(np.mean(coh_same[band]))

    assert mean_coh_pw < 0.5 * mean_coh_same, (
        f"plane-wave coherence not lowered as expected: "
        f"plane_wave={mean_coh_pw:.3f}  same_source={mean_coh_same:.3f}"
    )


def test_crosstalk_fir_lowpass_attenuation():
    """Phase 3C physicality: at ``corner_hz=500`` the FIR crosstalk
    path should attenuate low frequencies more than high frequencies
    (capacitive coupling = HPF in series with the leakage).
    """
    from acoustic_utils import apply_crosstalk_fir

    fs = 16000
    n = int(fs * 0.5)
    t = np.arange(n) / fs
    lo = np.sin(2 * np.pi * 100.0 * t)
    hi = np.sin(2 * np.pi * 2000.0 * t)
    zeros = np.zeros(n)

    # Two-channel array: channel 0 has a tone, channel 1 is silent; the
    # leakage into channel 1 is what we measure. coupling_db is high (-20)
    # so the result is measurable above numerical noise.
    sig_lo = np.stack([lo, zeros], axis=0)
    sig_hi = np.stack([hi, zeros], axis=0)

    out_lo = apply_crosstalk_fir(sig_lo, fs, coupling_db=-20.0, corner_hz=500.0)
    out_hi = apply_crosstalk_fir(sig_hi, fs, coupling_db=-20.0, corner_hz=500.0)

    leak_lo_rms = float(np.sqrt(np.mean(out_lo[1] ** 2)))
    leak_hi_rms = float(np.sqrt(np.mean(out_hi[1] ** 2)))
    # HPF at 500 Hz should attenuate 100 Hz by ~14 dB relative to 2 kHz
    # (20 dB/dec below the corner). Test with a conservative 6 dB margin.
    assert leak_hi_rms > 2.0 * leak_lo_rms, (
        f"FIR HPF leakage did not roll off at low-f: "
        f"leak@100Hz={leak_lo_rms:.4f}  leak@2kHz={leak_hi_rms:.4f}"
    )


def test_crosstalk_fir_limit_matches_simple():
    """Phase 3C superset property: with ``corner_hz`` driven to near-0,
    ``apply_crosstalk_fir`` must converge on ``apply_crosstalk`` (flat
    neighbour leakage). This guards the "strict superset" claim in the
    docs and means the FIR model is safe to leave on by default without
    changing the existing hardware-realistic preset.
    """
    from acoustic_utils import apply_crosstalk, apply_crosstalk_fir

    fs = 16000
    rng = np.random.default_rng(7)
    sigs = rng.standard_normal((6, 8000))

    out_simple = apply_crosstalk(sigs, coupling_db=-40.0)
    out_fir = apply_crosstalk_fir(sigs, fs, coupling_db=-40.0, corner_hz=0.1)

    err = float(np.max(np.abs(out_simple - out_fir)))
    # coupling_db=-40 means each neighbour's contribution is ~1e-2 of the
    # original signal; difference between the flat model and the
    # corner=0.1 Hz FIR should be several orders of magnitude smaller.
    assert err < 1e-3, f"FIR at corner_hz=0.1 did not match flat xtalk: {err:.2e}"


def test_ml_preview_int8_worse_than_int16():
    """Phase 3A: the MAX78000 preview at 8-bit resolution must report a
    lower ``ml_path_snr_db`` than the 16-bit preview. Theoretical
    quantization SNR difference is 6.02 * (16-8) = ~48 dB; we accept
    anything >= 30 dB to leave room for audio-dependent peak jitter.
    """
    from acoustic_utils import ml_path_quantize_audio, ml_path_snr_db as _snr

    rng = np.random.default_rng(3)
    sig = rng.standard_normal(16000)

    q8 = ml_path_quantize_audio(sig, bit_depth=8)
    q16 = ml_path_quantize_audio(sig, bit_depth=16)

    snr8 = _snr(sig, q8)
    snr16 = _snr(sig, q16)

    assert snr8 > 20.0, f"int8 ML SNR unexpectedly low: {snr8:.1f} dB"
    assert snr16 > 20.0, f"int16 ML SNR unexpectedly low: {snr16:.1f} dB"
    assert (snr16 - snr8) > 30.0, (
        f"int16 ML preview not meaningfully cleaner than int8: "
        f"snr8={snr8:.1f} dB  snr16={snr16:.1f} dB  delta={snr16 - snr8:.1f} dB"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
