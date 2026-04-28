"""Phase 2a + 2b + 3 end-to-end smoke -- hits sim_server.py over HTTP.

Checks:
  (a) RT60 mode still works.
  (b) Materials mode + Exhibition Hall preset returns rt60_actual.
  (c) Hardware-realistic impairments produce measurably different audio.
  (f) top_peaks has >=3 entries with (az, el, rel_db) and power grid exists.
  (g) Phase 2b moving source: trajectory has n_trajectory_chunks points
      and zero-speed collapses to the static DOA.
  (h) Phase 2b atmosphere: non-zero temp_gradient_c_per_m produces a
      non-zero atmospheric_bias_deg in the response.
  (i) Phase 3A MAX78000 ML preview: ml_audio_b64 decodes to a valid WAV,
      ml_path_snr_db + feature_snr_db are finite, spectrogram PNG decodes.
  (j) Phase 3B plane-wave crowd mode: returns a valid DOA response.
  (k) Phase 3C FIR capacitive crosstalk: returns a valid DOA response and
      the mic audio differs from the simple-crosstalk baseline.
  (l) Phase 3+ DOA-band + harmonic-comb: fmin/fmax override and
      harmonic_comb=True are echoed back by the server and produce a
      substantially smaller freq_bins set without wrecking DOA accuracy.

Run manually (server must be on 127.0.0.1:8766):
  python tests/_smoke_live.py
"""
from __future__ import annotations
import base64
import io
import json
import sys
import urllib.request
import wave

import numpy as np

BASE = "http://127.0.0.1:8766"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        url=f"{BASE}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def decode_wav_b64(b64: str) -> np.ndarray:
    wav_bytes = base64.b64decode(b64)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sampwidth]
    arr = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1)
    return arr


def base_payload() -> dict:
    return dict(
        geometry="CYLINDER",
        mic_count=12,
        radius=0.15,
        cylinder_sep=0.12,
        source_az_deg=60.0,
        source_el_deg=30.0,
        source_distance=6.0,
        rt60=1.0,
        drone_spl_db=78.0,
        crowd_spl_db=67.0,
        pa_spl_db=80.0,
        noise_floor_db=30.0,
        diffuse=False,
        mic_mismatch=False,
        crosstalk=False,
        quantization=False,
        crosstalk_db=-40.0,
        bit_depth=16,
        absorption_mode="rt60",
        room_dim=[20.0, 15.0, 10.0],
        duration=1.0,
    )


def _top_info(resp: dict) -> str:
    tops = resp.get("top_peaks") or []
    return " | ".join(
        f"#{i+1} az={p['az_deg']:.1f} el={p['el_deg']:.1f} rel={p['rel_db']:.1f}dB"
        for i, p in enumerate(tops[:3])
    )


def main() -> int:
    ok = True

    print("[a] RT60 mode, clean ...")
    r1 = post("/simulate", base_payload())
    az_err = abs((r1["est_az_deg"] - 60) % 360)
    az_err = min(az_err, 360 - az_err)
    el_err = abs(r1["est_el_deg"] - 30)
    print(f"    est az={r1['est_az_deg']} el={r1['est_el_deg']} "
          f"(err az={az_err:.1f} el={el_err:.1f})")
    print(f"    top peaks: {_top_info(r1)}")
    g2d = r1.get("power") or []
    print(f"    power grid shape: {len(g2d)}x"
          f"{len(g2d[0]) if g2d else 0}")
    if not (az_err < 10 and el_err < 10):
        print("    FAIL: clean RT60 DOA should be within 10 deg")
        ok = False
    if not (g2d and len(g2d) == 19 and len(g2d[0]) == 72):
        print("    FAIL: power grid must be 19x72")
        ok = False
    if len(r1.get("top_peaks", [])) < 3:
        print("    FAIL: need >=3 top peaks")
        ok = False

    print("[b] Materials mode + Exhibition Hall preset ...")
    body_mat = base_payload()
    body_mat["absorption_mode"] = "materials"
    body_mat["floor_material"] = "carpet_hairy"
    body_mat["ceiling_material"] = "ceiling_fissured_tile"
    body_mat["east_material"] = "plasterboard"
    body_mat["west_material"] = "plasterboard"
    body_mat["south_material"] = "curtains_cotton_0.5"
    body_mat["north_material"] = "plasterboard"
    r2 = post("/simulate", body_mat)
    rt60_actual = r2.get("rt60_actual")
    print(f"    rt60_actual = {rt60_actual}")
    print(f"    est az={r2['est_az_deg']} el={r2['est_el_deg']}")
    if rt60_actual is None:
        print("    FAIL: rt60_actual missing in materials mode")
        ok = False
    elif not (0.2 < float(rt60_actual) < 2.5):
        print(f"    WARN: rt60_actual={rt60_actual} outside [0.2, 2.5] s")

    print("[c] Hardware-realistic impairments change the audio ...")
    body_clean = base_payload()
    body_clean["diffuse"] = True
    body_clean["rt60"] = 0.8
    r_clean = post("/simulate", body_clean)
    body_hw = {**body_clean,
               "mic_mismatch": True,
               "crosstalk": True,
               "quantization": True,
               "crosstalk_db": -40.0,
               "bit_depth": 16}
    r_hw = post("/simulate", body_hw)
    raw_clean = decode_wav_b64(r_clean["raw_audio_b64"])
    raw_hw = decode_wav_b64(r_hw["raw_audio_b64"])
    # Same length check
    n = min(len(raw_clean), len(raw_hw))
    diff = raw_clean[:n] - raw_hw[:n]
    diff_rms = float(np.sqrt(np.mean(diff ** 2)))
    ref_rms = float(np.sqrt(np.mean(raw_clean[:n] ** 2))) + 1e-9
    rel_diff = diff_rms / ref_rms
    print(f"    raw-clean RMS={ref_rms:.1f} "
          f"diff RMS={diff_rms:.1f} (rel={rel_diff*100:.2f}%)")
    bf_clean = decode_wav_b64(r_clean["audio_b64"])
    bf_hw = decode_wav_b64(r_hw["audio_b64"])
    n2 = min(len(bf_clean), len(bf_hw))
    bf_diff = bf_clean[:n2] - bf_hw[:n2]
    bf_rms_diff = float(np.sqrt(np.mean(bf_diff ** 2)))
    bf_ref_rms = float(np.sqrt(np.mean(bf_clean[:n2] ** 2))) + 1e-9
    bf_rel = bf_rms_diff / bf_ref_rms
    print(f"    bf-clean RMS={bf_ref_rms:.1f} "
          f"diff RMS={bf_rms_diff:.1f} (rel={bf_rel*100:.2f}%)")
    if rel_diff < 0.01:
        print("    FAIL: HW impairments produced <1% audio difference")
        ok = False

    print("[f] Top-3 peaks + power grid sanity ...")
    tops = r1.get("top_peaks") or []
    for p in tops[:3]:
        missing = {"az_deg", "el_deg", "rel_db"} - set(p.keys())
        if missing:
            print(f"    FAIL: top peak missing keys: {missing}")
            ok = False

    print("[g] Phase 2b moving source + zero-speed equivalence ...")
    body_move = base_payload()
    body_move["moving_source"] = True
    body_move["trajectory_type"] = "straight"
    body_move["speed_mps"] = 10.0
    body_move["heading_deg"] = 45.0
    body_move["n_trajectory_chunks"] = 6
    r_move = post("/simulate", body_move)
    traj = r_move.get("trajectory") or []
    print(f"    trajectory len = {len(traj)} (expected 6)")
    print(f"    moving est az={r_move['est_az_deg']} el={r_move['est_el_deg']}")
    if len(traj) != 6:
        print(f"    FAIL: trajectory should have 6 points, got {len(traj)}")
        ok = False

    body_zero = dict(body_move)
    body_zero["speed_mps"] = 0.0
    r_zero = post("/simulate", body_zero)
    static_body = base_payload()
    r_static = post("/simulate", static_body)
    az_diff = abs(r_zero["est_az_deg"] - r_static["est_az_deg"])
    az_diff = min(az_diff, 360 - az_diff)
    el_diff = abs(r_zero["est_el_deg"] - r_static["est_el_deg"])
    print(f"    zero-speed DOA matches static within "
          f"az={az_diff:.1f} el={el_diff:.1f}")
    if az_diff > 5 or el_diff > 5:
        print(f"    FAIL: speed=0 moving source must match static DOA")
        ok = False

    print("[h] Phase 2b atmosphere (temp gradient) ...")
    body_atm = base_payload()
    body_atm["temperature_c"] = 22.0
    body_atm["humidity_pct"] = 55.0
    body_atm["temp_gradient_c_per_m"] = 2.0
    r_atm = post("/simulate", body_atm)
    atm_bias = r_atm.get("atmospheric_bias_deg")
    print(f"    atmospheric_bias_deg = {atm_bias}")
    if atm_bias is None:
        print("    FAIL: atmospheric_bias_deg missing from response")
        ok = False
    elif abs(float(atm_bias)) < 0.01:
        print(f"    FAIL: atm bias should be non-zero with grad=2.0 C/m")
        ok = False

    print("[i] Phase 3A MAX78000 ML-path preview ...")
    body_ml = base_payload()
    body_ml["ml_preview"] = True
    body_ml["ml_bit_depth"] = 8
    body_ml["ml_feature_bit_depth"] = 8
    body_ml["ml_n_mels"] = 64
    r_ml = post("/simulate", body_ml)
    ml_snr = r_ml.get("ml_path_snr_db")
    feat_snr = r_ml.get("feature_snr_db")
    ml_wav_b64 = r_ml.get("ml_audio_b64")
    ml_png_b64 = r_ml.get("ml_spectrogram_png_b64")
    print(f"    ml_path_snr_db = {ml_snr}  feature_snr_db = {feat_snr}")
    if ml_wav_b64 is None or ml_snr is None or feat_snr is None:
        print("    FAIL: ML preview response missing fields")
        ok = False
    else:
        try:
            ml_audio = decode_wav_b64(ml_wav_b64)
            print(f"    ml_audio samples = {len(ml_audio)}")
            if len(ml_audio) < 8000:
                print("    FAIL: ML audio shorter than 0.5 s")
                ok = False
        except Exception as e:
            print(f"    FAIL: ml_audio_b64 did not decode as WAV: {e}")
            ok = False
        if float(ml_snr) < 20.0 or float(ml_snr) > 120.0:
            print(f"    FAIL: ml_path_snr_db out of expected range (20..120): {ml_snr}")
            ok = False
    if ml_png_b64:
        try:
            png_raw = base64.b64decode(ml_png_b64)
            if not png_raw.startswith(b"\x89PNG\r\n\x1a\n"):
                print("    FAIL: ml_spectrogram_png_b64 is not a valid PNG")
                ok = False
            else:
                print(f"    spectrogram PNG ok ({len(png_raw)} bytes)")
        except Exception as e:
            print(f"    FAIL: ml_spectrogram_png_b64 did not decode: {e}")
            ok = False
    else:
        print("    FAIL: ml_spectrogram_png_b64 missing from response")
        ok = False

    # int8 vs int16 sanity: int16 should yield strictly higher SNR.
    body_ml16 = dict(body_ml, ml_bit_depth=16, ml_feature_bit_depth=16)
    r_ml16 = post("/simulate", body_ml16)
    ml_snr_16 = r_ml16.get("ml_path_snr_db")
    if ml_snr is not None and ml_snr_16 is not None:
        print(f"    int16 ml_path_snr_db = {ml_snr_16} (should exceed int8 {ml_snr})")
        if float(ml_snr_16) <= float(ml_snr):
            print("    FAIL: int16 should give higher ML SNR than int8")
            ok = False

    print("[j] Phase 3B plane-wave crowd model ...")
    body_pw = base_payload()
    body_pw["diffuse"] = True
    body_pw["crowd_model"] = "plane_wave"
    body_pw["n_plane_waves"] = 32
    r_pw = post("/simulate", body_pw)
    print(f"    est az={r_pw['est_az_deg']} el={r_pw['est_el_deg']} "
          f"elapsed={r_pw.get('elapsed_s')}s")
    if r_pw.get("est_az_deg") is None or r_pw.get("power") is None:
        print("    FAIL: plane_wave response missing fields")
        ok = False

    print("[k] Phase 3C FIR capacitive crosstalk ...")
    body_fir = base_payload()
    body_fir["crosstalk"] = True
    body_fir["crosstalk_db"] = -30.0
    body_fir["crosstalk_model"] = "fir_capacitive"
    body_fir["crosstalk_corner_hz"] = 500.0
    r_fir = post("/simulate", body_fir)
    body_simple = dict(body_fir, crosstalk_model="simple")
    r_simple = post("/simulate", body_simple)
    print(f"    fir est az={r_fir['est_az_deg']} el={r_fir['est_el_deg']}; "
          f"simple est az={r_simple['est_az_deg']} el={r_simple['est_el_deg']}")
    raw_fir = decode_wav_b64(r_fir["raw_audio_b64"])
    raw_simple = decode_wav_b64(r_simple["raw_audio_b64"])
    n = min(len(raw_fir), len(raw_simple))
    diff_rms = float(np.sqrt(np.mean((raw_fir[:n] - raw_simple[:n]) ** 2)))
    ref_rms = float(np.sqrt(np.mean(raw_simple[:n] ** 2))) + 1e-9
    rel = diff_rms / ref_rms
    print(f"    FIR vs simple raw-mic relative RMS diff = {rel*100:.2f}%")
    if rel < 1e-4:
        print("    FAIL: FIR crosstalk did not change the audio at all")
        ok = False

    print("[l] Phase 3+ DOA-band tuning + harmonic comb ...")
    body_band = base_payload()
    body_band["fmin_hz"] = 300.0
    body_band["fmax_hz"] = 2500.0
    body_band["harmonic_comb"] = True
    body_band["drone_fundamental_hz"] = 200.0
    r_band = post("/simulate", body_band)
    az_err = abs((r_band["est_az_deg"] - 60) % 360)
    az_err = min(az_err, 360 - az_err)
    el_err = abs(r_band["est_el_deg"] - 30)
    n_bins = r_band.get("n_freq_bins")
    echoed_band = f"{r_band.get('fmin_hz')}-{r_band.get('fmax_hz')} Hz"
    print(f"    est az={r_band['est_az_deg']} el={r_band['est_el_deg']} "
          f"(err az={az_err:.1f} el={el_err:.1f})")
    print(f"    server echoed band={echoed_band} comb={r_band.get('harmonic_comb')} "
          f"f0={r_band.get('drone_fundamental_hz')} Hz  n_bins={n_bins}")
    if not r_band.get("harmonic_comb", False):
        print("    FAIL: harmonic_comb was not echoed back as True")
        ok = False
    if n_bins is None or n_bins < 1:
        print("    FAIL: n_freq_bins missing or zero")
        ok = False
    # Comb at f0=200 Hz across a 300-2500 Hz band should produce far fewer
    # bins than the flat default (~143 at 15.6 Hz/bin). 40 is a generous
    # ceiling that still catches "comb silently fell back to flat".
    elif n_bins > 40:
        print(f"    FAIL: comb kept too many bins ({n_bins}) -- fallback bug?")
        ok = False
    if az_err > 10 or el_err > 20:
        print(f"    FAIL: DOA drifted too far under comb: "
              f"az_err={az_err:.1f} el_err={el_err:.1f}")
        ok = False

    if ok:
        print("\nSMOKE PASS")
        return 0
    print("\nSMOKE FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
