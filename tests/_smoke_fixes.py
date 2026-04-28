"""Ad-hoc smoke for the two UX fixes:
  1. normalize_audio=False preserves absolute amplitude (1/r^2 visible in RMS).
  2. ML spectrogram band follows the DOA band.
"""
import base64
import io
import json
import sys
import urllib.request
import wave

import numpy as np

BASE = "http://127.0.0.1:8766"


def post(body):
    req = urllib.request.Request(
        url=BASE + "/simulate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def rms_of_wav(b64):
    wav_bytes = base64.b64decode(b64)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n = wf.getnframes()
        data = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(data ** 2)))


def main():
    # Fix the seed and use a long integration so drone-audio segment
    # variation doesn't dominate the RMS we're comparing across trials.
    base = dict(
        geometry="CYLINDER", rt60=0.0, diffuse=False, integration_ms=1000,
        source_az_deg=60, source_el_deg=30, drone_spl_db=78,
        seed=42, mic_noise_floor_db=0,  # silence mic floor so drone dominates
    )

    print("### Fix 1: 1/r^2 with normalize_audio=False ###")
    prev_db = None
    for d in (2, 4, 8, 16):
        r = post({**base, "source_distance": float(d), "normalize_audio": False})
        rms = rms_of_wav(r["raw_audio_b64"])
        db = 20 * np.log10(max(rms, 1e-9))
        step = f" (delta {db - prev_db:+.1f} dB)" if prev_db is not None else ""
        print(f"  d={d}m: raw.wav RMS dBFS = {db:.1f}{step}  "
              f"(expected step: -6.0 dB)")
        prev_db = db

    print()
    print("### Legacy: normalize_audio=True -> equal loudness across trials ###")
    for d in (2, 8):
        r = post({**base, "source_distance": float(d), "normalize_audio": True})
        rms = rms_of_wav(r["raw_audio_b64"])
        db = 20 * np.log10(max(rms, 1e-9))
        print(f"  d={d}m: raw.wav RMS dBFS = {db:.1f}")

    print()
    print("### Fix 2: ML spectrogram band follows DOA band ###")
    for (fmin, fmax) in [(200, 2000), (50, 5000), (800, 1200)]:
        r = post({
            **base, "source_distance": 6.0,
            "ml_preview": True, "fmin_hz": fmin, "fmax_hz": fmax,
            "normalize_audio": True,
        })
        png = base64.b64decode(r["ml_spectrogram_png_b64"])
        print(f"  DOA {fmin}-{fmax} Hz: "
              f"ml_path_snr_db={r['ml_path_snr_db']} "
              f"feature_snr={r['feature_snr_db']} "
              f"png={len(png)} bytes")
    print()
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
