"""Sanity check the mel-axis fix.

Renders a log-mel spectrogram of a 500 Hz pure tone at two different DOA
bands (50-2000 Hz and 50-7000 Hz). For each render, the brightest row
must land at the mel position that corresponds to 500 Hz -- regardless of
the fmin / fmax of the plot axis.

Also writes both PNGs to disk (as <scratch>/ml_spec_narrow.png and
ml_spec_wide.png) so a human can eyeball that the 500 Hz tick aligns
with the bright stripe in both cases.
"""
from __future__ import annotations

import base64
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acoustic_utils import (  # noqa: E402
    _hz_to_mel,
    log_mel_features,
    render_spectrogram_png_b64,
)


def main() -> int:
    fs = 16000
    n_mels = 64
    duration_s = 1.0
    tone_hz = 500.0

    t = np.arange(int(fs * duration_s)) / fs
    tone = np.sin(2 * np.pi * tone_hz * t)

    scratch = ROOT / "tests" / "_smoke_out"
    scratch.mkdir(exist_ok=True)

    cases = [
        ("narrow", 50.0, 2000.0),
        ("wide", 50.0, 7000.0),
    ]

    ok = True
    for tag, fmin, fmax in cases:
        mel = log_mel_features(tone, fs=fs, n_mels=n_mels,
                               fmin=fmin, fmax=fmax)
        # Find the brightest row, averaged across time.
        peak_row = int(np.argmax(mel.mean(axis=1)))

        # Expected row for 500 Hz under this (fmin, fmax, n_mels):
        mel_min = float(_hz_to_mel(fmin))
        mel_max = float(_hz_to_mel(fmax))
        expected_frac = (float(_hz_to_mel(tone_hz)) - mel_min) / (mel_max - mel_min)
        expected_row = int(round(expected_frac * (n_mels - 1)))

        drift = abs(peak_row - expected_row)
        status = "OK" if drift <= 2 else "FAIL"
        print(f"[{tag:6s}] fmin-fmax={fmin:.0f}-{fmax:.0f} Hz  "
              f"peak_row={peak_row}  expected_row={expected_row}  "
              f"drift={drift} bins  [{status}]")
        if drift > 2:
            ok = False

        png_b64 = render_spectrogram_png_b64(
            mel, fs=fs, n_mels=n_mels, fmin=fmin, fmax=fmax,
        )
        out = scratch / f"ml_spec_{tag}.png"
        out.write_bytes(base64.b64decode(png_b64))
        print(f"            wrote {out}  ({out.stat().st_size} bytes)")

    print()
    print("SPECTROGRAM SMOKE " + ("PASS" if ok else "FAIL"))
    print(f"Eyeball the PNGs in {scratch} -- in both, the bright stripe "
          f"should land right on the '500 Hz' y-tick.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
