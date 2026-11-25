import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heimdall_ml import model_input  # noqa: E402


def sdk_add_transpose(samples: np.ndarray) -> np.ndarray:
    """AddTranspose of the SDK kws20 demo (C:/MaximSDK/Examples/MAX78002/CNN/kws20_demo/main.c), one call."""
    out = np.zeros(model_input.KWS20_SAMPLES, dtype=np.uint8)
    row = col = 0
    for value in samples:
        group = (col % 64) // 4
        word_row = (1 if col >= 64 else 0) + (row << 1)
        out[1024 * group + 4 * word_row + col % 4] = value
        col += 1
        if col >= 128:
            col, row = 0, row + 1
    return out


def test_parity_with_the_firmware_checksums():
    # The C test pushes 16,384 + 1000 samples into the ring: the window is the last 16,384.
    x = model_input.parity_signal(model_input.KWS20_SAMPLES + 1000)
    gain = model_input.kws20_layout(model_input.window_int8(x, "gain", 1.0))
    peak = model_input.kws20_layout(model_input.window_int8(x, "peak"))
    assert model_input.checksum(gain) == model_input.CHECKSUM_GAIN_1
    assert model_input.checksum(peak) == model_input.CHECKSUM_PEAK


def test_rounding_and_clipping_follow_the_firmware():
    s = model_input.to_int16(np.array([0.0, 0.5, -0.5, 1.0, -1.0, 1 / 65536, -1 / 65536]))
    assert s.tolist() == [0, 16384, -16384, 32767, -32768, 1, -1]
    # round(s / 256), halves away from zero (numpy alone would round 0.5 to 0).
    q = model_input.to_int8(np.array([128, -128, 127, 384, 32767, -32768], dtype=np.int16))
    assert q.tolist() == [1, -1, 0, 2, 127, -128]
    assert model_input.to_int8(np.array([16384, -32768], dtype=np.int16), "peak").tolist() == [64, -127]
    assert not model_input.to_int8(np.zeros(4, dtype=np.int16), "peak").any()


def test_layout_is_the_sdk_transpose():
    q = (np.arange(model_input.KWS20_SAMPLES) * 31 + np.arange(model_input.KWS20_SAMPLES) // 256).astype(np.uint8)
    assert np.array_equal(model_input.kws20_layout(q.view(np.int8)), sdk_add_transpose(q))
    assert len(set(model_input.kws20_offsets().tolist())) == model_input.KWS20_SAMPLES


def test_window_takes_the_last_samples_and_shapes_the_tensor():
    x = np.concatenate([np.full(100, 0.9), np.linspace(-0.5, 0.5, model_input.KWS20_SAMPLES)])
    q = model_input.window_int8(x)
    assert q.shape == (model_input.KWS20_SAMPLES,) and q[0] == -64 and q[-1] == 64
    tensor = model_input.kws20_tensor(q)
    assert tensor.shape == (128, 128) and tensor[0, 1] == q[128] and tensor[1, 0] == q[1]
    with pytest.raises(ValueError):
        model_input.window_int8(np.zeros(100))


def test_tensor_is_the_ai8x_fold_and_the_accelerator_channel_order():
    q = (np.arange(model_input.KWS20_SAMPLES) % 251 - 125).astype(np.int8)
    tensor = model_input.kws20_tensor(q)
    # ai8x-training datasets/kws20.py: torch.transpose(audio.reshape((-1, 128)), 1, 0).
    assert np.array_equal(tensor, q.reshape(-1, 128).T)
    # The accelerator holds input channel c in byte c % 4 of processor quad (c % 64) // 4, pass c // 64,
    # and step t in word 2 t + pass: tensor[c, t] must land there.
    c, t = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
    offsets = 1024 * ((c % 64) // 4) + 4 * ((c >= 64) + 2 * t) + c % 4
    assert np.array_equal(model_input.kws20_layout(q).view(np.int8)[offsets], tensor)


def test_softmax_matches_the_firmware():
    top, p = model_input.softmax(np.array([-5000, -5000 + 16384]))
    assert top == 1 and abs(p[1] - 0.7310586) < 1e-6
