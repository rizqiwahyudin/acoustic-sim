import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heimdall_ml.cancel import CancelConfig, Canceller  # noqa: E402

RNG = np.random.default_rng(11)


def leaked(reference, lag=7, gain=0.4):
    """Ch0 holding the interferer of Ch1 through a short filter (a sidelobe and a path difference)."""
    path = np.zeros(lag + 3)
    path[lag], path[lag + 1], path[lag + 2] = gain, -0.5 * gain, 0.2 * gain
    return np.convolve(reference, path)[:len(reference)]


def power_db(x):
    return 10.0 * np.log10(np.mean(np.square(x.astype(np.float64))) + 1e-20)


def test_learns_and_removes_a_directional_interferer():
    x = RNG.standard_normal(16000 * 4).astype(np.float32)
    canceller = Canceller(CancelConfig(gate_ratio_db=None))
    out, _ = canceller.process(leaked(x), x)
    tail = slice(16000 * 3, None)
    delayed = np.concatenate([np.zeros(32), leaked(x)])[:len(out)]
    assert power_db(delayed[tail]) - power_db(out[tail]) > 30.0
    assert canceller.learning_blocks == canceller.blocks == len(x) // 256


def test_frozen_coefficients_do_not_change():
    x = RNG.standard_normal(4096).astype(np.float32)
    canceller = Canceller()
    out, _ = canceller.process(leaked(x), x, learn=False)
    assert not canceller.weights.any() and canceller.learning_blocks == 0
    assert np.allclose(out[32:], leaked(x)[:len(out) - 32])


def test_streaming_matches_one_call():
    x = RNG.standard_normal(5000).astype(np.float32)
    d = leaked(x) + 0.1 * RNG.standard_normal(5000).astype(np.float32)
    whole, pieces = Canceller(CancelConfig(gate_ratio_db=None)), Canceller(CancelConfig(gate_ratio_db=None))
    one, _ = whole.process(d, x)
    parts = [pieces.process(d[i:i + 333], x[i:i + 333])[0] for i in range(0, len(x), 333)]
    assert np.allclose(np.concatenate(parts), one, atol=1e-6)
    assert np.allclose(pieces.weights, whole.weights, atol=1e-6)


def test_shadows_split_the_output_exactly():
    n = 256 * 40
    x_target, x_noise = RNG.standard_normal(n), RNG.standard_normal(n)
    p_target, r_target = x_target, 0.2 * np.roll(x_target, 3)
    p_noise, r_noise = leaked(x_noise), x_noise
    canceller = Canceller(CancelConfig(gate_ratio_db=None))
    out, (out_target, out_noise) = canceller.process(p_target + p_noise, r_target + r_noise,
                                                     shadows=((p_target, r_target), (p_noise, r_noise)))
    assert np.allclose(out, out_target + out_noise, atol=1e-4)


def test_gate_learns_only_when_the_reference_dominates():
    x = RNG.standard_normal(256 * 20).astype(np.float32)
    canceller = Canceller(CancelConfig(gate_ratio_db=6.0))
    canceller.process(0.1 * x, x)            # Ch1 20 dB above Ch0: learn
    assert canceller.learning_blocks == 20
    canceller.process(x, x)                  # equal: freeze
    assert canceller.learning_blocks == 20 and canceller.blocks == 40
