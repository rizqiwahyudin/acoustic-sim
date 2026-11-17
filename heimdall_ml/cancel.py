"""Adaptive interference canceller: Ch0 (beam on the candidate) minus filtered Ch1 (beam on the interferer).

The core is the normalised LMS of CMSIS-DSP arm_lms_norm_f32: float32, the reference energy kept as a running
sum over the taps, the step mu * error / (energy + FLT_EPSILON). One difference: the energy is clamped at zero,
so float rounding in silence cannot give a negative energy. Processing is in blocks with one learn/freeze
decision per block, so the firmware can run the update on a learning block and only the FIR (arm_fir_f32, same
coefficients) on a frozen block. Ch0 is delayed by `delay` samples so that the filter can also model a lag in
which Ch1 is late.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

FLT_EPSILON = np.float32(1.1920929e-07)


@dataclass(frozen=True)
class CancelConfig:
    taps: int = 64                       # 4 ms at 16 kHz
    delay: int = 32                      # Ch0 delay in samples: lags of -2 to +2 ms
    mu: float = 0.05                     # normalised step size
    block: int = 256                     # 16 ms at 16 kHz; one learn/freeze decision per block
    gate_ratio_db: float | None = 6.0    # learn only while Ch1 is this far above Ch0 in a block; None = always


class _Lines:
    """The delay lines of one Ch0/Ch1 pair: the last `taps` Ch1 samples and the last `delay` Ch0 samples."""

    def __init__(self, config: CancelConfig):
        self.reference = np.zeros(config.taps, dtype=np.float32)
        self.primary = np.zeros(config.delay, dtype=np.float32)


class Canceller:
    """Streaming: call process() with consecutive pieces; a partial block waits for the next call."""

    def __init__(self, config: CancelConfig = CancelConfig()):
        if config.taps < 1 or config.delay < 0 or config.block < 1:
            raise ValueError("taps and block must be positive, delay not negative")
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.weights = np.zeros(self.config.taps, dtype=np.float32)   # weights[k] multiplies Ch1[n - k]
        self.energy = np.float32(0.0)
        self.blocks = 0
        self.learning_blocks = 0
        self._lines = _Lines(self.config)
        self._shadow_lines: list[_Lines] = []
        self._pending = (np.zeros(0, np.float32), np.zeros(0, np.float32))

    def gate(self, primary_block: np.ndarray, reference_block: np.ndarray) -> bool:
        if self.config.gate_ratio_db is None:
            return True
        ratio = (float(np.dot(reference_block, reference_block)) + 1e-20) / \
                (float(np.dot(primary_block, primary_block)) + 1e-20)
        return 10.0 * np.log10(ratio) >= self.config.gate_ratio_db

    def process(self, primary: np.ndarray, reference: np.ndarray, learn: bool | None = None,
                shadows: tuple = ()) -> tuple[np.ndarray, list[np.ndarray]]:
        """The cleaned Ch0 of the whole blocks. `learn` overrides the gate for these samples. `shadows` are
        (Ch0, Ch1) pairs that go through their own delay lines and the same, current coefficients, without
        changing them: for analysis, because the canceller is linear for given coefficients."""
        primary = np.concatenate([self._pending[0], np.asarray(primary, dtype=np.float32)])
        reference = np.concatenate([self._pending[1], np.asarray(reference, dtype=np.float32)])
        if len(primary) != len(reference):
            raise ValueError("Ch0 and Ch1 must have the same length")
        size = self.config.block
        whole = len(primary) // size * size
        if shadows and (whole != len(primary) or any(len(p) != whole or len(r) != whole for p, r in shadows)):
            raise ValueError("with shadows, give whole blocks of the same length as Ch0, with nothing pending")
        while len(self._shadow_lines) < len(shadows):
            self._shadow_lines.append(_Lines(self.config))
        self._pending = (primary[whole:], reference[whole:])
        output = np.zeros(whole, dtype=np.float32)
        shadow_outputs = [np.zeros(whole, dtype=np.float32) for _ in shadows]
        for start in range(0, whole, size):
            block = slice(start, start + size)
            learning = self.gate(primary[block], reference[block]) if learn is None else learn
            self.blocks += 1
            self.learning_blocks += int(learning)
            shadow_blocks = [(np.asarray(p[block], np.float32), np.asarray(r[block], np.float32), lines, out[block])
                             for (p, r), lines, out in zip(shadows, self._shadow_lines, shadow_outputs)]
            output[block] = self._block(primary[block], reference[block], learning, shadow_blocks)
        return output, shadow_outputs

    def _block(self, p_block, r_block, learning, shadow_blocks):
        taps, delay = self.config.taps, self.config.delay
        mu = np.float32(self.config.mu)
        ref = np.concatenate([self._lines.reference, r_block])     # ref[n + taps] is the new Ch1 sample
        pri = np.concatenate([self._lines.primary, p_block])       # pri[n] is Ch0 delayed by `delay`
        shadow = [(np.concatenate([lines.primary, p]), np.concatenate([lines.reference, r]), lines, out)
                  for p, r, lines, out in shadow_blocks]
        weights = self.weights
        energy = self.energy
        out = np.empty(len(p_block), dtype=np.float32)
        for n in range(len(p_block)):
            window = ref[n + taps:n:-1]                             # Ch1[n], Ch1[n-1], ..., Ch1[n-taps+1]
            energy = np.float32(max(np.float32(0.0), energy - ref[n] * ref[n] + ref[n + taps] * ref[n + taps]))
            for s_pri, s_ref, _, s_out in shadow:
                s_out[n] = s_pri[n] - np.dot(weights, s_ref[n + taps:n:-1])
            error = np.float32(pri[n] - np.dot(weights, window))
            out[n] = error
            if learning:
                weights += np.float32(error * mu / (energy + FLT_EPSILON)) * window
        self.energy = energy
        self._lines.reference = ref[-taps:]
        self._lines.primary = pri[len(pri) - delay:]
        for s_pri, s_ref, lines, _ in shadow:
            lines.reference = s_ref[-taps:]
            lines.primary = s_pri[len(s_pri) - delay:]
        return out


def fit_fir(primary: np.ndarray, reference: np.ndarray, taps: int, delay: int,
            loading: float = 1e-4) -> np.ndarray:
    """Least-squares weights of a frozen stage from a calibration recording: primary delayed by `delay` is
    about sum_k w[k] reference[n - k] (the Canceller convention). The MCU can solve the same normal equations
    with arm_mat_cholesky_f32 and the triangular solvers of CMSIS-DSP."""
    p = np.asarray(primary, dtype=np.float32)
    r = np.asarray(reference, dtype=np.float32)
    padded = np.concatenate([np.zeros(taps - 1, np.float32), r])
    rows = np.lib.stride_tricks.sliding_window_view(padded, taps)[:, ::-1]    # row n: r[n], ..., r[n-taps+1]
    target = np.concatenate([np.zeros(delay, np.float32), p])[:len(p)]
    matrix = rows.T @ rows
    matrix += np.float32(loading * np.trace(matrix) / taps) * np.eye(taps, dtype=np.float32)
    return cho_solve(cho_factor(matrix), rows.T @ target).astype(np.float32)


def cycles_per_second(config: CancelConfig, rate_hz: int = 16000, learning_share: float = 1.0) -> float:
    """A rough M4 budget: about 2 cycles per tap and sample for the filter, about 3 more while learning
    (a dot product and a scaled add, as CMSIS-DSP does them). An estimate, to be measured with HEALTH."""
    return rate_hz * config.taps * (2.0 + 3.0 * learning_share)
