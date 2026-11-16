"""Does the adaptive canceller help? A room simulation with the deployed array, and the balcony recordings.

    python scripts/canceller_study.py sim        the drone clip and an interferer (crowd clip from a point, as a
                                                 talker or a PA speaker) in a room; Ch0 on the drone, Ch1 on
                                                 the interferer; interferer alone for 4 s, then both for 8 s
    python scripts/canceller_study.py balcony    the real Ch0 and Ch1 of 2026-10-30 (drone and river in both)

The simulation is linear, so the drone, the interferer and the microphone noise go through the canceller as
separate copies with the same coefficients (Canceller shadows): the dB of interferer removed and of drone lost
are exact, not estimated from the mix.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
from scipy.signal import oaconvolve, welch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from acoustic_utils import air_absorption_kwargs  # noqa: E402
from heimdall_acoustic import load_audio_48k, load_beam_contract  # noqa: E402
from heimdall_ml import dataset, tonal  # noqa: E402
from heimdall_ml.cancel import CancelConfig, Canceller, cycles_per_second  # noqa: E402

FS = 48000
ROOM = (6.0, 6.0, 4.0)                 # the bench living room, about 6 x 6 x 4 m
CENTRE = np.array((3.0, 0.6, 1.2))     # array centre; the array plane faces +y
ALONE_S, BOTH_S, SETTLE_S = 4.0, 8.0, 2.0
CLIP_TONE = tonal.ToneConfig(min_hz=150.0, max_hz=260.0)   # the bench drone clip has f0 about 190 Hz


def mic_positions(contract):
    """Microphones in the room: physical [right, up] offsets on the plane y = CENTRE[1]."""
    offsets = contract.microphone_positions_mm / 1000.0
    offsets = offsets - offsets.mean(axis=0)
    return np.vstack((CENTRE[0] + offsets[:, 0], np.full(len(offsets), CENTRE[1]), CENTRE[2] + offsets[:, 1]))


def source_at(azimuth_deg, distance_m, elevation_deg=0.0):
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    return CENTRE + distance_m * np.array((np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)))


def microphone_signals(signal, source, mics, rt60_s):
    """The source at each microphone through the room (image sources; rt60_s None = free field)."""
    if rt60_s is None:
        room = pra.ShoeBox(list(ROOM), fs=FS, max_order=0)
    else:
        absorption, order = pra.inverse_sabine(rt60_s, list(ROOM))
        room = pra.ShoeBox(list(ROOM), fs=FS, materials=pra.Material(absorption), max_order=min(order, 12),
                           **air_absorption_kwargs(20.0, 50.0))
    room.add_microphone_array(pra.MicrophoneArray(mics, fs=FS))
    room.add_source(source)
    room.compute_rir()
    return np.stack([oaconvolve(signal, np.asarray(room.rir[m][0]))[:len(signal)] for m in range(mics.shape[1])])


def fractional_delay(x, delay, taps=33):
    """x delayed by `delay` samples (integer part shifted, fractional part by a windowed sinc)."""
    whole, fraction = int(np.floor(delay)), delay - np.floor(delay)
    k = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(k - fraction) * np.hamming(taps)
    shifted = np.concatenate([np.zeros(whole + (taps - 1) // 2), x])[:len(x) + (taps - 1) // 2]
    return oaconvolve(shifted, h / h.sum())[taps - 1:taps - 1 + len(x)]


def beam(signals, contract, azimuth_deg, elevation_deg=0.0):
    """Delay-and-sum with the contract's taper, steered by direction cosines as the generator does."""
    positions = contract.microphone_positions_mm / 1000.0
    positions = positions - positions.mean(axis=0)
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    u = np.array((np.cos(el) * np.sin(az), np.sin(el)))
    lead = positions @ u / contract.speed_of_sound_m_s * FS          # samples a microphone hears early
    bulk = np.max(np.linalg.norm(positions, axis=1)) / contract.speed_of_sound_m_s * FS
    gains = np.abs(np.asarray(contract.gains, dtype=np.float64))
    total = sum(g * fractional_delay(x, bulk + d) for g, x, d in zip(gains, signals, lead))
    return total / gains.sum()


def power_db(x):
    return 10.0 * np.log10(np.mean(np.square(np.asarray(x, dtype=np.float64))) + 1e-20)


def median_score(x16, config=CLIP_TONE):
    scores = [t.score_db for t in tonal.tone_reports(x16, tonal.RATE_HZ, config=config)]
    return float(np.median(scores)) if scores else float("nan")


def scene(contract, rt60_s, interferer_az, level_db, seed=0):
    """Ch0 and Ch1 at 16 kHz for each component: drone, interferer, microphone noise."""
    rng = np.random.default_rng(seed)
    n_alone, n_both = int(ALONE_S * FS), int(BOTH_S * FS)
    drone = np.concatenate([np.zeros(n_alone), load_audio_48k(ROOT / "audio" / "drone.wav", BOTH_S)])
    crowd = load_audio_48k(ROOT / "audio" / "crowd.wav", ALONE_S + BOTH_S)
    mics = mic_positions(contract)
    target = microphone_signals(drone, source_at(0.0, 2.0), mics, rt60_s)
    interferer = microphone_signals(crowd, source_at(interferer_az, 1.5), mics, rt60_s)
    reference = power_db(target[:, n_alone:])
    interferer *= 10.0 ** ((reference + level_db - power_db(interferer[:, n_alone:])) / 20.0)
    noise = rng.standard_normal(target.shape) * 10.0 ** ((reference - 40.0) / 20.0)
    parts = {}
    for name, signals in (("drone", target), ("interferer", interferer), ("noise", noise)):
        parts[name] = (tonal.decimate_48k(beam(signals, contract, 0.0)),
                       tonal.decimate_48k(beam(signals, contract, interferer_az)))
    return parts


def run(parts, config, mode):
    """mode: always (learn on every block), gate (Ch1/Ch0 power ratio), oracle (learn while the interferer is
    alone, then freeze)."""
    split = int(ALONE_S * tonal.RATE_HZ) // config.block * config.block
    total = len(parts["drone"][0]) // config.block * config.block
    canceller = Canceller(config if mode != "always" else CancelConfig(**{**config.__dict__, "gate_ratio_db": None}))
    names = list(parts)
    outputs = {name: [] for name in names + ["mix"]}
    for start, stop, learn in ((0, split, True if mode == "oracle" else None),
                               (split, total, False if mode == "oracle" else None)):
        pieces = {name: (parts[name][0][start:stop], parts[name][1][start:stop]) for name in names}
        mix = (sum(p[0] for p in pieces.values()), sum(p[1] for p in pieces.values()))
        out, shadows = canceller.process(*mix, learn=learn, shadows=tuple(pieces[name] for name in names))
        outputs["mix"].append(out)
        for name, shadow in zip(names, shadows):
            outputs[name].append(shadow)
    return {name: np.concatenate(chunks) for name, chunks in outputs.items()}, canceller


def delayed(x, config):
    return np.concatenate([np.zeros(config.delay), x])[:len(x)]


def command_sim(args):
    contract = load_beam_contract()
    config = CancelConfig(taps=args.taps, mu=args.mu)
    print(f"canceller: {config.taps} taps ({1000 * config.taps / 16000:.0f} ms), mu {config.mu}, "
          f"block {config.block}; M4 estimate {cycles_per_second(config) / 60e6:.0%} of 60 MHz while learning, "
          f"{cycles_per_second(config, learning_share=0) / 60e6:.0%} frozen")
    print(f"evaluation: the last {BOTH_S - SETTLE_S:.0f} s with drone and interferer; drone 2 m ahead, interferer "
          "1.5 m away; Ch0 and Ch1 delayed alike")
    print(f"{'room':10s} {'interf.':>8s} {'level':>6s} {'mode':7s} {'SIR in':>7s} {'interf.':>8s} {'drone':>6s} "
          f"{'SIR gain':>8s} {'tonal score':>12s} {'learned':>8s}")
    print(f"{'':10s} {'az':>8s} {'dB':>6s} {'':7s} {'dB':>7s} {'removed':>8s} {'lost':>6s} {'dB':>8s} "
          f"{'before/after':>12s} {'blocks':>8s}")
    cases = [(rt60, az, level) for rt60 in (None, 0.4) for az in (30.0,) for level in (0.0, 10.0)]
    cases += [(0.4, 15.0, 10.0)]
    for rt60, az, level in cases:
        parts = scene(contract, rt60, az, level)
        start = int((ALONE_S + SETTLE_S) * tonal.RATE_HZ)
        for mode in ("always", "gate", "oracle"):
            out, canceller = run(parts, config, mode)
            window = slice(start, len(out["mix"]))
            before = {name: delayed(parts[name][0], config)[:len(out["mix"])][window] for name in parts}
            sir_in = power_db(before["drone"]) - power_db(before["interferer"])
            removed = power_db(before["interferer"]) - power_db(out["interferer"][window])
            lost = power_db(before["drone"]) - power_db(out["drone"][window])
            mix_before = sum(before.values())
            print(f"{'free field' if rt60 is None else f'RT60 {rt60} s':10s} {az:+7.0f}° {level:+6.0f} {mode:7s} "
                  f"{sir_in:+7.1f} {removed:+8.1f} {lost:+6.1f} {removed - lost:+8.1f} "
                  f"{median_score(mix_before):5.1f}/{median_score(out['mix'][window]):5.1f} "
                  f"{canceller.learning_blocks:4d}/{canceller.blocks:<4d}")


def command_balcony(args):
    config = CancelConfig(taps=args.taps, mu=args.mu, gate_ratio_db=None)
    recordings = [r for r in dataset.load_recordings() if r.get("date") == "2026-10-30"]
    print("balcony 2026-10-30: Ch0 = steered beam, Ch1 = a bank beam 33-44 deg away; drone and river in both")
    print(f"{'recording':34s} {'dist':>5s} {'total':>7s} {'20-500 Hz':>10s} {'650-820 Hz':>11s} "
          f"{'1.3-3.4 kHz':>12s} {'tonal score':>12s} {'gate would':>11s}")
    print(f"{'':34s} {'m':>5s} {'dB':>7s} {'(river) dB':>10s} {'(drone) dB':>11s} {'(drone) dB':>12s} "
          f"{'before/after':>12s} {'learn':>11s}")
    for recording in recordings:
        data, rate = dataset.read_wav(recording.wav)
        ch0, ch1 = tonal.to_16k(data[:, 0], rate), tonal.to_16k(data[:, 1], rate)
        canceller = Canceller(config)
        out, _ = canceller.process(ch0, ch1)
        before = delayed(ch0, config)[:len(out)]
        window = slice(int(5 * tonal.RATE_HZ), len(out))
        f, p_before = welch(before[window], tonal.RATE_HZ, nperseg=4096)
        _, p_after = welch(out[window], tonal.RATE_HZ, nperseg=4096)

        def change(low, high):
            band = (f >= low) & (f < high)
            return 10.0 * np.log10(p_after[band].sum() / p_before[band].sum())

        gate = Canceller(CancelConfig(taps=args.taps, mu=args.mu))
        gate.process(ch0, ch1)
        print(f"{recording.wav.name:34s} {recording.get('distance_m') or 0:5.0f} "
              f"{power_db(out[window]) - power_db(before[window]):+7.1f} {change(20, 500):+10.1f} "
              f"{change(650, 820):+11.1f} {change(1300, 3400):+12.1f} "
              f"{median_score(before[window], tonal.ToneConfig()):5.1f}/"
              f"{median_score(out[window], tonal.ToneConfig()):5.1f} "
              f"{gate.learning_blocks:5d}/{gate.blocks:<5d}")
    print("negative dB = removed by the canceller (always learning). 'gate would learn': blocks in which Ch1 is "
          "6 dB above Ch0")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("sim", "balcony"))
    parser.add_argument("--taps", type=int, default=64)
    parser.add_argument("--mu", type=float, default=0.05)
    args = parser.parse_args(argv)
    (command_sim if args.command == "sim" else command_balcony)(args)


if __name__ == "__main__":
    main()
