"""Does the adaptive canceller help? A room simulation with the deployed array, and the balcony recordings.

    python scripts/canceller_study.py sim        the drone clip and an interferer (crowd clip from a point, as a
                                                 talker or a PA speaker) in a room; Ch0 on the drone, Ch1 on
                                                 the interferer; interferer alone for 4 s, then both for 8 s
    python scripts/canceller_study.py conference the conference hall: drone speaker, a visitor near the array,
                                                 a diffuse crowd; references without the drone (option 1) and
                                                 fixed nulls from the geometry (option 2) against the old one
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
from scipy.signal import oaconvolve, resample_poly, welch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from acoustic_utils import air_absorption_kwargs  # noqa: E402
from heimdall_acoustic import load_audio_48k, load_beam_contract  # noqa: E402
from heimdall_ml import dataset, tonal  # noqa: E402
from heimdall_ml.cancel import CancelConfig, Canceller, cycles_per_second, fit_fir  # noqa: E402

FS = 48000
ROOM = (6.0, 6.0, 4.0)                 # the bench living room, about 6 x 6 x 4 m
CENTRE = np.array((3.0, 0.6, 1.2))     # array centre; the array plane faces +y
ALONE_S, BOTH_S, SETTLE_S = 4.0, 8.0, 2.0
CLIP_TONE = tonal.ToneConfig(min_hz=150.0, max_hz=260.0)   # the bench drone clip has f0 about 190 Hz


def mic_positions(contract, centre=CENTRE):
    """Microphones in the room: physical [right, up] offsets on the plane y = centre[1]."""
    offsets = contract.microphone_positions_mm / 1000.0
    offsets = offsets - offsets.mean(axis=0)
    return np.vstack((centre[0] + offsets[:, 0], np.full(len(offsets), centre[1]), centre[2] + offsets[:, 1]))


def source_at(azimuth_deg, distance_m, elevation_deg=0.0, centre=CENTRE):
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    return centre + distance_m * np.array((np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)))


def microphone_signals(signal, source, mics, rt60_s, room_size=ROOM, max_order=12, fs=FS):
    """The source at each microphone through the room (image sources; rt60_s None = free field)."""
    if rt60_s is None:
        room = pra.ShoeBox(list(room_size), fs=fs, max_order=0)
    else:
        absorption, order = pra.inverse_sabine(rt60_s, list(room_size))
        room = pra.ShoeBox(list(room_size), fs=fs, materials=pra.Material(absorption),
                           max_order=min(order, max_order), **air_absorption_kwargs(20.0, 50.0))
    room.add_microphone_array(pra.MicrophoneArray(mics, fs=fs))
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


def centred_positions(contract):
    positions = contract.microphone_positions_mm / 1000.0
    return positions - positions.mean(axis=0)


def direction(azimuth_deg, elevation_deg=0.0):
    """[right, up] components of a unit vector: all a planar array can see of a direction."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array((np.cos(el) * np.sin(az), np.sin(el)))


def channel_amplitudes(contract, spec):
    """Per-microphone amplitudes of a channel: ("beam", az, el) is the delay-and-sum beam with the contract's
    taper; ("difference", az, el) is the left half minus the right half, each normalised, so a plane wave from
    the steering direction cancels; ("mic", m) is one microphone."""
    gains = np.abs(np.asarray(contract.gains, dtype=np.float64))
    if spec[0] == "mic":
        return np.eye(len(gains))[spec[1]]
    if spec[0] == "beam":
        return gains / gains.sum()
    right = centred_positions(contract)[:, 0]
    left_half, right_half = right < -1e-3, right > 1e-3
    return np.where(left_half, gains / gains[left_half].sum(), 0.0) - \
        np.where(right_half, gains / gains[right_half].sum(), 0.0)


def channel(signals, contract, spec, fs=FS):
    """A Ch0/Ch1 candidate from the microphone signals (see channel_amplitudes)."""
    if spec[0] == "mic":
        return signals[spec[1]].copy()
    positions = centred_positions(contract)
    lead = positions @ direction(spec[1], spec[2]) / contract.speed_of_sound_m_s * fs
    bulk = np.max(np.linalg.norm(positions, axis=1)) / contract.speed_of_sound_m_s * fs
    amplitudes = channel_amplitudes(contract, spec)
    return sum(a * fractional_delay(x, bulk + d) for a, x, d in zip(amplitudes, signals, lead) if a != 0.0)


def beam(signals, contract, azimuth_deg, elevation_deg=0.0):
    """Delay-and-sum with the contract's taper, steered by direction cosines as the generator does."""
    return channel(signals, contract, ("beam", azimuth_deg, elevation_deg))


def response(contract, spec, toward, freqs_hz):
    """Complex response of a beam channel to a plane wave from `toward` (az, el), without the common bulk
    delay: what the firmware can calculate from the array geometry alone."""
    positions = centred_positions(contract)
    path = positions @ (direction(*toward) - direction(spec[1], spec[2])) / contract.speed_of_sound_m_s
    return channel_amplitudes(contract, spec) @ np.exp(2j * np.pi * np.outer(path, freqs_hz))


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


# The conference: a drone speaker 1.5 m ahead, a visitor talking near the array, a diffuse crowd.
HALL = ((20.0, 40.0, 10.0), 1.2, np.array((10.0, 0.5, 1.0)))   # the acoustic emulator's conference hall
LIVING_ROOM = (ROOM, 0.4, CENTRE)
FREE_FIELD = ((20.0, 40.0, 10.0), None, np.array((10.0, 0.5, 1.0)))
CAL_S, RUN_S, RUN_SETTLE_S = 5.0, 12.0, 4.0
CAL_LEVEL_DB, CROWD_DB, NOISE_DB = 10.0, -5.0, -40.0     # re the drone clip at the microphones
CROWD_WAVES = 48
RATE = tonal.RATE_HZ
BANDS = {"<1k": (100.0, 1000.0), "1-4k": (1000.0, 4000.0)}
COMPONENTS = ("drone", "visitor", "crowd", "noise")
_CROWD_CACHE = {}


def pink_noise(n, rng):
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / RATE)
    spectrum[1:] /= np.sqrt(freqs[1:])
    spectrum[(freqs < 80.0) | (freqs > 7000.0)] = 0.0
    x = np.fft.irfft(spectrum, n)
    return x / np.sqrt(np.mean(x * x))


def load_16k(name, duration_s):
    """Both clips are 16 kHz recordings, so the conference runs at the 16 kHz of the firmware stream."""
    x = resample_poly(load_audio_48k(ROOT / "audio" / name, duration_s), 1, 3)
    return x / np.sqrt(np.mean(x * x))


def diffuse_crowd(contract, n, rng, babble):
    """Many talkers far away: independent babble excerpts as plane waves from all around, mostly level."""
    positions = centred_positions(contract)
    signals = np.zeros((len(positions), n))
    for _ in range(CROWD_WAVES):
        u = direction(rng.uniform(-180.0, 180.0), np.degrees(np.arcsin(rng.uniform(-0.2, 0.5))))
        start = int(rng.integers(20 * RATE, len(babble) - n))
        lead = positions @ u / contract.speed_of_sound_m_s * RATE
        for m in range(len(positions)):
            signals[m] += fractional_delay(babble[start:start + n], 20.0 - lead[m])
    return signals


def nearest_bank_cell(contract, azimuth_deg, elevation_deg):
    az = contract.azimuth_deg[np.argmin(np.abs(contract.azimuth_deg - azimuth_deg))]
    el = contract.elevation_deg[np.argmin(np.abs(contract.elevation_deg - elevation_deg))]
    return float(az), float(el)


def conference_scene(contract, room, visitor, visitor_db, aims, seed=0):
    """Each component (drone, visitor, crowd, noise) in each channel at 16 kHz: a calibration part (the drone
    speaker plays pink noise, no visitor) and then the run (the drone clip, the visitor, the crowd)."""
    size, rt60, centre = room
    rng = np.random.default_rng(seed)
    n_cal, n_run = int(CAL_S * RATE), int(RUN_S * RATE)
    v_az, v_el, v_range = visitor
    mics = mic_positions(contract, centre)
    inner = int(np.argmin(np.linalg.norm(centred_positions(contract), axis=1)))
    specs = ([("mic", inner), ("beam", *nearest_bank_cell(contract, v_az, v_el)), ("beam", v_az, v_el)]
             + [("beam", a, 0.0) for a in aims] + [("difference", a, 0.0) for a in aims])
    babble = load_16k("crowd.wav", 180.0)
    drone = np.concatenate([pink_noise(n_cal, rng) * 10.0 ** (CAL_LEVEL_DB / 20.0), load_16k("drone.wav", RUN_S)])
    talk = np.concatenate([np.zeros(n_cal), babble[:n_run]])
    parts, reference = {}, None
    for name in COMPONENTS:
        if name == "drone":
            signals = microphone_signals(drone, source_at(0.0, 1.5, 0.0, centre), mics, rt60, size, 10, RATE)
            reference = power_db(signals[:, n_cal:])
        elif name == "visitor":
            signals = microphone_signals(talk, source_at(v_az, v_range, v_el, centre), mics, rt60, size, 10, RATE)
            signals *= 10.0 ** ((reference + visitor_db - power_db(signals[:, n_cal:])) / 20.0)
        elif name == "crowd":
            if "crowd" not in _CROWD_CACHE:
                _CROWD_CACHE["crowd"] = diffuse_crowd(contract, n_cal + n_run, np.random.default_rng(seed + 1), babble)
            signals = _CROWD_CACHE["crowd"].copy()
            signals *= 10.0 ** ((reference + CROWD_DB - power_db(signals)) / 20.0)
        else:
            signals = rng.standard_normal((len(mics.T), n_cal + n_run)) * 10.0 ** ((reference + NOISE_DB) / 20.0)
        parts[name] = {spec: channel(signals, contract, spec, RATE) for spec in specs}
        print(f"  {name} rendered", flush=True)
        del signals
    return parts, specs


def frozen(primary, reference, weights, delay):
    """What a frozen stage outputs (arm_fir_f32 on Ch1, subtracted from the delayed Ch0)."""
    return np.concatenate([np.zeros(delay), primary])[:len(primary)] - \
        np.convolve(reference, weights)[:len(reference)]


def null_filter(contract, ch0, ch1, toward, protect, taps=64, delay=32, protect_weight=1.0, nfft=1024):
    """Fixed filter from the geometry: Ch1 filtered by it matches Ch0 for a plane wave from `toward`, with a
    penalty on what it would take from `protect` (the drone): H = B0 B1* / (|B1|^2 + w |P1|^2)."""
    freqs = np.fft.rfftfreq(nfft, 1.0 / RATE)
    b0, b1 = response(contract, ch0, toward, freqs), response(contract, ch1, toward, freqs)
    p1 = response(contract, ch1, protect, freqs)
    h = b0 * np.conj(b1) / (np.abs(b1) ** 2 + protect_weight * np.abs(p1) ** 2 + 1e-3)
    h *= np.exp(-2j * np.pi * freqs * delay / RATE)
    return (np.fft.irfft(h, nfft)[:taps] * np.hanning(taps + 2)[1:-1]).astype(np.float32)


def adaptive(ch0, ref, config, run):
    """Over the run part only; the components go through as shadows. Returns them and the learning share."""
    canceller = Canceller(config)
    _, shadows = canceller.process(sum(ch0[n][run] for n in COMPONENTS), sum(ref[n][run] for n in COMPONENTS),
                                   shadows=tuple((ch0[n][run], ref[n][run]) for n in COMPONENTS))
    return dict(zip(COMPONENTS, shadows)), canceller.learning_blocks / canceller.blocks


def conference_methods(contract, parts, specs, aim, mu):
    """Each method: (name, output components, reference components, Ch0 delay, M4 share of 60 MHz)."""
    n_cal = int(CAL_S * RATE)
    total = (len(parts["drone"][specs[0]]) - n_cal) // 256 * 256
    run = slice(n_cal, n_cal + total)
    mic, bank, exact = specs[0], specs[1], specs[2]
    ch0_spec, diff_spec = ("beam", aim, 0.0), ("difference", aim, 0.0)
    ch0 = {n: parts[n][ch0_spec] for n in COMPONENTS}
    learn64 = CancelConfig(taps=64, delay=32, mu=mu, gate_ratio_db=None)
    learn128 = CancelConfig(taps=128, delay=64, mu=mu, gate_ratio_db=None)
    # Learn only while the drone-free reference is within 6 dB of Ch0: someone near is talking.
    gated128 = CancelConfig(taps=128, delay=64, mu=mu, gate_ratio_db=-6.0)
    m4 = lambda config, share: cycles_per_second(config, learning_share=share) / 60e6   # noqa: E731
    results = []

    def add(label, adapted, ref, delay, fixed_share):
        out, share = adapted
        config = learn64 if delay == 32 else learn128
        results.append((label, out, ref, delay, fixed_share + m4(config, share)))

    ref = {n: parts[n][bank] for n in COMPONENTS}
    add("beam reference (old)", adaptive(ch0, ref, learn64, run), ref, 32, 0.0)

    # Option 1, MCU only: Ch1 = one microphone; a frozen stage, fitted while the speaker plays pink noise,
    # takes out what the beam predicts of it (the drone direction); the canceller then always learns.
    cal = slice(0, n_cal)
    calibration = fit_fir(sum(parts[n][mic][cal] for n in COMPONENTS),
                          sum(ch0[n][cal] for n in COMPONENTS), taps=64, delay=32)
    blocked = {n: frozen(parts[n][mic], ch0[n], calibration, 32) for n in COMPONENTS}
    add("1 mic minus beam (MCU)", adaptive(ch0, blocked, learn128, run), blocked, 64, m4(learn64, 0))

    # The same with Ch1 = the bank beam on the visitor: less crowd and microphone noise in the reference.
    calibration = fit_fir(sum(parts[n][bank][cal] for n in COMPONENTS),
                          sum(ch0[n][cal] for n in COMPONENTS), taps=64, delay=32)
    blocked = {n: frozen(parts[n][bank], ch0[n], calibration, 32) for n in COMPONENTS}
    add("1 bank minus beam (MCU)", adaptive(ch0, blocked, learn128, run), blocked, 64, m4(learn64, 0))
    add("1 bank minus beam, gated", adaptive(ch0, blocked, gated128, run), blocked, 64, m4(learn64, 0))

    ref = {n: parts[n][diff_spec] for n in COMPONENTS}
    add("1 difference beam (DSP)", adaptive(ch0, ref, learn64, run), ref, 32, 0.0)

    for label, spec, toward in (("2 null, bank cell", bank, bank[1:]), ("2 null, exact (DSP)", exact, exact[1:])):
        weights = null_filter(contract, ch0_spec, spec, toward, (aim, 0.0))
        out = {n: frozen(ch0[n], parts[n][spec], weights, 32)[run] for n in COMPONENTS}
        results.append((label, out, {n: parts[n][spec] for n in COMPONENTS}, 32, m4(learn64, 0)))
    return results, ch0, run


def spectra(x):
    freqs, psd = welch(np.asarray(x, dtype=np.float64), RATE, nperseg=2048)
    return freqs, psd + 1e-30


def sir_gains(p_in, p_out, freqs):
    """SIR gain (dB) in each band, and at the drone: the gain per frequency bin, averaged with the drone's own
    spectrum as the weight (100 Hz - 4 kHz). Interference = visitor + crowd."""
    def sir(p, mask):
        return p["drone"][mask].sum() / (p["visitor"][mask].sum() + p["crowd"][mask].sum())
    in_band = (freqs >= 100.0) & (freqs < 4000.0)
    per_bin = 10.0 * np.log10((p_out["drone"] / (p_out["visitor"] + p_out["crowd"]))
                              / (p_in["drone"] / (p_in["visitor"] + p_in["crowd"])))
    at_drone = float(np.average(per_bin[in_band], weights=p_in["drone"][in_band]))
    bands = [10.0 * np.log10(sir(p_out, (freqs >= lo) & (freqs < hi)) / sir(p_in, (freqs >= lo) & (freqs < hi)))
             for lo, hi in BANDS.values()]
    return [at_drone] + bands


def command_conference(args):
    contract = load_beam_contract()
    print(f"conference: hall 20 x 40 x 10 m, RT60 1.2 s; drone clip 1.5 m ahead; a visitor (babble from one "
          f"point) near the array; a diffuse crowd ({CROWD_WAVES} plane waves) at {CROWD_DB:+.0f} dB re the drone; "
          f"calibration: {CAL_S:.0f} s of pink noise from the drone speaker, {CAL_LEVEL_DB:+.0f} dB, crowd on")
    print(f"evaluation: the last {RUN_S - RUN_SETTLE_S:.0f} s of {RUN_S:.0f} s; all cancellers always learn; "
          "'drone in ref' = drone in Ch1 re drone in Ch0")
    header = (f"{'method':24s} {'drone in':>8s} {'visitor':>8s} {'crowd':>7s} {'drone':>6s} "
              f"{'SIR gain dB':>20s} {'M4':>5s}")
    units = (f"{'':24s} {'ref dB':>8s} {'removed':>8s} {'removed':>7s} {'lost':>6s} "
             f"{'drone':>6s} {'<1k':>6s} {'1-4k':>6s} {'':>5s}")
    cases = [("hall", HALL, (-30.0, 20.0, 1.0), 10.0, (0.0, 10.0, 20.0)),
             ("hall", HALL, (-30.0, 20.0, 1.0), 0.0, (0.0,)),
             ("hall", HALL, (-30.0, 20.0, 1.0), -40.0, (0.0,)),
             ("hall", HALL, (-15.0, 20.0, 1.0), 10.0, (0.0,)),
             ("living room", LIVING_ROOM, (-30.0, 20.0, 1.0), 10.0, (0.0,)),
             ("free field", FREE_FIELD, (-30.0, 20.0, 1.0), 10.0, (0.0,))]
    for room_name, room, visitor, visitor_db, aims in cases:
        parts, specs = conference_scene(contract, room, visitor, visitor_db, aims)
        for aim in aims:
            results, ch0, run = conference_methods(contract, parts, specs, aim, args.mu)
            evaluated = slice(int(RUN_SETTLE_S * RATE), None)
            who = "no visitor (crowd only)" if visitor_db < -30.0 else \
                f"visitor at az {visitor[0]:+.0f}, el {visitor[1]:+.0f}, {visitor[2]:.1f} m, {visitor_db:+.0f} dB"
            print(f"\n{room_name}; {who}; Ch0 aimed {aim:+.0f} deg off the drone; Ch1 bank cell "
                  f"{specs[1][1]:+.1f}/{specs[1][2]:+.1f}")
            print(header)
            print(units)
            p_in = None
            for label, out, ref, delay, m4 in results:
                before = {n: np.concatenate([np.zeros(delay), ch0[n][run]])[:len(out[n])][evaluated]
                          for n in COMPONENTS}
                p_in = {n: spectra(before[n])[1] for n in COMPONENTS}
                freqs, _ = spectra(before["drone"])
                p_out = {n: spectra(out[n][evaluated])[1] for n in COMPONENTS}
                gains = sir_gains(p_in, p_out, freqs)
                leak = power_db(ref["drone"][run][evaluated]) - power_db(ch0["drone"][run][evaluated])
                band = (freqs >= 100.0) & (freqs < 6000.0)
                removed = {n: 10.0 * np.log10(p_in[n][band].sum() / p_out[n][band].sum()) for n in COMPONENTS}
                print(f"{label:24s} {leak:+8.1f} {removed['visitor']:+8.1f} {removed['crowd']:+7.1f} "
                      f"{removed['drone']:+6.1f} {gains[0]:+6.1f} {gains[1]:+6.1f} {gains[2]:+6.1f} {m4:5.0%}")
            sir_in = [10.0 * np.log10(p_in["drone"][(freqs >= lo) & (freqs < hi)].sum()
                                      / (p_in["visitor"][(freqs >= lo) & (freqs < hi)].sum()
                                         + p_in["crowd"][(freqs >= lo) & (freqs < hi)].sum()))
                      for lo, hi in BANDS.values()]
            print(f"{'(Ch0 SIR before, dB)':24s} {'':8s} {'':8s} {'':7s} {'':6s} {'':6s} "
                  + " ".join(f"{s:+6.1f}" for s in sir_in))


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
    parser.add_argument("command", choices=("sim", "conference", "balcony"))
    parser.add_argument("--taps", type=int, default=64)
    parser.add_argument("--mu", type=float, default=0.05)
    args = parser.parse_args(argv)
    {"sim": command_sim, "conference": command_conference, "balcony": command_balcony}[args.command](args)


if __name__ == "__main__":
    main()
