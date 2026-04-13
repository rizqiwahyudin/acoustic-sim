import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra

def drone_like_signal(fs: int, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(fs * seconds)) / fs
    # simple tonal + harmonics + a bit of noise (not a real drone recording, but good for debugging)
    freqs = [220, 440, 880, 1760]
    sig = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    sig *= 0.5
    sig += 0.05 * np.random.randn(len(t))
    return sig.astype(np.float32)

def make_uca(center_xy, mics=12, radius=0.15):
    return pra.circular_2D_array(center_xy, mics, 0.0, radius)

def make_cross(center_xy, mics_per_axis=6, half_length=0.15):
    # 2 orthogonal ULAs crossing at center, but no mic at exact center (avoids duplicate point)
    offsets = np.linspace(-half_length, half_length, mics_per_axis + 1)  # includes 0
    offsets = offsets[np.abs(offsets) > 1e-9]  # remove 0
    pts = []
    for o in offsets:
        pts.append([center_xy[0] + o, center_xy[1]])
    for o in offsets:
        pts.append([center_xy[0], center_xy[1] + o])
    R = np.array(pts).T  # shape (2, M)
    return R

def simulate_and_estimate(array_R, true_az_deg=60.0, use_reverb=False):
    fs = 16000
    c = 343.0
    nfft = 256
    hop = nfft // 2

    room_dim = np.array([10.0, 10.0])
    center = room_dim / 2.0

    # source position (far-ish field in a 10x10m room)
    distance = 3.0
    az = np.deg2rad(true_az_deg)
    src = center + distance * np.array([np.cos(az), np.sin(az)])

    # noise level (AWGN) to make it slightly less ideal
    SNR_db = 10.0
    sigma2 = 10 ** (-SNR_db / 10.0) / (4.0 * np.pi * distance) ** 2

    if not use_reverb:
        room = pra.AnechoicRoom(2, fs=fs, sigma2_awgn=sigma2)
    else:
        # Reverberant shoebox. Use inverse Sabine to get absorption/max_order for a target RT60.
        rt60 = 1.2  # seconds (tune this)
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            sigma2_awgn=sigma2,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )

    room.add_source(src, signal=drone_like_signal(fs))
    room.add_microphone_array(pra.MicrophoneArray(array_R, fs=fs))

    room.simulate()

    # STFT per mic, stack like in the upstream example
    X = np.array(
        [pra.transform.stft.analysis(sig, nfft, hop).T for sig in room.mic_array.signals]
    )  # shape (M, F, T)

    # choose a frequency band (bins) for localization
    fmin, fmax = 500, 4000
    df = fs / nfft
    freq_bins = np.arange(int(fmin / df), int(fmax / df))

    doa = pra.doa.SRP(array_R, fs, nfft, c=c, num_src=1)
    doa.locate_sources(X, freq_bins=freq_bins)

    est_az_deg = float(doa.azimuth_recon[0] * 180.0 / np.pi)
    return est_az_deg, doa

def wrap_angle_deg(a):
    return (a + 180) % 360 - 180

if __name__ == "__main__":
    room_dim = np.array([10.0, 10.0])
    center = room_dim / 2.0
    true_az = 60.0

    uca_R = make_uca(center, mics=12, radius=0.15)
    cross_R = make_cross(center, mics_per_axis=6, half_length=0.15)  # total 12 mics

    for use_reverb in [False, True]:
        tag = "REVERB" if use_reverb else "ANECHOIC"
        for name, R in [("UCA", uca_R), ("CROSS", cross_R)]:
            est, doa = simulate_and_estimate(R, true_az_deg=true_az, use_reverb=use_reverb)
            err = wrap_angle_deg(est - true_az)
            print(f"{tag:8s} {name:5s}  true={true_az:6.1f}°, est={est:7.2f}°, err={err:7.2f}°")

            az = float(np.atleast_1d(doa.azimuth_recon)[0])  # radians

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="polar")
            ax.plot([az, az], [0.0, 1.0], linewidth=3)
            ax.set_rlim(0, 1.0)
            ax.set_title(f"{name} - {tag} (SRP-PHAT)")
            plt.show()