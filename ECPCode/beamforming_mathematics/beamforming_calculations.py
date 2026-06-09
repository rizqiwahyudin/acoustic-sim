"""
beamforming_calculations.py

Array geometry generation, delay-and-sum beamforming calculations,
beam pattern analysis, and MAX78000 lookup table export.

Usage:
    1. Define array geometry using generator functions
    2. Optionally rotate with rotate_positions_3d()
    3. Visualize with plot_positions() or plot_beam()
    4. Generate SigmaStudio delay values with calculate_delays()
    5. Generate MAX78000 lookup table with generate_beam_table()
    6. Export as C header with export_beam_table_c()
"""

import numpy as np
import matplotlib.pyplot as plt





def hex_lattice_positions(rings, spacing_mm, center=(0, 0)):
    """Hexagon-shaped lattice of points up to given ring radius."""
    positions = []
    R = int(rings)
    for q in range(-R, R + 1):
        r1 = max(-R, -q - R)
        r2 = min(R, -q + R)
        for r in range(r1, r2 + 1):
            x = spacing_mm * np.sqrt(3) * (q + r / 2.0)
            y = spacing_mm * 1.5 * r
            positions.append((center[0] + float(x), center[1] + float(y)))
    return positions


def hexagon_clustered_array(cluster_center_rings=1, cluster_mic_rings=1,
                            cluster_spacing_mm=50.0, mic_spacing_mm=10.0,
                            center=(0, 0), exclude_center=True, max_clusters=None):
    """Array of hexagon-shaped mic clusters on a larger hex lattice."""
    centres = hex_lattice_positions(cluster_center_rings, cluster_spacing_mm, center=center)
    if exclude_center:
        centres = [c for c in centres
                   if not (np.isclose(c[0], center[0]) and np.isclose(c[1], center[1]))]
    if len(centres) > 0:
        angles = [np.arctan2(c[1] - center[1], c[0] - center[0]) for c in centres]
        centres = [c for _, c in sorted(zip(angles, centres))]
    if max_clusters is not None:
        centres = centres[:int(max_clusters)]

    all_mics = []
    for c in centres:
        all_mics.extend(hex_lattice_positions(cluster_mic_rings, mic_spacing_mm, center=c))
    return all_mics


# ============================================================
# GEOMETRY UTILITIES
# ============================================================

def rotate_positions_3d(mic_positions_mm, rot_x_deg=0.0, rot_y_deg=0.0, rot_z_deg=0.0,
                        center_mm=(0, 0, 0)):
    """Rotate mic positions about center by Euler angles (degrees). Input/output in mm."""
    pts = np.array(mic_positions_mm, dtype=float)
    if pts.ndim == 1 and pts.size == 2:
        pts = pts.reshape((1, 2))
    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((pts.shape[0], 1))])

    pts_m = pts / 1000.0
    center_m = np.array(center_mm, dtype=float) / 1000.0
    pts_centered = pts_m - center_m

    rx, ry, rz = np.radians(rot_x_deg), np.radians(rot_y_deg), np.radians(rot_z_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

    rotated = ((Rz @ Ry @ Rx) @ pts_centered.T).T + center_m
    rotated_mm = rotated * 1000.0
    return [(float(x), float(y), float(z)) for x, y, z in rotated_mm]


def positions_to_3d_string(mic_positions_mm, z_mm=0.0, precision=3):
    """Format positions as [x, y, z] string in metres for copy/paste."""
    fmt = f"{{:.{precision}f}}"
    triplets = []
    for p in mic_positions_mm:
        x, y = p[0], p[1]
        z = p[2] if len(p) == 3 else z_mm
        triplets.append(f"[{fmt.format(x/1000)}, {fmt.format(y/1000)}, {fmt.format(z/1000)}]")
    return ", ".join(triplets)


# ============================================================
# DELAY CALCULATIONS
# ============================================================

def steeringVector(mic_positions_mm, speed_of_sound=343, sampling_rate=48000, theta_deg=0, phi_deg = 0, freq = 1000):
    """
    Calculate per-mic delays in samples for a given steering direction.
    Returns non-negative float array.
    """
    # keep sampling_rate parameter referenced to avoid linter "unused" hint
    _ = sampling_rate
    mics = np.array([np.insert(x, 1, 0) / 1000.0 for x in mic_positions_mm])  # convert to meters and add z=0 for 3D calculations
    centroid = np.min(mics)
    mics = mics - centroid
    # interpret angles as conventional azimuth (theta) and elevation (phi)
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    u = np.asmatrix([np.sin(theta) * np.cos(phi),
                    np.cos(theta) * np.cos(phi),
                    np.sin(phi)]).T

    wavelength = speed_of_sound / freq
    steering_vector = np.exp(2j * np.pi * (mics @ u) / wavelength)  # Nx3 * 3x1 = Nx1 steering vector

    return steering_vector
def calculateWeights(mic_positions_mm, speed_of_sound=343, sampling_rate=48000, theta_deg=0, phi_deg = 0, freq = 1000, method='DAS'):
    """
    Calculate beamforming weights for a given steering direction.
    """
    case = method.upper()
    if case == 'DAS':
        weights = steeringVector(mic_positions_mm, speed_of_sound, sampling_rate, theta_deg, phi_deg, freq) # simple delay-and-sum weights (normalized)
        return weights
    if case == 'MVDR':
        pass # MVDR weights would require noise covariance matrix estimation, which is more complex and not implemented here
        return 0
    if case == 'LCMV':
        pass # LCMV weights would require constraints and noise covariance matrix, also more complex
        return 0
    if case == 'RLS':
        pass # RLS is an adaptive algorithm that would require iterative updates based on incoming signal, not just a static calculation
        return 0
    raise ValueError(f"Unknown method '{method}'. Supported methods: DAS, MVDR, LCMV, RLS.")

def plot_beam_pattern(mic_positions_mm, speed_of_sound=343, sampling_rate=48000, theta_deg=0, phi_deg=0, freq=1000, resolution=100, method='DAS', h=16, w=6): 
    thetaLimits = np.linspace(-np.pi, np.pi, resolution)
    phiLimits = np.linspace(-np.pi/2, np.pi/2, resolution)
    arrayFactor = np.zeros((resolution, resolution))
    weights = calculateWeights(mic_positions_mm, speed_of_sound, sampling_rate, theta_deg, phi_deg, freq, method)
    for i, theta_i in enumerate(thetaLimits):
        for j, phi_i in enumerate(phiLimits):
            amplitude = steeringVector(mic_positions_mm, speed_of_sound, sampling_rate, np.degrees(theta_i), np.degrees(phi_i), freq)
            arrayFactor[i, j] = np.abs(weights.conj().T @ amplitude)[0, 0]# normalized array factor (linear scale)

    
    plt.figure(figsize=(h, w))

    # plot in polar coordinates
    # plt.subplot(1, 2, 1, projection='polar')
    # Theta, Phi = np.meshgrid(thetaLimits, phiLimits)
    # plt.pcolormesh(Theta, Phi, arrayFactor.T, shading='auto', cmap='viridis')
    # plt.colorbar(label='Power [linear]')
    # plt.scatter(np.deg2rad(theta_deg), np.deg2rad(phi_deg), color='red', s=50) # Add a dot at the correct theta/phi
    

    # plt.title(f'Beam Pattern ({method}) at {freq} Hz, Steering: {theta_deg}\u00b0, {phi_deg}\u00b0')
    plt.subplot(1, 2, 1)
    plt.imshow(arrayFactor.T, extent=(np.degrees(thetaLimits[0]), np.degrees(thetaLimits[-1]), np.degrees(phiLimits[0]), np.degrees(phiLimits[-1])), origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Power [linear]')
    plt.scatter(theta_deg, phi_deg, color='red', s=50) # Add a dot at the correct theta/phi

    
    plt.subplot(1, 2, 2)
    mic_positions = np.array(mic_positions_mm)
    plt.scatter(mic_positions[:, 0], mic_positions[:, 1], c='blue', s=80, zorder=5, label='Microphones')
    plt.title('Microphone Array Geometry')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Y Position (mm)')
    plt.show()
    

def sigmaStudioFractionalDelay(mic_positions_mm, speed_of_sound = 343, sampling_rate = 48000, theta_deg = 0, phi_deg = 0, freq = 2000, method='DAS'):
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    u = np.asmatrix([np.sin(theta) * np.cos(phi),
                    np.cos(theta) * np.cos(phi),
                    np.sin(phi)]).T    
    delays = (mic_positions_mm @ u) / speed_of_sound * sampling_rate
    delays -= delays.min()
    max_delay_samples = int(np.ceil(np.max(delays))) + 2
    percentages = (delays / max_delay_samples) * 100.0
    n_mics = len(delays)
    print(f"Max (same for all {n_mics} blocks): {max_delay_samples} samples")
    print(f"Gain: 1/{n_mics} = {20 * np.log10(1 / n_mics):.2f} dB\n")
    for i in range(n_mics):
        print(f"Mic {i}: Delay = {delays[i]:.2f} samples, Percentage = {percentages[i]:.2f}%")


######### Old functions below#########
# def compute_beam_pattern(mic_positions_mm, theta_deg=0, freq=1000, speed_of_sound=343, n_angles=720):
#     mics = np.array(mic_positions_mm) / 1000.0
#     centroid = np.mean(mics, axis=0)
#     mics_centered = mics - centroid
#     N = len(mics_centered)

#     theta_steer = np.radians(theta_deg)
#     k = 2 * np.pi * freq / speed_of_sound

#     phi = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

#     af = np.zeros(len(phi))
#     for idx, p in enumerate(phi):
#         phase = k * (mics_centered[:, 0] * (np.cos(p) - np.cos(theta_steer)) +
#                       mics_centered[:, 1] * (np.sin(p) - np.sin(theta_steer)))
#         af[idx] = np.abs(np.sum(np.exp(1j * phase))) / N

#     return phi, af
# def plot_microphone_array_and_beam(mic_positions_mm, theta_deg=0, freq=1000, speed_of_sound=343):
#     mics = np.array(mic_positions_mm) / 1000.0
#     centroid = np.mean(mics, axis=0)
#     phi, af = compute_beam_pattern(mic_positions_mm, theta_deg, freq, speed_of_sound)

#     array_radius = np.max(np.linalg.norm(mics - centroid, axis=1))
#     scale = array_radius * 2.0

#     beam_x = centroid[0] + scale * af * np.cos(phi)
#     beam_y = centroid[1] + scale * af * np.sin(phi)

#     fig, ax = plt.subplots(1, 1, figsize=(8, 8))

#     ax.fill(beam_x, beam_y, alpha=0.15, color='red')
#     ax.plot(beam_x, beam_y, color='red', linewidth=1.5, label=f'Beam pattern ({freq} Hz)')

#     ax.scatter(mics[:, 0], mics[:, 1], c='blue', s=80, zorder=5, label='Microphones')
#     for i, (x, y) in enumerate(mics):
#         ax.annotate(f'{i}', (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, color='blue')

#     arrow_len = array_radius * 2.5
#     theta_rad = np.radians(theta_deg)
#     ax.annotate('',
#                 xy=(centroid[0] + arrow_len * np.cos(theta_rad),
#                     centroid[1] + arrow_len * np.sin(theta_rad)),
#                 xytext=(centroid[0], centroid[1]),
#                 arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
#     ax.text(centroid[0] + arrow_len * 1.1 * np.cos(theta_rad),
#             centroid[1] + arrow_len * 1.1 * np.sin(theta_rad),
#             f'\u03b8 = {theta_deg}\u00b0', color='green', fontsize=11, fontweight='bold',
#             ha='center', va='center')

#     ax.plot(*centroid, 'k+', markersize=12, markeredgewidth=2, label='Centroid')

#     n_mics = len(mics)
#     ax.set_title(f'{n_mics}-mic array | DAS beam | f = {freq} Hz, \u03b8 = {theta_deg}\u00b0')
#     ax.set_xlabel('X Position (m)')
#     ax.set_ylabel('Y Position (m)')
#     ax.legend(loc='upper right')
#     ax.grid(True, alpha=0.3)
#     ax.set_aspect('equal')

#     plt.tight_layout()
#     plt.show()

# # def calculate_delays_adaptive(mic_positions_mm, speed_of_sound=343, sampling_rate=48000, theta_deg=0):
#     """
#     Core delay calculation. Returns raw non-negative delays in samples (float).
#     """
#     mics = np.array([np.append(x, 0) / 1000.0 for x in mic_positions_mm])  # to meters, z=0
#     centroid = np.mean(mics, axis=0)
#     mics_centered = mics - centroid

#     theta = np.radians(theta_deg)
#     u = np.array([np.cos(theta), np.sin(theta)])

#     delays = (mics_centered @ u) / speed_of_sound * sampling_rate
#     delays -= delays.min()

#     return delays
# def delays_to_sigmastudio(delays, max_delay_samples=None):
#     """
#     Convert delays (float samples) to SigmaStudio fractional delay block settings.

#     SigmaStudio fractional delay block:
#         - Max: buffer size in samples (compile-time, same for all blocks)
#         - Percentage: (actual_delay / Max) * 100

#     If max_delay_samples is not given, it is auto-calculated as ceil(max(delays)) + 2.
#     """
#     if max_delay_samples is None:
#         max_delay_samples = int(np.ceil(np.max(delays))) + 2

#     percentages = (delays / max_delay_samples) * 100.0
#     n_mics = len(delays)

#     print(f"Max (same for all {n_mics} blocks): {max_delay_samples} samples")
#     print(f"Gain: 1/{n_mics} = {20 * np.log10(1 / n_mics):.2f} dB\n")
#     for i in range(n_mics):
#         print(f"Mic {i:2d}: delay = {delays[i]:6.3f} samples  ->  percentage = {percentages[i]:7.3f}%")

#     return max_delay_samples, percentages
# def calculate_and_print(mic_positions_mm, theta_deg=0, max_delay_samples=None,
#                         speed_of_sound=343, sampling_rate=48000):
#     """
#     Full pipeline: mic positions + angle -> SigmaStudio block settings.
#     """
#     delays = calculate_delays(mic_positions_mm, speed_of_sound, sampling_rate, theta_deg)
#     max_val, pcts = delays_to_sigmastudio(delays, max_delay_samples)
#     return delays, max_val, pcts






