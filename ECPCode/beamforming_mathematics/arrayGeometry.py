import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# ARRAY GEOMETRY GENERATORS
# ============================================================

def circular_array(n_mics, radius_mm, center=(0, 0), start_angle_deg=0):
    angles = np.linspace(0, 2 * np.pi, n_mics, endpoint=False) + np.radians(start_angle_deg)
    return [(center[0] + radius_mm * np.cos(a), center[1] + radius_mm * np.sin(a)) for a in angles]

def linear_array(n_mics, spacing_mm, axis='x', center=(0, 0)):
    offsets = np.arange(n_mics, dtype=float) * spacing_mm
    offsets -= offsets.mean()
    if axis == 'x':
        return [(center[0] + o, center[1]) for o in offsets]
    else:
        return [(center[0], center[1] + o) for o in offsets]

def rectangular_array(n_rows, n_cols, spacing_x_mm, spacing_y_mm, center=(0, 0)):
    xs = np.arange(n_cols, dtype=float) * spacing_x_mm
    ys = np.arange(n_rows, dtype=float) * spacing_y_mm
    xs -= xs.mean()
    ys -= ys.mean()
    return [(center[0] + x, center[1] + y) for y in ys for x in xs]

def concentric_rings(rings, center=(0, 0)):
    mics = []
    for n, r, a in rings:
        if r == 0:
            mics.append(center)
        else:
            mics.extend(circular_array(n, r, center, a))
    return mics

def cross_array(arm_length_mm, n_per_arm, spacing_mm, center=(0, 0)):
    mics = [center]
    for i in range(1, n_per_arm + 1):
        d = i * spacing_mm
        if d <= arm_length_mm:
            mics.append((center[0] + d, center[1]))
            mics.append((center[0] - d, center[1]))
            mics.append((center[0], center[1] + d))
            mics.append((center[0], center[1] - d))
    return mics

def sunflower_array(n_mics, s):
    "Generates a sunflower array with n_mics with a spacing s (mm). Diameter is approximately s * sqrt(n_mics)."
    mics = []
    for i in range(n_mics):
        r = s * np.sqrt(i) / np.sqrt(np.pi) 
        theta = 2*np.pi * i * 1.618 # golden ratio = 1.618
        mics.append((r * np.cos(theta), r * np.sin(theta)))
    return mics



def spiral_array(n_mics, n_arms, diameter_mm, turns=3, center=(0, 0)):
    n_mics = int(max(0, n_mics))
    n_arms = int(max(1, n_arms))

    if n_mics == 0:
        return []

    r_max = diameter_mm / 2.0
    t_max = max(1e-12, turns * 2.0 * np.pi)  # avoid div by zero
    c = r_max / t_max  # r = c * t

    # Distribute elements as evenly as possible across arms
    base = n_mics // n_arms
    rem = n_mics % n_arms

    mics = []
    for arm in range(n_arms):
        count = base + (1 if arm < rem else 0)
        if count == 0:
            continue
        # parameter t runs from 0..t_max for each arm; first element at center when count>1
        if count == 1:
            t_vals = np.array([0.0])
        else:
            t_vals = np.linspace(0.0, t_max, count)
        angle_offset = 2.0 * np.pi * arm / n_arms
        r = c * t_vals
        thetas = t_vals + angle_offset
        xs = center[0] + r * np.cos(thetas)
        ys = center[1] + r * np.sin(thetas)
        mics.extend([(float(x), float(y)) for x, y in zip(xs, ys)])

    return mics