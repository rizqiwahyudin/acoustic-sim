"""
Shared acoustic model helpers used by sim_server.py (live) and
run_comparison.py (batch).

Keeps the physical model (dB SPL conversion, source placement, mic
mismatch model, air absorption defaults, per-wall materials, hardware
impairments, top-N DOA peaks, moving-source trajectories, first-order
atmospheric beam-bending) in one place so the live and batch
simulators stay consistent.
"""

import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve

# ── SPL conversion ─────────────────────────────────────────────────────────────

P_REF = 20e-6  # 20 uPa, 0 dB SPL reference pressure in air

DRONE_SPL_DB = 78.0        # small-medium drone at 1 m
CROWD_SPL_DB = 67.0        # convention-voice talker at 1 m
PA_SPL_DB = 80.0           # ambient convention PA at 1 m (was 92 -- too loud)
MIC_NOISE_FLOOR_DB = 30.0  # typical MEMS self-noise equivalent SPL

# ── Hardware-impairment defaults ──────────────────────────────────────────────

CROSSTALK_COUPLING_DB_DEFAULT = -40.0  # neighbour channel leakage (voltage dB)
CODEC_BIT_DEPTH_DEFAULT = 16           # typical I2S/TDM codec


def spl_to_amplitude(spl_db):
    """Convert dB SPL at 1 m to RMS pressure in Pascals.

    Pyroomacoustics' ISM RIR has a direct-path amplitude of 1/dist, so if
    the source signal is scaled to spl_to_amplitude(X), the mic signal at
    distance d metres will have SPL ≈ X − 20·log10(d), matching free-field
    physics.
    """
    return P_REF * 10.0 ** (spl_db / 20.0)


# ── Source placement ───────────────────────────────────────────────────────────

def crowd_positions_mixed(
    room_dim,
    n,
    z_height,
    array_center,
    rng,
    margin=0.3,
    exclusion_radius=2.0,
    booth_inset=3.0,
    booth_sigma=2.0,
    cluster_fraction=0.6,
):
    """Generate realistic crowd positions: clustered booths + uniform fill.

    A fraction (`cluster_fraction`) of the sources are distributed around 4-8
    random "booth" centres with Gaussian offsets, simulating groups of
    people clustering at demo stands. The remainder are uniformly scattered
    across the floor. Sources within `exclusion_radius` of the array are
    resampled so none of them land under the mics.

    Parameters
    ----------
    room_dim : array-like of 3 floats
        Room dimensions in metres (x, y, z).
    n : int
        Number of crowd sources to generate.
    z_height : float
        Height of the noise sources (standing head height ≈ 1.5 m).
    array_center : array-like of 3 floats
        Mic-array centroid. Used for the exclusion zone test.
    rng : np.random.Generator
    margin : float
        Wall exclusion margin in metres.
    exclusion_radius : float
        Horizontal radius around the array from which crowd sources are
        rejected and resampled.

    Returns
    -------
    list of [x, y, z] positions (plain Python floats for JSON-friendliness).
    """
    room_dim = np.asarray(room_dim, dtype=float)
    array_center = np.asarray(array_center, dtype=float)

    if n <= 0:
        return []

    xmin, xmax = margin, room_dim[0] - margin
    ymin, ymax = margin, room_dim[1] - margin

    n_booths = int(rng.integers(4, 9))
    bx_min = min(booth_inset, (xmax - xmin) / 2.0 - 0.1)
    by_min = min(booth_inset, (ymax - ymin) / 2.0 - 0.1)
    bx_min = max(bx_min, margin)
    by_min = max(by_min, margin)
    booth_centers = np.column_stack([
        rng.uniform(bx_min, room_dim[0] - bx_min, n_booths),
        rng.uniform(by_min, room_dim[1] - by_min, n_booths),
    ])

    n_clustered = int(round(n * cluster_fraction))
    n_uniform = n - n_clustered

    positions = []
    max_tries = 50  # prevent pathological infinite loops

    for i in range(n_clustered):
        booth = booth_centers[i % n_booths]
        for _ in range(max_tries):
            off = rng.normal(0.0, booth_sigma, size=2)
            x = float(np.clip(booth[0] + off[0], xmin, xmax))
            y = float(np.clip(booth[1] + off[1], ymin, ymax))
            dx = x - array_center[0]
            dy = y - array_center[1]
            if dx * dx + dy * dy >= exclusion_radius ** 2:
                positions.append([x, y, float(z_height)])
                break
        else:
            # fall through: place at booth centre clipped
            x = float(np.clip(booth[0], xmin, xmax))
            y = float(np.clip(booth[1], ymin, ymax))
            positions.append([x, y, float(z_height)])

    for _ in range(n_uniform):
        for _ in range(max_tries):
            x = float(rng.uniform(xmin, xmax))
            y = float(rng.uniform(ymin, ymax))
            dx = x - array_center[0]
            dy = y - array_center[1]
            if dx * dx + dy * dy >= exclusion_radius ** 2:
                positions.append([x, y, float(z_height)])
                break
        else:
            positions.append([xmax, ymax, float(z_height)])

    return positions


def wall_adjacent_positions(room_dim, n, z_height, rng, margin=0.3):
    """PA-speaker positions sampled along the walls at a given height.

    Uses the caller's `rng` so results track the user's seed (unlike the
    previous version with a hardcoded seed=42).
    """
    room_dim = np.asarray(room_dim, dtype=float)
    positions = []
    for _ in range(n):
        wall = int(rng.integers(0, 4))
        if wall == 0:
            pos = [margin, float(rng.uniform(margin, room_dim[1] - margin)), z_height]
        elif wall == 1:
            pos = [float(room_dim[0] - margin),
                   float(rng.uniform(margin, room_dim[1] - margin)), z_height]
        elif wall == 2:
            pos = [float(rng.uniform(margin, room_dim[0] - margin)), margin, z_height]
        else:
            pos = [float(rng.uniform(margin, room_dim[0] - margin)),
                   float(room_dim[1] - margin), z_height]
        positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
    return positions


# ── Microphone mismatch v2 ────────────────────────────────────────────────────
#
# The previous sim_server.py mic-mismatch block applied a single phase shift
# to the instantaneous phase via a Hilbert transform. That only makes physical
# sense for narrowband signals and distorts broadband audio. The model below
# replaces it with:
#
#   (a) Per-channel random gain (dB ripple)       - models preamp tolerance
#   (b) Per-channel random fractional delay       - models TDM/PDM clock skew
#   (c) Small per-channel DC offset               - models amp DC imbalance
#
# The fractional delay is implemented as a 31-tap windowed-sinc FIR, which
# is broadband-correct (constant group delay across the pass band).

def _fractional_delay_filter(delay_samples, n_taps=31):
    """Build a windowed-sinc fractional-delay FIR filter.

    The filter group delay is ``(n_taps - 1) / 2 + delay_samples`` samples.
    The caller should therefore trim the leading ``(n_taps - 1) // 2``
    samples of the convolution output so that only the fractional portion
    of the delay remains.
    """
    center = (n_taps - 1) / 2.0
    n = np.arange(n_taps)
    h = np.sinc(n - center - delay_samples)
    h *= np.blackman(n_taps)
    s = np.sum(h)
    if abs(s) > 1e-12:
        h = h / s
    return h


def apply_mic_mismatch_v2(
    signals,
    fs,
    rng,
    gain_db_std=0.5,
    frac_delay_samples_std=0.2,
    dc_offset_std=1e-5,
):
    """Apply realistic per-channel gain + fractional delay + DC offset.

    Parameters
    ----------
    signals : ndarray, shape (n_mics, n_samples)
        Microphone signals (in Pa if the SPL model is used).
    fs : int
        Sample rate in Hz.  Accepted for API completeness; currently only
        used implicitly via `signals` length.
    rng : np.random.Generator
    gain_db_std : float
        Standard deviation of per-channel gain in dB (~±0.5 dB is typical
        for matched MEMS mics after trim).
    frac_delay_samples_std : float
        Standard deviation of per-channel time offset in samples.  At
        fs=16 kHz, 0.2 samples ≈ 12 µs, which matches reasonable
        TDM/PDM clock-skew budgets.
    dc_offset_std : float
        Standard deviation of DC bias added per channel (in the same units
        as ``signals``).

    Returns
    -------
    ndarray, shape (n_mics, n_samples)
        Modified signals.
    """
    signals = np.asarray(signals)
    n_mics, n_samples = signals.shape
    out = np.empty_like(signals, dtype=np.float64)
    n_taps = 31
    center_tap = (n_taps - 1) // 2

    for m in range(n_mics):
        gain_db = rng.normal(0.0, gain_db_std)
        gain = 10.0 ** (gain_db / 20.0)
        delay = rng.normal(0.0, frac_delay_samples_std)
        dc = rng.normal(0.0, dc_offset_std)

        h = _fractional_delay_filter(delay, n_taps=n_taps)
        y = fftconvolve(signals[m], h, mode="full")
        y = y[center_tap:center_tap + n_samples]
        if y.shape[0] < n_samples:
            pad = np.zeros(n_samples - y.shape[0])
            y = np.concatenate([y, pad])
        out[m] = y * gain + dc

    return out


# ── Air absorption defaults ────────────────────────────────────────────────────

def air_absorption_kwargs(temperature_c=20.0, humidity_pct=50.0):
    """Return kwargs to pass to pra.ShoeBox constructor to enable air
    absorption at typical convention-hall conditions.

    Expected usage::

        room = pra.ShoeBox(
            room_dim, fs=FS, sigma2_awgn=sigma2,
            materials=pra.Material(e_absorption),
            max_order=max_order,
            **air_absorption_kwargs(),
        )
    """
    return dict(
        air_absorption=True,
        temperature=float(temperature_c),
        humidity=float(humidity_pct),
    )


# ── Per-wall materials ────────────────────────────────────────────────────────
#
# Uses pyroomacoustics' built-in material database (pra.materials_data) so
# users can pick acoustician-style names like "carpet_hairy" rather than
# raw absorption coefficients. Unknown strings fall back to a uniform
# coefficient of 0.3 so the ShoeBox constructor never crashes.

MATERIAL_CHOICES = [
    # Hard surfaces
    "hard_surface",
    "rough_concrete",
    "unpainted_concrete",
    "brickwork",
    "marble_floor",
    # Floor coverings
    "concrete_floor",
    "linoleum_on_concrete",
    "wood_1.6cm",
    "carpet_thin",
    "carpet_hairy",
    "carpet_tufted_9.5mm",
    # Walls / linings
    "plasterboard",
    "gypsum_board",
    "wooden_lining",
    # Ceilings
    "ceiling_plasterboard",
    "ceiling_fissured_tile",
    "ceiling_metal_panel",
    "ceiling_perforated_gypsum_board",
    "mineral_wool_50mm_40kgm3",
    # Glass
    "glass_window",
    "double_glazing_30mm",
    # Soft / curtains
    "curtains_0.2",
    "curtains_cotton_0.33",
    "curtains_cotton_0.5",
    "curtains_velvet",
    # Audience / seating
    "audience_1_m2",
    "chairs_medium_upholstered",
]

EXHIBITION_HALL_MATERIALS = {
    "floor":   "carpet_hairy",
    "ceiling": "ceiling_fissured_tile",
    "east":    "plasterboard",
    "west":    "plasterboard",
    "south":   "curtains_cotton_0.5",
    "north":   "plasterboard",
}


def _safe_material(x, fallback=0.3):
    """Construct a pra.Material from either a named preset or a float.

    Any exception (e.g. unknown preset name) falls back to a uniform
    absorption coefficient so the ShoeBox constructor always succeeds.
    """
    try:
        if isinstance(x, str):
            return pra.Material(energy_absorption=x)
        return pra.Material(float(x))
    except Exception:
        return pra.Material(fallback)


def build_materials(floor, ceiling, east, west, south, north):
    """Build the dict expected by ``pra.ShoeBox(materials=...)``.

    Each argument is either a named preset string from
    :data:`MATERIAL_CHOICES` or a float absorption coefficient.
    """
    return {
        "floor":   _safe_material(floor),
        "ceiling": _safe_material(ceiling),
        "east":    _safe_material(east),
        "west":    _safe_material(west),
        "south":   _safe_material(south),
        "north":   _safe_material(north),
    }


def measure_rt60_from_rir(rir, fs, decay_db=20):
    """Return the measured RT60 of an RIR via ``pra.experimental.measure_rt60``.

    Returns ``None`` if the RIR is too short or the estimator raises.
    """
    try:
        rir = np.asarray(rir, dtype=np.float64)
        if rir.size < int(0.1 * fs):
            return None
        return float(pra.experimental.measure_rt60(rir, fs=fs, decay_db=decay_db))
    except Exception:
        return None


# ── Hardware impairments: crosstalk + codec quantization ──────────────────────

def apply_crosstalk(signals, coupling_db=CROSSTALK_COUPLING_DB_DEFAULT):
    """Add nearest-neighbour channel leakage.

    Models PCB routing / preamp crosstalk as a linear mix of each
    channel with its two index-neighbours. Wrap-around indexing is used
    so the model is geometry-agnostic (for UCA it is physically
    correct; for linear arrays it is a mild over-estimate at the edges).

    ``coupling_db`` is the voltage coupling ratio in dB; -40 dB is a
    reasonable budget for well-laid-out multi-channel MEMS boards.
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim != 2 or signals.shape[0] < 2:
        return signals.copy()
    alpha = 10.0 ** (coupling_db / 20.0)
    left = np.roll(signals, 1, axis=0)
    right = np.roll(signals, -1, axis=0)
    return signals + alpha * (left + right)


def apply_codec_quantization(signals, bit_depth=CODEC_BIT_DEPTH_DEFAULT,
                             full_scale=None):
    """Quantize mic signals to ``bit_depth``-bit integer LSBs.

    When ``full_scale`` is ``None`` the joint peak across all channels
    is used (models a codec with a single shared PGA / auto-range).
    Theoretical quantization SNR ≈ 6.02·N + 1.76 dB.
    """
    signals = np.asarray(signals, dtype=np.float64)
    if full_scale is None:
        full_scale = float(np.max(np.abs(signals)))
    if full_scale < 1e-15:
        return signals.copy()
    scale = (2 ** (max(int(bit_depth), 2) - 1)) - 1
    normalized = np.clip(signals / full_scale, -1.0, 1.0)
    quantized = np.round(normalized * scale) / scale
    return quantized * full_scale


# ── Moving-source trajectories ────────────────────────────────────────────────
#
# Pyroomacoustics has no native moving-source concept. Phase 2b implements
# a moving drone by rendering the integration window as K phantom static
# sources, each at a slightly different position along the trajectory and
# fed a Hann-crossfaded chunk of the same drone signal. pra.ShoeBox then
# renders all K phantoms + crowd + PA in one simulate() call; Doppler and
# direction change fall out of the differing ISM delays between chunks.
#
# See LIMITATIONS.md §2b for the assumptions and when this model breaks.

C_SOUND = 343.0  # nominal speed of sound in m/s at 20 C


def make_trajectory(start_pos, speed_mps, heading_deg, trajectory_type,
                    duration_s, n_chunks, array_center):
    """Return a list of ``n_chunks`` 3D source positions along a trajectory.

    ``trajectory_type`` is one of:

    - ``"straight"``: constant-velocity horizontal motion. ``heading_deg`` is
      the direction of motion in the horizontal plane (degrees, 0 = +x,
      90 = +y). Vertical position is held constant at ``start_pos[2]``.
    - ``"arc"``: horizontal circular orbit around ``array_center`` at the
      start-position radius; ``speed_mps`` is the tangential speed, so
      angular speed = speed / radius. Direction of rotation follows the
      sign of ``speed_mps``; negative speed reverses the orbit.

    A static source is recovered exactly by passing ``speed_mps = 0`` (all
    K positions collapse to ``start_pos``).
    """
    start = np.asarray(start_pos, dtype=np.float64)
    arc_center = np.asarray(array_center, dtype=np.float64)
    n = max(int(n_chunks), 1)

    if n == 1 or abs(speed_mps) < 1e-9:
        return [start.copy() for _ in range(n)]

    chunk_dur = float(duration_s) / n
    t_mid = (np.arange(n) + 0.5) * chunk_dur

    if trajectory_type == "arc":
        dx = float(start[0] - arc_center[0])
        dy = float(start[1] - arc_center[1])
        radius = float(np.hypot(dx, dy))
        if radius < 1e-6:
            return [start.copy() for _ in range(n)]
        theta0 = float(np.arctan2(dy, dx))
        omega = float(speed_mps) / radius  # rad/s
        positions = []
        for t in t_mid:
            th = theta0 + omega * t
            positions.append(np.array([
                arc_center[0] + radius * np.cos(th),
                arc_center[1] + radius * np.sin(th),
                start[2],
            ], dtype=np.float64))
        return positions

    head = np.deg2rad(float(heading_deg))
    vx = float(speed_mps) * np.cos(head)
    vy = float(speed_mps) * np.sin(head)
    positions = []
    for t in t_mid:
        positions.append(np.array([
            start[0] + vx * t,
            start[1] + vy * t,
            start[2],
        ], dtype=np.float64))
    return positions


def chunk_signal_with_crossfade(signal, n_chunks, fs, crossfade_ms=10.0):
    """Split ``signal`` into ``n_chunks`` full-length, time-gated copies.

    Each returned chunk has the same length as ``signal`` but is non-zero
    only within its own segment plus an overlap of ``crossfade_ms`` into
    each neighbour. At every boundary the two adjacent chunks form a
    raised-cosine crossfade (sin²-in / cos²-out) so their weights sum to
    1.0. Therefore summing the K returned chunks reproduces the original
    ``signal`` to within floating-point epsilon at every sample -- the
    property the static-equivalence test relies on.

    If ``n_chunks <= 1`` the signal is returned unchanged as a single-element
    list.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n_total = signal.shape[0]
    n = max(int(n_chunks), 1)
    if n == 1 or n_total == 0:
        return [signal.copy()]

    xf = max(0, int(round(float(crossfade_ms) * 1e-3 * float(fs))))
    chunk_len = n_total // n
    if chunk_len <= 0:
        return [signal.copy()]
    # Keep the overlap shorter than half a chunk on each side so neighbours
    # don't collide.
    xf = min(xf, chunk_len // 2)
    half = xf // 2  # overlap extends ±half samples past each boundary

    # Boundary sample indices between consecutive chunks: bounds[k] is
    # the nominal start of chunk k, bounds[n] = n_total.
    bounds = [i * chunk_len for i in range(n)] + [n_total]

    if half > 0:
        t = (np.arange(2 * half, dtype=np.float64) + 0.5) / (2.0 * half)
        # t runs 0 -> 1 over the boundary window; sin^2(pi/2 * t) + cos^2(..) = 1
        fade_in = np.sin(0.5 * np.pi * t) ** 2
        fade_out = np.cos(0.5 * np.pi * t) ** 2
    else:
        fade_in = np.array([], dtype=np.float64)
        fade_out = np.array([], dtype=np.float64)

    chunks = []
    for k in range(n):
        seg_start = bounds[k]
        seg_end = bounds[k + 1]
        # Widen by ``half`` on the inward side of each boundary so adjacent
        # chunks overlap; the very first boundary (left) and very last
        # boundary (right) don't have a neighbour, so clamp there.
        win_start = max(0, seg_start - (half if k >= 1 else 0))
        win_end = min(n_total, seg_end + (half if k <= n - 2 else 0))

        w = np.zeros(n_total, dtype=np.float64)
        w[win_start:win_end] = 1.0

        if k >= 1 and half > 0:
            a = max(0, seg_start - half)
            b = min(n_total, seg_start + half)
            length = b - a
            if length > 0:
                w[a:b] = fade_in[:length]
        if k <= n - 2 and half > 0:
            a = max(0, seg_end - half)
            b = min(n_total, seg_end + half)
            length = b - a
            if length > 0:
                w[a:b] = fade_out[:length]

        chunks.append(signal * w)
    return chunks


# ── First-order atmospheric beam-bending ──────────────────────────────────────
#
# A vertical temperature gradient dT/dz (K/m) gives a linear speed-of-sound
# gradient dc/dz ~= 0.6 * dT/dz (m/s per K/m, from c ∝ sqrt(T) at 293 K).
# A ray travelling horizontally through this gradient bends toward the
# slower (cooler) side. The apparent vertical shift of the source as seen
# from the array, for a horizontal range L and true elevation angle alpha,
# is approximately
#
#     delta_z ≈ -(dc/dz) / (2 * c0) * L^2 * sin(alpha)
#
# (first-order curvature, small-angle approximation). The sign is such
# that a positive upward gradient (ceiling hotter than floor, typical of
# lit-up convention halls) shifts the apparent source *down*, which maps
# to a lower apparent elevation.
#
# Azimuth is unaffected by a purely vertical gradient, so only the z
# coordinate of the source is modified.

DCSOUND_PER_DT = 0.6  # m/s per K, linearised from c(T) ~ 20.05*sqrt(T) at 293 K


def atmospheric_z_bias(source_pos, array_center, temp_gradient_c_per_m,
                       c0=C_SOUND):
    """Return a source position with its z-coordinate shifted by the
    first-order gradient beam-bending model described in the module
    docstring.

    If ``temp_gradient_c_per_m`` is zero or the horizontal range is zero,
    the input position is returned unchanged (up to a float copy).
    """
    src = np.asarray(source_pos, dtype=np.float64).copy()
    ctr = np.asarray(array_center, dtype=np.float64)
    if abs(float(temp_gradient_c_per_m)) < 1e-9:
        return src

    dx = float(src[0] - ctr[0])
    dy = float(src[1] - ctr[1])
    dz = float(src[2] - ctr[2])
    L = float(np.hypot(dx, dy))
    if L < 1e-6:
        return src

    r = float(np.hypot(L, dz))
    sin_alpha = dz / r if r > 1e-9 else 0.0

    dc_dz = DCSOUND_PER_DT * float(temp_gradient_c_per_m)
    delta_z = -dc_dz / (2.0 * float(c0)) * (L ** 2) * sin_alpha
    src[2] = src[2] + delta_z
    return src


def atmospheric_elevation_bias_deg(source_pos, array_center,
                                   temp_gradient_c_per_m, c0=C_SOUND):
    """Report the first-order apparent-elevation bias (degrees, signed)
    implied by :func:`atmospheric_z_bias` for display purposes.

    Positive values mean the source *appears* higher than the truth.
    """
    src = np.asarray(source_pos, dtype=np.float64)
    ctr = np.asarray(array_center, dtype=np.float64)
    shifted = atmospheric_z_bias(src, ctr, temp_gradient_c_per_m, c0=c0)
    dx = float(src[0] - ctr[0])
    dy = float(src[1] - ctr[1])
    L = float(np.hypot(dx, dy))
    if L < 1e-6:
        return 0.0
    dz_true = float(src[2] - ctr[2])
    dz_seen = float(shifted[2] - ctr[2])
    el_true = float(np.rad2deg(np.arctan2(dz_true, L)))
    el_seen = float(np.rad2deg(np.arctan2(dz_seen, L)))
    return el_seen - el_true


# ── Top-N DOA peaks with non-max suppression ──────────────────────────────────

def compute_top_n_peaks(power_grid, az_grid, colat_grid, n=3,
                        min_angular_sep_deg=15.0):
    """Return the top ``n`` non-max-suppressed peaks of an SRP-PHAT power map.

    Parameters
    ----------
    power_grid : array-like, shape (len(colat_grid), len(az_grid)) or flat
        The SRP-PHAT spatial power grid.
    az_grid, colat_grid : 1-D arrays
        Azimuth (radians) and colatitude (radians) grid axes.
    n : int
        Number of peaks to return.
    min_angular_sep_deg : float
        Minimum great-circle-style separation (degrees) between accepted
        peaks. Peaks closer than this are suppressed.

    Returns
    -------
    list of dicts with keys ``az_deg``, ``el_deg``, ``power``, ``rel_db``.
    """
    az_grid = np.asarray(az_grid)
    colat_grid = np.asarray(colat_grid)
    P = np.asarray(power_grid, dtype=np.float64)
    P = P.reshape(len(colat_grid), len(az_grid))

    flat_order = np.argsort(P.ravel())[::-1]
    peaks = []
    for idx in flat_order:
        if len(peaks) >= n:
            break
        ci, ai = np.unravel_index(idx, P.shape)
        az = float(np.rad2deg(az_grid[ai]))
        el = float(90.0 - np.rad2deg(colat_grid[ci]))

        ok = True
        for p in peaks:
            d_az = abs(az - p["az_deg"])
            d_az = min(d_az, 360.0 - d_az)
            d_el = abs(el - p["el_deg"])
            if float(np.hypot(d_az, d_el)) < min_angular_sep_deg:
                ok = False
                break
        if not ok:
            continue

        peaks.append({
            "az_deg": round(az, 1),
            "el_deg": round(el, 1),
            "power": float(P[ci, ai]),
        })

    if peaks:
        top = peaks[0]["power"]
        for p in peaks:
            ratio = max(p["power"] / max(top, 1e-15), 1e-12)
            p["rel_db"] = round(float(10.0 * np.log10(ratio)), 2)
    return peaks


# ── Phase 3A: MAX78000 fixed-point ML-path preview ────────────────────────────
#
# These helpers simulate what the beamformed signal looks like after an
# int8/int16 capture and a log-mel feature-extraction stage, so the user
# can preview the audio stream the CNN on the MAX78000 will actually see.
#
# The ML audio quantization is symmetric about 0 and uses the joint peak
# as the full-scale reference (matching the `apply_codec_quantization`
# convention). The feature quantizer operates on log-mel-dB tensors.

ML_DEFAULT_BIT_DEPTH = 8
ML_DEFAULT_FEATURE_BIT_DEPTH = 8
ML_DEFAULT_N_MELS = 64
ML_DEFAULT_NFFT = 512
ML_DEFAULT_HOP = 256
ML_DEFAULT_FMIN = 200.0
ML_DEFAULT_FMAX = 2000.0


def ml_path_quantize_audio(signal, bit_depth=ML_DEFAULT_BIT_DEPTH,
                           full_scale=None):
    """Quantize a beamformed audio signal to `bit_depth`-bit integers.

    Returns a float array holding the dequantized values. With
    ``bit_depth=8`` this mirrors what a MAX78000 ML front-end would see
    if it captured the beamformer output at int8 resolution.
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        return signal.copy()
    if full_scale is None:
        full_scale = float(np.max(np.abs(signal)))
    if full_scale < 1e-15:
        return signal.copy()
    scale = (2 ** (max(int(bit_depth), 2) - 1)) - 1
    normalized = np.clip(signal / full_scale, -1.0, 1.0)
    quantized = np.round(normalized * scale) / scale
    return quantized * full_scale


def _mel_filterbank(n_mels, n_fft, fs, fmin, fmax):
    """Build a triangular mel-filterbank matrix of shape (n_mels, n_fft//2 + 1)."""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    mel_min = hz_to_mel(float(fmin))
    mel_max = hz_to_mel(float(fmax))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / float(fs)).astype(np.int64)
    bin_points = np.clip(bin_points, 0, n_bins - 1)

    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        lo, mid, hi = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        if mid == lo:
            mid = lo + 1
        if hi == mid:
            hi = mid + 1
        if hi >= n_bins:
            hi = n_bins - 1
        if mid >= n_bins:
            mid = n_bins - 1
        if lo >= mid:
            continue
        for k in range(lo, mid):
            fb[m, k] = (k - lo) / float(mid - lo)
        if hi > mid:
            for k in range(mid, hi):
                fb[m, k] = (hi - k) / float(hi - mid)
    return fb


def log_mel_features(signal, fs, n_mels=ML_DEFAULT_N_MELS,
                     n_fft=ML_DEFAULT_NFFT, hop=ML_DEFAULT_HOP,
                     fmin=ML_DEFAULT_FMIN, fmax=ML_DEFAULT_FMAX):
    """Return a (n_mels, T) log-mel-dB spectrogram of ``signal``.

    Uses a Hann window, magnitude STFT, triangular mel filterbank, and
    10*log10 conversion with a small eps floor. Frequency range is
    clamped to [fmin, fmax] to match the SRP-PHAT band used upstream.
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        return np.zeros((int(n_mels), 0), dtype=np.float64)
    n_fft = int(n_fft)
    hop = int(hop)
    if signal.size < n_fft:
        pad = np.zeros(n_fft - signal.size, dtype=np.float64)
        signal = np.concatenate([signal, pad])

    window = np.hanning(n_fft)
    n_frames = 1 + (signal.size - n_fft) // hop
    if n_frames < 1:
        n_frames = 1
    frames = np.stack([
        signal[i * hop:i * hop + n_fft] * window
        for i in range(n_frames)
    ], axis=0)
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    mag = np.abs(spec) ** 2  # power spectrum

    fb = _mel_filterbank(int(n_mels), n_fft, float(fs),
                         float(fmin), float(fmax))
    mel_power = mag @ fb.T  # shape (n_frames, n_mels)
    mel_db = 10.0 * np.log10(mel_power + 1e-10)
    return mel_db.T  # (n_mels, n_frames)


def ml_path_quantize_features(features_db, bit_depth=ML_DEFAULT_FEATURE_BIT_DEPTH):
    """Symmetric quantization of a log-mel-dB tensor to ``bit_depth`` bits.

    The tensor is centered on its mean before scaling so the sign range
    of the quantizer is used efficiently.
    """
    features_db = np.asarray(features_db, dtype=np.float64)
    if features_db.size == 0 or int(bit_depth) >= 16:
        return features_db.copy()
    peak = float(np.max(np.abs(features_db)))
    if peak < 1e-12:
        return features_db.copy()
    scale = (2 ** (max(int(bit_depth), 2) - 1)) - 1
    normalized = np.clip(features_db / peak, -1.0, 1.0)
    quantized = np.round(normalized * scale) / scale
    return quantized * peak


def ml_path_snr_db(reference, quantized):
    """Energy-ratio SNR in dB between ``reference`` and its quantized copy."""
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    q = np.asarray(quantized, dtype=np.float64).reshape(-1)
    n = min(ref.size, q.size)
    if n == 0:
        return 0.0
    noise = ref[:n] - q[:n]
    signal_power = float(np.mean(ref[:n] ** 2))
    noise_power = float(np.mean(noise ** 2))
    if noise_power < 1e-30:
        return 120.0
    if signal_power < 1e-30:
        return 0.0
    return float(10.0 * np.log10(signal_power / noise_power))


def feature_snr_db(reference_features, quantized_features):
    """Energy-ratio SNR in dB between two (n_mels, T) log-mel tensors."""
    r = np.asarray(reference_features, dtype=np.float64).reshape(-1)
    q = np.asarray(quantized_features, dtype=np.float64).reshape(-1)
    return ml_path_snr_db(r, q)


def render_spectrogram_png_b64(features_db, fs, n_mels,
                               fmin=ML_DEFAULT_FMIN, fmax=ML_DEFAULT_FMAX):
    """Render a (n_mels, T) log-mel spectrogram to a small PNG as base64.

    Uses matplotlib's Agg backend so it works headless inside FastAPI.
    Returns the base64 string without the ``data:image/png;base64,`` prefix.
    """
    import base64 as _base64
    import io as _io
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    features_db = np.asarray(features_db, dtype=np.float64)
    if features_db.size == 0:
        features_db = np.zeros((int(n_mels), 1), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(4.8, 2.4), dpi=100)
    duration_s = float(features_db.shape[1]) * ML_DEFAULT_HOP / float(fs)
    im = ax.imshow(
        features_db, aspect="auto", origin="lower",
        extent=[0.0, max(duration_s, 1e-3), float(fmin), float(fmax)],
        cmap="magma",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("freq (Hz)")
    ax.set_title("ML path: log-mel features")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()

    buf = _io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return _base64.b64encode(buf.read()).decode("ascii")


# ── Phase 3B: isotropic-diffuse-field plane-wave crowd generator ──────────────
#
# Instead of placing discrete crowd point sources inside ShoeBox, we
# synthesize a bulk diffuse field as a sum of N plane waves arriving from
# random unit-sphere directions, each with its own (uncorrelated) source
# signal drawn from the concatenated crowd corpus. Per-mic delays are
# implemented as fractional-sample shifts in the frequency domain via
# FFT phase multiplication, so inter-mic coherence closely approaches
# the ideal sinc(k*d) curve as `n_planes` grows.

def _random_unit_vectors(n, rng):
    """Return ``n`` random unit vectors uniformly on the sphere."""
    v = rng.standard_normal((int(n), 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


def _apply_fractional_delay_fd(signal, delay_samples):
    """Apply a fractional-sample delay to ``signal`` in the frequency domain."""
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.size
    if n == 0:
        return signal.copy()
    spec = np.fft.rfft(signal)
    k = np.arange(spec.size)
    phase = np.exp(-1j * 2.0 * np.pi * float(delay_samples) * k / float(n))
    return np.fft.irfft(spec * phase, n=n)


def synthesize_diffuse_crowd_plane_waves(mic_positions, duration_s, fs,
                                         n_planes, source_signals, rng,
                                         c=C_SOUND):
    """Generate an (n_mics, n_samples) diffuse crowd-noise field.

    Parameters
    ----------
    mic_positions : ndarray, shape (3, n_mics)
        Absolute mic coordinates (metres). Only inter-mic differences
        matter.
    duration_s : float
        Output length in seconds.
    fs : int
        Sample rate (Hz).
    n_planes : int
        Number of plane waves to sum. 32-64 is usually enough to
        approach the ideal sinc(k*d) coherence.
    source_signals : list of 1-D arrays
        Independent crowd-voice signals (already RMS-normalized). The
        synthesizer cycles through them; the caller is expected to pass
        enough distinct segments to avoid obvious repetition.
    rng : np.random.Generator
        Used to sample random directions and signal rotations.
    c : float
        Speed of sound (m/s).

    Returns
    -------
    ndarray, shape (n_mics, n_samples)
    """
    mic_positions = np.asarray(mic_positions, dtype=np.float64)
    if mic_positions.ndim != 2 or mic_positions.shape[0] != 3:
        raise ValueError("mic_positions must be shape (3, n_mics)")
    n_mics = mic_positions.shape[1]
    n_samples = max(int(round(float(duration_s) * float(fs))), 1)
    n_p = max(int(n_planes), 1)

    if not source_signals:
        return np.zeros((n_mics, n_samples), dtype=np.float64)

    center = mic_positions.mean(axis=1)
    rel = mic_positions - center[:, None]  # (3, n_mics)

    directions = _random_unit_vectors(n_p, rng)  # (n_p, 3)

    out = np.zeros((n_mics, n_samples), dtype=np.float64)
    n_sources = len(source_signals)

    for p in range(n_p):
        base = np.asarray(source_signals[p % n_sources], dtype=np.float64)
        if base.size == 0:
            continue
        if base.size >= n_samples:
            start = int(rng.integers(0, base.size - n_samples + 1))
            s = base[start:start + n_samples].copy()
        else:
            reps = (n_samples // base.size) + 1
            s = np.tile(base, reps)[:n_samples].copy()
            shift = int(rng.integers(0, n_samples))
            s = np.roll(s, shift)

        d = directions[p]  # unit vector pointing toward source
        # Per-mic acoustic delay of the incoming plane wave (seconds):
        # tau_m = -(r_m . d) / c, so mics closer to the source (along d)
        # see the signal earlier (negative delay).
        tau_samples = -(rel.T @ d) * (float(fs) / float(c))

        # Normalize per-plane amplitude so the sum of N uncorrelated
        # plane waves has expected unit variance overall (the final
        # scaling is done by the caller via spl_to_amplitude).
        amp = 1.0 / np.sqrt(float(n_p))

        for m in range(n_mics):
            out[m] += amp * _apply_fractional_delay_fd(s, tau_samples[m])

    return out


# ── Phase 3C: FIR capacitive-coupling crosstalk model ─────────────────────────
#
# `apply_crosstalk` models neighbour leakage as a memoryless gain. Real
# PCB coupling is dominated by parasitic capacitance, which adds a
# high-pass character to the leakage path (df = 1/(2*pi*R*C)). The model
# below implements this as a 1-pole IIR applied in series with the
# neighbour tap; a very low `corner_hz` collapses back to the flat
# `apply_crosstalk` behaviour, so the FIR model is a strict superset.

def _highpass_1pole(signal, fs, corner_hz):
    """1-pole recursive high-pass. Unity passband gain above `corner_hz`."""
    signal = np.asarray(signal, dtype=np.float64)
    if corner_hz <= 0.0 or signal.size == 0:
        return signal.copy()
    # Bilinear-transform 1st-order HPF:
    #   y[n] = a * (y[n-1] + x[n] - x[n-1])
    # with a = 1 / (1 + 2*pi*fc/fs). DC gain = 0, Nyquist gain ≈ 1.
    a = 1.0 / (1.0 + 2.0 * np.pi * float(corner_hz) / float(fs))
    y = np.zeros_like(signal)
    prev_x = 0.0
    prev_y = 0.0
    for n in range(signal.size):
        y[n] = a * (prev_y + signal[n] - prev_x)
        prev_x = signal[n]
        prev_y = y[n]
    return y


def apply_crosstalk_fir(signals, fs,
                        coupling_db=CROSSTALK_COUPLING_DB_DEFAULT,
                        corner_hz=500.0, neighbor_only=True,
                        measured_fir=None):
    """Nearest-neighbour leakage with a capacitive (HPF) transfer path.

    ``coupling_db`` sets the passband voltage coupling ratio; at
    frequencies far above ``corner_hz`` each neighbour contributes
    ``10**(coupling_db/20)`` times its signal. Below the corner the
    coupling rolls off at ~6 dB/oct, matching a simple parasitic-C model.

    If ``measured_fir`` is not ``None``, the supplied FIR is convolved
    with each neighbour contribution instead of the 1-pole HPF (hook for
    real measured pair traces; see :func:`load_crosstalk_fir`).
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim != 2 or signals.shape[0] < 2:
        return signals.copy()
    alpha = 10.0 ** (float(coupling_db) / 20.0)
    left = np.roll(signals, 1, axis=0)
    right = np.roll(signals, -1, axis=0)

    out = signals.copy()
    for m in range(signals.shape[0]):
        if measured_fir is not None:
            fir = np.asarray(measured_fir, dtype=np.float64).reshape(-1)
            leak_l = fftconvolve(left[m], fir, mode="full")[:signals.shape[1]]
            leak_r = fftconvolve(right[m], fir, mode="full")[:signals.shape[1]]
        else:
            leak_l = _highpass_1pole(left[m], fs, corner_hz)
            leak_r = _highpass_1pole(right[m], fs, corner_hz)
        out[m] += alpha * (leak_l + leak_r)
    return out


def load_crosstalk_fir(path):
    """Load a measured FIR impulse response for the crosstalk path.

    Accepts either a JSON list of floats or an NPZ file containing a
    ``"fir"`` array. Returns ``None`` if the path is empty, missing, or
    unreadable, so the caller can transparently fall back to the
    analytic capacitive model.
    """
    import json as _json
    import pathlib as _pathlib

    if not path:
        return None
    p = _pathlib.Path(path)
    if not p.exists():
        return None
    try:
        if p.suffix.lower() == ".json":
            with p.open("r", encoding="utf-8") as fh:
                data = _json.load(fh)
            if isinstance(data, dict) and "fir" in data:
                data = data["fir"]
            return np.asarray(data, dtype=np.float64).reshape(-1)
        if p.suffix.lower() in (".npz", ".npy"):
            loaded = np.load(p)
            if hasattr(loaded, "files") and "fir" in loaded.files:
                return np.asarray(loaded["fir"], dtype=np.float64).reshape(-1)
            return np.asarray(loaded, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    return None
