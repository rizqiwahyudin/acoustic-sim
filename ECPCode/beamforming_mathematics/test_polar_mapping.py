import numpy as np

steer_az_deg = 0.0
steer_el_deg = 0.0
azs_deg = np.array([0, 20, 40, 60, 90])
el_deg = np.array([0.0])
az_rad = np.radians(azs_deg)
el_rad = np.radians(el_deg)
AZ_grid, EL_grid = np.meshgrid(az_rad, el_rad)
ux = np.cos(EL_grid) * np.cos(AZ_grid)
uy = np.cos(EL_grid) * np.sin(AZ_grid)
uz = np.sin(EL_grid)
U = np.vstack([ux.ravel(), uy.ravel(), uz.ravel()])
azs_rad = np.radians(steer_az_deg)
els_rad = np.radians(steer_el_deg)
a = np.array([np.cos(els_rad) * np.cos(azs_rad), np.cos(els_rad) * np.sin(azs_rad), np.sin(els_rad)])
b = np.array([0.0, 0.0, 1.0])
v = np.cross(a, b)
s = np.linalg.norm(v)
c = float(np.dot(a, b))
if s < 1e-12:
    if c > 0.0:
        R = np.eye(3)
    else:
        ort = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            ort = np.array([0.0, 1.0, 0.0])
        v2 = np.cross(a, ort)
        v2 = v2 / (np.linalg.norm(v2) + 1e-12)
        K = np.array([[0, -v2[2], v2[1]], [v2[2], 0, -v2[0]], [-v2[1], v2[0], 0]])
        R = np.eye(3) + 2.0 * K @ K
else:
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + K + (K @ K) * (1.0 / (1.0 + c))
U_rot = R @ U
xr = U_rot[0, :]
yr = U_rot[1, :]
zr = np.clip(U_rot[2, :], -1.0, 1.0)
theta = np.arccos(zr)
phi = np.arctan2(yr, xr)
print('azs_deg:', azs_deg)
print('phi_deg:', np.degrees(phi))
print('theta_deg:', np.degrees(theta))
