# spline_augmentation.py

import numpy as np
from scipy.interpolate import griddata


def cartesian_to_spherical(points: np.ndarray):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.where(r == 0, 1e-12, r)

    theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))      # [0, pi]
    phi = np.arctan2(y, x)                                # [-pi, pi]
    phi = (phi + 2.0 * np.pi) % (2.0 * np.pi)             # [0, 2pi)

    return theta, phi, r


def upsample_with_spline(points_curv: np.ndarray,
                         target_n: int) -> np.ndarray:
    """
    Use griddata to interpolate r(theta, phi) from the reduced point cloud
    and sample a denser point set of size ~target_n.
    """
    theta, phi, r = cartesian_to_spherical(points_curv)

    # 1. Choose a grid approximately of size target_n
    n_theta = int(np.sqrt(target_n))
    if n_theta < 8:
        n_theta = 8
    n_phi = int(np.ceil(target_n / n_theta))
    if n_phi < 8:
        n_phi = 8

    theta_min, theta_max = theta.min(), theta.max()
    phi_min,   phi_max   = phi.min(),   phi.max()

    theta_grid = np.linspace(theta_min, theta_max, n_theta)
    phi_grid   = np.linspace(phi_min,   phi_max,   n_phi)

    Theta, Phi = np.meshgrid(theta_grid, phi_grid, indexing="ij")

    # 2. Interpolate r on this grid using cubic griddata
    points_param = np.stack([theta, phi], axis=1)
    grid_points  = np.stack([Theta.ravel(), Phi.ravel()], axis=1)

    R_hat = griddata(points_param, r, grid_points, method="cubic")

    # griddata can produce NaNs where it cannot interpolate; fill with mean radius
    r_mean = r.mean()
    R_hat = np.where(np.isnan(R_hat), r_mean, R_hat)

    # clamp radii a bit around the observed range to avoid wild extrapolation
    r_min, r_max = r.min(), r.max()
    pad = 0.1 * (r_max - r_min + 1e-6)
    R_hat = np.clip(R_hat, r_min - pad, r_max + pad)

    R_hat = R_hat.reshape(Theta.shape)

    # 3. Convert back to Cartesian
    X = R_hat * np.sin(Theta) * np.cos(Phi)
    Y = R_hat * np.sin(Theta) * np.sin(Phi)
    Z = R_hat * np.cos(Theta)

    points_full = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # 4. If we overshot, randomly choose exactly target_n points
    if points_full.shape[0] > target_n:
        idx = np.random.choice(points_full.shape[0], size=target_n, replace=False)
        points_full = points_full[idx]

    return points_full



