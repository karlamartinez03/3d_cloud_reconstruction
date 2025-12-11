import numpy as np


def resample_points_curvature(
    points: np.ndarray,
    curvature: np.ndarray,
    flat_keep: float = 0.3,
    mid_keep: float = 0.7,
    high_dup: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """
    Resample a point cloud based on curvature.

    - Low curvature (flat regions): keep only a fraction of points.
    - Medium curvature: keep more points.
    - High curvature (features): duplicate points to increase density.
    """
    if points.shape[0] != curvature.shape[0]:
        raise ValueError("points and curvature must have same length")

    c = curvature.astype(float).copy()
    c -= c.min()
    if c.max() > 0:
        c /= c.max()

    flat_thresh = 0.33
    high_thresh = 0.66

    low_mask = c < flat_thresh
    mid_mask = (c >= flat_thresh) & (c < high_thresh)
    high_mask = c >= high_thresh

    pts_low = points[low_mask]
    pts_mid = points[mid_mask]
    pts_high = points[high_mask]

    rng = np.random.default_rng(seed)

    # Downsample low-curvature
    if len(pts_low) > 0 and flat_keep > 0:
        n_low_keep = max(1, int(flat_keep * len(pts_low)))
        n_low_keep = min(n_low_keep, len(pts_low))
        idx_low = rng.choice(len(pts_low), size=n_low_keep, replace=False)
        pts_low_resampled = pts_low[idx_low]
    else:
        pts_low_resampled = np.empty((0, 3))

    # Keep some mid-curvature
    if len(pts_mid) > 0 and mid_keep > 0:
        n_mid_keep = max(1, int(mid_keep * len(pts_mid)))
        n_mid_keep = min(n_mid_keep, len(pts_mid))
        idx_mid = rng.choice(len(pts_mid), size=n_mid_keep, replace=False)
        pts_mid_resampled = pts_mid[idx_mid]
    else:
        pts_mid_resampled = np.empty((0, 3))

    # Duplicate high-curvature points
    if len(pts_high) > 0 and high_dup > 0:
        pts_high_resampled = np.repeat(pts_high, repeats=high_dup, axis=0)
    else:
        pts_high_resampled = np.empty((0, 3))

    new_points = np.vstack(
        [pts_low_resampled, pts_mid_resampled, pts_high_resampled]
    )
    return new_points

