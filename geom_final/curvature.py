import numpy as np
from sklearn.neighbors import NearestNeighbors


def estimate_normals_and_curvature(points: np.ndarray, k: int = 20):
    """
    Estimate normals and a simple curvature measure at each point via local PCA.

    Parameters
    ----------
    points : (N, 3) array
        3D point cloud.
    k : int
        Number of neighbors to use for local PCA.

    Returns
    -------
    normals : (N, 3) array
        Estimated normals (not globally oriented).
    curvature : (N,) array
        Curvature estimate: lambda_min / (lambda0 + lambda1 + lambda2),
        where lambda_min is the smallest eigenvalue of the local covariance.
    """
    n_points = points.shape[0]
    if n_points < k:
        raise ValueError(f"Not enough points ({n_points}) for k = {k}")

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(points)
    _, indices = nbrs.kneighbors(points)

    normals = np.zeros_like(points)
    curvature = np.zeros(n_points)

    for i in range(n_points):
        neighbor_idx = indices[i]
        neighborhood = points[neighbor_idx]

        mean = np.mean(neighborhood, axis=0)
        centered = neighborhood - mean
        cov = centered.T @ centered / (centered.shape[0] - 1)

        eigvals, eigvecs = np.linalg.eigh(cov)
        lambda0, lambda1, lambda2 = eigvals  # ascending
        normal = eigvecs[:, 0]

        normals[i] = normal

        denom = lambda0 + lambda1 + lambda2
        curvature[i] = lambda0 / denom if denom > 0 else 0.0

    return normals, curvature

