# main.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from curvature import estimate_normals_and_curvature
from resampling import resample_points_curvature
from reconstruction_delaunay import delaunay_surface_triangles

from spline_augmentation import upsample_with_spline
from sklearn.neighbors import NearestNeighbors


def generate_sphere_points(
    n_points: int = 1500,
    radius: float = 1.0,
    noise: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate random points near the surface of a sphere.
    """
    rng = np.random.default_rng(seed)

    phi = rng.uniform(0, 2 * np.pi, n_points)
    costheta = rng.uniform(-1, 1, n_points)
    theta = np.arccos(costheta)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    pts = np.vstack((x, y, z)).T
    pts += noise * rng.normal(size=pts.shape)
    return pts


def plot_point_cloud(points: np.ndarray, title: str):
    """
    Simple 3D scatter plot using matplotlib.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Equal aspect ratio
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.show()


def plot_delaunay_surface(points: np.ndarray, triangles: np.ndarray, title: str):
    """
    Plot a triangulated surface using matplotlib (triangles from Delaunay).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ax.plot_trisurf(
        x,
        y,
        z,
        triangles=triangles,
        linewidth=0.2,
        antialiased=True,
        alpha=0.7,
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.show()


def sphere_radial_error(points: np.ndarray, radius: float = 1.0):
    """
    Compute how far points are from the ideal sphere of given radius.

    Returns:
        rms_error: root-mean-square radial error
        max_error: maximum radial error
    """
    r = np.linalg.norm(points, axis=1)
    diff = r - radius
    rms_error = np.sqrt(np.mean(diff**2))
    max_error = np.max(np.abs(diff))
    return rms_error, max_error


def hull_surface_area(points: np.ndarray) -> float:
    """
    Approximate surface area of the reconstructed shape
    via the convex hull of the points.
    """
    hull = ConvexHull(points)
    return hull.area
    
def plot_history(history):
    """
    Plot how key metrics change over iterations.
    """
    iters = [h["iter"] for h in history]
    n_points = [h["n_points"] for h in history]
    n_tris = [h["n_triangles"] for h in history]
    rms_err = [h["rms_error"] for h in history]
    hull_area = [h["hull_area"] for h in history]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].plot(iters, n_points, marker="o")
    axs[0, 0].set_title("# points")
    axs[0, 0].set_xlabel("iteration")
    axs[0, 0].set_ylabel("points")

    axs[0, 1].plot(iters, n_tris, marker="o")
    axs[0, 1].set_title("# surface triangles")
    axs[0, 1].set_xlabel("iteration")
    axs[0, 1].set_ylabel("triangles")

    axs[1, 0].plot(iters, rms_err, marker="o")
    axs[1, 0].set_title("RMS radial error")
    axs[1, 0].set_xlabel("iteration")
    axs[1, 0].set_ylabel("RMS error")

    axs[1, 1].plot(iters, hull_area, marker="o")
    axs[1, 1].set_title("Approx hull surface area")
    axs[1, 1].set_xlabel("iteration")
    axs[1, 1].set_ylabel("area")

    fig.tight_layout()
    plt.show()
    
def plot_all_point_clouds(history_points):
    """
    history_points = list of point arrays, one per iteration.
    Plots all iterations in one figure with subplots.
    """
    num = len(history_points)
    cols = min(3, num)
    rows = int(np.ceil(num / cols))

    fig = plt.figure(figsize=(5 * cols, 5 * rows))

    for i, pts in enumerate(history_points):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)
        ax.set_title(f"Iteration {i}")

        max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2.0
        mid = pts.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.show()
    
def plot_all_delaunay_surfaces(history_points, history_tris):
    """
    Plot Delaunay surfaces for all iterations in a single figure.

    history_points: list of (N_i, 3) arrays of points.
    history_tris:   list of (M_i, 3) arrays of triangle indices.
    """
    num = len(history_points)
    cols = min(3, num)
    rows = int(np.ceil(num / cols))

    fig = plt.figure(figsize=(5 * cols, 5 * rows))

    for i, (pts, tris) in enumerate(zip(history_points, history_tris)):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        if tris.size > 0:
            ax.plot_trisurf(
                x,
                y,
                z,
                triangles=tris,
                linewidth=0.2,
                antialiased=True,
                alpha=0.7,
            )
        else:
            # Fallback: just scatter if no triangles
            ax.scatter(x, y, z, s=1)

        ax.set_title(f"Delaunay surface (iter {i})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2.0
        mid = pts.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.show()




def run_iterations(
    initial_points: np.ndarray,
    n_iters: int = 3,
    flat_keep: float = 0.2,
    mid_keep: float = 0.7,
    high_dup: int = 3,
    k_neighbors: int = 30,
):
    """
    Run multiple iterations of curvature-aware resampling and reconstruction.

    Visualizes each iteration and returns history + final point set.
    """
    history = []
    current_points = initial_points.copy()
    history_points = []
    history_tris = []

    for it in range(n_iters + 1):
        print(f"\n=== Iteration {it} ===")
        n_pts = current_points.shape[0]

        # Sphere error
        rms, max_err = sphere_radial_error(current_points, radius=1.0)

        # Curvature
        normals, curvature = estimate_normals_and_curvature(
            current_points, k=k_neighbors
        )
        curv_min = curvature.min()
        curv_max = curvature.max()
        curv_mean = curvature.mean()

        # Delaunay surface
        triangles = delaunay_surface_triangles(current_points)
        n_tris = triangles.shape[0]

        # Surface area approx
        area = hull_surface_area(current_points)

        metrics = {
            "iter": it,
            "n_points": n_pts,
            "n_triangles": n_tris,
            "rms_error": rms,
            "max_error": max_err,
            "curv_min": curv_min,
            "curv_max": curv_max,
            "curv_mean": curv_mean,
            "hull_area": area,
        }
        history.append(metrics)
        history_points.append(current_points.copy())
        history_tris.append(triangles.copy())



        print(
            f"points = {n_pts}, tris = {n_tris}, "
            f"RMS err = {rms:.5f}, max err = {max_err:.5f}, "
            f"area = {area:.5f}"
        )

        #  Visualize this iteration's point cloud and surface
        plot_point_cloud(current_points, f"Point cloud (iteration {it})")
        if n_tris > 0:
            plot_delaunay_surface(
                current_points,
                triangles,
                f"Delaunay surface (iteration {it})",
            )
        else:
            print("No surface triangles for this iteration.")

        # If this is the last iteration, don't resample further
        if it == n_iters:
            break

        # Curvature-based resampling to create next point set
        current_points = resample_points_curvature(
            current_points,
            curvature,
            flat_keep=flat_keep,
            mid_keep=mid_keep,
            high_dup=high_dup,
            seed=it,  # different seed per iter
        )

    return history, current_points, history_points, history_tris
    
def nn_error(P_ref: np.ndarray, P_test: np.ndarray):
    """
    Compute nearest-neighbor RMS and max distance from P_test to P_ref.

    P_ref : reference cloud (e.g., original)
    P_test: cloud to evaluate (curvature-resampled or spline-augmented)
    """
    nbrs = NearestNeighbors(n_neighbors=1).fit(P_ref)
    dists, _ = nbrs.kneighbors(P_test)
    dists = dists.ravel()
    rms = np.sqrt(np.mean(dists**2))
    maxd = np.max(dists)
    return rms, maxd

    



def main():
    # 1. Generate synthetic sphere point cloud
    points0 = generate_sphere_points(
        n_points=1500, radius=1.0, noise=0.01, seed=0
    )
    print("Initial points:", points0.shape)

    # 2. Run multiple curvature-aware refinement iterations,
    #    visualizing each one inside run_iterations.
    history, final_points, history_points, history_tris = run_iterations(
        points0,
        n_iters=3,        # change this if you want more/fewer iterations
        flat_keep=0.2,
        mid_keep=0.7,
        high_dup=3,
        k_neighbors=30,
    )

    # 3. Print a summary table of iterations
    print("\n=== Iteration summary ===")
    for h in history:
        print(
            f"iter {h['iter']}: "
            f"points={h['n_points']}, tris={h['n_triangles']}, "
            f"RMS={h['rms_error']:.5f}, max={h['max_error']:.5f}, "
            f"area={h['hull_area']:.5f}"
        )
        
    print("\n=== Iteration summary ===")
    for h in history:
        print(
            f"iter {h['iter']}: "
            f"points={h['n_points']}, tris={h['n_triangles']}, "
            f"RMS={h['rms_error']:.5f}, max={h['max_error']:.5f}, "
            f"area={h['hull_area']:.5f}"
        )

    # plot metrics over iterations
    plot_history(history)
    plot_all_point_clouds(history_points)
    plot_all_delaunay_surfaces(history_points, history_tris)

 # original dense cloud (iteration 0) and final reduced cloud
    P_orig = history_points[0]      # iteration 0 cloud
    P_curv = final_points           # last iteration cloud

    print("\nOriginal vs curvature-resampled sizes:")
    print("  |P_orig| =", P_orig.shape[0])
    print("  |P_curv| =", P_curv.shape[0])

    # choose a spline target size that lies between the two
    target_n = int(0.75 * P_orig.shape[0])
    if target_n <= P_curv.shape[0]:
        target_n = P_curv.shape[0] + 10  # ensure growth

    print(f"\nSpline augmentation: target_n = {target_n}")

    # 4. Use spline to upsample, generating P_spline
    P_spline = upsample_with_spline(P_curv, target_n=target_n)
    print("|P_spline| =", P_spline.shape[0])

    # 5. Compare P_curv and P_spline to the original using NN error
    rms_curv, max_curv = nn_error(P_orig, P_curv)
    rms_spl,  max_spl  = nn_error(P_orig, P_spline)

    print("\nNearest-neighbor error vs original:")
    print(f"  Curvature-resampled: RMS = {rms_curv:.5f}, max = {max_curv:.5f}")
    print(f"  Spline-augmented:    RMS = {rms_spl:.5f}, max = {max_spl:.5f}")

    # 6. Visualize spline-augmented cloud
    plot_point_cloud(P_spline, "Spline-augmented point cloud")

    # optionally: visualize Delaunay surface of the spline-augmented cloud
    tris_spl = delaunay_surface_triangles(P_spline)
    plot_delaunay_surface(P_spline, tris_spl, "Delaunay surface (spline-augmented)")




if __name__ == "__main__":
    main()




