import numpy as np
import open3d as o3d
from pathlib import Path


def points_to_pointcloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Convert numpy (N, 3) array to Open3D PointCloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def reconstruct_surface_poisson(points: np.ndarray, depth: int = 8) -> o3d.geometry.TriangleMesh:
    """
    Poisson surface reconstruction using Open3D, WITHOUT visualization.
    """
    pcd = points_to_pointcloud(points)

    # Estimate normals required for Poisson
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Optional: remove low-density artifacts
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_keep = densities > density_threshold
    mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])

    return mesh


def save_mesh(mesh: o3d.geometry.TriangleMesh, path: str):
    """
    Save mesh to disk as .ply or .stl etc.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh)
    print(f"Saved mesh to {path}")


