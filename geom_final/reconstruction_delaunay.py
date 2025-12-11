# reconstruction_delaunay.py

import numpy as np
from scipy.spatial import Delaunay
import itertools


def delaunay_surface_triangles(points: np.ndarray) -> np.ndarray:
    """
    Compute the 3D Delaunay triangulation of the points and
    extract the outer surface triangles (faces that belong
    to exactly one tetrahedron).

    Parameters
    ----------
    points : (N, 3) array

    Returns
    -------
    triangles : (M, 3) int array
        Each row is a triple of point indices forming a triangle on the surface.
    """
    if points.shape[1] != 3:
        raise ValueError("Points must be 3D for 3D Delaunay.")

    tri = Delaunay(points)
    tets = tri.simplices  # (num_tets, 4)

    face_count = {}

    # each tetrahedron has 4 triangular faces
    for tet in tets:
        for face in itertools.combinations(tet, 3):
            face = tuple(sorted(face))  # order-independent
            face_count[face] = face_count.get(face, 0) + 1

    # keep faces that appear exactly once ⇒ boundary of the Delaunay complex
    surface_faces = [face for face, count in face_count.items() if count == 1]

    return np.array(surface_faces, dtype=int)




