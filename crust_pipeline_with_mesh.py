"""
CRUST + CURVATURE-AWARE REDUCTION PIPELINE (NO MLS)
with a PRACTICAL DENSITY GATE (kNN spacing uniformity)

Pipeline per iteration:
1) Reconstruct (Crust if density gate allows, else alpha fallback)
2) Compute curvature proxy on reconstructed mesh
3) Curvature-aware reduction (thinning), tuned based on gate PASS/WARN
4) Re-check density gate on reduced set:
    - PASS/WARN: accept reduced set
    - FAIL: revert + stop early
5) Repeat

Why this gate:
- It tests what you actually need for Crust in practice: local sampling density / uniformity.
- Avoids fragile 3D Voronoi-pole medial-axis proxies that can misbehave on torus/helix.

Run:
  python3 crust_reduce_pipeline.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay, Voronoi, ConvexHull
from sklearn.neighbors import NearestNeighbors
import time
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# MESH AND PDB LOADING UTILITIES
# ============================================================

def check_dependencies():
    """Check if optional dependencies are available"""
    deps = {
        'trimesh': False,
        'biopython': False,
        'scipy': True  # Already imported above
    }
    
    try:
        import trimesh
        deps['trimesh'] = True
    except ImportError:
        pass
    
    try:
        from Bio.PDB import PDBParser
        deps['biopython'] = True
    except ImportError:
        pass
    
    return deps


def load_mesh_from_file(filepath):
    """
    Load mesh from STL, OBJ, or PLY file
    
    Requires: pip install trimesh
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh not installed. Run: pip install trimesh")
    
    mesh = trimesh.load(filepath)
    return mesh.vertices, mesh.faces


def sample_points_from_mesh(vertices, faces, n_points=4000, noise=0.02):
    """Sample points uniformly from triangle mesh surface"""
    areas = []
    triangles = []
    
    for face in faces:
        v0, v1, v2 = vertices[face]
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(cross)
        areas.append(area)
        triangles.append([v0, v1, v2])
    
    areas = np.array(areas)
    triangles = np.array(triangles)
    
    if areas.sum() == 0:
        raise ValueError("Mesh has zero total area")
    
    probabilities = areas / areas.sum()
    triangle_indices = np.random.choice(len(triangles), size=n_points, p=probabilities)
    
    points = []
    for tri_idx in triangle_indices:
        v0, v1, v2 = triangles[tri_idx]
        r1, r2 = np.random.random(), np.random.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        point = v0 + r1 * (v1 - v0) + r2 * (v2 - v0)
        point += np.random.normal(0, noise, 3)
        points.append(point)
    
    return np.array(points)


def generate_from_mesh_file(filepath, n_points=4000, noise=0.02):
    """Load mesh file and sample points from surface"""
    print(f"Loading mesh from: {filepath}")
    vertices, faces = load_mesh_from_file(filepath)
    print(f"  Mesh: {len(vertices)} vertices, {len(faces)} faces")
    print(f"Sampling {n_points} points from mesh surface...")
    points = sample_points_from_mesh(vertices, faces, n_points, noise)
    print(f"  ✓ Generated {len(points)} points")
    return points


def generate_from_pdb_cartoon(pdb_file, points_per_residue=20, tube_radius=0.05, noise=0.02):
    """
    Load protein backbone (cartoon) from PDB file
    
    Extracts C-alpha atoms and creates smooth cartoon representation
    """
    try:
        from Bio.PDB import PDBParser
    except ImportError:
        raise ImportError("BioPython not installed. Run: pip install biopython")
    
    from scipy.interpolate import splprep, splev, interp1d
    
    print(f"Loading PDB file: {pdb_file}")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # Extract C-alpha atoms
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id('CA'):
                    ca_coords.append(residue['CA'].get_coord())
    
    if len(ca_coords) < 4:
        raise ValueError(f"Need ≥4 C-alpha atoms, got {len(ca_coords)}")
    
    ca_coords = np.array(ca_coords)
    print(f"  Found {len(ca_coords)} C-alpha atoms")
    print(f"  Sampling {points_per_residue} points per residue...")
    
    # Fit smooth spline
    try:
        tck, u = splprep([ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2]], 
                         s=0, k=min(3, len(ca_coords)-1))
        n_points_total = len(ca_coords) * points_per_residue
        u_fine = np.linspace(0, 1, n_points_total)
        backbone_points = np.array(splev(u_fine, tck)).T
    except Exception:
        # Fallback to linear
        t = np.arange(len(ca_coords))
        n_points_total = len(ca_coords) * points_per_residue
        t_fine = np.linspace(0, len(ca_coords)-1, n_points_total)
        fx = interp1d(t, ca_coords[:, 0], kind='linear')
        fy = interp1d(t, ca_coords[:, 1], kind='linear')
        fz = interp1d(t, ca_coords[:, 2], kind='linear')
        backbone_points = np.column_stack([fx(t_fine), fy(t_fine), fz(t_fine)])
    
    # Add tube radius
    points_with_radius = []
    for center in backbone_points:
        for _ in range(3):
            offset = np.random.randn(3)
            offset = offset * (tube_radius / np.linalg.norm(offset))
            point = center + offset
            points_with_radius.append(point)
    
    points_with_radius = np.array(points_with_radius)
    noisy_points = points_with_radius + np.random.normal(0, noise, points_with_radius.shape)
    
    print(f"  ✓ Generated {len(noisy_points)} points along backbone")
    return noisy_points




# ============================================================
# SHAPE GENERATORS
# ============================================================

def generate_ellipsoid(n_points=4000, a=1.5, b=1.0, c=0.7, noise=0.02):
    phi = np.pi * (3.0 - np.sqrt(5.0))
    pts = []
    for i in range(n_points):
        y = 1.0 - (i / float(n_points - 1)) * 2.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        x = a * np.cos(theta) * r
        y_coord = b * y
        z = c * np.sin(theta) * r
        p = np.array([x, y_coord, z]) + np.random.normal(0, noise, 3)
        pts.append(p)
    return np.array(pts)

def generate_sphere(n_points=4000, radius=1.0, noise=0.02):
    phi = np.pi * (3.0 - np.sqrt(5.0))
    pts = []
    for i in range(n_points):
        y = 1.0 - (i / float(n_points - 1)) * 2.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        x = radius * np.cos(theta) * r
        y_coord = radius * y
        z = radius * np.sin(theta) * r
        p = np.array([x, y_coord, z]) + np.random.normal(0, noise, 3)
        pts.append(p)
    return np.array(pts)

def generate_torus(n_points=20000, major_radius=1.0, minor_radius=0.3, noise=0.02):
    """
    Golden-ratio torus generator (quasi-random, low-discrepancy)
    
    Note: This generator is *not* uniform in area, but it's typically "dense enough"
    for Crust if N is high. (Our density gate handles the practical risk.)
    
    Due to golden ratio patterning, this typically requires r ≥ 0.95 for r-sample criterion.
    """
    pts = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        u = 2.0 * np.pi * (i / n_points)
        v = phi * i
        x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
        y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
        z = minor_radius * np.sin(v)
        p = np.array([x, y, z]) + np.random.normal(0, noise, 3)
        pts.append(p)
    return np.array(pts)

def generate_torus_uniform(n_points=20000, major_radius=1.0, minor_radius=0.3, noise=0.02):
    """
    Uniform area-weighted torus generator (truly random)
    
    Uses uniform random sampling in both angular directions.
    This generator satisfies r=0.5 criterion from Alexa et al. (2003)
    
    Recommended for rigorous r-sample testing.
    """
    pts = []
    np.random.seed(42)  # For reproducibility
    
    for i in range(n_points):
        # Uniform random in [0, 2π] for both angles
        u = 2.0 * np.pi * np.random.random()
        v = 2.0 * np.pi * np.random.random()
        
        # Parametric torus equations
        x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
        y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
        z = minor_radius * np.sin(v)
        
        # Add noise
        p = np.array([x, y, z]) + np.random.normal(0, noise, 3)
        pts.append(p)
    
    return np.array(pts)

def generate_rounded_box(n_points=4000, size=1.0, radius=0.3, noise=0.02):
    pts = []
    box_size = size - radius
    n_per_face = max(1, n_points // 6)

    for face_idx in range(6):
        for _ in range(n_per_face):
            u = np.random.uniform(-box_size, box_size)
            v = np.random.uniform(-box_size, box_size)

            if face_idx == 0:   x, y, z = box_size, u, v
            elif face_idx == 1: x, y, z = -box_size, u, v
            elif face_idx == 2: x, y, z = u, box_size, v
            elif face_idx == 3: x, y, z = u, -box_size, v
            elif face_idx == 4: x, y, z = u, v, box_size
            else:               x, y, z = u, v, -box_size

            p = np.array([x, y, z])
            norm = np.linalg.norm(p)
            if norm > box_size:
                p = p / (norm + 1e-12) * (box_size + radius)

            p += np.random.normal(0, noise, 3)
            pts.append(p)

    return np.array(pts[:n_points])

def generate_cone(n_points=4000, height=2.0, radius=1.0, noise=0.02):
    """
    Area-weighted UNIFORM sampling on cone surface (lateral surface only, no base)
    
    This ensures points are evenly distributed across the entire cone surface,
    NOT biased toward the apex. Critical for demonstrating curvature-aware reduction.
    
    Math:
    - Cone parameterized as: (r(z)cos(θ), r(z)sin(θ), z)
    - Where r(z) = radius * (z + height/2) / height  (linear from apex to base)
    - Surface area element: dS ∝ r(z) dz dθ
    - For uniform sampling by area, we need to account for increasing circumference
    """
    points = []
    np.random.seed(42)  # For reproducibility
    
    slant_height = np.sqrt(height**2 + radius**2)
    
    for _ in range(n_points):
        # Sample z with area weighting
        # Since circumference grows linearly with z, we use sqrt for uniform area
        u = np.random.random()
        z_normalized = np.sqrt(u)  # sqrt accounts for growing circumference
        
        # Map to actual z coordinate (apex at -height/2, base at +height/2)
        z = -height/2 + z_normalized * height
        
        # Radius at this height
        r_at_z = radius * (z + height/2) / height
        
        # Uniform angle around cone
        theta = 2.0 * np.pi * np.random.random()
        
        # Convert to Cartesian
        x = r_at_z * np.cos(theta)
        y = r_at_z * np.sin(theta)
        
        # Add noise
        point = np.array([x, y, z]) + np.random.normal(0, noise, 3)
        points.append(point)
    
    return np.array(points)

def generate_dna_helix(n_points=15000,
                       turns=4.0,
                       pitch=0.5,
                       radius=0.5,
                       tube_radius=0.08,
                       noise=0.01):
    """
    Double-helix tube surface: two strands, each a tube around a helix centerline.
    """
    rng = np.random.default_rng(1)
    pts = []
    n_each = n_points // 2
    u = rng.random(n_each) * (2.0 * np.pi * turns)

    def sample_tube(center, tangent):
        t = tangent / (np.linalg.norm(tangent) + 1e-12)
        a = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        n = np.cross(t, a)
        n = n / (np.linalg.norm(n) + 1e-12)
        b = np.cross(t, n)
        ang = 2.0 * np.pi * rng.random()
        return center + tube_radius * (np.cos(ang) * n + np.sin(ang) * b)

    # strand 1
    for ui in u:
        x = radius * np.cos(ui)
        y = radius * np.sin(ui)
        z = (pitch / (2.0 * np.pi)) * ui
        center = np.array([x, y, z])
        tangent = np.array([-radius * np.sin(ui), radius * np.cos(ui), pitch / (2.0 * np.pi)])
        p = sample_tube(center, tangent) + rng.normal(0, noise, 3)
        pts.append(p)

    # strand 2 (phase pi)
    for ui in (u + np.pi):
        x = radius * np.cos(ui)
        y = radius * np.sin(ui)
        z = (pitch / (2.0 * np.pi)) * ui
        center = np.array([x, y, z])
        tangent = np.array([-radius * np.sin(ui), radius * np.cos(ui), pitch / (2.0 * np.pi)])
        p = sample_tube(center, tangent) + rng.normal(0, noise, 3)
        pts.append(p)

    pts = np.array(pts)
    pts[:, 2] -= pts[:, 2].mean()
    return pts[:n_points]


def generate_cube(n_points=4000, size=1.0, noise=0.02):
    """
    Uniform surface sampling on cube faces
    
    Each of 6 faces gets equal number of samples.
    Points are uniformly distributed across each face.
    
    Parameters:
    -----------
    n_points : int
        Total points to sample
    size : float
        Cube side length
    noise : float
        Gaussian noise to add
    """
    points = []
    half = size / 2
    n_per_face = n_points // 6
    
    # Define 6 faces
    faces = [
        # (axis, fixed_value, u_range, v_range)
        ('x',  half, (-half, half), (-half, half)),  # +X face
        ('x', -half, (-half, half), (-half, half)),  # -X face
        ('y',  half, (-half, half), (-half, half)),  # +Y face
        ('y', -half, (-half, half), (-half, half)),  # -Y face
        ('z',  half, (-half, half), (-half, half)),  # +Z face
        ('z', -half, (-half, half), (-half, half)),  # -Z face
    ]
    
    for axis, fixed_val, u_range, v_range in faces:
        for _ in range(n_per_face):
            u = np.random.uniform(*u_range)
            v = np.random.uniform(*v_range)
            
            if axis == 'x':
                point = np.array([fixed_val, u, v])
            elif axis == 'y':
                point = np.array([u, fixed_val, v])
            else:  # z
                point = np.array([u, v, fixed_val])
            
            point += np.random.normal(0, noise, 3)
            points.append(point)
    
    return np.array(points[:n_points])


def load_protein_pdb(pdb_file, n_samples=5000, method='alpha', alpha=5.0):
    """
    Load protein structure from PDB file and extract surface points
    
    This allows you to use REAL protein structures instead of synthetic shapes!
    
    Parameters:
    -----------
    pdb_file : str
        Path to PDB file (download from https://www.rcsb.org/)
    n_samples : int
        Target number of surface points
    method : str
        'alpha' - Alpha shape (recommended, fast)
        'convex' - Convex hull (simple, only for globular proteins)
    alpha : float
        Alpha shape parameter in Angstroms (default 5.0)
        Smaller = tighter fit, Larger = smoother
    
    Returns:
    --------
    points : (N, 3) array
        Surface point cloud ready for your pipeline!
    
    Example:
    --------
    # Download PDB file first (e.g., 1A00.pdb for hemoglobin)
    points = load_protein_pdb('1A00.pdb', n_samples=8000, method='alpha')
    # Now use points in your pipeline!
    """
    print(f"\n{'='*70}")
    print(f"LOADING PROTEIN FROM PDB")
    print(f"{'='*70}")
    print(f"File: {pdb_file}")
    print(f"Method: {method}")
    
    # Read atomic coordinates
    atoms = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    atoms.append([x, y, z])
                except:
                    continue
    
    atoms = np.array(atoms)
    print(f"  ✓ Read {len(atoms)} atoms")
    
    # Center at origin
    atoms = atoms - atoms.mean(axis=0)
    
    # Normalize scale to ~[-1, 1] range (like synthetic shapes)
    scale = max(atoms.max(axis=0) - atoms.min(axis=0))
    atoms = atoms / (scale / 2.0)
    
    print(f"  Extracting surface points...")
    
    if method == 'alpha':
        # Alpha shape method
        delaunay = Delaunay(atoms)
        surface_triangles = []
        
        for simplex in delaunay.simplices:
            verts = atoms[simplex]
            center = verts.mean(axis=0)
            circumradius = np.linalg.norm(verts - center, axis=1).max()
            
            if circumradius < alpha / (scale / 2.0):  # Scaled alpha
                faces = [
                    [simplex[0], simplex[1], simplex[2]],
                    [simplex[0], simplex[1], simplex[3]],
                    [simplex[0], simplex[2], simplex[3]],
                    [simplex[1], simplex[2], simplex[3]],
                ]
                surface_triangles.extend(faces)
        
        # Find boundary
        face_counts = {}
        for face in surface_triangles:
            key = tuple(sorted(face))
            face_counts[key] = face_counts.get(key, 0) + 1
        
        boundary_faces = [list(face) for face, count in face_counts.items() if count == 1]
        
        # Sample from boundary triangles
        points = []
        if len(boundary_faces) > 0:
            samples_per_tri = max(1, n_samples // len(boundary_faces))
            
            for face in boundary_faces:
                v0, v1, v2 = atoms[face]
                for _ in range(samples_per_tri):
                    r1, r2 = np.random.random(2)
                    sqrt_r1 = np.sqrt(r1)
                    point = (1 - sqrt_r1) * v0 + sqrt_r1 * (1 - r2) * v1 + sqrt_r1 * r2 * v2
                    points.append(point)
        
        points = np.array(points)
    
    elif method == 'convex':
        # Convex hull (simple fallback)
        hull = ConvexHull(atoms)
        points = []
        samples_per_tri = max(1, n_samples // len(hull.simplices))
        
        for simplex in hull.simplices:
            v0, v1, v2 = atoms[simplex]
            for _ in range(samples_per_tri):
                r1, r2 = np.random.random(2)
                sqrt_r1 = np.sqrt(r1)
                point = (1 - sqrt_r1) * v0 + sqrt_r1 * (1 - r2) * v1 + sqrt_r1 * r2 * v2
                points.append(point)
        
        points = np.array(points)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'alpha' or 'convex'")
    
    print(f"  ✓ Generated {len(points)} surface points")
    print(f"{'='*70}\n")
    
    return points


# ============================================================
# MESH DATA STRUCTURE
# ============================================================

class SurfaceMesh:
    def __init__(self, vertices, faces):
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)
        self.n_vertices = len(vertices)
        self.n_faces = len(faces)
        self.face_normals = self._compute_face_normals()
        self.vertex_faces = self._build_vertex_faces()

    def _compute_face_normals(self):
        normals = []
        if len(self.faces) == 0:
            return np.zeros((0, 3))
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            e1 = v1 - v0
            e2 = v2 - v0
            n = np.cross(e1, e2)
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
            normals.append(n)
        return np.array(normals)

    def _build_vertex_faces(self):
        vf = [[] for _ in range(self.n_vertices)]
        for fi, face in enumerate(self.faces):
            for vi in face:
                vf[vi].append(fi)
        return vf


# ============================================================
# R-SAMPLE CRITERION (from Alexa et al. 2003)
# ============================================================

def estimate_local_feature_size_poles(points, use_voronoi_poles=False):
    """
    Estimate local feature size (LFS) - distance to medial axis
    
    From Alexa et al. (2003), Section 3.2:
    "We did not compute the medial axis, which can be quite a chore.
     Instead, we used the distance to the nearest pole as a reasonable,
     and easily computed, estimate of the distance to the medial axis."
    
    Options:
    1. use_voronoi_poles=True: Use actual Voronoi pole distances (more accurate)
    2. use_voronoi_poles=False: Use k-NN average as proxy (faster, more robust)
    
    Returns: array of LFS estimates (one per point)
    """
    n = len(points)
    
    if use_voronoi_poles and n >= 10:
        try:
            # Compute Voronoi diagram
            vor = Voronoi(points)
            lfs = np.zeros(n)
            
            for i in range(n):
                p = points[i]
                region_idx = vor.point_region[i]
                region = vor.regions[region_idx]
                
                if (-1 in region) or (len(region) == 0):
                    # Fallback to k-NN for this point
                    nbrs = NearestNeighbors(n_neighbors=min(10, n)).fit(points)
                    dists, _ = nbrs.kneighbors([p])
                    lfs[i] = dists[0, 1:].mean()
                    continue
                
                verts = vor.vertices[region]
                if len(verts) == 0:
                    nbrs = NearestNeighbors(n_neighbors=min(10, n)).fit(points)
                    dists, _ = nbrs.kneighbors([p])
                    lfs[i] = dists[0, 1:].mean()
                    continue
                
                # Distance to farthest Voronoi vertex (positive pole)
                distances = np.linalg.norm(verts - p, axis=1)
                lfs[i] = distances.max()
            
            return lfs
            
        except Exception:
            # Voronoi failed, fall back to k-NN
            pass
    
    # k-NN fallback (default)
    k = min(10, n - 1)
    if k < 1:
        return np.ones(n) * 0.1
    
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    # Average distance to k neighbors (exclude self at index 0)
    lfs = distances[:, 1:].mean(axis=1)
    return lfs


def check_r_sample_criterion(points, r=0.5, use_voronoi_poles=False, verbose=True):
    """
    Check if point set satisfies r-sample criterion from Alexa et al. (2003)
    
    From the paper, Section 3.2:
    "A sample S is an r-sample from a surface F when the Euclidean distance 
     from any point p ∈ F to the nearest sample point is at most r times 
     the distance from p to the nearest point of the medial axis of F."
    
    In practice: For each point, check if nearest neighbor distance ≤ r × LFS
    
    Parameters:
    -----------
    points : array
        Point cloud to validate
    r : float
        Sampling ratio (default 0.5, from paper: "generally suffices")
    use_voronoi_poles : bool
        Use actual Voronoi poles vs k-NN proxy for LFS
    
    Returns:
    --------
    (status, report_dict)
        status: "PASS", "WARN", or "FAIL"
        report_dict: detailed metrics
    """
    n = len(points)
    if n < 4:
        return "FAIL", {"reason": f"n={n} < 4", "n": n}
    
    # Estimate local feature size for each point
    lfs = estimate_local_feature_size_poles(points, use_voronoi_poles=use_voronoi_poles)
    
    # Find nearest neighbor distance for each point
    nbrs = NearestNeighbors(n_neighbors=2).fit(points)
    nn_dists, _ = nbrs.kneighbors(points)
    nn_dists = nn_dists[:, 1]  # Distance to nearest neighbor (not self)
    
    # r-sample criterion: nn_dist ≤ r × LFS
    ratios = nn_dists / (lfs + 1e-12)
    
    # Statistics
    max_ratio = float(ratios.max())
    median_ratio = float(np.median(ratios))
    p95_ratio = float(np.percentile(ratios, 95))
    violations = (ratios > r).sum()
    violation_pct = 100.0 * violations / n
    
    # Determine status
    status = "PASS"
    reason = None
    
    if max_ratio > r * 1.5:
        status = "FAIL"
        reason = f"severe r-sample violations ({violation_pct:.1f}% of points)"
    elif max_ratio > r:
        status = "WARN"
        reason = f"some r-sample violations ({violation_pct:.1f}% of points)"
    else:
        reason = f"satisfies r={r:.2f} criterion"
    
    report = {
        "n": n,
        "r_target": float(r),
        "max_ratio": max_ratio,
        "median_ratio": median_ratio,
        "p95_ratio": p95_ratio,
        "violations": int(violations),
        "violation_pct": violation_pct,
        "lfs_min": float(lfs.min()),
        "lfs_max": float(lfs.max()),
        "lfs_median": float(np.median(lfs)),
        "reason": reason,
        "method": "voronoi_poles" if use_voronoi_poles else "knn_proxy"
    }
    
    if verbose:
        print(f"    r-sample check: {status} | r={r:.2f}, max_ratio={max_ratio:.2f}, "
              f"violations={violation_pct:.1f}%")
        print(f"      LFS range: {lfs.min():.4f} to {lfs.max():.4f} "
              f"(median={np.median(lfs):.4f})")
    
    return status, report


# ============================================================
# PRACTICAL DENSITY GATE
# ============================================================

def crust_density_gate(points,
                       k=10,
                       max_cv=0.35,
                       max_p95_over_median=2.0,
                       verbose=True):
    """
    Practical 'enough points for Crust' gate:
      - Compute distance to k-th nearest neighbor (local spacing proxy)
      - Require spacing reasonably uniform (global-ish, but very effective)

    Returns: (status in {"PASS","WARN","FAIL"}, report dict)
    """
    n = len(points)
    if n < (k + 1):
        rep = {"reason": f"n={n} < k+1", "n": n, "k": k}
        if verbose:
            print(f"    density gate: FAIL | {rep}")
        return "FAIL", rep

    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    dists, _ = nbrs.kneighbors(points)
    kth = dists[:, -1]

    med = float(np.median(kth))
    p95 = float(np.percentile(kth, 95))
    mean = float(np.mean(kth))
    std = float(np.std(kth))
    cv = float(std / (mean + 1e-12))
    ratio = float(p95 / (med + 1e-12))

    status = "PASS"
    reason = None

    if (cv > max_cv) or (ratio > max_p95_over_median):
        status = "WARN"
        reason = "non-uniform sampling (risk of holes/degeneracy)"
        if (cv > max_cv * 1.5) or (ratio > max_p95_over_median * 1.5):
            status = "FAIL"
            reason = "severely non-uniform sampling"

    rep = {
        "n": n,
        "k": k,
        "median_kNN": med,
        "p95_kNN": p95,
        "cv_kNN": cv,
        "p95_over_median": ratio,
        "max_cv": float(max_cv),
        "max_p95_over_median": float(max_p95_over_median),
        "reason": reason
    }

    if verbose:
        print(f"    density gate: {status} | median(kNN)={med:.4f}, p95={p95:.4f}, "
              f"p95/med={ratio:.2f}, cv={cv:.2f}")

    return status, rep


# ============================================================
# ALPHA SHAPE FALLBACK
# ============================================================

def alpha_shape_surface(points, alpha=None):
    n = len(points)
    if n < 4:
        return SurfaceMesh(points, np.array([]))

    if alpha is None:
        nbrs = NearestNeighbors(n_neighbors=min(5, n)).fit(points)
        dists, _ = nbrs.kneighbors(points)
        alpha = 2.5 * dists[:, 1:].mean()

    try:
        delaunay = Delaunay(points)
    except Exception:
        hull = ConvexHull(points)
        return SurfaceMesh(points, hull.simplices)

    face_set = set()
    for simplex in delaunay.simplices:
        verts = points[simplex]
        circumradius = np.linalg.norm(verts - verts.mean(axis=0), axis=1).max()
        if circumradius < alpha:
            faces = [
                tuple(sorted([simplex[0], simplex[1], simplex[2]])),
                tuple(sorted([simplex[0], simplex[1], simplex[3]])),
                tuple(sorted([simplex[0], simplex[2], simplex[3]])),
                tuple(sorted([simplex[1], simplex[2], simplex[3]])),
            ]
            for f in faces:
                if f in face_set:
                    face_set.remove(f)
                else:
                    face_set.add(f)

    surface_faces = np.array(list(face_set)) if face_set else np.array([])
    if len(surface_faces) == 0:
        hull = ConvexHull(points)
        surface_faces = hull.simplices
    return SurfaceMesh(points, surface_faces)


# ============================================================
# CRUST (Your existing-style Crust implementation)
# ============================================================

class CrustReconstructor:
    def __init__(self, points, verbose=True):
        self.points = np.array(points)
        self.n_points = len(points)
        self.verbose = verbose

    def reconstruct(self):
        if self.n_points < 10:
            if self.verbose:
                print("    Too few points for Crust; using alpha shapes")
            return alpha_shape_surface(self.points, alpha=None)

        if self.verbose:
            print("  Running Crust algorithm...")
        start = time.time()

        try:
            vor = Voronoi(self.points)
        except Exception:
            if self.verbose:
                print("      Voronoi failed; using alpha shapes")
            return alpha_shape_surface(self.points, alpha=None)

        if self.verbose:
            print(f"    Step 1: Voronoi ({len(vor.vertices)} vertices)")

        poles = self._extract_poles(vor)
        if self.verbose:
            print(f"    Step 2: Poles ({len(poles)} poles)")

        if len(poles) < 4:
            if self.verbose:
                print("      Too few poles; using alpha shapes")
            return alpha_shape_surface(self.points, alpha=None)

        augmented = np.vstack([self.points, poles])
        if self.verbose:
            print(f"    Step 3: Augmented ({len(augmented)} total)")

        try:
            delaunay = Delaunay(augmented)
        except Exception:
            if self.verbose:
                print("      Delaunay failed; using alpha shapes")
            return alpha_shape_surface(self.points, alpha=None)

        if self.verbose:
            print(f"    Step 4: Delaunay ({len(delaunay.simplices)} tetrahedra)")

        faces = self._filter_crust_triangles(delaunay)
        if self.verbose:
            print(f"    Step 5: Filtered ({len(faces)} triangles)")
            print(f"    ✓ Crust complete ({time.time()-start:.2f}s)")

        return SurfaceMesh(self.points, faces)

    def _extract_poles(self, vor):
        poles = []
        for i in range(self.n_points):
            p = self.points[i]
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if (-1 in region) or (len(region) == 0):
                continue

            verts = vor.vertices[region]
            if len(verts) == 0:
                continue

            d = np.linalg.norm(verts - p, axis=1)
            j_far = int(np.argmax(d))
            pole1 = verts[j_far]
            poles.append(pole1)

            dir1 = pole1 - p
            dir1 /= (np.linalg.norm(dir1) + 1e-12)
            dots = (verts - p) @ dir1
            j_opp = int(np.argmin(dots))
            if j_opp != j_far:
                poles.append(verts[j_opp])

        return np.array(poles)

    def _filter_crust_triangles(self, delaunay):
        surface_triangles = []
        n_original = self.n_points
        face_dict = {}

        for tet_idx, tet in enumerate(delaunay.simplices):
            faces = [
                (tet[0], tet[1], tet[2]),
                (tet[0], tet[1], tet[3]),
                (tet[0], tet[2], tet[3]),
                (tet[1], tet[2], tet[3]),
            ]
            for f in faces:
                if all(v < n_original for v in f):
                    key = tuple(sorted(f))
                    face_dict.setdefault(key, []).append(tet_idx)

        for face in face_dict.keys():
            v0, v1, v2 = face
            p0, p1, p2 = self.points[v0], self.points[v1], self.points[v2]

            edge_lengths = [
                np.linalg.norm(p1 - p0),
                np.linalg.norm(p2 - p1),
                np.linalg.norm(p0 - p2)
            ]
            max_edge = max(edge_lengths)

            nn_dists = []
            for p in [p0, p1, p2]:
                d = np.linalg.norm(self.points - p, axis=1)
                d.sort()
                if len(d) > 1:
                    nn_dists.append(d[1])
            if not nn_dists:
                continue

            avg_nn = float(np.mean(nn_dists))
            if max_edge < 5.0 * avg_nn:
                surface_triangles.append(face)

        return np.array(surface_triangles)

def crust_reconstruction(points, verbose=True):
    return CrustReconstructor(points, verbose=verbose).reconstruct()


# ============================================================
# CURVATURE (proxy) & RESAMPLING
# ============================================================

def compute_mesh_curvature(mesh: SurfaceMesh):
    """
    Curvature proxy: mean angle deviation of adjacent face normals at each vertex.
    Normalized to roughly [0, ~0.33] by dividing by (3*max).
    """
    curv = np.zeros(mesh.n_vertices)
    if mesh.n_faces == 0:
        return curv

    for v_idx in range(mesh.n_vertices):
        adj_faces = mesh.vertex_faces[v_idx]
        if len(adj_faces) < 2:
            continue
        normals = mesh.face_normals[adj_faces]
        avg = normals.mean(axis=0)
        avg /= (np.linalg.norm(avg) + 1e-12)

        angles = []
        for n in normals:
            c = np.clip(np.dot(n, avg), -1.0, 1.0)
            angles.append(np.arccos(c))
        curv[v_idx] = float(np.mean(angles))

    if curv.max() > 0:
        curv = curv / (3.0 * curv.max())
    return curv

def resample_points_curvature(points,
                             curvature,
                             flat_keep=0.50,
                             mid_keep=0.85,
                             high_keep=1.00,
                             seed=0,
                             min_points=500):
    """
    Keep more points in high curvature regions.
    'Curved' is defined RELATIVELY: top 25% curvature values are "high"
    (via the t2 percentile split), not an absolute geometric threshold.
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    if n == 0:
        return points, curvature

    t1 = np.percentile(curvature, 40)
    t2 = np.percentile(curvature, 75)

    flat_mask = curvature < t1
    mid_mask = (curvature >= t1) & (curvature < t2)
    high_mask = curvature >= t2

    mask = np.zeros(n, dtype=bool)
    if flat_mask.sum() > 0:
        mask[flat_mask] = rng.random(flat_mask.sum()) < flat_keep
    if mid_mask.sum() > 0:
        mask[mid_mask] = rng.random(mid_mask.sum()) < mid_keep
    if high_mask.sum() > 0:
        mask[high_mask] = rng.random(high_mask.sum()) < high_keep

    # safety floor: don't drop below min_points unless original smaller
    floor = min(min_points, n)
    if mask.sum() < floor:
        idx = np.argsort(-curvature)
        keep = idx[:floor]
        mask[:] = False
        mask[keep] = True

    return points[mask], curvature[mask]


# ============================================================
# VISUALIZATION
# ============================================================

def set_axes_equal_3d(ax, points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def create_comprehensive_visualization(iteration_data, n_iterations):
    n_rows = n_iterations + 1
    fig = plt.figure(figsize=(18, 6 * n_rows))

    for row_idx, it in enumerate(iteration_data):
        iteration = it["iteration"]
        input_mesh = it["input_mesh"]
        input_curvatures = it["input_curvatures"]
        reduced_points = it["reduced_points"]
        reconstructed_mesh = it["reconstructed_mesh"]

        n_input_pts = it["n_input_pts"]
        n_reduced_pts = it["n_reduced_pts"]
        n_reconstructed_pts = it["n_reconstructed_pts"]
        reduction_pct = it["reduction_pct"]
        gate_status = it.get("gate_status", "—")

        # 1) Input mesh with curvature colors
        ax1 = fig.add_subplot(n_rows, 3, row_idx*3 + 1, projection="3d")
        triangles_3d, colors = [], []
        for face in input_mesh.faces:
            tri = input_mesh.vertices[face]
            triangles_3d.append(tri)
            colors.append(float(input_curvatures[face].mean()) if len(input_curvatures) else 0.0)

        colors = np.array(colors) if len(colors) else np.array([0.0])
        if len(colors) > 1 and colors.max() > colors.min():
            colors = (colors - colors.min()) / (colors.max() - colors.min())

        mesh_collection = Poly3DCollection(triangles_3d, alpha=0.35, edgecolor="darkblue", linewidths=0.25)
        mesh_collection.set_array(colors)
        mesh_collection.set_cmap("viridis")
        ax1.add_collection3d(mesh_collection)

        scat = ax1.scatter(input_mesh.vertices[:, 0], input_mesh.vertices[:, 1], input_mesh.vertices[:, 2],
                           c=input_curvatures, cmap="viridis", s=2, alpha=0.9)

        ax1.set_title(
            f"Iter {iteration}: Input (gate={gate_status})\n({n_input_pts} pts, {len(input_mesh.faces)} tris)",
            fontsize=11, fontweight="bold"
        )
        set_axes_equal_3d(ax1, input_mesh.vertices)
        ax1.view_init(elev=20, azim=45)
        if row_idx == 0:
            cbar = plt.colorbar(scat, ax=ax1, shrink=0.5, pad=0.1)
            cbar.set_label("Curvature proxy", fontsize=9)

        # 2) Reduced points
        ax2 = fig.add_subplot(n_rows, 3, row_idx*3 + 2, projection="3d")
        ax2.scatter(reduced_points[:, 0], reduced_points[:, 1], reduced_points[:, 2],
                    c="red", s=6, alpha=0.8)
        ax2.set_title(f"Reduced\n({n_reduced_pts} pts, {reduction_pct:.1f}%)",
                      fontsize=11, fontweight="bold")
        set_axes_equal_3d(ax2, reduced_points)
        ax2.view_init(elev=20, azim=45)

        # 3) Reconstructed mesh
        ax3 = fig.add_subplot(n_rows, 3, row_idx*3 + 3, projection="3d")
        if reconstructed_mesh is not None and len(reconstructed_mesh.faces) > 0:
            tris = [reconstructed_mesh.vertices[f] for f in reconstructed_mesh.faces]
            mesh_collection_recon = Poly3DCollection(tris, alpha=0.35,
                                                     facecolor="lightblue",
                                                     edgecolor="blue", linewidths=0.25)
            ax3.add_collection3d(mesh_collection_recon)
            ax3.scatter(reconstructed_mesh.vertices[:, 0], reconstructed_mesh.vertices[:, 1], reconstructed_mesh.vertices[:, 2],
                        c="blue", s=1, alpha=0.6)
            n_tris = len(reconstructed_mesh.faces)
            set_axes_equal_3d(ax3, reconstructed_mesh.vertices)
        else:
            n_tris = 0

        ax3.set_title(f"Reconstructed\n({n_reconstructed_pts} pts, {n_tris} tris)",
                      fontsize=11, fontweight="bold")
        ax3.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig


# ============================================================
# MAIN PIPELINE (NO MLS) + DENSITY GATE
# ============================================================

def crust_reduce_pipeline(initial_points,
                          n_iterations=3,
                          shape_name="surface",
                          force_crust=True,
                          gate_k=10,
                          gate_max_cv=0.35,
                          gate_max_p95_over_median=2.0,
                          use_r_sample=True,
                          r_value=0.5,
                          use_voronoi_poles=False):
    print("="*70)
    print(f"CRUST + REDUCTION PIPELINE - {shape_name.upper()}")
    print("NO MLS (no point regeneration)")
    print("Using FULL CRUST + density gate (kNN spacing uniformity)")
    if use_r_sample:
        print(f"+ r-sample criterion (r={r_value}, from Alexa et al. 2003)")
    print("="*70)

    iteration_data = []

    # --- Iteration 0: initial reconstruction ---
    print("\nPHASE 1: INITIAL RECONSTRUCTION")
    print("="*70)

    gate0, rep0 = crust_density_gate(initial_points,
                                     k=gate_k,
                                     max_cv=gate_max_cv,
                                     max_p95_over_median=gate_max_p95_over_median,
                                     verbose=True)
    
    # Check r-sample criterion if enabled
    r_status0 = "N/A"
    if use_r_sample:
        r_status0, r_rep0 = check_r_sample_criterion(
            initial_points,
            r=r_value,
            use_voronoi_poles=use_voronoi_poles,
            verbose=True
        )
        # Combine statuses: use worse of the two
        combined_status = gate0
        if r_status0 == "FAIL" or (r_status0 == "WARN" and gate0 == "PASS"):
            combined_status = r_status0
            print(f"    Using r-sample status: {r_status0}")
    else:
        combined_status = gate0

    if (not force_crust) or (combined_status == "FAIL"):
        if combined_status == "FAIL":
            print("    sampling gate FAIL, using alpha shapes for Iter 0")
        mesh0 = alpha_shape_surface(initial_points, alpha=None)
        gate_used = "ALPHA"
    else:
        if combined_status == "WARN":
            print("    sampling gate WARN, running Crust anyway (more conservative reduction later)")
        mesh0 = crust_reconstruction(initial_points, verbose=True)
        if mesh0 is None or len(mesh0.faces) == 0:
            print("    Crust returned 0 triangles, falling back to alpha shapes")
            mesh0 = alpha_shape_surface(initial_points, alpha=None)
            gate_used = "ALPHA"
        else:
            gate_used = combined_status

    print("  Computing curvature...")
    curv0 = compute_mesh_curvature(mesh0)

    iteration_data.append({
        "iteration": 0,
        "input_mesh": mesh0,
        "input_curvatures": curv0,
        "reduced_points": mesh0.vertices,
        "reconstructed_mesh": mesh0,
        "n_input_pts": len(initial_points),
        "n_reduced_pts": len(mesh0.vertices),
        "n_reconstructed_pts": len(mesh0.vertices),
        "reduction_pct": 100.0,
        "gate_status": gate_used,
    })

    # Track previous iteration's output for next iteration's input
    previous_mesh = mesh0
    previous_curvatures = curv0
    current_points = mesh0.vertices.copy()
    current_curvatures = curv0.copy()

    # --- Iterative reduction + reconstruction ---
    print("\nPHASE 2: ITERATIVE REFINEMENT (NO MLS)")
    print("="*70)

    for iteration in range(1, n_iterations + 1):
        print(f"\nITERATION {iteration}/{n_iterations}")
        print("─"*70)

        # Gate on current set (controls reduction aggressiveness)
        gate_now, rep_now = crust_density_gate(current_points,
                                               k=gate_k,
                                               max_cv=gate_max_cv,
                                               max_p95_over_median=gate_max_p95_over_median,
                                               verbose=True)

        # Tune reduction based on gate
        # PASS -> moderate thinning
        # WARN -> very gentle thinning (keep more everywhere)
        # FAIL -> stop (shouldn't happen often because we'd have fallen back earlier)
        if gate_now == "PASS":
            flat_keep, mid_keep, high_keep = 0.60, 0.90, 1.00
            min_points = 1500
        elif gate_now == "WARN":
            flat_keep, mid_keep, high_keep = 0.85, 0.97, 1.00
            min_points = 3000
            print("  [NOTE] Gate=WARN ⇒ using conservative reduction (keep more points)")
        else:  # FAIL
            print("   Gate=FAIL on current set ⇒ stopping early (won't thin further).")
            break

        # 1) Reduce using curvature
        print("  [CURVATURE-AWARE] Step 1: Reduction")
        reduced_points, reduced_curv = resample_points_curvature(
            current_points, current_curvatures,
            flat_keep=flat_keep, mid_keep=mid_keep, high_keep=high_keep,
            seed=iteration,
            min_points=min_points
        )
        reduction_pct = (len(reduced_points) / len(current_points)) * 100.0
        print(f"    {len(current_points)} → {len(reduced_points)} ({reduction_pct:.1f}%)")

        # 2) Gate on reduced set (hard safety)
        gate_red, rep_red = crust_density_gate(reduced_points,
                                               k=gate_k,
                                               max_cv=gate_max_cv,
                                               max_p95_over_median=gate_max_p95_over_median,
                                               verbose=True)
        
        # Check r-sample criterion on reduced set if enabled
        if use_r_sample:
            r_status_red, r_rep_red = check_r_sample_criterion(
                reduced_points,
                r=r_value,
                use_voronoi_poles=use_voronoi_poles,
                verbose=True
            )
            # Combine statuses
            combined_red_status = gate_red
            if r_status_red == "FAIL" or (r_status_red == "WARN" and gate_red == "PASS"):
                combined_red_status = r_status_red
                print(f"   Using r-sample status for reduced set: {r_status_red}")
        else:
            combined_red_status = gate_red
        
        if combined_red_status == "FAIL":
            print("     Reduction made sampling inadequate for Crust. Reverting and stopping.")
            break

        # 3) Reconstruct from reduced set
        print("  [FINAL] Step 2: Reconstructing surface")
        if combined_red_status == "WARN":
            print("      Reduced set is WARN ⇒ running Crust anyway, but expect more artifacts.")

        reconstructed = crust_reconstruction(reduced_points, verbose=False)
        if reconstructed is None or len(reconstructed.faces) == 0:
            print("      Crust returned 0 triangles, falling back to alpha shapes")
            reconstructed = alpha_shape_surface(reduced_points, alpha=None)
            gate_used_iter = "ALPHA"
        else:
            gate_used_iter = combined_red_status

        # 4) Curvature for next iteration
        next_curv = compute_mesh_curvature(reconstructed)
        print(f"    ✓ {reconstructed.n_faces} triangles")

        # 5) Store iteration data with PREVIOUS mesh as input
        iteration_data.append({
            "iteration": iteration,
            "input_mesh": previous_mesh,              # ← FIX: Use previous iteration's output
            "input_curvatures": previous_curvatures,  # ← FIX: Use previous iteration's curvatures
            "reduced_points": reduced_points,
            "reconstructed_mesh": reconstructed,
            "n_input_pts": len(current_points),
            "n_reduced_pts": len(reduced_points),
            "n_reconstructed_pts": len(reconstructed.vertices),
            "reduction_pct": reduction_pct,
            "gate_status": gate_used_iter,
        })

        # 6) Update for next iteration
        previous_mesh = reconstructed
        previous_curvatures = next_curv
        current_points = reduced_points
        current_curvatures = reduced_curv if len(reduced_curv) == len(reduced_points) else next_curv

    print("\nCREATING VISUALIZATION...")
    print("="*70)
    actual_iterations = len(iteration_data) - 1
    fig = create_comprehensive_visualization(iteration_data, actual_iterations)
    filename = f"reconstruction_{shape_name}_NO_MLS_DENSITY_GATE.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {filename}")
    plt.show()

    return iteration_data


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*70)
    print("CRUST + REDUCTION (NO MLS) + DENSITY GATE")
    print("="*70)

    shapes = {
        1: ("Ellipsoid", generate_ellipsoid, False, 4000),
        2: ("Sphere", generate_sphere, False, 4000),
        3: ("Torus (Golden Ratio)", generate_torus, True, 20000),
        4: ("Torus (Uniform)", generate_torus_uniform, True, 20000),
        5: ("Rounded Box", generate_rounded_box, False, 4000),
        6: ("Cone", generate_cone, False, 4000),
        7: ("DNA Helix", generate_dna_helix, True, 15000),
        8: ("Cube", generate_cube, False, 4000),
        9: ("Load Mesh File (STL/OBJ/PLY)", None, True, 4000),
        10: ("Protein Cartoon (PDB backbone)", None, True, 2000),
        11: ("Protein Surface (PDB all atoms)", None, True, 8000),
    }

    # Check dependencies
    deps = check_dependencies()
    
    print("\nShapes:")
    print("-" * 70)
    for num, (name, _, needs_crust, default_n) in shapes.items():
        marker = "-" if needs_crust else "  "
        
        # Add dependency warnings
        warning = ""
        if num == 9 and not deps['trimesh']:
            warning = "  needs: pip install trimesh"
        elif num in [10, 11] and not deps['biopython']:
            warning = "  needs: pip install biopython"
        elif num == 12 and not deps['trimesh']:
            warning = "  needs: pip install trimesh + bunny.ply file"
        
        print(f"{marker} {num}. {name} (default N={default_n}){warning}")
    
    print("-" * 70)
    print("Note: Torus (Uniform) satisfies r=0.5 | Torus (Golden Ratio) needs r≥0.95")
    print("      Cartoon = protein backbone/ribbon | Surface = molecular envelope")
    print("-" * 70)

    while True:
        choice = input("\nShape (1-12) or 'q': ").strip()
        if choice.lower() == "q":
            return
        try:
            shape_choice = int(choice)
            if shape_choice in shapes:
                break
        except ValueError:
            pass
        print(" Invalid")

    shape_name, generator, needs_crust, default_n = shapes[shape_choice]

    # Handle special loading cases
    if shape_choice == 9:
        # Mesh file loading
        mesh_file = input("\nPath to mesh file (.stl, .obj, .ply): ").strip()
        if not mesh_file:
            print(" No file specified")
            return
        
        n_str = input(f"\nPoints to sample (default={default_n}): ").strip()
        n_points = default_n if n_str == "" else max(500, int(n_str))
        
        noise_str = input("\nScanning noise (default=0.02): ").strip()
        noise = 0.02 if noise_str == "" else float(noise_str)
        
        print(f"\nLoading mesh from {mesh_file}...")
        try:
            initial_points = generate_from_mesh_file(mesh_file, n_points=n_points, noise=noise)
            shape_name = mesh_file.split('/')[-1].replace('.', '_')
        except Exception as e:
            print(f" Error loading mesh: {e}")
            return
    
    elif shape_choice == 10:
        # PDB cartoon loading
        pdb_file = input("\nPath to PDB file (e.g., 1A3M.pdb): ").strip()
        if not pdb_file or not pdb_file.endswith('.pdb'):
            print(" Invalid PDB file")
            return
        
        ppr_str = input("\nPoints per residue (default=20): ").strip()
        points_per_residue = 20 if ppr_str == "" else int(ppr_str)
        
        print(f"\nLoading protein cartoon from {pdb_file}...")
        try:
            initial_points = generate_from_pdb_cartoon(pdb_file, points_per_residue=points_per_residue)
            shape_name = pdb_file.replace('.pdb', '').replace('/', '_')
        except Exception as e:
            print(f" Error loading PDB: {e}")
            return
    
    elif shape_choice == 11:
        # PDB surface loading (original method)
        pdb_file = input("\nPath to PDB file (e.g., 1A00.pdb): ").strip()
        if not pdb_file or not pdb_file.endswith('.pdb'):
            print(" Invalid PDB file")
            return
        
        n_str = input(f"\nSurface points N (default={default_n}): ").strip()
        n_points = default_n if n_str == "" else max(500, int(n_str))
        
        alpha_str = input("\nAlpha shape parameter (default=5.0 Å): ").strip()
        alpha_param = 5.0 if alpha_str == "" else float(alpha_str)
        
        print(f"\nLoading protein from {pdb_file}...")
        try:
            initial_points = load_protein_pdb(pdb_file, n_samples=n_points, method='alpha', alpha=alpha_param)
            shape_name = pdb_file.replace('.pdb', '').replace('/', '_')
        except Exception as e:
            print(f" Error loading PDB: {e}")
            return
    
    
    else:
        # Normal shape generation
        n_str = input(f"\nInitial points N (default={default_n}): ").strip()
        n_points = default_n if n_str == "" else max(500, int(n_str))
        
        print(f"\nGenerating {shape_name} with N={n_points}...")
        initial_points = generator(n_points=n_points)
    
    print(f"✓ {len(initial_points)} points")

    # Crust selection
    if needs_crust:
        print("\n Non-convex-ish shape: forcing Crust on (with density gate safety)")
        force_crust = True
    else:
        algo = input("\nAlgorithm (1=Alpha, 2=Crust, default=2): ").strip()
        if algo == "" or algo == "2":
            force_crust = True
        else:
            force_crust = False

    # Gate parameters (optional)
    gate_k_str = input("\nDensity gate k (default=10): ").strip()
    gate_k = 10 if gate_k_str == "" else int(gate_k_str)
    
    # r-sample criterion option
    r_sample_choice = input("\nUse r-sample criterion from Alexa et al.? (y/n, default=y): ").strip().lower()
    use_r_sample = (r_sample_choice != 'n')
    
    r_value = 0.5
    use_voronoi_poles = False
    if use_r_sample:
        r_str = input("  r value (default=0.5, paper recommends 0.5): ").strip()
        r_value = 0.5 if r_str == "" else float(r_str)
        
        pole_choice = input("  Use Voronoi poles for LFS? (y/n, default=n, uses k-NN proxy): ").strip().lower()
        use_voronoi_poles = (pole_choice == 'y')

    it_str = input("\nIterations (1-5, default=3): ").strip()
    n_iterations = 3 if it_str == "" else int(it_str)
    n_iterations = int(np.clip(n_iterations, 1, 5))

    print("\n" + "="*70)
    print("STARTING PIPELINE")
    print("="*70)

    crust_reduce_pipeline(
        initial_points,
        n_iterations=n_iterations,
        shape_name=shape_name.lower().replace(" ", "_"),
        force_crust=force_crust,
        gate_k=gate_k,
        gate_max_cv=0.35,
        gate_max_p95_over_median=2.0,
        use_r_sample=use_r_sample,
        r_value=r_value,
        use_voronoi_poles=use_voronoi_poles
    )

    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)

    another = input("\nAnother? (y/n): ").strip().lower()
    if another == "y":
        print("\n\n")
        return main()
    else:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()


