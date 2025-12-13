"""
Hybrid Crust + Curvature-Aware Reconstruction Pipeline
WITH COMPREHENSIVE VISUALIZATION
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay, ConvexHull, Voronoi
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import time

# ============================================================
# [All the previous helper classes and functions remain the same]
# SurfaceMesh, alpha_shape_surface, compute_mesh_curvature, etc.
# ============================================================

class SurfaceMesh:
    """Simple mesh data structure"""
    def __init__(self, vertices, faces):
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)
        self.n_vertices = len(vertices)
        self.n_faces = len(faces)
        self.face_normals = self._compute_face_normals()
        self.vertex_faces = self._build_vertex_faces()
    
    def _compute_face_normals(self):
        normals = []
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            normals.append(normal)
        return np.array(normals)
    
    def _build_vertex_faces(self):
        vertex_faces = [[] for _ in range(self.n_vertices)]
        for face_idx, face in enumerate(self.faces):
            for vertex_idx in face:
                vertex_faces[vertex_idx].append(face_idx)
        return vertex_faces


def alpha_shape_surface(points, alpha=None):
    """Extract surface using alpha shapes"""
    n = len(points)
    
    if alpha is None:
        nbrs = NearestNeighbors(n_neighbors=2).fit(points)
        distances, _ = nbrs.kneighbors(points)
        avg_dist = distances[:, 1].mean()
        alpha = 2.0 * avg_dist
    
    delaunay = Delaunay(points)
    face_set = set()
    
    for simplex in delaunay.simplices:
        vertices = points[simplex]
        circumradius = np.linalg.norm(vertices - vertices.mean(axis=0), axis=1).max()
        
        if circumradius < alpha:
            faces = [
                tuple(sorted([simplex[0], simplex[1], simplex[2]])),
                tuple(sorted([simplex[0], simplex[1], simplex[3]])),
                tuple(sorted([simplex[0], simplex[2], simplex[3]])),
                tuple(sorted([simplex[1], simplex[2], simplex[3]])),
            ]
            for face in faces:
                if face in face_set:
                    face_set.remove(face)
                else:
                    face_set.add(face)
    
    surface_faces = np.array(list(face_set))
    return SurfaceMesh(points, surface_faces)


def compute_mesh_curvature(mesh):
    """Compute curvature from normal variation"""
    curvatures = np.zeros(mesh.n_vertices)
    
    for v_idx in range(mesh.n_vertices):
        adjacent_faces = mesh.vertex_faces[v_idx]
        
        if len(adjacent_faces) < 2:
            continue
        
        normals = mesh.face_normals[adjacent_faces]
        avg_normal = normals.mean(axis=0)
        norm = np.linalg.norm(avg_normal)
        if norm > 1e-10:
            avg_normal = avg_normal / norm
        
        deviations = []
        for normal in normals:
            cos_angle = np.clip(np.dot(normal, avg_normal), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            deviations.append(angle)
        
        curvatures[v_idx] = np.mean(deviations)
    
    if curvatures.max() > 0:
        curvatures = curvatures / (3 * curvatures.max())
    
    return curvatures


def resample_points_curvature(points, curvature, flat_keep=0.25, mid_keep=0.70, high_keep=1.0, seed=0):
    """Curvature-aware resampling"""
    rng = np.random.default_rng(seed)
    n = len(points)
    
    t1 = np.percentile(curvature, 40)
    t2 = np.percentile(curvature, 75)
    
    mask = np.zeros(n, dtype=bool)
    
    flat_mask = curvature < t1
    mid_mask = (curvature >= t1) & (curvature < t2)
    high_mask = curvature >= t2
    
    if flat_mask.sum() > 0:
        mask[flat_mask] = rng.random(flat_mask.sum()) < flat_keep
    if mid_mask.sum() > 0:
        mask[mid_mask] = rng.random(mid_mask.sum()) < mid_keep
    if high_mask.sum() > 0:
        mask[high_mask] = rng.random(high_mask.sum()) < high_keep
    
    return points[mask], curvature[mask]


def delaunay_surface_triangles(points):
    """Extract surface triangles"""
    if len(points) < 4:
        return np.array([])
    
    try:
        delaunay = Delaunay(points)
    except:
        return np.array([])
    
    face_counts = {}
    for simplex in delaunay.simplices:
        faces = [
            tuple(sorted([simplex[0], simplex[1], simplex[2]])),
            tuple(sorted([simplex[0], simplex[1], simplex[3]])),
            tuple(sorted([simplex[0], simplex[2], simplex[3]])),
            tuple(sorted([simplex[1], simplex[2], simplex[3]])),
        ]
        for face in faces:
            face_counts[face] = face_counts.get(face, 0) + 1
    
    surface_faces = [face for face, count in face_counts.items() if count == 1]
    return np.array(surface_faces)


def fit_polynomial_patches(reduced_points, triangles):
    """Fit quadratic polynomial patches"""
    patches = []
    for tri_indices in triangles:
        tri_verts = reduced_points[tri_indices]
        patch = fit_local_patch(tri_verts, reduced_points)
        patches.append({'indices': tri_indices, 'vertices': tri_verts, 'patch': patch})
    return patches


def fit_local_patch(tri_verts, all_points):
    """Fit quadratic patch"""
    center = tri_verts.mean(axis=0)
    
    edge1 = tri_verts[1] - tri_verts[0]
    edge2 = tri_verts[2] - tri_verts[0]
    normal = np.cross(edge1, edge2)
    normal_norm = np.linalg.norm(normal)
    
    if normal_norm > 1e-10:
        normal = normal / normal_norm
    else:
        normal = np.array([0, 0, 1])
    
    tangent1 = edge1 / (np.linalg.norm(edge1) + 1e-10)
    tangent2 = np.cross(normal, tangent1)
    tangent2 = tangent2 / (np.linalg.norm(tangent2) + 1e-10)
    
    radius = 2.5 * np.max(np.linalg.norm(tri_verts - center, axis=1))
    distances = np.linalg.norm(all_points - center, axis=1)
    nearby_mask = distances < radius
    nearby_points = all_points[nearby_mask]
    
    if len(nearby_points) < 6:
        nearby_points = all_points
    
    relative = nearby_points - center
    x = relative @ tangent1
    y = relative @ tangent2
    z = relative @ normal
    
    features = np.column_stack([np.ones(len(x)), x, y, x**2, x*y, y**2])
    coeffs, _, _, _ = np.linalg.lstsq(features, z, rcond=None)
    
    return {
        'coeffs': coeffs,
        'origin': center,
        'tangent1': tangent1,
        'tangent2': tangent2,
        'normal': normal,
        'tri_verts': tri_verts
    }


def sample_polynomial_patches(patches, target_size):
    """Sample polynomial patches"""
    total_area = 0
    areas = []
    
    for patch_info in patches:
        tri_verts = patch_info['patch']['tri_verts']
        v1 = tri_verts[1] - tri_verts[0]
        v2 = tri_verts[2] - tri_verts[0]
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        areas.append(area)
        total_area += area
    
    augmented = []
    for patch_info, area in zip(patches, areas):
        fraction = area / total_area
        n_samples = max(1, int(target_size * fraction))
        
        patch = patch_info['patch']
        tri_verts = patch['tri_verts']
        
        for _ in range(min(n_samples, target_size - len(augmented))):
            r1, r2 = np.random.rand(2)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            
            local_point = r1*tri_verts[0] + r2*tri_verts[1] + (1-r1-r2)*tri_verts[2]
            relative = local_point - patch['origin']
            x = relative @ patch['tangent1']
            y = relative @ patch['tangent2']
            
            coeffs = patch['coeffs']
            z = (coeffs[0] + coeffs[1]*x + coeffs[2]*y +
                 coeffs[3]*x**2 + coeffs[4]*x*y + coeffs[5]*y**2)
            
            point_3d = (patch['origin'] +
                       x * patch['tangent1'] +
                       y * patch['tangent2'] +
                       z * patch['normal'])
            
            augmented.append(point_3d)
            
            if len(augmented) >= target_size:
                break
        
        if len(augmented) >= target_size:
            break
    
    return np.array(augmented)


def generate_ellipsoid(n_points=1200, a=1.5, b=1.0, c=0.7, noise=0.02):
    """Generate ellipsoid"""
    phi = np.pi * (3. - np.sqrt(5.))
    points = []
    
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        radius_at_y = np.sqrt(1 - y * y)
        theta = phi * i
        
        x = a * np.cos(theta) * radius_at_y
        y_ell = b * y
        z = c * np.sin(theta) * radius_at_y
        
        point = np.array([x, y_ell, z])
        point += np.random.normal(0, noise, 3)
        points.append(point)
    
    return np.array(points)


def set_axes_equal_3d(ax, points):
    """Set equal aspect ratio"""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


# ============================================================
# NEW: COMPREHENSIVE VISUALIZATION FUNCTION
# ============================================================

def create_comprehensive_visualization(iteration_data, n_iterations):
    """
    Create comprehensive visualization showing all steps for all iterations
    
    Layout: 3 columns x (n_iterations + 1) rows
    - Column 1: Input (mesh with curvature)
    - Column 2: Reduced (points)
    - Column 3: Reconstructed (mesh)
    
    Row 0: Initial Crust reconstruction
    Rows 1-n: Iteration results
    """
    n_rows = n_iterations + 1
    fig = plt.figure(figsize=(18, 6 * n_rows))
    
    for row_idx, iter_data in enumerate(iteration_data):
        iteration = iter_data['iteration']
        input_mesh = iter_data['input_mesh']
        input_curvatures = iter_data['input_curvatures']
        reduced_points = iter_data['reduced_points']
        reduced_curvatures = iter_data['reduced_curvatures']
        reconstructed_mesh = iter_data['reconstructed_mesh']
        n_input_pts = iter_data['n_input_pts']
        n_reduced_pts = iter_data['n_reduced_pts']
        n_reconstructed_pts = iter_data['n_reconstructed_pts']
        reduction_pct = iter_data['reduction_pct']
        
        # ========================================
        # COLUMN 1: Input Mesh with Curvature
        # ========================================
        ax1 = fig.add_subplot(n_rows, 3, row_idx*3 + 1, projection='3d')
        
        # Draw mesh colored by curvature
        triangles_3d = []
        colors = []
        
        for face in input_mesh.faces:
            triangle = input_mesh.vertices[face]
            triangles_3d.append(triangle)
            avg_curv = input_curvatures[face].mean()
            colors.append(avg_curv)
        
        colors = np.array(colors)
        if len(colors) > 0 and colors.max() > colors.min():
            colors = (colors - colors.min()) / (colors.max() - colors.min())
        
        mesh_collection = Poly3DCollection(triangles_3d, alpha=0.4, 
                                          edgecolor='darkblue', linewidths=0.3)
        mesh_collection.set_array(colors)
        mesh_collection.set_cmap('viridis')
        ax1.add_collection3d(mesh_collection)
        
        # Add points colored by curvature
        scatter = ax1.scatter(input_mesh.vertices[:, 0],
                             input_mesh.vertices[:, 1],
                             input_mesh.vertices[:, 2],
                             c=input_curvatures, cmap='viridis',
                             s=3, alpha=0.8)
        
        if iteration == 0:
            title = f'Iter {iteration}: Input\n({n_input_pts} pts, {len(input_mesh.faces)} triangles)'
        else:
            title = f'Iter {iteration}: Input\n({n_input_pts} pts, {len(input_mesh.faces)} triangles)'
        
        ax1.set_title(title, fontsize=11, fontweight='bold')
        ax1.set_xlabel('X', fontsize=9)
        ax1.set_ylabel('Y', fontsize=9)
        ax1.set_zlabel('Z', fontsize=9)
        set_axes_equal_3d(ax1, input_mesh.vertices)
        ax1.view_init(elev=20, azim=45)
        
        # Colorbar
        if row_idx == 0:
            cbar = plt.colorbar(scatter, ax=ax1, shrink=0.5, pad=0.1)
            cbar.set_label('Curvature', fontsize=9)
        
        # ========================================
        # COLUMN 2: Reduced Points
        # ========================================
        ax2 = fig.add_subplot(n_rows, 3, row_idx*3 + 2, projection='3d')
        
        # Draw reduced points (red)
        ax2.scatter(reduced_points[:, 0],
                   reduced_points[:, 1],
                   reduced_points[:, 2],
                   c='red', s=15, alpha=0.8, 
                   edgecolors='darkred', linewidths=0.5)
        
        title = f'Reduced\n({n_reduced_pts} pts, {reduction_pct:.1f}%)'
        ax2.set_title(title, fontsize=11, fontweight='bold')
        ax2.set_xlabel('X', fontsize=9)
        ax2.set_ylabel('Y', fontsize=9)
        ax2.set_zlabel('Z', fontsize=9)
        set_axes_equal_3d(ax2, reduced_points)
        ax2.view_init(elev=20, azim=45)
        
        # ========================================
        # COLUMN 3: Reconstructed Mesh
        # ========================================
        ax3 = fig.add_subplot(n_rows, 3, row_idx*3 + 3, projection='3d')
        
        # Draw reconstructed mesh
        if reconstructed_mesh is not None:
            triangles_3d_recon = []
            
            for face in reconstructed_mesh.faces:
                triangle = reconstructed_mesh.vertices[face]
                triangles_3d_recon.append(triangle)
            
            mesh_collection_recon = Poly3DCollection(triangles_3d_recon, alpha=0.4,
                                                    facecolor='lightblue',
                                                    edgecolor='blue', linewidths=0.3)
            ax3.add_collection3d(mesh_collection_recon)
            
            # Add points
            ax3.scatter(reconstructed_mesh.vertices[:, 0],
                       reconstructed_mesh.vertices[:, 1],
                       reconstructed_mesh.vertices[:, 2],
                       c='blue', s=2, alpha=0.6)
            
            n_triangles = len(reconstructed_mesh.faces)
            set_axes_equal_3d(ax3, reconstructed_mesh.vertices)
        else:
            n_triangles = 0
        
        title = f'Reconstructed\n({n_reconstructed_pts} pts, {n_triangles} triangles)'
        ax3.set_title(title, fontsize=11, fontweight='bold')
        ax3.set_xlabel('X', fontsize=9)
        ax3.set_ylabel('Y', fontsize=9)
        ax3.set_zlabel('Z', fontsize=9)
        ax3.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig


# ============================================================
# MODIFIED PIPELINE WITH COMPREHENSIVE VISUALIZATION
# ============================================================

def hybrid_crust_pipeline_with_visualization(initial_points, n_iterations=3):
    """
    Complete pipeline with comprehensive visualization
    """
    
    print("="*70)
    print("HYBRID CRUST + CURVATURE-AWARE RECONSTRUCTION PIPELINE")
    print("="*70)
    
    iteration_data = []
    
    # ========================================
    # PHASE 1: Initial Crust Reconstruction
    # ========================================
    print("\n" + "="*70)
    print("PHASE 1: INITIAL SURFACE RECONSTRUCTION (CRUST)")
    print("="*70)
    
    initial_mesh = alpha_shape_surface(initial_points, alpha=None)
    print(f"  Initial mesh: {initial_mesh.n_vertices} vertices, {initial_mesh.n_faces} faces")
    
    # Compute mesh-based curvature
    initial_curvatures = compute_mesh_curvature(initial_mesh)
    
    # Store iteration 0 data
    iteration_data.append({
        'iteration': 0,
        'input_mesh': initial_mesh,
        'input_curvatures': initial_curvatures,
        'reduced_points': initial_mesh.vertices,  # Not reduced yet
        'reduced_curvatures': initial_curvatures,
        'reconstructed_mesh': initial_mesh,  # Same as input for iter 0
        'n_input_pts': len(initial_points),
        'n_reduced_pts': len(initial_points),
        'n_reconstructed_pts': len(initial_points),
        'reduction_pct': 100.0
    })
    
    # ========================================
    # PHASE 2: Iterative Refinement
    # ========================================
    print("\n" + "="*70)
    print("PHASE 2: ITERATIVE CURVATURE-AWARE REFINEMENT")
    print("="*70)
    
    current_points = initial_mesh.vertices.copy()
    current_curvatures = initial_curvatures.copy()
    
    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─'*70}")
        print(f"ITERATION {iteration}/{n_iterations}")
        print(f"{'─'*70}")
        
        # Create input mesh for this iteration (from previous augmented points)
        if iteration > 1:
            input_mesh = alpha_shape_surface(current_points, alpha=None)
            current_curvatures = compute_mesh_curvature(input_mesh)
        else:
            input_mesh = initial_mesh
        
        # Step 1: Reduce
        print(f"  Step 1: Curvature-aware reduction")
        print(f"    Input: {len(current_points)} points")
        
        reduced_points, reduced_curvatures = resample_points_curvature(
            current_points, current_curvatures,
            flat_keep=0.25, mid_keep=0.70, high_keep=1.0, seed=iteration
        )
        
        reduction_pct = (len(reduced_points) / len(current_points)) * 100
        print(f"    Reduced: {len(current_points)} → {len(reduced_points)} ({reduction_pct:.1f}%)")
        
        # Step 2: Delaunay triangulation
        print(f"  Step 2: Delaunay triangulation")
        triangles = delaunay_surface_triangles(reduced_points)
        print(f"    Extracted {len(triangles)} triangles")
        
        # Step 3: Polynomial fitting
        print(f"  Step 3: Fitting polynomial patches")
        patches = fit_polynomial_patches(reduced_points, triangles)
        print(f"    Fitted {len(patches)} patches")
        
        # Step 4: Sample patches
        print(f"  Step 4: Sampling polynomial patches")
        target_size = len(initial_points)
        augmented_points = sample_polynomial_patches(patches, target_size)
        print(f"    Generated {len(augmented_points)} points")
        
        # Create reconstructed mesh
        reconstructed_mesh = alpha_shape_surface(augmented_points, alpha=None)
        print(f"    Reconstructed mesh: {reconstructed_mesh.n_faces} triangles")
        
        # Store iteration data
        iteration_data.append({
            'iteration': iteration,
            'input_mesh': input_mesh,
            'input_curvatures': current_curvatures,
            'reduced_points': reduced_points,
            'reduced_curvatures': reduced_curvatures,
            'reconstructed_mesh': reconstructed_mesh,
            'n_input_pts': len(current_points),
            'n_reduced_pts': len(reduced_points),
            'n_reconstructed_pts': len(augmented_points),
            'reduction_pct': reduction_pct
        })
        
        # Prepare for next iteration
        current_points = augmented_points
        # Compute curvature for next iteration (will be done at start of next loop)
    
    # ========================================
    # CREATE COMPREHENSIVE VISUALIZATION
    # ========================================
    print("\n" + "="*70)
    print("CREATING COMPREHENSIVE VISUALIZATION")
    print("="*70)
    
    fig = create_comprehensive_visualization(iteration_data, n_iterations)
    
    plt.savefig('reconstruction_results_hybrid.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: reconstruction_results_hybrid.png")
    
    plt.show()
    
    # ========================================
    # PHASE 3: Final Crust (Optional)
    # ========================================
    print("\n" + "="*70)
    print("PHASE 3: FINAL CRUST RECONSTRUCTION")
    print("="*70)
    
    final_mesh = alpha_shape_surface(current_points, alpha=None)
    print(f"  Final mesh: {final_mesh.n_vertices} vertices, {final_mesh.n_faces} faces")
    
    # Visualize final mesh
    fig_final = plt.figure(figsize=(12, 10))
    ax = fig_final.add_subplot(111, projection='3d')
    
    triangles_3d = []
    for face in final_mesh.faces:
        triangle = final_mesh.vertices[face]
        triangles_3d.append(triangle)
    
    mesh_collection = Poly3DCollection(triangles_3d, alpha=0.7,
                                      facecolor='lightblue',
                                      edgecolor='darkblue', linewidths=0.5)
    ax.add_collection3d(mesh_collection)
    
    ax.set_title('Final High-Quality Reconstruction (Crust)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal_3d(ax, final_mesh.vertices)
    ax.view_init(elev=20, azim=45)
    
    plt.savefig('final_crust_mesh.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: final_crust_mesh.png")
    plt.show()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    
    return iteration_data, final_mesh


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution"""
    
    # Generate test surface
    print("Generating test surface...")
    initial_points = generate_ellipsoid(n_points=1200, noise=0.02)
    print(f"Generated ellipsoid with {len(initial_points)} points\n")
    
    # Run complete pipeline with visualization
    iteration_data, final_mesh = hybrid_crust_pipeline_with_visualization(
        initial_points,
        n_iterations=3
    )
    
    print("\n" + "="*70)
    print("FILES GENERATED:")
    print("="*70)
    print("  • reconstruction_results_hybrid.png - Complete pipeline visualization")
    print("  • final_crust_mesh.png - Final high-quality mesh")
    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()




