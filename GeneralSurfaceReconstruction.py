import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay, cKDTree

class GeneralSurfaceReconstruction:
    def __init__(self, original_cloud, k_neighbors=15):
        self.original_cloud = original_cloud
        self.k_neighbors = k_neighbors
    
    def full_pipeline(self, n_iterations=3, visualize=True):
        current_cloud = self.original_cloud.copy()
        metrics_history = []
        
        if visualize:
            fig = plt.figure(figsize=(18, 5 * n_iterations))  # Made wider for better viewing
        
        for iteration in range(n_iterations):
            print(f"\n{'='*50}")
            print(f"ITERATION {iteration}")
            print(f"{'='*50}")
            
            # STEP 1: Reduce points
            print(f"Starting points: {len(current_cloud)}")
            reduced_cloud, curvatures = self.curvature_aware_reduction(current_cloud)
            print(f"After reduction: {len(reduced_cloud)} points ({len(reduced_cloud)/len(current_cloud)*100:.1f}%)")
            
            # STEP 2: Reconstruct
            print("Reconstructing surface...")
            reconstructed_surface = self.spline_reconstruction(reduced_cloud)
            
            # STEP 3: Augment
            print("Generating augmented sampling...")
            target = len(self.original_cloud)
            augmented_cloud = self.sample_spline_surface(reconstructed_surface, target_size=target)
            print(f"Generated {len(augmented_cloud)} augmented points")
            
            # STEP 4: Evaluate
            metrics = self.evaluate_reconstruction(
                original=self.original_cloud,
                reduced=reduced_cloud,
                augmented=augmented_cloud
            )
            metrics_history.append(metrics)
            
            print(f"\nMetrics:")
            print(f"  Reduction: {len(current_cloud)} → {len(reduced_cloud)} ({len(reduced_cloud)/len(current_cloud)*100:.1f}%)")
            print(f"  RMS error: {metrics['rms_error_augmented']:.4f}")
            print(f"  Max error: {metrics['max_error_augmented']:.4f}")
            
            # STEP 5: Visualize with triangulation
            if visualize:
                self.visualize_iteration_with_mesh(
                    fig, iteration, n_iterations,
                    current_cloud, reduced_cloud, augmented_cloud, curvatures
                )
            
            current_cloud = augmented_cloud
        
        if visualize:
            plt.tight_layout()
            plt.savefig('reconstruction_results.png', dpi=150, bbox_inches='tight')
            plt.show()
        
        return metrics_history
    
    def extract_surface_triangles(self, point_cloud):
        """
        Extract surface triangles from point cloud using Delaunay triangulation
        Returns triangles as indices into the point cloud
        """
        try:
            delaunay = Delaunay(point_cloud)
            
            # Find boundary faces (faces that belong to only one tetrahedron)
            face_counts = {}
            
            for simplex in delaunay.simplices:
                # Each tetrahedron has 4 triangular faces
                faces = [
                    tuple(sorted([simplex[0], simplex[1], simplex[2]])),
                    tuple(sorted([simplex[0], simplex[1], simplex[3]])),
                    tuple(sorted([simplex[0], simplex[2], simplex[3]])),
                    tuple(sorted([simplex[1], simplex[2], simplex[3]]))
                ]
                
                for face in faces:
                    if face in face_counts:
                        face_counts[face] += 1
                    else:
                        face_counts[face] = 1
            
            # Surface faces appear exactly once
            surface_triangles = [face for face, count in face_counts.items() if count == 1]
            
            return np.array(surface_triangles)
        except:
            # If Delaunay fails, return empty array
            return np.array([])
    
    def visualize_iteration_with_mesh(self, fig, iteration, n_iterations, 
                                     original, reduced, augmented, curvatures):
        """Visualize with triangulated mesh surfaces"""
        row = iteration
        
        # Plot 1: Original with curvature coloring and mesh
        ax1 = fig.add_subplot(n_iterations, 3, row*3 + 1, projection='3d')
        
        # Extract and plot mesh
        print(f"  Computing triangulation for input cloud...")
        triangles = self.extract_surface_triangles(original)
        if len(triangles) > 0:
            # Create mesh
            tri_collection = []
            for tri_idx in triangles:
                triangle = original[list(tri_idx)]
                tri_collection.append(triangle)
            
            mesh = Poly3DCollection(tri_collection, alpha=0.3, facecolor='cyan', 
                                   edgecolor='darkblue', linewidths=0.1)
            ax1.add_collection3d(mesh)
        
        # Add colored points on top
        scatter = ax1.scatter(original[:, 0], original[:, 1], original[:, 2],
                            c=curvatures, cmap='viridis', s=2, alpha=0.8)
        ax1.set_title(f'Iter {iteration}: Input\n({len(original)} pts, {len(triangles)} triangles)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        plt.colorbar(scatter, ax=ax1, label='Curvature', shrink=0.5, pad=0.1)
        
        # Set equal aspect ratio
        self.set_axes_equal(ax1, original)
        
        # Plot 2: Reduced cloud (no mesh, just points)
        ax2 = fig.add_subplot(n_iterations, 3, row*3 + 2, projection='3d')
        ax2.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                   c='red', s=8, alpha=0.8, edgecolors='darkred', linewidths=0.5)
        ax2.set_title(f'Reduced\n({len(reduced)} pts, {len(reduced)/len(original)*100:.1f}%)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        self.set_axes_equal(ax2, reduced)
        
        # Plot 3: Augmented reconstruction with mesh
        ax3 = fig.add_subplot(n_iterations, 3, row*3 + 3, projection='3d')
        
        # Extract and plot mesh
        print(f"  Computing triangulation for reconstructed surface...")
        triangles_aug = self.extract_surface_triangles(augmented)
        if len(triangles_aug) > 0:
            tri_collection_aug = []
            for tri_idx in triangles_aug:
                triangle = augmented[list(tri_idx)]
                tri_collection_aug.append(triangle)
            
            mesh_aug = Poly3DCollection(tri_collection_aug, alpha=0.4, facecolor='lightblue',
                                       edgecolor='blue', linewidths=0.1)
            ax3.add_collection3d(mesh_aug)
        
        # Add points on top
        ax3.scatter(augmented[:, 0], augmented[:, 1], augmented[:, 2],
                   c='blue', s=1, alpha=0.6)
        ax3.set_title(f'Reconstructed\n({len(augmented)} pts, {len(triangles_aug)} triangles)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        self.set_axes_equal(ax3, augmented)
    
    def set_axes_equal(self, ax, points):
        """Set equal aspect ratio for 3D plot"""
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
    
    def curvature_aware_reduction(self, point_cloud):
        """PCA-based reduction"""
        curvatures = []
        
        for point in point_cloud:
            distances = np.linalg.norm(point_cloud - point, axis=1)
            neighbor_indices = np.argsort(distances)[1:self.k_neighbors+1]
            neighbors = point_cloud[neighbor_indices]
            
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered / (len(neighbors) - 1)
            eigenvalues = np.linalg.eigvalsh(cov)
            
            lambda_0, lambda_1, lambda_2 = np.sort(eigenvalues)
            kappa = lambda_0 / (lambda_0 + lambda_1 + lambda_2 + 1e-10)
            curvatures.append(kappa)
        
        curvatures = np.array(curvatures)
        t1, t2 = np.percentile(curvatures, [40, 75])
        
        reduced = []
        for point, kappa in zip(point_cloud, curvatures):
            if kappa < t1:  # Flat
                if np.random.rand() < 0.25:
                    reduced.append(point)
            elif kappa < t2:  # Medium
                if np.random.rand() < 0.70:
                    reduced.append(point)
            else:  # High curvature
                reduced.append(point)
        
        return np.array(reduced), curvatures
    
    def spline_reconstruction(self, reduced_cloud):
        """Build spline surface"""
        delaunay = Delaunay(reduced_cloud)
        
        patches = []
        for simplex in delaunay.simplices:
            tri_verts = reduced_cloud[simplex]
            patch = self.fit_local_patch(tri_verts, reduced_cloud)
            patches.append({
                'simplex': simplex,
                'vertices': tri_verts,
                'patch': patch
            })
        
        return {'delaunay': delaunay, 'patches': patches, 'points': reduced_cloud}
    
    def fit_local_patch(self, tri_verts, all_points):
        """Fit polynomial patch"""
        tri_center = tri_verts.mean(axis=0)
        radius = 2.5 * np.max(np.linalg.norm(tri_verts - tri_center, axis=1))
        
        distances = np.linalg.norm(all_points - tri_center, axis=1)
        nearby_mask = distances < radius
        nearby_points = all_points[nearby_mask]
        
        if len(nearby_points) < 6:
            nearby_points = all_points
        
        # Local coordinate frame
        v1 = tri_verts[1] - tri_verts[0]
        v2 = tri_verts[2] - tri_verts[0]
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-10:
            normal /= normal_norm
        else:
            normal = np.array([0, 0, 1])
        
        tangent1 = v1 / (np.linalg.norm(v1) + 1e-10)
        tangent2 = np.cross(normal, tangent1)
        tangent2 /= (np.linalg.norm(tangent2) + 1e-10)
        
        # Transform to local coords
        relative = nearby_points - tri_center
        x = relative @ tangent1
        y = relative @ tangent2
        z = relative @ normal
        
        # Fit quadratic
        features = np.column_stack([
            np.ones(len(x)),
            x, y,
            x**2, x*y, y**2
        ])
        
        coeffs, _, _, _ = np.linalg.lstsq(features, z, rcond=None)
        
        return {
            'coeffs': coeffs,
            'origin': tri_center,
            'tangent1': tangent1,
            'tangent2': tangent2,
            'normal': normal,
            'tri_verts': tri_verts
        }
    
    def sample_spline_surface(self, reconstructed_surface, target_size):
        """Sample spline surface"""
        patches = reconstructed_surface['patches']
        
        # Calculate areas
        total_area = 0
        triangle_areas = []
        for patch_info in patches:
            tri_verts = patch_info['patch']['tri_verts']
            v1 = tri_verts[1] - tri_verts[0]
            v2 = tri_verts[2] - tri_verts[0]
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            triangle_areas.append(area)
            total_area += area
        
        # Generate points
        augmented = []
        
        for patch_info, area in zip(patches, triangle_areas):
            fraction = area / total_area
            n_samples = max(1, int(target_size * fraction) + 1)
            
            patch = patch_info['patch']
            tri_verts = patch['tri_verts']
            
            for _ in range(min(n_samples, target_size - len(augmented))):
                r1, r2 = np.random.rand(2)
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                
                local_point = r1 * tri_verts[0] + r2 * tri_verts[1] + (1-r1-r2) * tri_verts[2]
                relative = local_point - patch['origin']
                x = relative @ patch['tangent1']
                y = relative @ patch['tangent2']
                
                coeffs = patch['coeffs']
                z = coeffs[0] + coeffs[1]*x + coeffs[2]*y + \
                    coeffs[3]*x**2 + coeffs[4]*x*y + coeffs[5]*y**2
                
                point_3d = patch['origin'] + \
                          x * patch['tangent1'] + \
                          y * patch['tangent2'] + \
                          z * patch['normal']
                
                augmented.append(point_3d)
        
        return np.array(augmented)
    
    def evaluate_reconstruction(self, original, reduced, augmented):
        """Evaluate quality"""
        tree = cKDTree(original)
        distances_aug, _ = tree.query(augmented)
        distances_red, _ = tree.query(reduced)
        
        return {
            'reduction_ratio': len(reduced) / len(original),
            'rms_error_reduced': np.sqrt(np.mean(distances_red**2)),
            'rms_error_augmented': np.sqrt(np.mean(distances_aug**2)),
            'max_error_augmented': np.max(distances_aug),
            'mean_error_augmented': np.mean(distances_aug),
            'n_original': len(original),
            'n_reduced': len(reduced),
            'n_augmented': len(augmented)
        }


# ============================================================
# SURFACE GENERATORS (keep all the same as before)
# ============================================================

def generate_ellipsoid(n_points=1500, a=1.5, b=1.0, c=0.7, noise=0.02):
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


def generate_egg_shape(n_points=1500, noise=0.02):
    """Generate egg-shaped surface"""
    phi = np.pi * (3. - np.sqrt(5.))
    
    points = []
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        radius_at_y = np.sqrt(1 - y * y)
        theta = phi * i
        
        scale_factor = 1.0 + 0.3 * y
        
        x = 1.5 * np.cos(theta) * radius_at_y * scale_factor
        z = 0.8 * np.sin(theta) * radius_at_y * scale_factor
        y_egg = 1.2 * y
        
        point = np.array([x, y_egg, z])
        point += np.random.normal(0, noise, 3)
        points.append(point)
    
    return np.array(points)


def generate_rounded_box(n_points=1500, a=1.2, b=0.8, c=0.6, r=0.3, noise=0.02):
    """Generate rounded box"""
    points = []
    
    while len(points) < n_points:
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        
        x_dir = np.sin(phi) * np.cos(theta)
        y_dir = np.sin(phi) * np.sin(theta)
        z_dir = np.cos(phi)
        
        abs_x = abs(x_dir)
        abs_y = abs(y_dir)
        abs_z = abs(z_dir)
        
        if abs_x >= abs_y and abs_x >= abs_z:
            scale = (a - r) / abs_x
            x = scale * x_dir + r * np.sign(x_dir)
            y = scale * y_dir
            z = scale * z_dir
            y = np.clip(y, -(b-r), (b-r))
            z = np.clip(z, -(c-r), (c-r))
        elif abs_y >= abs_x and abs_y >= abs_z:
            scale = (b - r) / abs_y
            x = scale * x_dir
            y = scale * y_dir + r * np.sign(y_dir)
            z = scale * z_dir
            x = np.clip(x, -(a-r), (a-r))
            z = np.clip(z, -(c-r), (c-r))
        else:
            scale = (c - r) / abs_z
            x = scale * x_dir
            y = scale * y_dir
            z = scale * z_dir + r * np.sign(z_dir)
            x = np.clip(x, -(a-r), (a-r))
            y = np.clip(y, -(b-r), (b-r))
        
        dx = max(0, abs(x) - (a - r))
        dy = max(0, abs(y) - (b - r))
        dz = max(0, abs(z) - (c - r))
        d = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if d > 0:
            x = np.sign(x) * ((a - r) + r * dx / d)
            y = np.sign(y) * ((b - r) + r * dy / d)
            z = np.sign(z) * ((c - r) + r * dz / d)
        
        point = np.array([x, y, z])
        point += np.random.normal(0, noise, 3)
        points.append(point)
    
    return np.array(points)


def generate_capsule(n_points=1500, radius=0.6, height=1.5, noise=0.02):
    """Generate capsule shape"""
    points = []
    
    total_surface_area = 2 * np.pi * radius * height + 4 * np.pi * radius**2
    cylinder_area = 2 * np.pi * radius * height
    hemisphere_area = 2 * np.pi * radius**2
    
    n_cylinder = int(n_points * cylinder_area / total_surface_area)
    n_per_hemisphere = int(n_points * hemisphere_area / total_surface_area)
    
    for i in range(n_cylinder):
        theta = np.random.uniform(0, 2*np.pi)
        y = np.random.uniform(-height/2, height/2)
        
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        
        point = np.array([x, y, z])
        point += np.random.normal(0, noise, 3)
        points.append(point)
    
    for i in range(n_per_hemisphere):
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi/2)
        
        x = radius * np.sin(phi) * np.cos(theta)
        z = radius * np.sin(phi) * np.sin(theta)
        y = radius * np.cos(phi) + height/2
        
        point = np.array([x, y, z])
        point += np.random.normal(0, noise, 3)
        points.append(point)
    
    for i in range(n_per_hemisphere):
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(np.pi/2, np.pi)
        
        x = radius * np.sin(phi) * np.cos(theta)
        z = radius * np.sin(phi) * np.sin(theta)
        y = radius * np.cos(phi) - height/2
        
        point = np.array([x, y, z])
        point += np.random.normal(0, noise, 3)
        points.append(point)
    
    return np.array(points)


def generate_superellipsoid(n_points=1500, a=1.5, b=1.0, c=0.8, n1=1.5, n2=1.5, noise=0.02):
    """Generate superellipsoid"""
    phi = np.pi * (3. - np.sqrt(5.))
    
    points = []
    for i in range(n_points):
        v = -np.pi/2 + np.pi * (i / float(n_points))
        u = phi * i
        
        def signed_pow(val, exp):
            return np.sign(val) * (np.abs(val) ** exp)
        
        cos_v = np.cos(v)
        sin_v = np.sin(v)
        cos_u = np.cos(u)
        sin_u = np.sin(u)
        
        x = a * signed_pow(cos_v, 2.0/n1) * signed_pow(cos_u, 2.0/n2)
        y = b * signed_pow(cos_v, 2.0/n1) * signed_pow(sin_u, 2.0/n2)
        z = c * signed_pow(sin_v, 2.0/n1)
        
        point = np.array([x, y, z])
        point += np.random.normal(0, noise, 3)
        points.append(point)
    
    return np.array(points)


def generate_kidney_bean(n_points=1500, noise=0.02):
    """Generate kidney bean shape"""
    phi = np.pi * (3. - np.sqrt(5.))
    
    points = []
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        radius_at_y = np.sqrt(1 - y * y)
        theta = phi * i
        
        x = 1.5 * np.cos(theta) * radius_at_y
        z = 0.8 * np.sin(theta) * radius_at_y
        
        if x < 0 and abs(y) < 0.5:
            indent_factor = np.cos(np.pi * y) * np.cos(np.pi * (x + 0.75) / 1.5)
            indent = 0.3 * max(0, indent_factor)
            x += indent
        
        x *= (1 + 0.2 * np.sin(2 * np.pi * y))
        
        point = np.array([x, y, z])
        point += np.random.normal(0, noise, 3)
        points.append(point)
    
    return np.array(points)


def generate_bumpy_sphere(n_points=1500, radius=1.0, noise=0.02):
    """Generate bumpy sphere"""
    phi = np.pi * (3. - np.sqrt(5.))
    
    points = []
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        radius_at_y = np.sqrt(1 - y * y)
        theta = phi * i
        
        bump = 1 + 0.2 * np.sin(5 * theta) * np.cos(3 * np.arcsin(y))
        
        x = np.cos(theta) * radius_at_y * bump
        z = np.sin(theta) * radius_at_y * bump
        y_bumpy = y * bump
        
        point = np.array([x, y_bumpy, z]) * radius
        point += np.random.normal(0, noise, 3)
        points.append(point)
    
    return np.array(points)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*60)
    print("CURVATURE-AWARE SURFACE RECONSTRUCTION")
    print("="*60)
    
    print("\nChoose a test surface:")
    print("1. Ellipsoid (simple, uniform curvature)")
    print("2. Egg Shape (varying curvature, tapered)")
    print("3. Rounded Box (flat faces + curved edges) RECOMMENDED")
    print("4. Capsule (cylinder with spherical caps)")
    print("5. Superellipsoid (sharp edges, very interesting)")
    print("6. Kidney Bean (complex curves)")
    print("7. Bumpy Sphere (many small bumps)")
    
    choice = input("Enter choice (1-7) or press Enter for rounded box [3]: ").strip()
    
    if choice == '1':
        print("\nGenerating ellipsoid...")
        point_cloud = generate_ellipsoid(n_points=1200, noise=0.02)
    elif choice == '2':
        print("\nGenerating egg shape...")
        point_cloud = generate_egg_shape(n_points=1200, noise=0.02)
    elif choice == '4':
        print("\nGenerating capsule...")
        point_cloud = generate_capsule(n_points=1200, noise=0.02)
    elif choice == '5':
        print("\nGenerating superellipsoid...")
        point_cloud = generate_superellipsoid(n_points=1200, n1=1.5, n2=1.5, noise=0.02)
    elif choice == '6':
        print("\nGenerating kidney bean...")
        point_cloud = generate_kidney_bean(n_points=1200, noise=0.02)
    elif choice == '7':
        print("\nGenerating bumpy sphere...")
        point_cloud = generate_bumpy_sphere(n_points=1200, noise=0.02)
    else:
        print("\nGenerating rounded box...")
        point_cloud = generate_rounded_box(n_points=1200, noise=0.02)
    
    print(f"Generated {len(point_cloud)} points")
    
    # Run reconstruction
    reconstructor = GeneralSurfaceReconstruction(point_cloud, k_neighbors=15)
    metrics_history = reconstructor.full_pipeline(n_iterations=3, visualize=True)
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    final = metrics_history[-1]
    print(f"Original:   {final['n_original']} points")
    print(f"Reduced:    {final['n_reduced']} points ({final['reduction_ratio']:.1%})")
    print(f"RMS Error:  {final['rms_error_augmented']:.4f}")
    print(f"Max Error:  {final['max_error_augmented']:.4f}")


if __name__ == "__main__":
    main()


