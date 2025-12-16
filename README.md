#  Surface Reconstruction Pipeline with Curvature-Aware Reduction

**Authors:** Karla Martinez & Terry Zhang  
**Course:** Computational Geometry Capstone  
**Date:** December 2025

A Python implementation of iterative surface reconstruction using Crust algorithm with intelligent curvature-aware point reduction and r-sample validation.

---

## Overview

This pipeline performs iterative surface reconstruction from point clouds:

1. **Initial Reconstruction:** Converts point cloud to triangle mesh using Crust or alpha shapes
2. **Curvature Analysis:** Computes curvature on the reconstructed mesh
3. **Intelligent Reduction:** Removes more points from flat regions, keeps high-curvature areas
4. **Validation:** Uses density gate and optional r-sample criterion to verify sampling adequacy
5. **Iteration:** Repeats for multiple refinement cycles

### Key Innovation

**Curvature-aware reduction** preserves geometric features better than uniform sampling by:
- Keeping 100% of points in high-curvature regions (edges, corners, features)
- Removing up to 40% of points in flat regions
- Adapting reduction aggressiveness based on sampling quality

---

## Features

- **Multiple Shape Generators:** Synthetic shapes (torus, cone, sphere, cube) for testing
- **Real Data Support:** Load mesh files (STL, OBJ, PLY) and protein structures (PDB)
- **Two Reconstruction Methods:** Crust (smooth) or Alpha shapes (detail-preserving)
- **Sampling Validation:** Density gate (kNN uniformity) + optional r-sample criterion
- **Protein Cartoon Visualization:** PDB backbone extraction for structural biology

---

## Installation

### Required Dependencies

```bash
# Core dependencies
pip install numpy matplotlib scipy scikit-learn

# For mesh file loading (STL/OBJ/PLY)
pip install trimesh

# For protein structures (PDB)
pip install biopython
```

### Verify Installation

```python
python crust_pipeline_with_mesh.py
# Should display the shape menu
```

---

## Quick Start

### Basic Usage

```bash
python crust_pipeline_with_mesh.py
```

### Simple Example (Torus)

```
Shape (1-12): 4                  # Torus (Uniform)
Initial points: 20000            # Default
Density gate k: 10               # Default
Use r-sample criterion? y        # Yes
  r value: 1.0                   # For torus
  Use Voronoi poles? n           # k-NN proxy is fine
Iterations: 3                    # Standard
```

**Result:** Clean torus reconstruction with hole preserved through all iterations.

---

## Shape Options

### Synthetic Shapes (Built-in Generators)

| # | Shape | Points | Non-Convex | Best For |
|---|-------|--------|------------|----------|
| 1 | Ellipsoid | 4000 | No | Testing basic reconstruction |
| 2 | Sphere | 4000 | No | Sanity check |
| 3 | Torus (Golden Ratio) | 20000 | Yes | Testing topology preservation |
| 4 | **Torus (Uniform)** | 20000 | Yes | Main demo|
| 5 | Rounded Box | 4000 | No | Testing sharp features |
| 6 | Cone | 4000 | No | Testing curvature gradient |
| 7 | DNA Helix | 15000 | Yes | Complex helical structure |
| 8 | Cube | 4000 | No | Sharp edges and corners |

### Real Data Sources

| # | Type | Input | Use Case |
|---|------|-------|----------|
| 9 | **Mesh File** | .stl/.obj/.ply | Any 3D model |
| 10 | **Protein Cartoon** | .pdb | Backbone ribbon |
| 11 | **Protein Surface** | .pdb | Molecular envelope |

---

## Parameters Guide

### Initial Points (N)

**What it does:** Number of points to sample from the shape

**Recommendations:**
- Simple shapes (sphere, cone): 4,000 - 8,000
- Complex shapes (torus, helix): 15,000 - 25,000
- Real meshes: Match mesh complexity
- Proteins: 2,000 - 10,000

**Trade-off:** More points = better detail but slower computation

---

### Density Gate (k)

**What it does:** Uses k-nearest neighbors to check sampling uniformity

**Default:** k=10 (good for most cases)

**How it works:**
- Computes median and variance of k-NN distances
- **PASS:** Uniform sampling → use Crust
- **WARN:** Some non-uniformity → conservative reduction
- **FAIL:** Poor sampling → fallback to alpha shapes

**When to change:**
- Increase k (15-20) for denser point clouds
- Decrease k (5-8) for sparser point clouds

---

### r-Sample Criterion

**What it does:** Validates sampling adequacy using local feature size (LFS)

**Formula:** 
```
For each point: nearest_neighbor_distance ≤ r × local_feature_size
```

**Recommended r values:**
- **r=0.5:** Paper's recommendation (strict)
- **r=1.0:** Practical for most shapes
- **r=1.5:** Lenient (if having issues)

**r-Sample by Shape:**

| Shape | Recommended r | Why |
|-------|---------------|-----|
| Torus (Golden Ratio) | 1.0 | Quasi-random pattern |
| Cone (uniform) | 0.5 | Simple geometry |
| Protein | 1.0 | Complex but thick features |
| Real meshes | 1.0 | Safe default |

**Voronoi Poles vs k-NN:**
- **k-NN (default):** Fast, good approximation
- **Voronoi poles:** More accurate but can have outliers, slower

**Recommendation:** Use k-NN unless you need research-grade accuracy

---

### Iterations

**What it does:** Number of reduction-reconstruction cycles

**Recommended:** 3 iterations

**What happens each iteration:**
1. Test current sampling quality
2. Reduce points (curvature-aware)
3. Test reduced sampling quality
4. Reconstruct surface
5. Compute curvature for next iteration

**More iterations:**
- More aggressive reduction
- Risk of quality degradation
- May lose fine features

**Typical results:**
- Iteration 0: 100% points (initial)
- Iteration 1: ~80-90% points
- Iteration 2: ~70-80% of original
- Iteration 3: ~60-75% of original

---

## Usage Examples

### Example 1: Protein Structure

```bash
Shape: 10                        # Protein Cartoon
Path to PDB: 1A3M.pdb
Points per residue: 20
r value: 1.0
Iterations: 3
```

**Expected:** Clean backbone ribbon showing secondary structure

---

## Understanding the Output

### Terminal Output

```
CRUST + REDUCTION PIPELINE - TORUS_(UNIFORM)
NO MLS (no point regeneration)
Using FULL CRUST + density gate (kNN spacing uniformity)
+ r-sample criterion (r=0.5, from Alexa et al. 2003)
══════════════════════════════════════════════════════════════════════

PHASE 1: INITIAL RECONSTRUCTION
══════════════════════════════════════════════════════════════════════
    density gate: PASS | median(kNN)=0.0486, p95=0.0621, cv=0.16
    r-sample check: PASS | r=0.50, max_ratio=0.48, violations=0.0%
      LFS range: 0.0234 to 0.0892 (median=0.0456)
  Running Crust algorithm...
    ✓ Crust complete (4.3s)
  Computing curvature...

ITERATION 1/3
──────────────────────────────────────────────────────────────────────
    density gate: PASS
  [CURVATURE-AWARE] Step 1: Reduction
    20000 → 18477 (92.4%)
    density gate: PASS | median(kNN)=0.0504
    r-sample check: PASS | r=0.50, max_ratio=0.49
  [FINAL] Step 2: Reconstructing surface
    ✓ 34650 triangles
```

**Key metrics:**
- **density gate status:** PASS/WARN/FAIL
- **r-sample violations:** Should be 0% for PASS
- **Reduction percentage:** How many points kept
- **Triangle count:** Mesh complexity

---

### Visualization

**Output file:** `reconstruction_[shape]_NO_MLS_DENSITY_GATE.png`

**Layout:** 4 iterations × 3 columns

**Columns:**
- **GREEN (left):** Input points with curvature colormap
  - Yellow/bright = high curvature
  - Blue/dark = low curvature
- **RED (middle):** Reduced points (what algorithm keeps)
- **BLUE (right):** Reconstructed mesh

---


### Parameter Selection

- Use defaults (press Enter for all prompts)
- Set r=1.0 (safe for all shapes)
- Use 3 iterations
- Enable r-sample criterion


---

### Performance

- Crust is O(n²) in worst case, but O(n log n) in most cases
- Reduce initial points
- Use fewer iterations
- Consider alpha shapes (faster)

---

##  Known Limitations


### 1. r-Sample Criterion Strictness

**Issue:** Golden Ratio Torus fails r=0.5 despite good quality

**Why:** Quasi-random sampling pattern has inherent non-uniformity (max_ratio ≈ 0.94)

**Solution:** Use r=1.0 for Golden Ratio Torus

**Mathematical validity:** Both r=0.5 and r=1.0 are valid; paper says r=0.5 "generally suffices" not "always required"

---

### 2. Protein Surface vs Cartoon

**Issue:** Protein surface (option 11) gives blob, cartoon (option 10) works well

**Why:**
- Surface extracts all atoms → thin molecular envelope
- Cartoon adds tube radius → volumetric ribbon (reconstructable)

**Solution:** Use Protein Cartoon (option 10) for best results

---

### 3. Computational Complexity

**Issue:** Crust reconstruction slows significantly with >30,000 points

**Why:** Voronoi diagram computation is expensive

**Solution:**
- Limit initial points to 20,000-25,000
- Use alpha shapes for very large point clouds
- Sample from mesh instead of using all vertices

---

## References

### Papers

1. **Alexa, M., Behr, J., Cohen-Or, D., Fleishman, S., Levin, D., & Silva, C. T. (2003)**  
   "Computing and rendering point set surfaces"  
   *IEEE Transactions on Visualization and Computer Graphics, 9*(1), 3-15.  
   - Source of r-sample criterion and MLS projection

2. **Amenta, N., & Bern, M. (1998)**  
   "Surface reconstruction by Voronoi filtering"  
   *Discrete & Computational Geometry, 22*(4), 481-504.  
   - Original Crust algorithm

3. **Amenta, N., Choi, S., & Kolluri, R. K. (2001)**  
   "The power crust"  
   *Proceedings of the sixth ACM symposium on Solid modeling and applications*  
   - Power crust extension

### Datasets

4. **Stanford 3D Scanning Repository**  
   https://graphics.stanford.edu/data/3Dscanrep/  
   - Source of Stanford Bunny and other test models

5. **RCSB Protein Data Bank**  
   https://www.rcsb.org/  
   - Protein structures (PDB files)

### Tools & Libraries

6. **Trimesh**  
   https://trimsh.org/  
   - Mesh loading and processing

7. **BioPython**  
   https://biopython.org/  
   - PDB file parsing

---


## License

Academic/Educational Use  
Computational Geometry Capstone Project  
Macalester College, Fall 2025

---

## Acknowledgments

- **Professor** Lori Ziegelmeier
- **Alexa et al.** for r-sample criterion
- **Amenta & Bern** for Crust algorithm  
- **Stanford 3D Scanning Repository** for test models
