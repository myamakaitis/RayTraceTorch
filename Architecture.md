# Project Blueprint: Differentiable 3D Ray Tracer (PyTorch)

## 1. Project Overview

* **Purpose:** A differentiable 3D ray tracing library for optical system design and optimization using PyTorch autograd.
* **Target Users:** Internal research, optical engineers, and future open-source contributors.
* **Core Dependencies:** `PyTorch` (primary engine), `NumPy` (utility), `PySide6` (GUI).

---

## 2. High-Level Architecture

* **Entry Point:** `__init__.py` (root)
* **Data Flow:**
  1. User defines ray sources via `Bundle` subclasses.
  2. User assembles a `Scene` or `SequentialScene` containing `Element` subclasses.
  3. The `Scene` iterates through ray-surface interactions (bounce loop or sequential pass).
  4. PyTorch tracks gradients from the loss (e.g., spot size) back to `Element` parameters (e.g., curvature, position).

---

## 3. Module Breakdown

### 3.1. `geom/` (Geometry Engine)

Defines spatial boundaries, intersection math, and optical shapes. All geometry classes inherit from `torch.nn.Module`.

* **`transform.py`**:
  * **Class `RayTransform(nn.Module)`**: Handles affine transformations using a rotation vector (Rodrigues/matrix-exponential encoding) and a translation vector.
    * **Parameters**: `trans` [3] (translation), `rot_vec` [3] (axis-angle rotation)
    * Optional gradient masks: `trans_mask`, `rot_mask` (registered buffers)
    * **Convention**: Translation is subtracted before rotation, so `trans` encodes the object origin in the parent (global) frame.
    * **Methods**:
      * `transform(rays)` / `transform_(pos, dir)`: Maps **Global → Local** (subtract translation, then rotate).
      * `invTransform(rays)` / `invTransform_(pos, dir)`: Maps **Local → Global** (rotate by R.T, then add translation).
      * `rot` (property): Computes 3×3 rotation matrix via `torch.linalg.matrix_exp`.
      * `paraxial()` / `paraxial_inv()`: Returns 5×5 paraxial transfer matrices.
  * **Class `RayTransformNoisy(RayTransform)`**: Adds per-ray Gaussian perturbations to both translation and rotation for tolerance analysis.
  * **Class `RayTransformBundle(RayTransform)`**: Variant used by `Bundle` sources; applies translation first (additive, not subtractive) so bundles can be placed in world space.

* **`primitives.py`**:
  * **Class `Surface(nn.Module)`**: Abstract base for infinite mathematical surfaces.
    * **Protocol**:
      * `intersectTest(rays)`: **Fast, detached.** Returns scalar $t$ for validity checks.
      * `forward(rays)`: **Differentiable.** Returns $(t$, `hit_point` [Global], `normal` [Global]).
    * **Implemented subclasses**:
      * `Plane`: Infinite flat surface ($z=0$ in local frame).
      * `Sphere`: Curvature-based surface; parameter `radius`.
      * `Quadric`: General conic section of revolution; parameters `c` (curvature), `k` (conic constant). Covers spheres, paraboloids, and hyperboloids.
      * `QuadricZY`: Cylindrical variant of `Quadric`; curvature along Y only, invariant in X.
      * `Cylinder`: Infinite tube ($x^2 + y^2 = R^2$); parameter `radius`.
      * `Cone`: Double cone along Z; parameter `slope` (dz/dr).

* **`bounded.py`**:
  * **Class `SurfaceBounded(Surface)`**: Base wrapper combining a mathematical surface with a finite boundary mask.
    * **Protocol**: `inBounds(local_pos)` — abstract boolean check.
    * `forward(rays)`: Calls parent intersection, then filters hits via `inBounds`.
  * **Implemented subclasses**:
    * `Disk(Plane, SurfaceBounded)`: Circular aperture; parameter `radius`.
    * `Rectangle(Plane, SurfaceBounded)`: Rectangular aperture; parameters `hx`, `hy`.
    * `Ellipse(Plane, SurfaceBounded)`: Elliptical aperture; parameters `r_major`, `r_minor`, `rot`.
    * `HalfSphere(Quadric, SurfaceBounded)`: Quadric clipped to a hemisphere (lens faces). `inBounds` checks that the Z-coordinate opposes the radius direction. Provides `sagittalZ(radius)`.
    * `HalfCyl(QuadricZY, SurfaceBounded)`: Cylindrical quadric clipped to a half-cylinder (cylindrical lens faces). Provides `sagittalZ(y_height)`.
    * `SingleCone(Cone, SurfaceBounded)`: Single nappe of a cone.

* **`shape.py`**:
  * **Class `Shape(nn.Module)`**: A 3D volume defined by a collection of `Surface` objects.
    * **Protocol**:
      * `intersectTest(rays)`: Returns `[N, K]` matrix of $t$ values for all $K$ surfaces.
      * `forward(rays, surf_idx)`: Returns differentiable intersection for a specific surface.
      * `inBounds(local_pos, surf_idx)`: Abstract volume check.
  * **Implemented subclasses**:
    * `CvxPolyhedron(Shape)`: Convex polyhedron from a list of `Plane` objects. `inBounds` checks all half-space inequalities.
    * `Box(CvxPolyhedron)`: Rectangular prism; auto-generates 6 planes from `length`, `width`, `height`.
    * `Box4Side(CvxPolyhedron)`: Rectangular prism with only 4 side planes (no front/back caps); used as cylindrical lens edges.

* **`spherics.py`**:
  * **Class `Spheric(Shape)`**: Abstract base for rotationally symmetric lens stacks.
    * `_make_surface(C, z_vertex)`: Factory creating a `HalfSphere` at a given vertex position.
    * `inBounds(local_pos, surf_idx)`: Checks optical faces (radial aperture) and cylindrical edges (axial thickness span).
  * **Implemented subclasses**:
    * `Singlet(Spheric)`: 2 optical surfaces + 1 cylindrical edge. Parameters: `C1`, `C2`, `D`, `T`.
    * `Doublet(Spheric)`: 3 optical surfaces + 2 cylindrical edges. Parameters: `C1`, `C2`, `C3`, `D`, `T1`, `T2`.
    * `Triplet(Spheric)`: 4 optical surfaces + 3 cylindrical edges. Parameters: `C1`–`C4`, `D`, `T1`–`T3`.

* **`cylindrics.py`**:
  * **Class `Cylindric(Shape)`**: Abstract base for cylindrical power lens stacks.
    * `_make_cyl_surface(C, z_vertex)`: Factory creating a `HalfCyl` surface.
    * `inBounds(local_pos, surf_idx)`: Checks optical faces (rectangular aperture) and box edges (axial bounds).
  * **Implemented subclasses**:
    * `CylSinglet(Cylindric)`: 2 cylindrical optical surfaces + 4-sided box edge. Parameters: `C1`, `C2`, `width`, `height`, `T`.

* **`complex.py`**:
  * **Class `Aspheric(Surface)`**: Placeholder stub — not yet implemented.

---

### 3.2. `phys/` (Physics Engine)

* **`std.py`**:
  * **Class `SurfaceFunction(nn.Module)`**: Abstract base functor for optical physics.
    * **Protocol**: `forward(local_intersect, ray_dir, normal, **kwargs)` → `(new_dir, intensity_mod)`.
  * **Implemented subclasses**:
    * `Linear`: ABCD-matrix style refraction; parameters `Cx`, `Cy`, `Dx`, `Dy`. Used by ideal elements.
    * `Reflect`: Perfect specular reflection (vector reflection formula).
    * `RefractSnell`: Differentiable Snell's Law with TIR handling; parameters `ior_in`, `ior_out`.
    * `RefractFresnel`: Stochastic Fresnel refraction/reflection; parameters `ior_in`, `ior_out`.
    * `Transmit`: Pass-through (no direction/intensity change).
    * `Block`: Absorber — zeros direction and intensity.

* **`filter.py`**:
  * `ApertureFilter(Transmit)`: Applies a caller-supplied `inBounds` callable to zero out out-of-aperture rays.
  * `Fuzzy(Transmit)`: Applies a caller-supplied intensity function to modulate ray intensity.

---

### 3.3. `elements/` (Optical Components)

* **`parent.py`**:
  * **Class `Element(nn.Module)`**: Base class for all optical elements.
    * **Structure**: `shape` (a `Shape`), `surface_functions` (a `ModuleList` of `SurfaceFunction`).
    * **Logic**: `forward(rays, surf_idx)` calls `shape.forward` for geometry, then applies the corresponding `SurfaceFunction` for physics.
    * `getParaxial()` → list of `(z_position, paraxial_matrix)` tuples.

* **`ideal.py`**:
  * Paraxial helper functions: `ParaxialLensMat`, `ParaxialDistMat`, `ParaxialRefractMat`, `ParaxialMirrorMat` — all return 5×5 matrices.
  * `LinearElement(Element)`: Base for ideal elements backed by a `Plane`/`Disk` shape and a `Linear` surface function.
  * `IdealThinLens(LinearElement)`: Ideal thin lens; parameter `P = -1/focal`. Property `f`.
  * `IdealCylThinLens(LinearElement)`: Ideal cylindrical thin lens; parameters `Px`, `Py`. Properties `fx`, `fy`.
  * `IdealMirror(LinearElement)`: Ideal mirror; parameters `Px = -2/Rx`, `Py = -2/Ry`. Properties `fx`, `fy`, `Rx`, `Ry`.

* **`lens.py`**:
  * `SingletLens(Element)`: Realistic singlet using `Singlet` shape + 2 `RefractSnell`/`RefractFresnel` + 1 `Block` edge. Properties: `Power`, `f`, `f_bfl`, `f_ffl`, `R1`, `R2`, `T`, `T_edge`, `P1z`, `P2z`. Method `Bend(delta_c)`.
  * `CylSingletLens(SingletLens)`: Cylindrical singlet using `CylSinglet` shape.
  * `DoubletLens(Element)`: Cemented doublet using `Doublet` shape + 3 refractions + 2 `Block` edges.
  * `TripletLens(Element)`: Three-element cemented lens using `Triplet` shape + 4 refractions + 3 `Block` edges.

* **`mirror.py`**:
  * `SphericalMirror(Element)`: `HalfSphere` shape + `Reflect`. Properties: `c1`, `R`, `f`.
  * `CylindricalMirror(Element)`: `HalfCyl` shape + `Reflect`.
  * `ParabolicMirror(Element)`: `Quadric(k=-1)` shape + `Reflect`.
  * `ParabolicMirrorXZ(Element)`: Parabolic mirror with curvature in the XZ plane (90° rotated `QuadricZY`).

* **`aperture.py`**:
  * `CircularAperture(Element)`: `Disk` shape + `ApertureFilter`. Property `radius`.
  * `RectangularAperture(Element)`: `Rectangle` shape + `ApertureFilter`. Properties `half_x`, `half_y`.
  * `EllipticAperture(Element)`: `Ellipse` shape + `ApertureFilter`. Properties `r_major`, `r_minor`.

* **`sensor.py`**:
  * `Sensor(Element)`: Records ray hit positions, intensities, and IDs. `Transmit` physics (rays pass through). Methods: `reset()`, `getHitsTensors(ray_id)`, `getSpotSizeID_xy(ray_id, target_xy)`, `getSpotSizeParallel_xy(query_ids, target_xy)`.

* **`solid.py`**: Reserved; currently empty.

---

### 3.4. `rays/` (Data Containers)

* **`ray.py`**:
  * **Class `Rays` (tensorclass)**: Vectorized ray batch. Fields: `pos` [N,3], `dir` [N,3], `intensity` [N], `id` [N], `wavelength` [N]. Key methods: `initialize(...)` (factory), `scatter_update(mask, ...)`, `with_coords(new_pos, new_dir)`.
  * **Class `Paths`**: Proxy wrapper around `Rays` that records position history for 3D path visualization. Methods: `scatter_update(...)` appends to history, `get_history()`, `unwrap()`.

* **`bundle.py`**:
  * **Class `Bundle(nn.Module)`**: Abstract ray source. Subclasses implement `sample_pos(N)` and/or `sample_dir(N)`. Carries a `RayTransformBundle` to place the source in world space.
  * `CollimatedDisk(Bundle)`: Uniform disk with all rays parallel to +Z.
  * `CollimatedLine(Bundle)`: Line segment, parallel rays.
  * `Fan(Bundle)`: 2D angular fan in the XZ plane.
  * `PointSource(Bundle)`: Diverging cone defined by `NA`.

* **`beam.py`**:
  * `GaussianBeam(Bundle)`: Collimated Gaussian beam; parameters `diameter_1e2_x`, `diameter_1e2_y` (1/e² diameters).

* **`particle.py`**:
  * `LambertianSphere(Bundle)`: Lambertian emitter on a sphere surface.
  * `RayleighScatter(Bundle)`: Rayleigh scattering phase function sampled via Cardano formula.
  * `MieScatter(Bundle)`: Mie scattering placeholder (not implemented).

* **`panels.py`**:
  * `PanelSource(Bundle)`: Base class for flat area lights with Lambertian directional distribution.
  * `RectangularPanel(PanelSource)`: Uniform rectangular area light.
  * `RingSource(PanelSource)`: Annular area light.

---

### 3.5. `scene/` (Simulation Manager)

* **`base.py`**:
  * **Class `Scene(nn.Module)`**: General 3D non-sequential scene.
    * Population: `add_element(element)`, `add_bundle(bundle, N_rays)`.
    * Simulation: `simulate()` runs a bounce loop up to `Nbounces`. `ray_cast(rays)` finds closest hits. `step()` performs one bounce.
    * `compile_elements()`: Wraps elements in `torch.compile`.

* **`sequential.py`**:
  * **Class `SequentialScene(Scene)`**: Fixed-order sequential traversal.
    * `simulate(rays)`: Propagates rays through elements in order.
    * `getParaxial()`: Computes the system paraxial matrix by chaining element matrices with propagation matrices.
    * `checkDimensions()`: Validates physical dimensions (partially implemented).

* **`naive.py`**: Reserved; currently empty.

---

### 3.6. `optim/` (Optimization Logic)

* **`goals.py`** — Loss functions:
  * `FocalLengthLoss`: MSE on system power ($1/f$) via `SequentialScene.getParaxial()`.
  * `SpotTargetLoss`: Centroid-to-target distance per bundle.
  * `SpotSizeLoss`: Weighted RMS spot radius across multiple bundles.

* **`constraints.py`** — Log-barrier penalty functions:
  * Helpers: `_log_barrier_lb`, `_log_barrier_ub`, `_log_barrier`.
  * `ThicknessConstraint`: Enforces min/max spacing between optical surfaces within an element.
  * `SpacingConstraint`: Enforces minimum air gap between elements.
  * `SystemLengthConstraint`: Enforces maximum total system length.

> **Note:** The optimization subsystem is functional but targeted for overhaul. `WavefrontError` loss and glass-material boundary constraints are not yet implemented.

---

### 3.7. `render/` (Visualization)

* **`camera.py`**:
  * `Camera`: Pinhole camera model. `generate_rays()` returns a `Rays` batch for all pixels.
  * `OrbitCamera(Camera)`: CAD-style orbiting camera. Methods: `orbit(d_yaw, d_pitch)`, `roll(angle)`, `pan(dx, dy)`, `zoom(delta)`.
  * `Renderer`: Single-bounce shaded renderer. `render_3d(camera)` returns an RGB image tensor. `scan_profile(element, axis)` returns a 2D cross-section.

---

### 3.8. `gui/` (Interactive Application)

PySide6-based desktop GUI. Not intended for programmatic use.

* `workbench.py`: Main application window.
* `viewport.py`: 3D rendering viewport using the `render/` module.
* `forms.py`: Qt form widgets for element creation and parameter editing.
* `project.py`: Project save/load management.

---

## 4. Technical Constraints

1. **Strict Vectorization**: No Python loops over ray indices. Use PyTorch batch operations.
2. **In-place Operations**: Avoid `x += y` to preserve the autograd graph.
3. **Coordinate Handling**: Always call `transform(rays)` (Global → Local) before intersection math in `primitives.py`, then `invTransform` (Local → Global) to return results.
4. **Inked Surfaces**: Optical faces get `RefractSnell`/`Reflect`; cylindrical/box edges get `Block`.
5. **Gradient Flags**: Every `nn.Parameter` has a corresponding `_grad` boolean argument in the constructor. Optimization targets must be created with the appropriate `_grad=True`.
