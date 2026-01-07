# Project Blueprint: Differentiable 3D Ray Tracer (PyTorch)

## 1. Project Overview

* **Purpose:** A differentiable 3D ray tracing library for optical system design and optimization using PyTorch autograd.
* **Target Users:** Internal research, optical engineers, and future open-source contributors.
* **Core Dependencies:** `PyTorch` (primary engine), `NumPy` (utility).

---

## 2. High-Level Architecture

* **Entry Point:** `src/__init__.py`
* **Data Flow:** 
1.  User defines `Rays` or `Bundles`.
2.  User assembles a `Scene` or `OpticalSystem` containing `Elements`.
3.  The `Scene` iterates through ray-surface interactions.
4.  PyTorch tracks gradients from the "Loss" (e.g., spot size) back to `Element` parameters (e.g., curvature, position).

---

## 3. Module Breakdown

### 3.1. `geom/` (Geometry Engine)

Defines the spatial boundaries and intersection math.

* **`transform.py`**: Helper functions for 4x4 matrix operations. Maps Global  Local coordinates using differentiable PyTorch tensors.
* **`primitives.py`**:
* **Class `Surface`**: Abstract Base Class.
    * **Protocol**:
        * `intersectTest(rays)`: **Fast, detached.** Returns distance $t$ (or matrix of $t$) for validity checks.
        * `intersect(rays)`: **Differentiable, detailed.** Returns $t$, `hit_point` (Global), and `normal` (Global).
    * **Implemented Children**:
        * `Plane`: Infinite flat surface.
        * `Sphere`: Standard curvature-based surface.
        * `Quadric`: General conic section of revolution (Spheres, Parabolas, Ellipses, Hyperbolas).
        * `Cylinder`: Infinite cylinder (for lens edges/barrels).


* **`composite.py`**:
    * **Class `Shape`**: A 3D volume defined by a collection of `Surface` objects.
        * `surfaces`: List of `Surface` instances (e.g., Front, Back, Edge).
        * `intersectTest(rays)`: Returns matrix `[N, K]` of $t$ values for all $K$ surfaces.
        * `intersect_surface(rays, index)`: Returns differentiable intersection for a specific surface index.
        * `inBounds(local_pos)`: Boolean volume check (True if point is inside).
        * **Children**: `Box` (auto-generates 6 planes), `ConvexPolyhedron`.
    
    * **Class `Shape2D`**: Finite boundaries acting on **Local Coordinates** (2D).
        * `inBounds(local_pos)`: Boolean mask for aperture checks.
        * **Children**: `Disk`, `Rectangle`, `Ellipse`.


* **`special.py`**: Complex geometries (e.g., `Aspheric`) requiring iterative solvers. Not implemented.

---

### 3.2. `phys/` (Physics Engine)

* **`perfPhys.py`**:
* **Class `SurfaceFunction**`: A callable object initialized with specific parameters (like refractive index ).
* **Logic**: Every surface in an element is assigned its own `SurfaceFunction`.
* **Children**:
* `Refract`: Differentiable Snell’s Law.
* `Reflect`: Standard vector reflection.
* `Block`: Terminates the ray (used for apertures or "inked" lens edges).
* `Transmit`: No change in direction.


---

### 3.3. `elements/` (Optical Components)

* **`element.py`**:
* **Class `Element**`: Parent class.
* **Structure**: Contains a `Shape` or `Shape2D`. Each individual surface within that shape holds a reference to a specific `SurfaceFunction`.
* **Logic**: During interaction, the element identifies which internal surface was hit and applies that surface's specific `SurfaceFunction` (e.g., front face refracts, side "inked" face blocks).


* **`lens.py`**: `SingletLens`, `DoubletLens`, `CylLens`.
* **`mirror.py`**: `FlatMirror`, `ParabolicMirror`, `CylMirror`.
* **`aperture.py`**: `CircAperture`, `RectAperture`.

---

### 3.4. `rays/` (Data Containers)

* **`ray.py`**:
* **Class `Rays**`: Vectorized container. Attributes: `.pos` [N,3], `.dir` [N,3], `.n` [N], `.active` [bool], `.wavelength`, `.intensity`.
* **Class `Paths**`: Child of `Rays`. Automatically appends `.pos` to a `.history` list for 3D path visualization.


* **`bundle.py`**: Factory functions: `create_point_source`, `create_collimated_bundle`.

---

### 3.5. `scene/` (Simulation Manager)

* **`scene.py`**:
* **Class `Scene**`: General 3D collection.
* **Class `OpticalSystem**`: Specialized for sequential systems. Calculates **Paraxial properties** (EFL, BFL), **Petzval sum**, and checks for sequential element intersections.



---

### 3.6. `optim/` (Optimization Logic)

* **`goals.py`**: Loss functions (e.g., `SpotSizeLoss`, `WavefrontError`).
* **`constraints.py`**: Penalty functions for physical constraints (e.g., `MinEdgeThickness`, `GlassBoundaryConstraint`).

---

## 4. Technical Constraints for AI Implementation

1. **Strict Vectorization**: No Python loops over ray indices. Use PyTorch batch operations.
2. **In-place Operations**: Avoid `x += y` to preserve the autograd graph.
3. **Coordinate Handling**: Always transform `Global Ray`  `Local Space` before calculating intersections in `primitives.py`.
4. **Inked Surfaces**: When building a `SingletLens`, the AI must assign `Refract` to the optical faces and `Block` to the cylindrical "edge" surface.

---