This completes the architectural blueprint. I have integrated your specific requirements: the `Inside()` method for volumetric/planar checks, the `Shape2D` class for apertures and mirrors, geometry validation logic, and the per-surface physics model for "inked" lens edges.

---

# Project Blueprint: Differentiable 3D Ray Tracer (PyTorch)

## 1. Project Overview

* **Purpose:** A differentiable 3D ray tracing library for optical system design and optimization using PyTorch autograd.
* **Target Users:** Internal research, optical engineers, and future open-source contributors.
* **Core Dependencies:** `PyTorch` (primary engine), `NumPy` (utility).

---

## 2. High-Level Architecture

* **Entry Point:** `src/__init__.py`
* **Data Flow:** 1.  User defines `Rays` or `Bundles`.
2.  User assembles a `Scene` or `OpticalSystem` containing `Elements`.
3.  The `Scene` iterates through ray-surface interactions.
4.  PyTorch tracks gradients from the "Loss" (e.g., spot size) back to `Element` parameters (e.g., curvature, position).

---

## 3. Module Breakdown

### 3.1. `geom/` (Geometry Engine)

Defines the spatial boundaries and intersection math.

* **`transform.py`**: Helper functions for 4x4 matrix operations. Maps Global  Local coordinates using differentiable PyTorch tensors.
* **`primitives.py`**:
* **Class `Surface**`: Parent class.
* **Methods**:
* `IntersectTest(ray)`: Returns distance .
* `Intersect(ray)`: Returns  and Local Normal .


* **Children**: `Plane`, `Cylinder`, `Sphere`, `Quadric`, `Hyperbola`.


* **`composite.py`**:
* **Class `Shape**`: A collection of surfaces defining a 3D volume.
* **`Inside(pos)`**: Returns boolean tensor; `True` if point is within the bounded volume.
* **Validation**: Constructor must check for non-physical geometry (e.g., thickness resulting in self-intersecting surfaces).


* **Class `Shape2D**`: A specialized `Shape` consisting of a single surface with a finite boundary (e.g., a disk or rectangle).
* **`Inside(pos)`**: Checks if the intersection point lies within the 2D bounds.


* **Children**: `Box`, `Singlet`, `Doublet`, `Polyhedron`.


* **`special.py`**: Complex geometries (e.g., `Aspheric`) requiring iterative solvers.

---

### 3.2. `phys/` (Physics Engine)

* **`perfPhys.py`**:
* **Class `SurfaceFunction**`: A callable object initialized with specific parameters (like refractive index ).
* **Logic**: Every surface in an element is assigned its own `SurfaceFunction`.
* **Methods**:
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