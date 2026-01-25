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

Defines the spatial boundaries, intersection math, and complex optical shapes. All geometry classes inherit from `torch.nn.Module`.

* **`transform.py`**:
    * **Class `RayTransform(nn.Module)`**: Handles 4x4 affine transformations.
        * **Attributes**: `trans` (Translation parameter), `rot_vec` (Rotation parameter/buffer).
            * Translation is always applied first, so that the vector always indicates relative positions in the parent frame
        * **Methods**:
            * `transform(rays)`: Maps Global $\to$ Local coordinates.
            * `invTransform(rays)`: Maps Local $\to$ Global coordinates.

* **`primitives.py`**:
    * **Class `Surface(nn.Module)`**: Abstract Base Class for infinite mathematical surfaces.
        * **Protocol**:
            * `intersectTest(rays)`: **Fast, detached.** Returns distance $t$ (or matrix of $t$) for validity checks.
            * `forward(rays)`: **Differentiable.** Returns $t$, `hit_point` (Global), and `normal` (Global).
        * **Implemented Children**:
            * `Plane`: Infinite flat surface ($z=0$ in local).
            * `Sphere`: Curvature-based surface ($|P|^2 = R^2$).
            * `Quadric`: General conic section of revolution (Parabolas, Hyperbolas).
            * `Cylinder`: Infinite tube ($x^2 + y^2 = R^2$).

* **`bounded.py`** (Replaces `Shape2D`):
    * **Class `SurfaceBounded(Surface)`**: Base wrapper class combining a mathematical surface with a finite boundary mask.
        * **Protocol**:
            * `inBounds(local_pos)`: Abstract boolean check. Returns `True` if point lies within the physical aperture.
            * `forward(rays)`: Calls parent intersection, then filters hits based on `inBounds`.
    * **Implemented Children**:
        * `Disk(Plane, SurfaceBounded)`: Circular aperture defined by `radius`.
        * `Rectangle(Plane, SurfaceBounded)`: Rectangular aperture defined by `half_x`, `half_y`.
        * `Ellipse(Plane, SurfaceBounded)`: Elliptical aperture defined by `r_major`, `r_minor`.
        * **`HalfSphere(Quadric, SurfaceBounded)`**: A `Sphere` or `Quadric` clipped to a hemisphere (used for lens faces).
            * Logic: `inBounds` checks if the $Z$ coordinate opposes the Radius direction.

* **`spherics.py`**:
    * **Class `Spheric(Shape)`**: Abstract base for rotationally symmetric lens stacks.
        * **Methods**:
            * `_make_surface(C, z_vertex)`: Factory method to create a `HalfSphere` optical surface at a specific position.
            * `inBounds(local_pos, surf_idx)`: Logic to determine if a hit is valid based on whether it hit an **Optical Face** (aperture check) or the **Lens Edge** (thickness check).
    * **Implemented Children**:
        * `Singlet(Spheric)`: A single element with 2 optical surfaces (Front/Back) and 1 cylindrical edge.
        * `Doublet(Spheric)`: 2 cemented elements (3 optical surfaces, 2 edges).
        * `Triplet(Spheric)`: 3 cemented elements (4 optical surfaces, 3 edges).

* **`cylindrics.py`** (Proposed Template):
    * **Class `Cylindric(Shape)`**: Abstract base for lenses with cylindrical power (anamorphic or rod lenses).
        * **Methods**:
            * `_make_cyl_surface(Cx, Cy, z_vertex)`: Factory method to create a toric or cylindrical surface. unlike `Spheric`, this requires distinct curvatures for X and Y axes.
    * **Class `CylSinglet(Cylindric)`**:
        * **Constructor**: `__init__(Cy1, Cy2, Dx, Dy, T, ...)`
        * **Structure**:
            1.  **Front Surface**: Defined by Y-curvature `Cy1`.
            2.  **Back Surface**: Defined by `Cy2`.
            3.  **Edge**: Use a `CvxPolhedron` (rectangular sides) or `Cylinder` depending on the "cut" of the lens.
        * **Logic**: Standard singlet construction but utilizing `Toroidal` or `Cylindrical` primitive math for the faces.

* **`shape.py`**:
    * **Class `Shape(nn.Module)`**: A 3D volume defined by a collection of `Surface` objects.
        * **Protocol**:
            * `forward(rays, surf_idx)`: Calculates differentiable intersection for a specific surface index.
            * `intersectTest(rays)`: Returns intersection candidates for *all* surfaces in the volume.
    * **Implemented Children**:
        * `Box`: Auto-generates 6 `Plane` surfaces to form a rectangular prism.
        * `CvxPolyhedron`: A generic convex shape defined by a list of planes.


* **`shape.py`**:
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

* **`phys_std.py`**:
* **Class `SurfaceFunction(nn.Module)`**: Base functor for optical physics.
* **Logic**: Every surface in an element is assigned its own `SurfaceFunction`. The interaction is computed via the `forward` method.
* **Protocol**:
    * `forward(local_intersect, ray_dir, normal, **kwargs)`: Returns new direction and intensity.
* **Children**:
* `RefractSnell`: Differentiable Snell’s Law.
* `RefractFresnel`: Stochastic Fresnel reflection/refraction.
* `Reflect`: Standard vector reflection.
* `Block`: Terminates the ray (used for apertures or "inked" lens edges).
* `Transmit`: No change in direction.

---

### 3.3. `elements/` (Optical Components)

* **`element.py`**:
* **Class `Element(nn.Module)`**: Parent class.
* **Structure**: Contains a `Shape` (Module) and a list of `SurfaceFunctions`.
* **Logic**: During interaction, the element identifies which internal surface was hit and applies that surface's specific `SurfaceFunction` (e.g., front face refracts, side "inked" face blocks).


* **`lens.py`**: `SingletLens`, `DoubletLens`, `CylLens`.
* **`mirror.py`**: `FlatMirror`, `ParabolicMirror`, `CylMirror`.
* **`aperture.py`**: `CircAperture`, `RectAperture`.


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