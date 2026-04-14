# RayTraceTorch — Detailed Code Reference

This file maps every public class and key function to its exact file and line number, constructor signature, key attributes, and method signatures. Use it as a navigation guide when making targeted edits.

---

## Table of Contents

1. [geom/ — Geometry Engine](#1-geom--geometry-engine)
   - [transform.py](#11-transformpy)
   - [primitives.py](#12-primitivespy)
   - [bounded.py](#13-boundedpy)
   - [shape.py](#14-shapepy)
   - [spherics.py](#15-sphericspy)
   - [cylindrics.py](#16-cylindricspy)
   - [complex.py](#17-complexpy)
2. [phys/ — Physics Engine](#2-phys--physics-engine)
   - [std.py](#21-stdpy)
   - [filter.py](#22-filterpy)
3. [elements/ — Optical Components](#3-elements--optical-components)
   - [parent.py](#31-parentpy)
   - [ideal.py](#32-idealpy)
   - [lens.py](#33-lenspy)
   - [mirror.py](#34-mirrorpy)
   - [aperture.py](#35-aperturepy)
   - [sensor.py](#36-sensorpy)
4. [rays/ — Ray Data](#4-rays--ray-data)
   - [ray.py](#41-raypy)
   - [bundle.py](#42-bundlepy)
   - [beam.py](#43-beampy)
   - [particle.py](#44-particlepy)
   - [panels.py](#45-panelspy)
5. [scene/ — Simulation Manager](#5-scene--simulation-manager)
   - [base.py](#51-basepy)
   - [sequential.py](#52-sequentialpy)
6. [optim/ — Optimization](#6-optim--optimization)
   - [goals.py](#61-goalspy)
   - [constraints.py](#62-constraintspy)
7. [render/ — Visualization](#7-render--visualization)
   - [camera.py](#71-camerapy)
8. [gui/ — Desktop Application](#8-gui--desktop-application)

---

## 1. `geom/` — Geometry Engine

### 1.1 `transform.py`

**`geom/transform.py`**

#### `RayTransform(nn.Module)` — line 10

Central coordinate-frame transform used by every `Surface` and `Shape`.

```
__init__(
    rotation=None,       # [3] axis-angle vector (Rodrigues); encodes Local→Global rotation
    translation=None,    # [3] object origin in the parent (Global) frame
    dtype=torch.float32,
    trans_grad=False,    # enable gradient on trans
    trans_mask=None,     # [3] bool mask limiting which axes carry gradients
    rot_grad=False,
    rot_mask=None,
)
```

| Attribute | Type | Description |
|---|---|---|
| `trans` | `nn.Parameter [3]` | Translation (object origin in Global frame) |
| `rot_vec` | `nn.Parameter [3]` | Axis-angle rotation vector |
| `trans_mask` | buffer `[3]` | Optional gradient mask for `trans` |
| `rot_mask` | buffer `[3]` | Optional gradient mask for `rot_vec` |

| Method | Line | Signature | Notes |
|---|---|---|---|
| `_compute_matrix` | 48 | `() → Tensor[3,3]` | Rotation matrix via `matrix_exp` |
| `rot` (property) | 56 | `→ Tensor[3,3]` | Calls `_compute_matrix` |
| `transform` | 64 | `(rays) → (pos, dir)` | **Global → Local**; calls `transform_` |
| `transform_` | 68 | `(pos, dir) → (local_pos, local_dir)` | Subtracts `trans`, multiplies by `rot` |
| `invTransform` | 90 | `(rays) → (pos, dir)` | **Local → Global**; calls `invTransform_` |
| `invTransform_` | 94 | `(pos, dir) → (global_pos, global_dir)` | Multiplies by `rot.T`, adds `trans` |
| `paraxial` | 114 | `() → Tensor[5,5]` | Paraxial transfer matrix |
| `paraxial_inv` | 124 | `() → Tensor[5,5]` | Inverse paraxial transfer matrix |

> **Coordinate convention**: `transform` = Global→Local (for intersection math). `invTransform` = Local→Global (to convert results back). Note: the in-code docstrings on `transform_`/`invTransform_` have the labels swapped — trust this document and the actual math.

---

#### `RayTransformNoisy(RayTransform)` — line 134

Adds per-ray Gaussian noise to translation and rotation for tolerance analysis.

```
__init__(
    rotation=None, translation=None, dtype=torch.float32,
    std_translation=(0,0,0),   # [3] std-dev of translation noise
    std_rotation=(0,0,0),      # [3] std-dev of rotation noise
    trans_grad=False, trans_mask=None, rot_grad=False, rot_mask=None,
)
```

| Method | Line | Notes |
|---|---|---|
| `_compute_matrix_batch` | 165 | Batch rotation matrix for N perturbed rot_vecs |
| `addNoise` | 176 | `(N) → (trans_noise, rot_noise)` — samples N perturbations |
| `transform_` | 183 | Applies per-ray noisy transform (Global→Local) |
| `invTransform_` | 198 | Applies per-ray noisy inverse transform (Local→Global) |

---

#### `RayTransformBundle(RayTransform)` — line 218

Variant used by `Bundle` sources. Translation is **additive** (places source in world space).

| Method | Line | Notes |
|---|---|---|
| `transform_` | 220 | `pos + trans`; applies `rot.T` to directions |
| `invTransform_` | 242 | `pos - trans`; applies `rot` to directions |

---

### 1.2 `primitives.py`

**`geom/primitives.py`**

#### `Surface(nn.Module)` — line 9

Abstract base for all infinite mathematical surfaces.

```
__init__(transform=None)   # RayTransform or None
```

| Attribute | Type | Description |
|---|---|---|
| `epsilon` | `nn.Parameter` scalar | Minimum valid $t$ threshold |
| `transform` | `RayTransform` | Coordinate frame |

| Method | Line | Signature | Notes |
|---|---|---|---|
| `_check_t` | 28 | `(t_list, *args) → t` | Filters invalid $t$, returns minimum |
| `intersectTest` | 38 | `(rays) → t` | No-grad fast test |
| `forward` | 59 | `(rays, *args) → (t, hit_point, normal)` | Differentiable; hit_point and normal in Global frame |
| `_getNormal` | 100 | `(local_pos) → normal` | **Abstract** |
| `_solve_t` | 104 | `(local_pos, local_dir) → t` | **Abstract** |
| `z` (property) | 108 | `→ float` | Z position of surface |
| `surfaces` (property) | 115 | `→ tuple` | Returns `(self,)` |

---

#### `Plane(Surface)` — line 119

Infinite flat surface at $z=0$ in local frame.

```
__init__(transform=None)
```

| Method | Line | Notes |
|---|---|---|
| `_solve_t` | 124 | Solves $z=0$: $t = -p_z / d_z$ |
| `_getNormal` | 138 | Returns `[0, 0, 1]` |

---

#### `Sphere(Surface)` — line 146

```
__init__(radius, radius_grad=False, transform=None)
```

| Parameter | Type | Description |
|---|---|---|
| `radius` | `nn.Parameter` scalar | Sphere radius |

| Method | Line | Notes |
|---|---|---|
| `_solve_t` | 155 | Quadratic equation; returns smallest positive $t$ |
| `_getNormal` | 183 | `local_pos / ‖local_pos‖` |

---

#### `Cylinder(Surface)` — line 190

Infinite cylinder along Z-axis ($x^2 + y^2 = R^2$).

```
__init__(radius, transform=None, radius_grad=False)
```

| Method | Line | Notes |
|---|---|---|
| `_solve_t` | 201 | Quadratic in XY only |
| `_getNormal` | 233 | Radial normal (XY components only) |

---

#### `Quadric(Surface)` — line 244

General conic section of revolution: $c r^2 / (1 + \sqrt{1-(1+k)c^2 r^2}) - z = 0$.

```
__init__(c, k, transform=None, c_grad=False, k_grad=False)
```

| Parameter | Type | Description |
|---|---|---|
| `c` | `nn.Parameter` scalar | Curvature ($1/R$) |
| `k` | `nn.Parameter` scalar | Conic constant ($k=0$: sphere, $k=-1$: paraboloid) |

| Method | Line | Notes |
|---|---|---|
| `_get_coeffs` | 266 | `(local_pos, local_dir) → (A, B, C)` quadratic coefficients |
| `_solve_quadratic` | 290 | Differentiable quadratic solver |
| `_solve_t` | 322 | Wraps `_solve_quadratic` |
| `_getNormal` | 330 | Gradient of the sag equation |

---

#### `QuadricZY(Quadric)` — line 346

Cylindrical variant: curvature along Y only, invariant in X.

```
__init__(c, k, transform=None, c_grad=False, k_grad=False)
```

| Method | Line | Notes |
|---|---|---|
| `_get_coeffs` | 356 | Projects onto Y-Z plane for coefficient extraction |
| `_getNormal` | 379 | Normal has zero X component |

---

#### `Cone(Surface)` — line 398

Double cone along Z: $x^2 + y^2 = (z / \text{slope})^2$.

```
__init__(slope, slope_grad=False, transform=None)
```

| Parameter | Type | Description |
|---|---|---|
| `slope` | `nn.Parameter` scalar | dz/dr ratio |

| Method | Line | Notes |
|---|---|---|
| `_solve_t` | 416 | Quadratic; returns both nappes |
| `_getNormal` | 468 | Handles apex singularity |

---

### 1.3 `bounded.py`

**`geom/bounded.py`**

#### `SurfaceBounded(Surface)` — line 9

Mixin/base for surfaces with finite aperture. Overrides `_check_t` to call `inBounds`.

```
__init__(transform=None, invert=False)
```

| Method | Line | Notes |
|---|---|---|
| `_check_t` | 20 | Filters $t$ values where `inBounds` is False |
| `inBounds` | 39 | **Abstract** `(local_pos) → bool mask [N]` |

---

#### `Disk(Plane, SurfaceBounded)` — line 51

```
__init__(radius, invert=False, transform=None)
```

`inBounds` (line 60): $x^2 + y^2 \leq R^2$

---

#### `Rectangle(Plane, SurfaceBounded)` — line 67

```
__init__(half_x, half_y, invert=False, transform=None)
```

Parameters: `hx`, `hy` (stored as `nn.Parameter`)

`inBounds` (line 77): $|x| \leq hx$ AND $|y| \leq hy$

---

#### `Ellipse(Plane, SurfaceBounded)` — line 85

```
__init__(r_major, r_minor, rot, r_major_grad=False, r_minor_grad=False, rot_grad=False, invert=False, transform=None)
```

Parameters: `r_major`, `r_minor`, `rot` (all `nn.Parameter`)

`inBounds` (line 98): rotated ellipse inequality

---

#### `HalfSphere(Quadric, SurfaceBounded)` — line 109

Quadric clipped to a hemisphere. Used as optical face of spherical lenses.

```
__init__(curvature, curvature_grad, transform=None)
# Internally: Quadric(c=curvature, k=0)
```

| Method | Line | Notes |
|---|---|---|
| `inBounds` | 123 | `local_pos_z * sign(curvature) <= 0` |
| `sagittalZ` | 129 | `(radius) → z` — sag height at given aperture radius |

---

#### `HalfCyl(QuadricZY, SurfaceBounded)` — line 142

Cylindrical quadric clipped to half-cylinder. Used as optical face of cylindrical lenses.

```
__init__(curvature, curvature_grad, transform=None)
# Internally: QuadricZY(c=curvature, k=0)
```

| Method | Line | Notes |
|---|---|---|
| `inBounds` | 151 | `local_pos_z * sign(curvature) <= 0` |
| `sagittalZ` | 156 | `(y_height) → z` — sag height at given Y half-height |

---

#### `SingleCone(Cone, SurfaceBounded)` — line 169

Single nappe of a cone (positive or negative Z side).

```
__init__(slope, slope_grad=False, invert=False, transform=None)
```

`inBounds` (line 188): selects one nappe by checking Z sign.

---

### 1.4 `shape.py`

**`geom/shape.py`**

#### `Shape(nn.Module)` — line 8

Abstract base for 3D volumetric shapes. Manages a `ModuleList` of `Surface` objects.

```
__init__(transform=None)
```

| Attribute | Type | Description |
|---|---|---|
| `epsilon` | `nn.Parameter` scalar | Minimum valid $t$ |
| `surfaces` | `nn.ModuleList` | List of `Surface` objects |
| `transform` | `RayTransform` | Frame of the shape |

| Method | Line | Signature | Notes |
|---|---|---|---|
| `intersectTest` | 25 | `(rays) → Tensor[N, K]` | Fast no-grad test for all K surfaces |
| `forward` | 61 | `(rays, surf_idx) → (t, hit_point, normal)` | Differentiable for one surface |
| `inBounds` | 89 | `(local_pos, surf_idx) → bool[N]` | **Abstract** |
| `z` (property) | 96 | `→ float` | Z position of the shape |
| `__len__` | 100 | `() → int` | Number of surfaces |

---

#### `CvxPolyhedron(Shape)` — line 104

Convex polyhedron from a list of `Plane` objects.

```
__init__(planes_list=None, transform=None)
```

| Property | Line | Notes |
|---|---|---|
| `PlaneRotMat` | 115 | Stacked plane normals `[K, 3]` |
| `PlaneTrans` | 118 | Stacked plane positions `[K, 3]` |

`inBounds` (line 122): checks all half-space inequalities.

---

#### `Box(CvxPolyhedron)` — line 135

Rectangular prism with 6 planes (all faces).

```
__init__(length, width, height, transform=None, l_grad=False, w_grad=False, h_grad=False)
```

Properties `length`, `width`, `height` (lines 147–157): computed from surface positions.

`_build_surfaces` (line 159): creates 6 `Plane` objects.

---

#### `Box4Side(CvxPolyhedron)` — line 213

Rectangular prism with 4 side planes only (no front/back caps). Used as lens barrel for `CylSinglet`.

```
__init__(width, height, transform=None, w_grad=False, h_grad=False)
```

`_build_surfaces` (line 233): creates 4 side `Plane` objects.

---

### 1.5 `spherics.py`

**`geom/spherics.py`**

#### `Spheric(Shape)` — line 10

Abstract base for spherical-surface lens stacks. Surfaces are indexed as: `[0..N_optical-1]` = optical faces, `[N_optical..]` = cylindrical edges.

| Method | Line | Signature | Notes |
|---|---|---|---|
| `_make_surface` | 15 | `(C, z_vertex, c_grad, z_grad) → HalfSphere` | Factory for optical faces |
| `inBounds` | 27 | `(local_pos, surf_idx) → bool[N]` | Optical: radial check; Edge: axial span check |
| `T` (property) | 48 | `→ float` | Total center thickness |
| `T_edge` (property) | 51 | `→ float` | Minimum edge thickness |

---

#### `Singlet(Spheric)` — line 56

Single lens element: Front surface (index 0), Back surface (index 1), Edge cylinder (index 2).

```
__init__(
    C1, C2,           # front/back curvatures
    D,                # diameter
    T,                # center thickness
    C1_grad=True, C2_grad=True, D_grad=False, T_grad=True,
    transform=None,
)
```

Attributes: `N_optical=2`, `radius` ($= D/2$).

---

#### `Doublet(Spheric)` — line 116

Two-element cemented lens: Surf1 (0), Surf2 (1), Surf3 (2), Edge1 (3), Edge2 (4).

```
__init__(
    C1, C2, C3,       # surface curvatures
    D,                # diameter
    T1, T2,           # element thicknesses
    C1_grad=True, C2_grad=True, C3_grad=True,
    D_grad=False, T1_grad=True, T2_grad=True,
    transform=None,
)
```

Properties `T1`, `T2` (lines 199–206): thicknesses derived from surface Z positions.

---

#### `Triplet(Spheric)` — line 209

Three-element cemented lens: Surf1–4 (0–3), Edge1–3 (4–6).

```
__init__(
    C1, C2, C3, C4,
    D,
    T1, T2, T3,
    C1_grad=True, C2_grad=True, C3_grad=True, C4_grad=True,
    D_grad=False, T1_grad=True, T2_grad=True, T3_grad=True,
    transform=None,
)
```

Properties `T1`, `T2`, `T3` (lines 286–298).

---

### 1.6 `cylindrics.py`

**`geom/cylindrics.py`**

#### `Cylindric(Shape)` — line 10

Abstract base for cylindrical-surface lens stacks. Optical faces are `HalfCyl`; edges are sides of a `Box4Side`.

| Method | Line | Notes |
|---|---|---|
| `_make_cyl_surface` | 15 | `(C, z_vertex, c_grad, z_grad) → HalfCyl` |
| `inBounds` | 23 | Optical: rectangular aperture check; Edge: axial Z-bounds check |

---

#### `CylSinglet(Cylindric)` — line 58

Cylindrical singlet: Front (0), Back (1), Right (2), Left (3), Top (4), Bottom (5).

```
__init__(
    C1, C2,                # Y-curvatures
    width, height,         # aperture dimensions
    T,                     # center thickness
    C1_grad=True, C2_grad=True, T_grad=True,
    w_grad=False, h_grad=False,
    transform=None,
)
```

Properties `width`, `height` (lines 112–119): derived from edge surface positions.

---

### 1.7 `complex.py`

**`geom/complex.py`**

#### `Aspheric(Surface)` — line 6

Stub only. `_solve_t` and `_getNormal` are not implemented.

---

## 2. `phys/` — Physics Engine

### 2.1 `std.py`

**`phys/std.py`**

#### `SurfaceFunction(nn.Module)` — line 8

Abstract base.

```
forward(local_intersect, ray_dir, normal, **kwargs) → (new_dir [N,3], intensity_mod [N])
```

---

#### `Linear(SurfaceFunction)` — line 35

ABCD-matrix style ideal optic. Used by `LinearElement` subclasses.

```
__init__(Cx=0, Cy=0, Dx=1, Dy=1, Cx_grad=False, Cy_grad=False, Dx_grad=False, Dy_grad=False, transform=None)
```

Parameters: `Cx`, `Cy` (focal power per axis), `Dx`, `Dy` (magnification per axis).

`forward` (line 56): applies linear transformation to ray directions.

---

#### `Reflect(SurfaceFunction)` — line 91

Perfect specular reflection.

`forward` (line 97): $d' = d - 2(d \cdot n)n$

---

#### `RefractSnell(SurfaceFunction)` — line 111

Differentiable Snell's Law with TIR handling.

```
__init__(ior_in, ior_out, ior_in_grad=False, ior_out_grad=False)
```

Parameters: `ior_in`, `ior_out` (indices of refraction).

`forward` (line 123): computes refracted direction; zeroes intensity on TIR.

---

#### `RefractFresnel(SurfaceFunction)` — line 178

Stochastic Fresnel refraction/reflection. Probabilistically chooses refract or reflect per ray.

```
__init__(ior_in, ior_out, ior_in_grad=False, ior_out_grad=False)
```

| Method | Line | Notes |
|---|---|---|
| `_fresnel_reflectance` | 195 | `(cos_i, cos_t) → R` — Fresnel equations |
| `forward` | 219 | Stochastic branch on Fresnel R |

---

#### `Transmit(SurfaceFunction)` — line 277

Pass-through. Returns unchanged `ray_dir` and `intensity * 1.0`.

---

#### `Block(SurfaceFunction)` — line 293

Absorber. Returns zero direction and zero intensity.

---

### 2.2 `filter.py`

**`phys/filter.py`**

#### `ApertureFilter(Transmit)` — line 10

Applies an aperture mask from a callable.

```
__init__(inBounds: Callable)   # callable(local_pos) → bool[N]
```

`forward` (line 24): zeroes intensity where `inBounds` returns False.

---

#### `Fuzzy(Transmit)` — line 36

Variable intensity transmission.

```
__init__(intensity_function: Callable)   # callable(local_pos) → float[N]
```

`forward` (line 45): multiplies intensity by `intensity_function(local_intersect)`.

---

## 3. `elements/` — Optical Components

### 3.1 `parent.py`

**`elements/parent.py`**

#### `Element(nn.Module)` — line 8

Base class for all optical elements.

```
__init__()
# Subclasses must populate self.shape and self.surface_functions
```

| Attribute | Type | Description |
|---|---|---|
| `shape` | `Shape` | The geometric volume |
| `surface_functions` | `nn.ModuleList[SurfaceFunction]` | One per surface in `shape` |

| Method | Line | Signature | Notes |
|---|---|---|---|
| `intersectTest` | 30 | `(rays) → Tensor[N, K]` | Delegates to `shape.intersectTest` |
| `forward` | 44 | `(rays, surf_idx) → rays` | Geometry + physics for one surface |
| `_paraxial` | 61 | `() → Tensor[5,5]` | Returns identity; override in subclasses |
| `getParaxial` | 65 | `() → List[(z, mat)]` | Returns list of (z_position, 5×5 matrix) |

---

### 3.2 `ideal.py`

**`elements/ideal.py`**

#### Paraxial helper functions — lines 9–45

| Function | Line | Signature |
|---|---|---|
| `ParaxialLensMat` | 9 | `(lens_power_x, lens_power_y) → Tensor[5,5]` |
| `ParaxialDistMat` | 17 | `(dist) → Tensor[5,5]` |
| `ParaxialRefractMat` | 25 | `(Cx, Cy, ior_1, ior_2) → Tensor[5,5]` |
| `ParaxialMirrorMat` | 39 | `(Cx, Cy) → Tensor[5,5]` |

---

#### `LinearElement(Element)` — line 47

Base for ideal optics. Shape is `Plane` or `Disk`; physics is `Linear`.

```
__init__(shape: Plane, linSurfFunc: Linear)
```

`_paraxial` (line 58): returns the lens power matrix from `linSurfFunc`.

---

#### `IdealThinLens(LinearElement)` — line 65

```
__init__(focal, focal_grad=False, diameter=inf, transform=None)
```

| Attribute | Type | Description |
|---|---|---|
| `P` | `nn.Parameter` scalar | Lens power ($= -1/f$) |

Property `f` (line 85): returns $-1/P$.

Shape: `Plane` if `diameter=inf`, else `Disk(diameter/2)`.

---

#### `IdealCylThinLens(LinearElement)` — line 90

```
__init__(focal_x, focal_y, focal_x_grad=False, focal_y_grad=False, diameter=inf, transform=None)
```

Parameters: `Px` ($= -1/f_x$), `Py` ($= -1/f_y$).

Properties `fx` (line 112), `fy` (line 115).

---

#### `IdealMirror(LinearElement)` — line 121

```
__init__(radius_x, radius_y, radius_x_grad=False, radius_y_grad=False, diameter=inf, transform=None)
```

Parameters: `Px` ($= -2/R_x$), `Py` ($= -2/R_y$).

`_paraxial` (line 144): returns mirror matrix.

Properties `fx`, `fy`, `Rx`, `Ry` (lines 148–162).

---

### 3.3 `lens.py`

**`elements/lens.py`**

#### `SingletLens(Element)` — line 13

Realistic singlet backed by a `Singlet` shape.

```
__init__(
    c1, c2,                         # front/back curvatures
    d,                               # diameter
    t,                               # center thickness
    ior_glass, ior_media=1.0,
    c1_grad=False, c2_grad=False, t_grad=False, d_grad=False,
    ior_glass_grad=False, ior_media_grad=False,
    fresnel=False,                   # use RefractFresnel instead of RefractSnell
    inked=False,                     # assign Block to edge (vs. RefractSnell)
    transform=None,
)
```

Surface function assignment: `surface_functions[0]` = front refraction, `[1]` = back refraction, `[2]` = edge (`Block` if `inked`, else `RefractSnell`).

| Property | Line | Notes |
|---|---|---|
| `Power` | 60 | Thin-lens approximation of total power |
| `power1`, `power2` | 63–67 | Per-surface power |
| `f` | 72 | EFL ($= -1 / \text{Power}$) |
| `f_bfl` | 76 | Back focal length |
| `f_ffl` | 81 | Front focal length |
| `R1`, `R2` | 94–102 | Radii ($= 1/c$) |
| `T`, `T_edge` | 104–110 | Center and edge thicknesses |
| `P1z`, `P2z` | 112–127 | Principle plane Z positions |

| Method | Line | Signature | Notes |
|---|---|---|---|
| `getParaxial` | 129 | `() → List[(z, mat)]` | Builds paraxial sequence |
| `Bend` | 150 | `(delta_c) → None` | Adjusts curvatures preserving power |

---

#### `CylSingletLens(SingletLens)` — line 185

Cylindrical singlet. Same interface as `SingletLens` but backed by `CylSinglet` shape.

```
__init__(
    c1, c2,
    height, width,
    t,
    ior_glass, ior_media=1.0,
    c1_grad=False, c2_grad=False, t_grad=False,
    ior_glass_grad=False, ior_media_grad=False,
    fresnel=False, inked=False,
    transform=None,
)
```

---

#### `DoubletLens(Element)` — line 231

Cemented doublet backed by `Doublet` shape.

```
__init__(
    c1, c2, c3,
    d, t1, t2,
    ior_glass1, ior_glass2, ior_media=1.0,
    c1_grad=False, c2_grad=False, c3_grad=False,
    d_grad=False, t1_grad=False, t2_grad=False,
    ior_glass1_grad=False, ior_glass2_grad=False, ior_media_grad=False,
    fresnel=False, inked=False,
    transform=None,
)
```

Surface functions: `[0]` media→glass1, `[1]` glass1→glass2, `[2]` glass2→media, `[3]` `Block`, `[4]` `Block`.

Properties `R1`, `R2`, `R3` (lines 284–294), `T1`, `T2` (lines 296–301).

---

#### `TripletLens(Element)` — line 325

Three-element cemented lens backed by `Triplet` shape.

```
__init__(
    c1, c2, c3, c4,
    d, t1, t2, t3,
    ior_glass1, ior_glass2, ior_glass3, ior_media=1.0,
    c1_grad=False, c2_grad=False, c3_grad=False, c4_grad=False,
    d_grad=False, t1_grad=False, t2_grad=False, t3_grad=False,
    ior_glass1_grad=False, ior_glass2_grad=False, ior_glass3_grad=False,
    ior_media_grad=False,
    fresnel=False, inked=False,
    transform=None,
)
```

Surface functions: 4 refractions + 3 `Block` edges.

Properties `R1`–`R4` (lines 387–398), `T1`–`T3` (lines 400–408).

---

### 3.4 `mirror.py`

**`elements/mirror.py`**

#### `Mirror(Element)` — line 14

Base mirror. Assigns `Reflect` as the single surface function.

#### `SphericalMirror(Mirror)` — line 22

```
__init__(c1, d, c1_grad=False, d_grad=False, transform=None)
```

Shape: `HalfSphere`. Properties `c1` (line 33), `R` (line 36), `f` (line 39).

---

#### `CylindricalMirror(Mirror)` — line 55

```
__init__(c1, d, c1_grad=False, d_grad=False, transform=None)
```

Shape: `HalfCyl`.

---

#### `ParabolicMirror(Mirror)` — line 91

```
__init__(c1, d, c1_grad=False, d_grad=False, transform=None)
```

Shape: `Quadric(c=c1, k=-1.0)` (parabola).

---

#### `ParabolicMirrorXZ(Mirror)` — line 126

Parabolic mirror with curvature in the XZ plane. Uses `QuadricZY` rotated 90° around Z.

```
__init__(c1, d, c1_grad=False, d_grad=False, transform=None)
```

---

### 3.5 `aperture.py`

**`elements/aperture.py`**

#### `CircularAperture(Element)` — line 8

```
__init__(radius, invert=False, transform=None)
```

Shape: `Disk`. Physics: `ApertureFilter(Disk.inBounds)`. Property `radius` (line 19).

---

#### `RectangularAperture(Element)` — line 24

```
__init__(half_x, half_y, invert=False, transform=None)
```

Shape: `Rectangle`. Properties `half_x` (line 35), `half_y` (line 38).

---

#### `EllipticAperture(Element)` — line 44

```
__init__(r_major, r_minor, rot=0.0, invert=False, transform=None)
```

Shape: `Ellipse`. Properties `r_major` (line 56), `r_minor` (line 59).

---

### 3.6 `sensor.py`

**`elements/sensor.py`**

#### `Sensor(Element)` — line 9

Records ray hits. Physics: `Transmit` (rays pass through). Accumulates `hitLocs`, `hitIntensity`, `hitID` as lists across multiple `forward` calls.

```
__init__(shape: Union[Shape, Surface])
```

| Method | Line | Signature | Notes |
|---|---|---|---|
| `forward` | 22 | `(rays, surf_idx)` | Records hits, applies Transmit |
| `reset` | 41 | `()` | Clears hit lists |
| `getHitsTensors` | 46 | `(ray_id=None) → (pos, intensity)` | Returns concatenated tensors |
| `getSpotSizeID_xy` | 67 | `(ray_id, target_xy=None, norm_ord=2) → scalar` | RMS spot radius for one bundle ID |
| `getSpotSizeParallel_xy` | 87 | `(query_ids, target_xy=None, norm_ord=2) → Tensor[K]` | Vectorized spot size for K bundle IDs |

---

## 4. `rays/` — Ray Data

### 4.1 `ray.py`

**`rays/ray.py`**

#### `Rays` (tensorclass) — line 7

Vectorized ray batch. Implemented as a `tensordict.tensorclass`.

| Field | Shape | Description |
|---|---|---|
| `pos` | `[N, 3]` | Ray origins |
| `dir` | `[N, 3]` | Ray directions (normalized in `__post_init__`) |
| `intensity` | `[N]` | Per-ray intensity |
| `id` | `[N]` | Bundle ID (integer) |
| `wavelength` | `[N]` | Wavelength |

| Method | Line | Signature | Notes |
|---|---|---|---|
| `__post_init__` | 22 | — | Normalizes `dir` |
| `initialize` | 42 | `(origins, directions, wavelengths, intensities, ray_id, device, dtype) → Rays` | Factory / class method |
| `scatter_update` | 29 | `(mask, new_pos, new_dir, intensity_mod)` | Non-in-place masked update |
| `with_coords` | 84 | `(new_pos, new_dir) → Rays` | Returns copy with updated geometry |

---

#### `Paths` — line 100

Proxy wrapper around `Rays` that appends `pos` to `_history` after each `scatter_update`.

```
__init__(rays: Rays)
```

| Method | Line | Notes |
|---|---|---|
| `scatter_update` | 177 | Records `pos` to history before delegating to `_rays` |
| `get_history` | 196 | `() → List[Tensor[N,3]]` |
| `unwrap` | 192 | `() → Rays` |
| `to` | 204 | `(device) → Paths` |

Property accessors `pos`, `dir`, `intensity`, `id`, `wavelength`, `batch_size` delegate to `_rays`.

---

### 4.2 `bundle.py`

**`rays/bundle.py`**

#### `Bundle(nn.Module)` — line 9

Abstract ray source. Subclasses implement `sample_pos(N)` and/or `sample_dir(N)`.

```
__init__(ray_id, device='cpu', dtype=torch.float32, transform=None)
```

| Method | Line | Notes |
|---|---|---|
| `sample_pos` | 27 | `(N) → Tensor[N,3]` — **Abstract** |
| `sample_dir` | 24 | `(N) → Tensor[N,3]` — default returns `[0,0,1]` repeated |
| `sample` | 30 | `(N) → Rays` — calls `sample_pos` + `sample_dir`, applies `transform` |

---

#### `DiskSample` — line 40

Utility for uniform annular disk sampling.

```
__init__(radius_inner_2, radius_outer_2, theta_min, theta_max)
```

`sample(N)` (line 47): returns `[N, 3]` positions on an annular disk.

---

#### `SolidAngleSample` — line 58

Solid angle distribution helper.

| Method | Line | Notes |
|---|---|---|
| `sample` | 65 | `(N) → (phi, theta)` |
| `invCDF_phi` | 72 | Inverse CDF for azimuth |
| `CDF_phi` | 77 | CDF for azimuth |

---

#### `CollimatedDisk(Bundle)` — line 83

Uniform disk, all rays parallel to +Z.

```
__init__(radius, ray_id, device='cpu', dtype=torch.float32, transform=None)
```

`sample_pos` (line 96): uniform disk sampling.

---

#### `CollimatedLine(Bundle)` — line 101

Line segment, parallel rays.

```
__init__(length, ray_id, device='cpu', dtype=torch.float32, transform=None)
```

`sample_pos` (line 111): uniform line sampling.

---

#### `Fan(Bundle)` — line 121

2D angular fan in the XZ plane.

```
__init__(angle, ray_id, device='cpu', dtype=torch.float32, transform=None)
```

`sample_dir` (line 136): uniform angular distribution over `[-angle/2, angle/2]`.

---

#### `PointSource(Bundle)` — line 143

Diverging cone defined by numerical aperture.

```
__init__(NA, ray_id, device='cpu', dtype=torch.float32, transform=None)
```

`sample_dir` (line 162): samples directions from solid-angle distribution matching `NA`.

---

### 4.3 `beam.py`

**`rays/beam.py`**

#### `GaussianBeam(Bundle)` — line 9

Collimated Gaussian beam.

```
__init__(diameter_1e2_x, diameter_1e2_y, ray_id, device='cpu', dtype=torch.float32, transform=None)
# sigma_x = diameter_1e2_x / 4,  sigma_y = diameter_1e2_y / 4
```

`sample_pos` (line 37): samples from `Normal(0, sigma_x/y)` in X and Y.

---

### 4.4 `particle.py`

**`rays/particle.py`**

#### `LambertianSphere(Bundle)` — line 9

Lambertian emitter on a sphere surface.

```
__init__(radius, ray_id, device='cpu', dtype=torch.float32, transform=None)
```

| Method | Line | Notes |
|---|---|---|
| `sample` | 33 | `(N) → Rays` — samples surface positions + Lambertian directions |
| `_lambertian_around_normals` | 53 | Frisvad ONB construction for cosine-weighted hemisphere |

---

#### `RayleighScatter(Bundle)` — line 86

Rayleigh scattering phase function sampled analytically (Cardano's formula).

`sample_dir` (line 110): returns Rayleigh-distributed directions.

---

#### `MieScatter(Bundle)` — line 127

Mie scattering — constructor stores particle parameters; `sample_dir` raises `NotImplementedError`.

---

### 4.5 `panels.py`

**`rays/panels.py`**

#### `LambertianSample` — line 9

Cosine-weighted hemisphere sampling utility.

`sample(N, device, dtype)` (line 20): returns `[N, 3]` Lambertian directions.

---

#### `PanelSource(Bundle)` — line 33

Base class for flat area lights with Lambertian directional distribution.

`sample_dir` (line 49): delegates to `LambertianSample.sample`.

---

#### `RectangularPanel(PanelSource)` — line 53

```
__init__(width, height, ray_id, device='cpu', dtype=torch.float32, transform=None)
```

`sample_pos` (line 82): uniform rectangle.

---

#### `RingSource(PanelSource)` — line 89

```
__init__(radius_inner, radius_outer, ray_id, device='cpu', dtype=torch.float32, transform=None)
```

`sample_pos` (line 119): uniform annular disk.

---

## 5. `scene/` — Simulation Manager

### 5.1 `base.py`

**`scene/base.py`**

#### `Scene(nn.Module)` — line 7

General non-sequential ray-tracing scene.

```
__init__()
```

| Attribute | Type | Description |
|---|---|---|
| `elements` | `nn.ModuleList` | All optical elements |
| `bundles` | `nn.ModuleList` | All ray sources |
| `_bundle_N_rays` | `list[int]` | Parallel to `bundles`; rays per bundle |
| `rays` | `Rays` or `None` | Current ray batch |
| `Nbounces` | `int` (default 100) | Maximum bounce count |

| Method | Line | Signature | Notes |
|---|---|---|---|
| `add_element` | 24 | `(element: Element)` | Appends to `elements` |
| `add_bundle` | 28 | `(bundle: Bundle, N_rays=200)` | Appends bundle + N |
| `clear_elements` | 37 | `()` | Resets `elements` and index maps |
| `clear_bundles` | 42 | `()` | Resets `bundles`, `_bundle_N_rays`, `rays` |
| `clear_rays` | 48 | `()` | Clears `rays` without removing bundles |
| `_build_rays` | 56 | `()` | Samples all bundles, concatenates to `self.rays` |
| `_build_index_maps` | 96 | `()` | Creates surface index buffers |
| `simulate` | 129 | `()` | Runs bounce loop until `Nbounces` or all rays dead |
| `ray_cast` | 144 | `(rays) → (elem_idx, surf_idx, t)` | Finds closest hit across all elements |
| `step` | 180 | `()` | One bounce: ray_cast + forward on each hit element |
| `compile_elements` | 237 | `()` | Wraps all elements in `torch.compile` |

---

### 5.2 `sequential.py`

**`scene/sequential.py`**

#### `SequentialScene(Scene)` — line 6

Fixed-order sequential traversal. Elements are passed in order at construction time.

```
__init__(elements)   # list of Element objects
```

| Method | Line | Signature | Notes |
|---|---|---|---|
| `simulate` | 12 | `(rays) → rays` | Propagates through all elements sequentially |
| `getParaxial` | 38 | `() → Tensor[5,5]` | Chains element paraxial matrices with `ParaxialDistMat` between them |
| `checkDimensions` | 64 | `()` | Validates physical dimensions (partially implemented) |

---

## 6. `optim/` — Optimization

### 6.1 `goals.py`

**`optim/goals.py`**

#### `FocalLengthLoss(nn.Module)` — line 11

MSE on system power ($1/f$).

```
__init__(f_target)
# Stores P_target = 1/f_target as buffer
```

`forward(scene: SequentialScene)` (line 31): `(P_actual - P_target)²`

---

#### `SpotTargetLoss(nn.Module)` — line 37

Centroid-to-target distance per bundle.

```
__init__(sensor: Sensor, target_xy)
# target_xy: [K, 2] buffer — one target per bundle
```

`forward(scene, bundles: List[Bundle], N_rays=128)` (line 58): returns mean centroid distance across all bundles.

---

#### `SpotSizeLoss(nn.Module)` — line 94

Weighted RMS spot radius across multiple bundles.

```
__init__(sensor: Sensor, bundles: List[Bundle], N_rays=128, target_xy=None)
```

`forward(scene)` (line 139): calls `scene.simulate`, reads sensor hits, returns weighted RMS spot radius.

---

### 6.2 `constraints.py`

**`optim/constraints.py`**

#### Log-barrier helpers — lines 10–22

| Function | Line | Signature | Notes |
|---|---|---|---|
| `_log_barrier_lb` | 10 | `(x, lb) → scalar` | Penalty for $x \leq lb$ |
| `_log_barrier_ub` | 15 | `(x, ub) → scalar` | Penalty for $x \geq ub$ |
| `_log_barrier` | 20 | `(x, lb, ub) → scalar` | Two-sided penalty |

#### `_get_optical_z_list(elem)` — line 29

Returns list of optical surface Z positions for a lens element.

---

#### `ThicknessConstraint(nn.Module)` — line 57

Log-barrier on inter-surface spacing within elements.

```
__init__(elements: List, t_min, t_max=None, weight=1.0)
```

`forward()` (line 84): sums barrier penalties for all adjacent surface pairs.

---

#### `SpacingConstraint(nn.Module)` — line 104

Log-barrier on air gaps between elements.

```
__init__(elements: List, d_min, weight=1.0)
```

`forward()` (line 123): sums barrier penalties for gaps between adjacent elements.

---

#### `SystemLengthConstraint(nn.Module)` — line 138

Log-barrier on total system length.

```
__init__(elements: List, L_max, weight=1.0)
```

`forward()` (line 159): `_log_barrier_ub(total_length, L_max)`.

---

## 7. `render/` — Visualization

### 7.1 `camera.py`

**`render/camera.py`**

#### `Camera` — line 16

Pinhole camera model.

```
__init__(position, look_at, up_vector, fov_deg, width, height, device='cpu')
```

| Attribute | Type | Description |
|---|---|---|
| `origin` | `Tensor[3]` | Camera position |
| `forward` | `Tensor[3]` | View direction |
| `right` | `Tensor[3]` | Right axis |
| `up_cam` | `Tensor[3]` | Camera-up axis |

`generate_rays()` (line 39): returns `Rays` for all `width × height` pixels.

---

#### `OrbitCamera(Camera)` — line 75

CAD-style orbiting camera with pivot point.

```
__init__(pivot=(0,0,0), **kwargs)
```

| Method | Line | Signature | Notes |
|---|---|---|---|
| `update_view_matrix` | 87 | `()` | Recomputes `forward`, `right`, `up_cam` |
| `orbit` | 108 | `(d_yaw, d_pitch)` | Orbits around `pivot` |
| `roll` | 143 | `(angle)` | Rolls around forward axis |
| `pan` | 158 | `(dx, dy)` | Translates pivot and camera |
| `zoom` | 164 | `(delta)` | Moves camera toward/away from pivot |

---

#### `Renderer` — line 172

Single-bounce shaded renderer.

```
__init__(scene, background_color=(1.0,1.0,1.0), light_dir=(-0.5,1.0,-1.0))
```

| Method | Line | Signature | Notes |
|---|---|---|---|
| `render_3d` | 191 | `(camera) → Tensor[H,W,3]` | Single-bounce ray cast + shading |
| `_compute_color` | 259 | `(phys_func, normals) → Tensor[N,3]` | Maps surface type to RGB color |
| `scan_profile` | 320 | `(target_element, axis='x', num_points=200, bounds=None) → (positions, colors)` | 2D cross-section slice |

---

## 8. `gui/` — Desktop Application

PySide6-based interactive GUI. Not intended for programmatic use.

| File | Size | Description |
|---|---|---|
| `workbench.py` | ~32 KB | Main application window; scene setup, simulation controls |
| `forms.py` | ~36 KB | Qt form widgets for element creation and parameter editing |
| `viewport.py` | ~11 KB | 3D rendering viewport; integrates `OrbitCamera` and `Renderer` |
| `project.py` | ~1.4 KB | Project save/load |

---

## Quick-Reference: Surface Index Conventions

| Shape class | Index 0 | Index 1 | Index 2 | Index 3 | Index 4 | Index 5 | Index 6 |
|---|---|---|---|---|---|---|---|
| `Singlet` | Front optical | Back optical | Edge cyl | — | — | — | — |
| `Doublet` | Surf 1 | Surf 2 | Surf 3 | Edge 1 | Edge 2 | — | — |
| `Triplet` | Surf 1 | Surf 2 | Surf 3 | Surf 4 | Edge 1 | Edge 2 | Edge 3 |
| `CylSinglet` | Front optical | Back optical | Right | Left | Top | Bottom | — |

## Quick-Reference: `nn.Parameter` Grad Flags

Every numerical parameter in a geometry or physics class has a corresponding `_grad` boolean constructor argument (e.g., `c1_grad`, `ior_glass_grad`). Set to `True` to include in gradient computation. Gradient masks (`trans_mask`, `rot_mask`) restrict which axes of a parameter carry gradients.
