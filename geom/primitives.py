import torch
from .transform import RayTransform


class Surface:
    """
    Base class for all optical surfaces.
    Implements the dual-method intersection protocol:
    1. intersectTest: Fast, no gradients, returns distance t.
    2. intersect: Detailed, differentiable, returns t, hit_point, and global_normal.
    """

    def __init__(self, transform=None, device='cpu'):
        self.device = device
        if transform is None:
            self.transform = RayTransform(device=device)
        else:
            self.transform = transform

    def to(self, device):
        self.device = device
        # Move any internal tensors (radius, etc.) in child classes
        for attr, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))

        self.transform = self.transform.to(device)
        return self

    def intersectTest(self, rays):
        """
        Performs a lightweight intersection check to find distance t.
        Strictly NO GRADIENTS (detached).

        Args:
            rays (Rays): Global ray batch.
        Returns:
            t (Tensor): [N] Distance to intersection.
        """

        # 1. Transform Global -> Local
        local_pos, local_dir = self.transform.invTransform(rays)

        # 2. Solve only for t
        t = self._solve_t(local_pos, local_dir)

        return t

    def intersect(self, rays):
        """
        Performs a full intersection calculation tracking gradients.

        Args:
            rays (Rays): Global ray batch.
        Returns:
            t (Tensor): [N] Intersection distance.
            hit_point (Tensor): [N, 3] Global coordinates of intersection.
            global_normal (Tensor): [N, 3] Surface normal at hit point (Global frame).
        """
        # 1. Transform Global -> Local
        local_pos, local_dir = self.transform.invTransform(rays)

        # 2. Solve for t and Local Normal (Differentiable)
        t, local_normal = self._solve_geometric_properties(local_pos, local_dir)

        # 3. Compute Global Hit Point
        # P_hit = P_origin + t * Direction
        hit_point = rays.pos + t.unsqueeze(1) * rays.dir

        # 4. Transform Normal Local -> Global
        # Normals are direction vectors, so we apply the rotation.
        # Consistent with transform.py: D_global = D_local @ R.T
        global_normal = local_normal @ self.transform.rot.T

        return t, hit_point, global_normal

    # --- Abstract Methods ---

    def _solve_t(self, local_pos, local_dir):
        """Child classes must implement the math to find t."""
        raise NotImplementedError

    def _solve_geometric_properties(self, local_pos, local_dir):
        """Child classes must implement the math to find t and local_normal."""
        raise NotImplementedError


class Plane(Surface):
    """
    An infinite flat plane.
    Canonical definition: The XY plane (z=0) facing +Z.
    """
    def _solve_t(self, local_pos, local_dir):
        # Ray equation: P(t) = O + t*D
        # Plane equation: z = 0
        # Substitute: O_z + t*D_z = 0  =>  t = -O_z / D_z

        oz = local_pos[:, 2]
        dz = local_dir[:, 2]

        # Epsilon for stability (avoid div by zero for parallel rays)
        epsilon = 1e-8
        safe_dz = torch.where(torch.abs(dz) < epsilon, torch.tensor(epsilon, device=self.device), dz)

        t = -oz / safe_dz
        return t

    def _solve_geometric_properties(self, local_pos, local_dir):
        # 1. Re-calculate t (Differentiable)
        # We duplicate the code to ensure the graph is constructed from scratch
        # rather than trying to re-attach the no_grad result.
        oz = local_pos[:, 2]
        dz = local_dir[:, 2]

        epsilon = 1e-8
        safe_dz = torch.where(torch.abs(dz) < epsilon, torch.tensor(epsilon, device=self.device), dz)
        t = -oz / safe_dz

        # 2. Local Normal
        # For a standard plane z=0, the normal is always (0, 0, 1).
        zeros = torch.zeros_like(oz)
        ones = torch.ones_like(oz)
        local_normal = torch.stack([zeros, zeros, ones], dim=1)

        return t, local_normal


class Sphere(Surface):
    """
    A sphere centered at (0,0,0) in Local Space.
    Radius is defined by curvature (c = 1/R).
    Using curvature avoids infinity issues for flat surfaces (R=inf),
    though for a pure Sphere class, R is often more intuitive.
    Here we stick to Radius R as a parameter for explicit Spheres.
    """

    def __init__(self, radius, transform=None, device='cpu'):
        super().__init__(transform, device)
        self.radius = torch.tensor(radius, dtype=torch.float32, device=device)

    def _solve_t(self, local_pos, local_dir):
        # Ray-Sphere Intersection: |O + tD|^2 = R^2
        # (O+tD).(O+tD) = R^2
        # O.O + 2t(O.D) + t^2(D.D) - R^2 = 0
        # a*t^2 + b*t + c = 0

        # a = 1.0 (since D is normalized)
        # b = 2 * (O . D)
        # c = O.O - R^2

        # Dot products [N]
        # (N,3) * (N,3) -> sum -> (N)
        b = 2.0 * torch.sum(local_pos * local_dir, dim=1)
        c = torch.sum(local_pos * local_pos, dim=1) - self.radius ** 2

        discriminant = b ** 2 - 4.0 * c

        # Initialize t as infinity (miss)
        t = torch.full_like(b, float('inf'))

        # Mask for valid hits (discriminant >= 0)
        hit_mask = discriminant >= 0

        if hit_mask.any():
            sqrt_delta = torch.sqrt(discriminant[hit_mask])
            b_valid = b[hit_mask]

            # Two solutions
            t1 = (-b_valid - sqrt_delta) / 2.0
            t2 = (-b_valid + sqrt_delta) / 2.0

            # Logic to find smallest POSITIVE t
            # If t1 > epsilon, pick t1. Else if t2 > epsilon, pick t2.
            # (Standard "ray trace" logic usually implies moving forward)
            epsilon = 1e-4

            # Create a localized t for valid rays
            t_sol = torch.full_like(b_valid, float('inf'))

            # Case 1: Both positive, t1 is smaller
            # Case 2: t1 negative (inside sphere looking out?), t2 positive

            # Simple vector check:
            # We prefer t1 if t1 > eps.
            # Otherwise we take t2 if t2 > eps.

            mask_t1 = t1 > epsilon
            t_sol[mask_t1] = t1[mask_t1]

            # Where t1 was invalid, check t2
            mask_t2 = (~mask_t1) & (t2 > epsilon)
            t_sol[mask_t2] = t2[mask_t2]

            t[hit_mask] = t_sol

        return t

    def _solve_geometric_properties(self, local_pos, local_dir):
        # 1. Re-calculate t differentiably
        b = 2.0 * torch.sum(local_pos * local_dir, dim=1)
        c = torch.sum(local_pos ** 2, dim=1) - self.radius ** 2
        delta = b ** 2 - 4.0 * c

        # We assume hit validity was checked by the caller using intersectTest ideally,
        # or we just handle the math. If delta < 0, sqrt(delta) is NaN -> gradients break.
        # We clamp delta to 0 to avoid NaNs in the gradient pass for missing rays,
        # realizing those rays should be marked "inactive" by the Scene manager anyway.
        delta_safe = torch.relu(delta)
        sqrt_delta = torch.sqrt(delta_safe)

        t1 = (-b - sqrt_delta) / 2.0
        t2 = (-b + sqrt_delta) / 2.0

        # Selection logic (differentiable approx using torch.where)
        epsilon = 1e-4

        # Prefer t1 if t1 > epsilon, else t2
        # Note: If both are negative (behind ray), we return t1 (negative)
        # but the ray is effectively invalid.
        t = torch.where(t1 > epsilon, t1, t2)

        # 2. Local Normal
        # For a sphere, Normal = (Hit_Point - Center) / Radius
        # Center is (0,0,0)
        hit_point_local = local_pos + t.unsqueeze(1) * local_dir

        # outward facing normal
        local_normal = hit_point_local / self.radius

        # In concave cases (inside sphere), we might want the normal pointing inward.
        # However, standard convention is surface normal always points "out" of the solid.
        # Snell's law derivation in phys/ will handle the dot(ray, normal) sign check.

        return t, local_normal


class Cylinder(Surface):
    """
    An infinite cylinder aligned along the local Z-axis.
    Equation: x^2 + y^2 = R^2
    """

    def __init__(self, radius, transform=None, device='cpu'):
        super().__init__(transform, device)
        self.radius = torch.tensor(radius, dtype=torch.float32, device=device)

    def _solve_t(self, local_pos, local_dir):
        # Ray: P(t) = O + tD
        # Substitute into x^2 + y^2 = R^2
        # (Ox + tDx)^2 + (Oy + tDy)^2 = R^2
        # Expand:
        # (Dx^2 + Dy^2)t^2 + 2(OxDx + OyDy)t + (Ox^2 + Oy^2 - R^2) = 0
        # A*t^2 + B*t + C = 0

        ox, oy = local_pos[:, 0], local_pos[:, 1]
        dx, dy = local_dir[:, 0], local_dir[:, 1]

        A = dx ** 2 + dy ** 2
        B = 2.0 * (ox * dx + oy * dy)
        C = (ox ** 2 + oy ** 2) - self.radius ** 2

        discriminant = B ** 2 - 4.0 * A * C

        # Initialize t as infinity (miss)
        t = torch.full_like(A, float('inf'))

        # 1. Mask for valid hits (real roots)
        hit_mask = discriminant >= 0

        if hit_mask.any():
            sqrt_delta = torch.sqrt(discriminant[hit_mask])
            A_valid = A[hit_mask]
            B_valid = B[hit_mask]

            # Solve quadratic
            t1 = (-B_valid - sqrt_delta) / (2.0 * A_valid)
            t2 = (-B_valid + sqrt_delta) / (2.0 * A_valid)

            # Selection Logic: Smallest POSITIVE t
            epsilon = 1e-4

            # Temporary holder for solutions on the masked subset
            t_subset = torch.full_like(t1, float('inf'))

            # Check t1
            mask_t1 = t1 > epsilon
            t_subset[mask_t1] = t1[mask_t1]

            # Check t2 (if t1 invalid OR t2 better)
            mask_t2 = (t2 > epsilon) & ((~mask_t1) | (t2 < t_subset))
            t_subset[mask_t2] = t2[mask_t2]

            t[hit_mask] = t_subset

        return t

    def _solve_geometric_properties(self, local_pos, local_dir):
        # 1. Re-calculate A, B, C for gradients
        ox, oy = local_pos[:, 0], local_pos[:, 1]
        dx, dy = local_dir[:, 0], local_dir[:, 1]

        A = dx ** 2 + dy ** 2
        B = 2.0 * (ox * dx + oy * dy)
        C = (ox ** 2 + oy ** 2) - self.radius ** 2

        discriminant = B ** 2 - 4.0 * A * C

        # Safe sqrt for gradients
        delta_safe = torch.relu(discriminant)
        sqrt_delta = torch.sqrt(delta_safe)

        # 2. Re-calculate roots
        # Add epsilon to A to avoid div/0 for rays perfectly parallel to Z-axis
        A_safe = torch.where(A.abs() < 1e-8, torch.tensor(1e-8, device=self.device), A)

        t1 = (-B - sqrt_delta) / (2.0 * A_safe)
        t2 = (-B + sqrt_delta) / (2.0 * A_safe)

        # 3. Selection Logic (Differentiable)
        epsilon = 1e-4

        # Default to inf
        t_final = torch.full_like(t1, float('inf'))

        # t1 valid?
        mask_t1 = t1 > epsilon
        t_final = torch.where(mask_t1, t1, t_final)

        # t2 valid and better?
        mask_t2 = (t2 > epsilon) & ((~mask_t1) | (t2 < t_final))
        t_final = torch.where(mask_t2, t2, t_final)

        # Mask out misses (discriminant < 0)
        hit_mask = discriminant >= 0
        t = torch.where(hit_mask, t_final, torch.full_like(t_final, float('inf')))

        # 4. Local Normal
        # For cylinder along Z: Normal is (x/R, y/R, 0)
        # We need the hit point in local coordinates
        hit_point = local_pos + t.unsqueeze(1) * local_dir

        nx = hit_point[:, 0] / self.radius
        ny = hit_point[:, 1] / self.radius
        nz = torch.zeros_like(nx)

        local_normal = torch.stack([nx, ny, nz], dim=1)

        return t, local_normal

class Quadric(Surface):
    """
    A General Quadric Surface (Conic Section of Revolution).
    Aligned with the optical axis Z.

    Equation:
    z = (c * r^2) / (1 + sqrt(1 - (1+k) * c^2 * r^2))

    Implicit form for Intersection:
    c(x^2 + y^2) + c(1+k)z^2 - 2z = 0

    Parameters:
        c (float): Curvature (1/Radius). c=0 implies a plane.
        k (float): Conic constant.
                   k=0 (Sphere), k=-1 (Parabola), k<-1 (Hyperbola).
    """

    def __init__(self, c, k=0.0, transform=None, device='cpu'):
        super().__init__(transform, device)
        self.c = c
        self.k = k

    def _get_coeffs(self, local_pos, local_dir):
        """
        Substitutes Ray P = O + tD into Quadric equation to get quadratic coeffs:
        A*t^2 + B*t + C = 0
        """
        ox, oy, oz = local_pos[:, 0], local_pos[:, 1], local_pos[:, 2]
        dx, dy, dz = local_dir[:, 0], local_dir[:, 1], local_dir[:, 2]

        # Derived coefficients from substituting (O+tD) into c(x^2+y^2) + c(1+k)z^2 - 2z = 0
        # Common term: c
        c = self.c
        k = self.k

        # A = c(Dx^2 + Dy^2) + c(1+k)Dz^2
        A = c * (dx ** 2 + dy ** 2) + c * (1 + k) * dz ** 2

        # B = 2c(OxDx + OyDy) + 2c(1+k)OzDz - 2Dz
        B = 2 * c * (ox * dx + oy * dy) + 2 * c * (1 + k) * oz * dz - 2 * dz

        # C = c(Ox^2 + Oy^2) + c(1+k)Oz^2 - 2Oz
        C = c * (ox ** 2 + oy ** 2) + c * (1 + k) * oz ** 2 - 2 * oz

        return A, B, C

    def _solve_quadratic(self, A, B, C):
        """
        Solves At^2 + Bt + C = 0 differentiably.
        Correctly handles misses (discriminant < 0).
        """
        discriminant = B ** 2 - 4 * A * C

        # 1. Create a mask for valid intersections (Real roots exist)
        # Rays with discriminant < 0 effectively miss the surface.
        hit_mask = discriminant >= 0

        t = torch.full_like(discriminant, float('inf'))

        if hit_mask.any():
            # 2. Safe sqrt for gradient stability
            # We still use relu to ensure the sqrt calculation doesn't produce NaNs
            # for the 'False' entries in hit_mask during the backward pass.

            disc_valid = discriminant[hit_mask]

            A = A[hit_mask]
            B = B[hit_mask]
            C = C[hit_mask]

            t_sol = torch.zeros_like(disc_valid)
            sqrt_delta = torch.sqrt(disc_valid)

            # Standard Quadratic Formula
            t1 = (-B - sqrt_delta) / (2 * A)
            t2 = (-B + sqrt_delta) / (2 * A)

            # 3. Handle Linear Fallback (A ~ 0)
            epsilon_a = 1e-7
            mask_linear = torch.abs(A) < epsilon_a

            # Avoid div/0 for linear case
            safe_B = torch.where(torch.abs(B) < 1e-8, torch.tensor(1e-8, device=self.device), B)
            t_linear = -C / safe_B

            # If A is effectively zero, use the linear solution
            t1 = torch.where(mask_linear, t_linear, t1)
            t2 = torch.where(mask_linear, t_linear, t2)

            # 4. Selection Logic (Smallest Positive t)
            epsilon_t = 1e-5

            # Check t1 validity (must be > epsilon)
            mask_t1_valid = t1 > epsilon_t
            t_sol = torch.where(mask_t1_valid, t1, t_sol)

            # Check t2 validity:
            # Accept t2 if it is positive AND (t1 was invalid OR t2 is closer than t1)
            mask_t2_valid = t2 > epsilon_t
            mask_t2_better = mask_t2_valid & ((~mask_t1_valid) | (t2 < t_sol))
            t_sol = torch.where(mask_t2_better, t2, t_sol)

            # 5. FINAL MASK: Apply the discriminant check
            # If the ray mathematically missed (discriminant < 0), force t to infinity.

            t[hit_mask] = t_sol

        return t

    def _solve_t(self, local_pos, local_dir):
        A, B, C = self._get_coeffs(local_pos, local_dir)

        # Detached calculation for speed
        with torch.no_grad():
            t = self._solve_quadratic(A, B, C)
        return t

    def _solve_geometric_properties(self, local_pos, local_dir):
        # 1. Re-calculate A, B, C (Graph attached)
        A, B, C = self._get_coeffs(local_pos, local_dir)

        # 2. Re-calculate t
        t = self._solve_quadratic(A, B, C)

        # 3. Compute Local Hit Point
        hit_point = local_pos + t.unsqueeze(1) * local_dir
        hx, hy, hz = hit_point[:, 0], hit_point[:, 1], hit_point[:, 2]

        # 4. Compute Local Normal
        # Gradient of Implicit F(x,y,z) = c(x^2+y^2) + c(1+k)z^2 - 2z
        # Nx = 2cx
        # Ny = 2cy
        # Nz = 2c(1+k)z - 2

        nx = 2 * self.c * hx
        ny = 2 * self.c * hy
        nz = 2 * self.c * (1 + self.k) * hz - 2.0

        raw_normal = torch.stack([nx, ny, nz], dim=1)

        # Normalize
        # Avoid div zero if normal is zero vector (shouldn't happen on valid surface)
        norm_len = torch.norm(raw_normal, dim=1, keepdim=True)
        local_normal = raw_normal / (norm_len + 1e-8)

        # Note: The gradient of F points in the -Z direction at the vertex.
        # F = ... - 2z. dF/dz = -2.
        # For a standard surface facing +Z (ray incoming from -Z),
        # we usually want the normal pointing against the incoming ray (towards -Z).
        # Our gradient gives (0,0,-1) at origin. This is correct for reflection math.
        # If your physics engine expects normals pointing OUT (+Z), flip this sign.
        # Based on your Plane class ((0,0,1)), you likely want normals pointing +Z.
        # If so, negate the normal:

        local_normal = -local_normal

        return t, local_normal