import torch
import torch.nn as nn
from .transform import RayTransform


class Surface(nn.Module):
    """
    Base class for all optical surfaces.
    Implements the dual-method intersection protocol:
    1. intersectTest: Fast, no gradients, returns distance t.
    2. intersect: Detailed, differentiable, returns t, hit_point, and global_normal.
    """

    def __init__(self, transform=None):

        super().__init__()

        if transform is None:
            self.transform = RayTransform()
        else:
            self.transform = transform

    def _check_t(self, t_list, *args):

        t_stack = torch.stack(t_list)

        t_stack = t_stack.masked_fill(t_stack <=0, float('inf'))

        t_min, _ = torch.min(t_stack, dim=0)

        return t_min

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
        local_pos, local_dir = self.transform.transform(rays)

        # 2. Solve only for t
        t_list = self._solve_t(local_pos, local_dir)

        t = self._check_t(t_list, local_pos, local_dir)

        return t

    def forward(self, rays, *args):
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
        local_pos, local_dir = self.transform.transform(rays)

        # 2. Solve only for t
        t_list = self._solve_t(local_pos, local_dir)

        t = self._check_t(t_list, local_pos, local_dir)

        # 3. Compute Global Hit Point
        # P_hit = P_origin + t * Direction
        hit_global = rays.pos + t.unsqueeze(1) * rays.dir
        hit_local = rays.pos + t.unsqueeze(1) * local_dir

        # 4
        normal_local = self._getNormal(hit_local)

        # 5. Transform Normal Local -> Global
        # Normals are direction vectors, so we apply the rotation.
        # Consistent with transform.py: D_global = D_local @ R.T
        normal_global = normal_local @ self.transform.rot.T

        return t, hit_global, normal_global

    # --- Abstract Methods ---

    def _getNormal(self, local_pos):

        raise NotImplementedError

    def _solve_t(self, local_pos, local_dir):
        """Child classes must implement the math to find t."""
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

        safe_dz = torch.where(torch.abs(dz) < epsilon, 1e-8, dz)

        t = -oz / safe_dz

        return [t]

    def _getNormal(self, local_pos):

        local_normal = torch.zeros_like(local_pos)
        local_normal[:, 2] = torch.ones_like(local_pos[:, 2])

        return local_normal


class Sphere(Surface):
    """
    A sphere centered at (0,0,0) in Local Space.
    Radius is defined by curvature (c = 1/R).
    Using curvature avoids infinity issues for flat surfaces (R=inf),
    though for a pure Sphere class, R is often more intuitive.
    Here we stick to Radius R as a parameter for explicit Spheres.
    """
    def __init__(self, radius, radius_grad = False, transform=None):
        super().__init__(transform=transform)
        self.radius = torch.nn.Parameter(torch.tensor(radius), requires_grad=radius_grad)

    def _solve_t(self, local_pos, local_dir):
        # |O + td|^2 = R^2
        # (d.d)t^2 + 2(O.d)t + (O.O - R^2) = 0
        # d is normalized, so a = 1

        # a = 1.0
        b = 2.0 * torch.sum(local_pos * local_dir, dim=1)
        c = torch.sum(local_pos * local_pos, dim=1) - self.radius ** 2

        discriminant = b ** 2 - 4 * c

        # Shape [N]
        # Roots: (-b +/- sqrt(D)) / 2

        hit_mask = discriminant >= 0
        sqrt_delta = torch.sqrt(torch.where(hit_mask, discriminant, torch.zeros_like(discriminant)))

        t1 = (-b - sqrt_delta) / 2.0
        t2 = (-b + sqrt_delta) / 2.0

        # Mark complex roots as Inf
        inf = float('inf')
        t1 = torch.where(hit_mask, t1, torch.full_like(t1, inf))
        t2 = torch.where(hit_mask, t2, torch.full_like(t2, inf))

        # Stack [N, 2]
        return [t1, t2]

    def _getNormal(self, local_pos):

        local_normal = local_pos / self.radius

        return local_normal


class Cylinder(Surface):
    """
    An infinite cylinder aligned along the local Z-axis.
    Equation: x^2 + y^2 = R^2
    """

    def __init__(self, radius, transform=None,
                 radius_grad = False):
        super().__init__(transform)
        self.radius = nn.Parameter(torch.tensor(radius), requires_grad=radius_grad)

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

        hit_mask = discriminant >= 0
        sqrt_delta = torch.sqrt(torch.abs(discriminant))

        t1 = (-B - sqrt_delta) / (2.0 * A)
        t2 = (-B + sqrt_delta) / (2.0 * A)

        inf = float('inf')
        t1 = torch.where(hit_mask, t1, torch.full_like(t1, inf))
        t2 = torch.where(hit_mask, t2, torch.full_like(t2, inf))

        return [t1, t2]

    def _getNormal(self, local_pos):

        nx = local_pos[:, 0] / self.radius
        ny = local_pos[:, 1] / self.radius
        nz = torch.zeros_like(nx)

        local_normal = torch.stack([nx, ny, nz], dim=1)

        return local_normal


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

    def __init__(self, c, k, transform=None,
                 c_grad = False, k_grad = False):

        super().__init__(transform = transform)

        self.c = nn.Parameter(torch.tensor(c), requires_grad=c_grad)
        self.k = nn.Parameter(torch.tensor(k), requires_grad=k_grad)

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
        sqrt_delta = torch.sqrt(torch.abs(discriminant))

        t1 = (-B - sqrt_delta) / (2.0 * A)
        t2 = (-B + sqrt_delta) / (2.0 * A)

        inf = float('inf')
        t1 = torch.where(hit_mask, t1, torch.full_like(t1, inf))
        t2 = torch.where(hit_mask, t2, torch.full_like(t2, inf))

        epsilon_a = 1e-7
        mask_linear = torch.abs(A) < epsilon_a

        # Avoid div/0 for linear case
        safe_B = torch.where(torch.abs(B) < 1e-8, torch.tensor(1e-8, device=self.device), B)
        t_linear = -C / safe_B

        # If A is effectively zero, use the linear solution
        t1 = torch.where(mask_linear, t_linear, t1)
        t2 = torch.where(mask_linear, t_linear, t2)

        return [t1, t2]

    def _solve_t(self, local_pos, local_dir):

        A, B, C = self._get_coeffs(local_pos, local_dir)

        t_list = self._solve_quadratic(A, B, C)

        return t_list

    def _getNormal(self, local_pos):

        nx = 2 * self.c * local_pos[:, 0]
        ny = 2 * self.c * local_pos[:, 1]
        nz = 2 * self.c * (1 + self.k) * local_pos[:, 2] - 2.0

        raw_normal = torch.stack([nx, ny, nz], dim=1)

        # Normalize
        # Avoid div zero if normal is zero vector (shouldn't happen on valid surface)
        norm_len = torch.norm(raw_normal, dim=1, keepdim=True)
        local_normal = raw_normal / (norm_len + 1e-8)

        return -local_normal

