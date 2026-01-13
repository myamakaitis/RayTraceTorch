
import torch
from .transform import RayTransform
from .primitives import Surface, Plane, Sphere, Quadric, QuadricZY

class SurfaceBounded(Surface):
    """
    Base class for bounded surfaces
    """

    def __init__(self, transform=None, invert=False):

        super().__init__(transform=transform)
        self.surface = None
        self.invert = invert

    def _check_t(self, t_list, local_pos, local_dir):

        t_stack = torch.stack(t_list)

        hits_local = local_pos[None, :, :] + t_stack[:, :, None] * local_dir[None, :, :]
        M, N, _ = hits_local.shape

        keep = self.inBounds(hits_local.view(-1, 3)).view(M, N)

        if self.invert:
            keep = ~keep

        t_stack = t_stack.masked_fill((t_stack <= 0) | ~keep , float('inf'))

        t_min, _ = torch.min(t_stack, dim=0)

        return t_min


    def inBounds(self, local_pos):
        """
        Determines if points lie within the 2D boundary.

        Args:
            local_pos (Tensor): [N, 3] Points in local surface coordinates.
                                Z-component is typically ignored.
        Returns:
            mask (Tensor): [N] Bool tensor (True = Inside, False = Blocked).
        """
        raise NotImplementedError

class Disk(Plane, SurfaceBounded):
    """
    Circular aperture defined by a radius.
    """

    def __init__(self, radius, transform = None):
        super().__init__()
        self.radius = torch.tensor(radius, dtype=torch.float32)

    def inBounds(self, local_pos):
        # r^2 = x^2 + y^2
        # Check: r^2 <= R^2
        xy_sq = local_pos[:, 0] ** 2 + local_pos[:, 1] ** 2
        return xy_sq <= self.radius ** 2


class Rectangle(Plane, SurfaceBounded):
    """
    Rectangular aperture defined by half-widths in X and Y.
    """

    def __init__(self, half_x, half_y):
        super().__init__()
        self.hx = nn.Parameter(torch.as_tensor(half_x, dtype=torch.float32))
        self.hy = nn.Parameter(torch.as_tensor(half_y, dtype=torch.float32))

    def inBounds(self, local_pos):
        # |x| <= hx  AND  |y| <= hy
        abs_x = torch.abs(local_pos[:, 0])
        abs_y = torch.abs(local_pos[:, 1])

        return (abs_x <= self.hx) & (abs_y <= self.hy)


class Ellipse(Plane, SurfaceBounded):
    """
    Elliptical aperture defined by semi-axes X and Y.
    """

    def __init__(self, r_major, r_minor, rot,
                       r_major_grad = False, r_minor_grad = False, rot_grad = False):
        super().__init__()
        self.r_minor = nn.Parameter(torch.as_tensor(r_minor), requires_grad=r_minor_grad)
        self.r_major = nn.Parameter(torch.as_tensor(r_major), requires_grad=r_major_grad)
        self.rot = nn.Parameter(torch.as_tensor(rot), requires_grad=rot_grad)

    def inBounds(self, local_pos):
        # (x/rx)^2 + (y/ry)^2 <= 1

        cos_rot, sin_rot = torch.cos(self.rot), torch.sin(self.rot)

        dir_maj = local_pos[:, 0] * cos_rot - local_pos[:, 1] * sin_rot
        dir_minor = local_pos[:, 0] * sin_rot + local_pos[:, 1] * cos_rot

        return ((dir_major/self.r_major)**2 + (dir_minor/r_minor)**2) <= 1.0


class HalfSphere(Quadric, SurfaceBounded):
    """
    A Sphere clipped to a hemisphere.
    Logic: The valid surface is the one where the local Z coordinate
    has the OPPOSITE sign of the Radius.

    Example:
    R > 0 (Convex Front): Center is to Right. Valid surface is Left of Center (Z < 0).
    R < 0 (Convex Back): Center is to Left. Valid surface is Right of Center (Z > 0).
    """

    def __init__(self, curvature, curvature_grad, transform=None):
        super().__init__(c = curvature, c_grad=curvature_grad, k = 0.0, k_grad=False, transform=transform)

    def inBounds(self, local_pos):
        # Check: sign(z) != sign(R)
        # Equivalent to: z * R < 0
        z = local_pos[:, 2]
        return torch.abs(z * self.c) < 1.001

    def sagittalZ(self, radius):
        """Calculates Z-coordinate of the surface edge relative to vertex Z."""
        r_sq = (radius / 2.0) ** 2

        # Sag equation for vertex formulation
        term = 1.0 - self.c ** 2 * r_sq
        term = torch.relu(term)
        denom = 1.0 + torch.sqrt(term)

        sag = (self.c * r_sq) / denom
        return sag + self.transform.trans[2]


class HalfCyl(QuadricZY, SurfaceBounded):
    """
    A Cylindrical surface clipped to the valid hemisphere (relative to curvature).
    """

    def __init__(self, curvature, curvature_grad, transform=None):
        # Initialize as QuadricZY with k=0 (Cylinder)
        super().__init__(c=curvature, c_grad=curvature_grad, k=0.0, k_grad=False, transform=transform)

    def inBounds(self, local_pos):
        # Check validity similar to HalfSphere: sign(z) != sign(R) => z*c < 1 (approx)
        z = local_pos[:, 2]
        return torch.abs(z * self.c) < 1.001

    def sagittalZ(self, y_height):
        """Calculates Z-coordinate of the surface at a specific Y height."""
        # Standard sag equation applied to y
        r_sq = y_height ** 2

        term = 1.0 - self.c ** 2 * r_sq
        term = torch.relu(term)
        denom = 1.0 + torch.sqrt(term)

        sag = (self.c * r_sq) / denom
        return sag + self.transform.trans[2]