from mpmath import inverse

from .transform import RayTransform
from .primitives import Plane

class Shape2D(Plane):
    """
    Base class for 2D finite boundaries (apertures).
    Operates on the XY components of Local Space coordinates.
    """

    def __init__(self, device='cpu', transform=None, mode=inverse):

        super().__init__(transform, device=device)
        self.device = device
        self.surface = None

    def intersectTest(self, rays):
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
        local_pos, local_dir = self.transform.Transform(rays)

        # 2. Solve for t and Local Normal (Differentiable)
        t = self._solve_t(local_pos, local_dir)

        local_hit = local_pos + t * local_dir

        inside = self.inside(local_hit)

        if inverse:
            # points outside of the region are intersected
            t = torch.where(inside, t, torch.full_like(t, torch.inf))
        else:
            t = torch.where(torch.inverse(inside), t, torch.full_like(t, torch.inf))

        return

    def inside(self, local_pos):
        """
        Determines if points lie within the 2D boundary.

        Args:
            local_pos (Tensor): [N, 3] Points in local surface coordinates.
                                Z-component is typically ignored.
        Returns:
            mask (Tensor): [N] Bool tensor (True = Inside, False = Blocked).
        """
        raise NotImplementedError

class Disk(Shape2D):
    """
    Circular aperture defined by a radius.
    """

    def __init__(self, radius, transform = None, device='cpu', ):
        super().__init__(device)
        self.radius = torch.tensor(radius, dtype=torch.float32, device=device)

    def inside(self, local_pos):
        # r^2 = x^2 + y^2
        # Check: r^2 <= R^2
        xy_sq = local_pos[:, 0] ** 2 + local_pos[:, 1] ** 2
        return xy_sq <= self.radius ** 2


class Rectangle(Shape2D):
    """
    Rectangular aperture defined by half-widths in X and Y.
    """

    def __init__(self, half_x, half_y, device='cpu'):
        super().__init__(device)
        self.hx = torch.tensor(half_x, dtype=torch.float32, device=device)
        self.hy = torch.tensor(half_y, dtype=torch.float32, device=device)

    def inside(self, local_pos):
        # |x| <= hx  AND  |y| <= hy
        abs_x = torch.abs(local_pos[:, 0])
        abs_y = torch.abs(local_pos[:, 1])

        return (abs_x <= self.hx) & (abs_y <= self.hy)


class Ellipse(Shape2D):
    """
    Elliptical aperture defined by semi-axes X and Y.
    """

    def __init__(self, r_major, r_minor, rot, device='cpu'):
        super().__init__(device)
        self.r_minor = r_minor.to(device)
        self.r_major = r_major.to(device)
        self.rot = rot.to(device)

    def inside(self, local_pos):
        # (x/rx)^2 + (y/ry)^2 <= 1

        cos_rot, sin_rot = torch.cos(self.rot), torch.sin(self.rot)

        dir_maj = local_pos[:, 0] * cos_rot - local_pos[:, 1] * sin_rot
        dir_minor = local_pos[:, 0] * sin_rot + local_pos[:, 1] * cos_rot

        return ((dir_major/self.r_major)**2 + (dir_minor/r_minor)**2) <= 1.0