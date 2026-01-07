import torch
import math
from .shape import Shape
from .primitives import Cylinder, Plane
from .bounded import HalfSphere
from .transform import RayTransform


class Singlet(Shape):
    """
    A 3D Singlet Lens defined by two optical surfaces (Sphere or Plane) and a cylindrical edge.

    Standard Cartesian Sign Convention:
    - Light propagates in +Z direction.
    - Lens is centered at (0,0,0) in Local space.
    - Front Vertex (V1) at z = -T/2.
    - Back Vertex (V2) at z = +T/2.
    - R > 0: Center of curvature is to the Right (+Z) of the vertex.
    - R < 0: Center of curvature is to the Left (-Z) of the vertex.
    - R = Inf: Surface is a flat plane.
    """

    def __init__(self,
                 R1: torch.Tensor,
                 R2: torch.Tensor,
                 D: torch.Tensor,
                 T: torch.Tensor,
                 transform=None,
                 device='cpu'):
        """
        Args:
            R1 (Tensor): Radius of curvature of front surface (Inf for Plane).
            R2 (Tensor): Radius of curvature of back surface (Inf for Plane).
            D (Tensor): Diameter of the lens.
            T (Tensor): Center thickness.
        """
        super().__init__(transform, device)

        self.R1 = R1.to(device)
        self.R2 = R2.to(device)
        self.D = D.to(device)
        self.T = T.to(device)
        self.half_D = self.D / 2.0

        # Validation
        with torch.no_grad():
            if not torch.isinf(self.R1) and abs(2 * self.R1.item()) < self.D.item():
                raise ValueError(f"|R1| must be larger than D/2")
            if not torch.isinf(self.R2) and abs(2 * self.R2.item()) < self.D.item():
                raise ValueError(f"|R2| must be larger than D/2")
            if self.T.item() <= 1e-6:
                raise ValueError("Thickness T must be positive")

        # Vertices
        z_v1 = -self.T / 2.0
        z_v2 = self.T / 2.0

        # Front Surface (V1)
        if torch.isinf(self.R1):
            rot_matrix = torch.tensor([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]], device=device)
            t1 = RayTransform(translation=torch.tensor([0., 0., z_v1], device=device),
                              rotation=rot_matrix, device=device)
            self.surf1 = Plane(transform=t1, device=device)
            self.z_edge_front = z_v1
        else:
            c1 = z_v1 + self.R1
            t1 = RayTransform(translation=torch.tensor([0., 0., c1], device=device), device=device)
            self.surf1 = HalfSphere(torch.abs(self.R1), transform=t1, device=device)
            self.z_edge_front = self._get_sag_z(c1, self.R1, self.D)

        # Back Surface (V2)
        if torch.isinf(self.R2):
            t2 = RayTransform(translation=torch.tensor([0., 0., z_v2], device=device), device=device)
            self.surf2 = Plane(transform=t2, device=device)
            self.z_edge_back = z_v2
        else:
            c2 = z_v2 + self.R2
            t2 = RayTransform(translation=torch.tensor([0., 0., c2], device=device), device=device)
            self.surf2 = HalfSphere(torch.abs(self.R2), transform=t2, device=device)
            self.z_edge_back = self._get_sag_z(c2, self.R2, self.D)

        # Check Edge Thickness
        with torch.no_grad():
            if self.z_edge_front >= self.z_edge_back:
                raise ValueError(f"Shape has Zero or Negative Edge Thickness.")

        # Edge (Cylinder)
        self.edge = Cylinder(self.half_D, device=device)

        # Order MUST be [Front, Back, Edge] for inside() logic
        self.surfaces = [self.surf1, self.surf2, self.edge]

    def _get_sag_z(self, c, R, diameter):
        r_sq = (diameter / 2.0) ** 2
        root = torch.sqrt(R ** 2 - r_sq)
        sign_R = torch.sign(R)
        return c - sign_R * root

    def inBounds(self, local_pos, surf_idx=None):
        """
        Validates hits based on physical boundaries.
        surf_idx: 0 (Front), 1 (Back), 2 (Edge)
        """
        z = local_pos[:, 2]
        if surf_idx == 2:
            # Edge Hit: Must be between the Z-planes of the lens edge
            return (z >= self.z_edge_front) & (z <= self.z_edge_back)
        else:
            # Face Hit: Must be within the Aperture Diameter
            # r^2 <= (D/2)^2
            r_sq = local_pos[:, 0] ** 2 + local_pos[:, 1] ** 2
            in_aperture = r_sq <= self.half_D ** 2

            return in_aperture

class CylindricalSinglet(Singlet):
    """
    A 3D Singlet Lens with Cylindrical faces aligned along the X-axis.
    Inherits volume validation logic (inside) from Singlet.
    """

    def __init__(self,
                 R1: torch.Tensor,
                 R2: torch.Tensor,
                 D: torch.Tensor,
                 T: torch.Tensor,
                 transform=None,
                 device='cpu'):

        # Initialize as Shape directly to avoid running Singlet.__init__ logic
        # which sets up Spheres. We need Cylinders.
        # We manually set the attributes required by Singlet.inBounds()
        Shape.__init__(self, transform, device)

        self.R1 = R1.to(device)
        self.R2 = R2.to(device)
        self.D = D.to(device)
        self.T = T.to(device)
        self.half_D = self.D / 2.0

        # Validation
        with torch.no_grad():
            if not torch.isinf(self.R1) and abs(2 * self.R1.item()) < self.D.item():
                raise ValueError(f"|R1| must be larger than D")
            if not torch.isinf(self.R2) and abs(2 * self.R2.item()) < self.D.item():
                raise ValueError(f"|R2| must be larger than D")
            if self.T.item() <= 1e-6:
                raise ValueError("Thickness T must be positive")

        z_v1 = -self.T / 2.0
        z_v2 = self.T / 2.0

        rot_x_axis = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]], device=device)

        # Surface 1
        if torch.isinf(self.R1):
            rot_plane = torch.tensor([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]], device=device)
            t1 = RayTransform(translation=torch.tensor([0., 0., z_v1], device=device),
                              rotation=rot_plane, device=device)
            self.surf1 = Plane(transform=t1, device=device)
            self.z_edge_front = z_v1
        else:
            c1 = z_v1 + self.R1
            t1 = RayTransform(translation=torch.tensor([0., 0., c1], device=device),
                              rotation=rot_x_axis, device=device)
            self.surf1 = Cylinder(torch.abs(self.R1), transform=t1, device=device)
            self.z_edge_front = self._get_sag_z(c1, self.R1, self.D)

        # Surface 2
        if torch.isinf(self.R2):
            t2 = RayTransform(translation=torch.tensor([0., 0., z_v2], device=device), device=device)
            self.surf2 = Plane(transform=t2, device=device)
            self.z_edge_back = z_v2
        else:
            c2 = z_v2 + self.R2
            t2 = RayTransform(translation=torch.tensor([0., 0., c2], device=device),
                              rotation=rot_x_axis, device=device)
            self.surf2 = Cylinder(torch.abs(self.R2), transform=t2, device=device)
            self.z_edge_back = self._get_sag_z(c2, self.R2, self.D)

        with torch.no_grad():
            if self.z_edge_front >= self.z_edge_back:
                raise ValueError(f"CylindricalSinglet has Zero/Negative Edge Thickness.")

        self.edge = Cylinder(self.half_D, device=device)
        self.surfaces = [self.surf1, self.surf2, self.edge]