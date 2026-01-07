import torch
import math
from .shape import Shape
from .primitives import Plane, Cylinder
from .bounded import HalfSphere
from .transform import RayTransform


class Lens(Shape):
    """
    Base class for lens stacks to share utility methods.
    """

    def _make_surface(self, z_vertex, R, diameter, device):
        """
        Creates an optical surface (Sphere or Plane) oriented with Normal pointing +Z.
        Returns: (SurfaceObject, EdgeZPosition)
        """
        # 1. Plane (Inf Radius)
        if torch.isinf(R):
            # Standard Plane: Normal points +Z (0,0,1).
            # We use Identity rotation (default).
            t = RayTransform(translation=torch.tensor([0., 0., z_vertex], device=device), device=device)
            surf = Plane(transform=t, device=device)
            z_edge = z_vertex

        # 2. Sphere (Finite Radius)
        else:
            # Center of Curvature: C = V + R
            c = z_vertex + R
            t = RayTransform(translation=torch.tensor([0., 0., c], device=device), device=device)

            # OPTICAL CONVENTION: Normal points +Z.
            # The Sphere primitive calculates Normal = (Hit - Center) / Radius.
            # To ensure the normal points +Z (Forward), we pass -R.
            # Logic:
            # - Convex Front (R>0): Center is Right. Hit is Left. (Hit-Center) is -Z.
            #   Divide by -R (neg) -> Result is +Z.
            # - Convex Back (R<0): Center is Left. Hit is Right. (Hit-Center) is +Z.
            #   Divide by -R (pos) -> Result is +Z.
            surf = HalfSphere(R, transform=t, device=device)

            # Calculate Sag Z (Edge Z position)
            # z_edge = C - sign(R) * sqrt(R^2 - (D/2)^2)
            r_sq = (diameter / 2.0) ** 2
            root = torch.sqrt(R ** 2 - r_sq)
            # Note: We use the actual sign of R for geometry, not the flipped one for normals
            z_edge = c - torch.sign(R) * root

        return surf, z_edge


class Doublet(Lens):
    """
    A Cemented Doublet Lens with 3 Optical Surfaces and 2 Edge Surfaces.
    """

    def __init__(self,
                 R1: torch.Tensor,
                 R2: torch.Tensor,
                 R3: torch.Tensor,
                 T1: torch.Tensor,
                 T2: torch.Tensor,
                 D: torch.Tensor,
                 transform=None,
                 device='cpu'):

        super().__init__(transform=transform, device=device)

        self.R1, self.R2, self.R3 = R1.to(device), R2.to(device), R3.to(device)
        self.T1, self.T2 = T1.to(device), T2.to(device)
        self.D = D.to(device)
        self.half_D = self.D / 2.0

        # 1. Validation
        with torch.no_grad():
            for R, name in [(self.R1, 'R1'), (self.R2, 'R2'), (self.R3, 'R3')]:
                if not torch.isinf(R) and abs(2 * R.item()) < self.D.item():
                    raise ValueError(f"|{name}| must be larger than D")
            if self.T1.item() <= 1e-6 or self.T2.item() <= 1e-6:
                raise ValueError("Thicknesses T1 and T2 must be positive")

        # 2. Vertices (Centered at Local Z=0)
        T_total = self.T1 + self.T2
        z_v1 = -T_total / 2.0
        z_v2 = z_v1 + self.T1
        z_v3 = z_v2 + self.T2

        # 3. Construct Optical Surfaces & Get Edge Bounds
        # Note: z_e is the Z-coordinate where the surface intersects the cylinder of diameter D
        self.surf1, z_e1 = self._make_surface(z_v1, self.R1, self.D.item(), device)
        self.surf2, z_e2 = self._make_surface(z_v2, self.R2, self.D.item(), device)
        self.surf3, z_e3 = self._make_surface(z_v3, self.R3, self.D.item(), device)

        # 4. Check Element Edge Thicknesses
        with torch.no_grad():
            if z_e1 >= z_e2: raise ValueError(f"Element 1 (Front) has negative edge thickness")
            if z_e2 >= z_e3: raise ValueError(f"Element 2 (Back) has negative edge thickness")

        # Store bounds for inside() checks
        self.bounds = [
            (z_e1, z_e2),  # Element 1 Range
            (z_e2, z_e3)  # Element 2 Range
        ]

        # 5. Construct Separate Edge Surfaces
        self.edge1 = Cylinder(self.half_D, device=device)
        self.edge2 = Cylinder(self.half_D, device=device)

        # Surface Order: [Optical 1, Optical 2, Optical 3, Edge 1, Edge 2]
        self.surfaces = [self.surf1, self.surf2, self.surf3, self.edge1, self.edge2]

    def inBounds(self, local_pos, surf_idx=None):
        """
        Validates intersection based on surface type.
        Indices 0-2: Optical Surfaces -> Check Aperture Radius.
        Indices 3-4: Edge Surfaces -> Check Z-Bounds for specific element.
        """
        # Optical Surfaces (0, 1, 2)
        if surf_idx < 3:
            r_sq = local_pos[:, 0] ** 2 + local_pos[:, 1] ** 2
            return r_sq <= self.half_D ** 2

        # Edge Surfaces (3, 4)
        else:
            # Map surface index to element bounds
            # surf_idx 3 (Edge 1) -> bounds[0]
            # surf_idx 4 (Edge 2) -> bounds[1]
            bounds_idx = surf_idx - 3
            z_min, z_max = self.bounds[bounds_idx]

            z = local_pos[:, 2]
            return (z >= z_min) & (z <= z_max)


class Triplet(Lens):
    """
    A Cemented Triplet Lens with 4 Optical Surfaces and 3 Edge Surfaces.
    """

    def __init__(self,
                 R1: torch.Tensor, R2: torch.Tensor, R3: torch.Tensor, R4: torch.Tensor,
                 T1: torch.Tensor, T2: torch.Tensor, T3: torch.Tensor,
                 D: torch.Tensor,
                 transform=None,
                 device='cpu'):

        super().__init__(transform, device)

        self.R1, self.R2 = R1.to(device), R2.to(device)
        self.R3, self.R4 = R3.to(device), R4.to(device)
        self.T1, self.T2, self.T3 = T1.to(device), T2.to(device), T3.to(device)
        self.D = D.to(device)
        self.half_D = self.D / 2.0

        # 1. Validation
        with torch.no_grad():
            for R in [self.R1, self.R2, self.R3, self.R4]:
                if not torch.isinf(R) and abs(2 * R.item()) < self.D.item():
                    raise ValueError(f"Radius {R.item()} too small for diameter {self.D.item()}")
            if any(t.item() <= 1e-6 for t in [self.T1, self.T2, self.T3]):
                raise ValueError("Thicknesses must be positive")

        # 2. Vertices
        T_total = self.T1 + self.T2 + self.T3
        z_v1 = -T_total / 2.0
        z_v2 = z_v1 + self.T1
        z_v3 = z_v2 + self.T2
        z_v4 = z_v3 + self.T3

        # 3. Optical Surfaces & Bounds
        self.surf1, z_e1 = self._make_surface(z_v1, self.R1, self.D.item(), device)
        self.surf2, z_e2 = self._make_surface(z_v2, self.R2, self.D.item(), device)
        self.surf3, z_e3 = self._make_surface(z_v3, self.R3, self.D.item(), device)
        self.surf4, z_e4 = self._make_surface(z_v4, self.R4, self.D.item(), device)

        # 4. Check Edge Thicknesses
        with torch.no_grad():
            if z_e1 >= z_e2: raise ValueError("Element 1 has negative edge thickness")
            if z_e2 >= z_e3: raise ValueError("Element 2 has negative edge thickness")
            if z_e3 >= z_e4: raise ValueError("Element 3 has negative edge thickness")

        self.bounds = [
            (z_e1, z_e2),  # Elem 1
            (z_e2, z_e3),  # Elem 2
            (z_e3, z_e4)  # Elem 3
        ]

        # 5. Separate Edges
        self.edge1 = Cylinder(self.half_D, device=device)
        self.edge2 = Cylinder(self.half_D, device=device)
        self.edge3 = Cylinder(self.half_D, device=device)

        # Order: [S1, S2, S3, S4, E1, E2, E3]
        self.surfaces = [
            self.surf1, self.surf2, self.surf3, self.surf4,
            self.edge1, self.edge2, self.edge3
        ]

    def inBounds(self, local_pos, surf_idx=None):
        # Optical Surfaces (0,1,2,3)
        if surf_idx < 4:
            r_sq = local_pos[:, 0] ** 2 + local_pos[:, 1] ** 2
            return r_sq <= self.half_D ** 2

        # Edge Surfaces (4,5,6)
        else:
            bounds_idx = surf_idx - 4
            z_min, z_max = self.bounds[bounds_idx]
            z = local_pos[:, 2]
            return (z >= z_min) & (z <= z_max)