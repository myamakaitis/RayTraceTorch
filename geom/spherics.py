import torch
import math
from .shape import Shape
from .primitives import Cylinder, Plane
from .bounded import HalfSphere
from .transform import RayTransform

class Spheric(Shape):
    """
    Base class for lens stacks to share utility methods.
    """

    def _make_surface(self, C, z_vertex, c_grad=False, z_grad=False):
        """
        Creates an optical surface (Quadric/HalfSphere) at the specified vertex.
        """
        # 2. Transform (Vertex-based)
        t = RayTransform(translation=[0., 0., z_vertex], trans_grad=z_grad, trans_mask=[False, False, True])

        # 3. Create Surface
        surf = HalfSphere(curvature=C, transform=t, curvature_grad=c_grad)

        return surf

    def inBounds(self, local_pos, surf_idx):
        """
        Validates hits based on physical boundaries.
        surf_idx: 0 (Front), 1 (Back), 2 (Edge)
        """
        z = local_pos[:, 2]

        if surf_idx >= self.N_optical:
            # Edge Hit: Must be between the Z-planes of the lens edge
            z1 = self.surfaces[surf_idx-self.N_optical].sagitalZ(self.radius)
            z2 = self.surfaces[surf_idx-self.N_optical+1].sagitalZ(self.radius)

            return (z >= z1) & (z <= z2)
        else:
            # Face Hit: Must be within the Aperture Diameter
            # r^2 <= (D/2)^2
            r_sq = local_pos[:, 0] ** 2 + local_pos[:, 1] ** 2
            in_aperture = r_sq <= self.radius ** 2

            return in_aperture

class Singlet(Spheric):
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
                 C1: float, C2: float,
                 D: float, T: float,
                 C1_grad = True, C2_grad = True,
                 D_grad = False, T_grad = True,
                 transform=None):
        """
        Args:
            C1 (Tensor): curvature of front surface (Inf for Plane).
            C2 (Tensor): curvature of back surface (Inf for Plane).
            D (Tensor): Diameter of the lens.
            T (Tensor): Center thickness.
        """
        super().__init__(transform=transform)

        self.N_optical = 2
        radius = nn.Parameter(torch.as_tensor(D / 2.0), requires_grad=D_grad)

        surf1 = self._make_surface(C1, -T/2, c_grad=C1_grad, z_grad=T_grad)
        surf2 = self._make_surface(C2, T/2, c_grad=C2_grad, z_grad=T_grad)

        # Edge (Cylinder)
        edge = Cylinder(radius)

        # Order MUST be [Front, Back, Edge] for inside() logic
        self.surfaces.append(surf1, surf2, edge)

        # Validation
        with torch.no_grad():
            if abs(0.5 * C1) > 1/D:
                raise ValueError(f"|R1| must be larger than D/2")
            if abs(0.5 * C2) > 1/D:
                raise ValueError(f"|R2| must be larger than D/2")
            if self.T <= 1e-6:
                raise ValueError("Thickness T must be positive")
            z1 = surf1.sagittalZ(self.radius)
            z2 = surf1.sagittalZ(self.radius)

            if z1 > z2:
                raise ValueError("Intersecting optical surfaces")

    @property
    def T:
        return self.surfaces[self.N_optical-1].transform.trans[2] - self.surfaces[0].transform.trans[s]


class Doublet(Spheric):
    """
    A Cemented Doublet Lens with 3 Optical Surfaces and 2 Edge Surfaces.
    """

    def __init__(self,
                 C1: float, C2: float, C3: float,
                 T1: float, T2: float, D: float,
                 c1_grad = True, c2_grad = True, c3_grad = True,
                 t1_grad = True, t2_grad=True,
                 transform=None):

        super().__init__(transform=transform)

        self.radius = nn.Parameter(torch.as_tensor(D / 2.0), requires_grad=False)

        # 1. Validation
        with torch.no_grad():
            for C, name in [(C1, 'R1'), (C2, 'R2'), (C3, 'R3')]:
                if abs(0.5 * C) > 1 / D:
                    raise ValueError(f"|{name}| must be larger than D")
            if T1 <= 1e-6 or T2 <= 1e-6:
                raise ValueError("Thicknesses T1 and T2 must be positive")

        # 2. Vertices (Centered at Local Z=0)
        T_total = T1 + T2
        z_v1 = 0
        z_v2 = z_v1 + T1
        z_v3 = z_v2 + T2

        # 3. Construct Optical Surfaces & Get Edge Bounds
        # Note: z_e is the Z-coordinate where the surface intersects the cylinder of diameter D
        self.surf1, z_e1 = self._make_surface(z_v1, self.C1, device)
        self.surf2, z_e2 = self._make_surface(z_v2, self.C2, device)
        self.surf3, z_e3 = self._make_surface(z_v3, self.C3, device)

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


class Triplet(Lens):
    """
    A Cemented Triplet Lens with 4 Optical Surfaces and 3 Edge Surfaces.
    """

    def __init__(self,
                 C1: float, C2: float, C3: float, C4: float,
                 T1: float, T2: float, T3: float,
                 D: float,
                 transform=None,
                 device='cpu'):

        super().__init__(transform, device)

        self.C1, self.C2 = C1.to(device), C2.to(device)
        self.C3, self.C4 = C3.to(device), C4.to(device)
        self.T1, self.T2, self.T3 = T1.to(device), T2.to(device), T3.to(device)
        self.D = D.to(device)
        self.half_D = self.D / 2.0

        # 1. Validation
        with torch.no_grad():
            for C, name in [(self.C1, 'R1'), (self.C2, 'R2'), (self.C3, 'R3')]:
                if abs(0.5 * C.item()) > 1 / self.D.item():
                    raise ValueError(f"|{name}| must be larger than D")
            if any(t.item() <= 1e-6 for t in [self.T1, self.T2, self.T3]):
                raise ValueError("Thicknesses must be positive")

        # 2. Vertices
        T_total = self.T1 + self.T2 + self.T3
        z_v1 = -T_total / 2.0
        z_v2 = z_v1 + self.T1
        z_v3 = z_v2 + self.T2
        z_v4 = z_v3 + self.T3

        # 3. Optical Surfaces & Bounds
        self.surf1, z_e1 = self._make_surface(z_v1, self.C1, device)
        self.surf2, z_e2 = self._make_surface(z_v2, self.C2, device)
        self.surf3, z_e3 = self._make_surface(z_v3, self.C3, device)
        self.surf4, z_e4 = self._make_surface(z_v4, self.C4, device)

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
