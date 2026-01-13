import torch
import torch.nn as nn
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
            z1 = self.surfaces[surf_idx-self.N_optical].sagittalZ(self.radius)
            z2 = self.surfaces[surf_idx-self.N_optical+1].sagittalZ(self.radius)

            return (z >= z1) & (z <= z2)
        else:
            # Face Hit: Must be within the Aperture Diameter
            # r^2 <= (D/2)^2
            r_sq = local_pos[:, 0] ** 2 + local_pos[:, 1] ** 2
            in_aperture = r_sq <= self.radius ** 2

            return in_aperture

    @property
    def T(self):
        return self.surfaces[self.N_optical-1].transform.trans[2] - self.surfaces[0].transform.trans[2]

    @property
    def T_edge(self):
        self.surfaces[self.N_optical-1].sagittalZ(self.radius) - self.surfaces[0].sagittalZ(self.radius)

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
        self.radius = nn.Parameter(torch.as_tensor(D / 2.0), requires_grad=D_grad)

        surf1 = self._make_surface(C1, -T/2, c_grad=C1_grad, z_grad=T_grad)
        surf2 = self._make_surface(C2, T/2, c_grad=C2_grad, z_grad=T_grad)

        # Edge (Cylinder)
        edge = Cylinder(D/2)
        edge.radius = self.radius

        # Order MUST be [Front, Back, Edge] for inside() logic
        for surf in [surf1, surf2, edge]:
            self.surfaces.append(surf)

        # Validation
        with torch.no_grad():
            if abs(0.5 * C1) > 1/D:
                raise ValueError(f"|R1| must be larger than D/2")
            if abs(0.5 * C2) > 1/D:
                raise ValueError(f"|R2| must be larger than D/2")
            if self.T <= 1e-6:
                raise ValueError("Thickness T must be positive")
            z_e1 = surf1.sagittalZ(self.radius)
            z_e2 = surf2.sagittalZ(self.radius)

            if z_e1 > z_e2:
                raise ValueError("Intersecting optical surfaces")




class Doublet(Spheric):
    """
    A 3D Doublet Lens defined by three optical surfaces and two cylindrical edges.
    The entire stack is centered at (0,0,0).
    """

    def __init__(self,
                 C1: float, C2: float, C3: float,
                 D: float,
                 T1: float, T2: float,
                 C1_grad=True, C2_grad=True, C3_grad=True,
                 D_grad=False,
                 T1_grad=True, T2_grad=True,
                 transform=None):
        """
        Args:
            C1, C2, C3: Curvatures of the 3 surfaces (Front, Middle, Back).
            D: Diameter of the lens.
            T1: Axial thickness of the first element (between C1 and C2).
            T2: Axial thickness of the second element (between C2 and C3).
        """
        super().__init__(transform=transform)

        self.N_optical = 3
        self.radius = nn.Parameter(torch.as_tensor(D / 2.0), requires_grad=D_grad)

        # Calculate Z positions to center the triplet
        # Total mechanical thickness
        T_total = T1 + T2

        # Z positions relative to the lens center (0,0,0)
        # V1 is at -Total/2
        z1 = -T_total / 2.0
        z2 = z1 + T1
        z3 = z2 + T2

        # Create Optical Surfaces
        # Note: We pass z_grad=True if the thickness determining this position has gradients enabled
        surf1 = self._make_surface(C1, z1, c_grad=C1_grad, z_grad=(T1_grad or T2_grad))
        surf2 = self._make_surface(C2, z2, c_grad=C2_grad, z_grad=(T1_grad or T2_grad))
        surf3 = self._make_surface(C3, z3, c_grad=C3_grad, z_grad=(T1_grad or T2_grad))

        # Create Edges (Cylinders)
        # Edge 1 connects Surf 1 and Surf 2
        edge1 = Cylinder(D / 2)
        edge1.radius = self.radius

        # Edge 2 connects Surf 2 and Surf 3
        edge2 = Cylinder(D / 2)
        edge2.radius = self.radius

        # Order MUST be [Optical Surfaces ..., Edges ...] for the base inBounds() logic
        optical_surfaces = [surf1, surf2, surf3]
        edges = [edge1, edge2]

        for s in optical_surfaces + edges:
            self.surfaces.append(s)

        # Validation
        with torch.no_grad():
            curvatures = [C1, C2, C3]
            thicknesses = [T1, T2]

            # Check curvature constraints
            for i, C in enumerate(curvatures):
                if abs(0.5 * C) > 1 / D:
                    raise ValueError(f"|R{i + 1}| must be larger than D/2")

            # Check thickness constraints
            for i, T_val in enumerate(thicknesses):
                if T_val <= 1e-6:
                    raise ValueError(f"Thickness T{i + 1} must be positive")

            # Check for intersection (crossing surfaces)
            # We check if the sagittal Z of the next surface is physically after the previous one
            z_sags = [s.sagittalZ(self.radius) for s in optical_surfaces]

            if z_sags[0] > z_sags[1]:
                raise ValueError("Optical surfaces 1 and 2 intersect")
            if z_sags[1] > z_sags[2]:
                raise ValueError("Optical surfaces 2 and 3 intersect")

    @property
    def T1(self):
        """Thickness of the first element."""
        return self.surfaces[1].transform.trans[2] - self.surfaces[0].transform.trans[2]

    @property
    def T2(self):
        """Thickness of the second element."""
        return self.surfaces[2].transform.trans[2] - self.surfaces[1].transform.trans[2]


class Triplet(Spheric):
    """
    A 3D Triplet Lens defined by four optical surfaces and three cylindrical edges.
    The entire stack is centered at (0,0,0).
    """

    def __init__(self,
                 C1: float, C2: float, C3: float, C4: float,
                 D: float,
                 T1: float, T2: float, T3: float,
                 C1_grad=True, C2_grad=True, C3_grad=True, C4_grad=True,
                 D_grad=False,
                 T1_grad=True, T2_grad=True, T3_grad=True,
                 transform=None):
        """
        Args:
            C1..C4: Curvatures of the 4 surfaces.
            D: Diameter of the lens.
            T1: Thickness between C1 and C2.
            T2: Thickness between C2 and C3.
            T3: Thickness between C3 and C4.
        """
        super().__init__(transform=transform)

        self.N_optical = 4
        self.radius = nn.Parameter(torch.as_tensor(D / 2.0), requires_grad=D_grad)

        # Calculate Z positions to center the triplet
        T_total = T1 + T2 + T3

        # Determine if any thickness gradients affect position
        any_T_grad = T1_grad or T2_grad or T3_grad

        z1 = -T_total / 2.0
        z2 = z1 + T1
        z3 = z2 + T2
        z4 = z3 + T3

        # Create Optical Surfaces
        surf1 = self._make_surface(C1, z1, c_grad=C1_grad, z_grad=any_T_grad)
        surf2 = self._make_surface(C2, z2, c_grad=C2_grad, z_grad=any_T_grad)
        surf3 = self._make_surface(C3, z3, c_grad=C3_grad, z_grad=any_T_grad)
        surf4 = self._make_surface(C4, z4, c_grad=C4_grad, z_grad=any_T_grad)

        # Create Edges
        edges = []
        for _ in range(3):
            e = Cylinder(D / 2)
            e.radius = self.radius
            edges.append(e)

        # Order MUST be [Optical Surfaces ..., Edges ...]
        optical_surfaces = [surf1, surf2, surf3, surf4]

        for s in optical_surfaces + edges:
            self.surfaces.append(s)

        # Validation
        with torch.no_grad():
            curvatures = [C1, C2, C3, C4]
            thicknesses = [T1, T2, T3]

            for i, C in enumerate(curvatures):
                if abs(0.5 * C) > 1 / D:
                    raise ValueError(f"|R{i + 1}| must be larger than D/2")

            for i, T_val in enumerate(thicknesses):
                if T_val <= 1e-6:
                    raise ValueError(f"Thickness T{i + 1} must be positive")

            z_sags = [s.sagittalZ(self.radius) for s in optical_surfaces]

            for i in range(len(z_sags) - 1):
                if z_sags[i] > z_sags[i + 1]:
                    raise ValueError(f"Optical surfaces {i + 1} and {i + 2} intersect")

    @property
    def T1(self):
        """Thickness of the first element."""
        return self.surfaces[1].transform.trans[2] - self.surfaces[0].transform.trans[2]

    @property
    def T2(self):
        """Thickness of the second element."""
        return self.surfaces[2].transform.trans[2] - self.surfaces[1].transform.trans[2]

    @property
    def T3(self):
        """Thickness of the third element."""
        return self.surfaces[3].transform.trans[2] - self.surfaces[2].transform.trans[2]