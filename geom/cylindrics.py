import torch
import torch.nn as nn
from .primitives import Surface, Quadric, Plane
from .bounded import HalfCyl
from .shape import Shape, Box4Side
from .transform import RayTransform

class Cylindric(Shape):
    """
    Base class for cylindrical/toroidal lens stacks.
    """

    def _make_cyl_surface(self, C, z_vertex, c_grad=False, z_grad=False):
        """
        Creates a HalfCyl optical surface at the specified vertex.
        """
        t = RayTransform(translation=[0., 0., z_vertex], trans_grad=z_grad, trans_mask=[False, False, True])
        surf = HalfCyl(curvature=C, transform=t, curvature_grad=c_grad)
        return surf

    def inBounds(self, local_pos, surf_idx):
        """
        Validates hits based on physical boundaries (Rectangular Box logic).
        Assumes surfaces are ordered: [Optical_Front, Optical_Back, Right, Left, Top, Bottom]
        """
        x, y, z = local_pos[:, 0], local_pos[:, 1], local_pos[:, 2]

        if surf_idx < self.N_optical:

            x_max = self.surfaces[2].transform.trans[0]
            x_min = self.surfaces[3].transform.trans[0]
            y_max = self.surfaces[4].transform.trans[1]
            y_min = self.surfaces[5].transform.trans[1]

            in_aperture = (x <= x_max + 1e-5) & (x >= x_min - 1e-5) & \
                          (y <= y_max + 1e-5) & (y >= y_min - 1e-5)
            # Check aperture
            # Hit on Optical Face: Must be within XY Aperture
            return in_aperture

        else:
            # Check z-limits
            z_front = self.surfaces[0].sagittalZ(y)
            z_back = self.surfaces[1].sagittalZ(y)

            # Ensure z_front is the 'left-most' in local coordinates if T/2 logic holds
            # But generally z_front < z_back
            in_z = (z >= z_front - 1e-4) & (z <= z_back + 1e-4)
            # Hit on Edge (Right/Left/Top/Bottom)
            # Must be within Z-bounds of the lens (between faces)

            return in_z


class CylSinglet(Cylindric):
    """
    A Cylindrical Singlet Lens.
    Defined by Y-curvature on front/back faces, and a rectangular aperture (Width x Height).
    """

    def __init__(self,
                 C1: float, C2: float,
                 width: float, height: float, T: float,
                 C1_grad: bool=True, C2_grad: bool=True,
                 T_grad: bool=True, w_grad: bool=False, h_grad: bool=False,
                 transform: RayTransform=None):
        """
        Args:
            C1, C2: Curvature in Y-direction.
            width: Size in X.
            height: Size in Y.
            T: Center thickness.
        """
        super().__init__(transform=transform)

        self.N_optical = 2

        # 1. Create Optical Surfaces (Front/Back)
        # Using T/2 centering convention
        surf1 = self._make_cyl_surface(C1, -T / 2, c_grad=C1_grad, z_grad=T_grad)
        surf2 = self._make_cyl_surface(C2, T / 2, c_grad=C2_grad, z_grad=T_grad)

        self.surfaces.append(surf1)
        self.surfaces.append(surf2)

        # 2. Create Edges (Using Plane logic from Box)
        edge_box = Box4Side(width, height, w_grad=w_grad, h_grad=h_grad)

        # Extend surfaces list with the 4 edge planes
        for s in edge_box.surfaces:
            self.surfaces.append(s)


        # Validation
        with torch.no_grad():
            # Check if curvature is too steep for height
            if abs(0.5 * C1) > 1 / height:
                raise ValueError(f"|R1| must be larger than Height/2")
            if abs(0.5 * C2) > 1 / height:
                raise ValueError(f"|R2| must be larger than Height/2")

            z_front = self.surfaces[0].sagittalZ(height/2)
            z_back = self.surfaces[1].sagittalZ(height/2)

            if z_front > z_back:
                raise ValueError(f"Front and back surfaces intersecting")

    @property
    def width(self):
        # Surfaces: 0=Front, 1=Back, 2=Right, 3=Left, 4=Top, 5=Bottom
        return self.surfaces[2].transform.trans[0] - self.surfaces[3].transform.trans[0]

    @property
    def height(self):
        # Surfaces: 0=Front, 1=Back, 2=Right, 3=Left, 4=Top, 5=Bottom
        return self.surfaces[4].transform.trans[1] - self.surfaces[5].transform.trans[1]