import torch
import torch.nn as nn
from .base import Scene
from ..elements import ParaxialDistMat

class SequentialScene(Scene):

    def __init__(self, elements):
        super().__init__()
        self.elements = nn.ModuleList(elements)

    def simulate(self, rays):
        """
        Propagate rays sequentially through all elements and surfaces.
        Assumes fixed traversal order.
        """
        for element in self.elements:

            for i in range(len(element.shape)):

                # Intersection test
                t = element.intersectTest(rays)[:, i]   # (N,)
                ray_mask = t < float('inf')

                if not torch.any(ray_mask):
                    continue

                # Extract valid rays
                rays_valid = rays.subset(ray_mask)

                # Apply surface interaction
                new_pos, new_dir, intensity_mod = element(rays_valid, i)

                rays.scatter_update(ray_mask, new_pos, new_dir, intensity_mod)

        return rays

    def getParaxial(self):
        """
        Compute the full system paraxial matrix.
        """

        all_Z = []
        all_M = []

        # Collect all paraxial surfaces in order
        for elem in self.elements:
            Zs, Mats = elem.getParaxial()
            all_Z.extend(Zs)
            all_M.extend(Mats)

        M_sys = all_M[0].clone()

        all_Z = torch.as_tensor(all_Z)
        dZ = all_Z[1:] - all_Z[:-1]

        for i in range(0, len(all_M)-1):
            # Apply surface matrix
            M_sys = ParaxialDistMat(dZ[i]) @ M_sys
            M_sys = all_M[i+1] @ M_sys

        return M_sys

    def checkDimensions(self):

        """
        Check that the physical dimensions of the system are physically realizable
        """

        center_thickness = self._getCenterThickness()
        edge_thickness = self._getEdgeThickness()

        center_clearance = self._getCenterClearance()
        edge_clearance = self._getEdgeClearance()


