import torch
import torch.nn as nn

from ..elements import ParaxialDistMat

class SequentialSystem(nn.Module):

    def __init__(self, elements):
        super().__init__()
        self.elements = nn.ModuleList(elements)

    def forward(self, rays):
        """
        Propagate rays sequentially through all elements and surfaces.
        Assumes fixed traversal order.
        """
        for element in self.elements:

            surfaces = element.shape.surfaces

            for i in range(len(element.shape)):

                # Intersection test
                t = surfaces[i].intersectTest(rays)   # (N,)
                ray_mask = t < float('inf')

                if not torch.any(ray_mask):
                    continue

                # Extract valid rays
                rays_valid = rays.subset(ray_mask)

                # Apply surface interaction
                new_pos, new_dir, intensity_mod = element(rays_valid, i)

                # Update valid rays
                rays_valid.update(new_pos, new_dir, intensity_mod)

                # Scatter back into full ray set
                rays.update_subset(rays_valid, ray_mask)

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

        for i in range(1, len(all_M)):

            # Apply surface matrix
            M_sys = all_M[i] @ M_sys

            D = ParaxialDistMat(dZ[i-1])
            M_sys = D @ M_sys

        return M_sys