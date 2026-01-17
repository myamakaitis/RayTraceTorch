import torch

from .parent import Element
from ..phys import Transmit

class Sensor(Element):

    def __init__(self, shape):

        super().__init__()

        self.shape = shape
        self.surface_functions.extend([Transmit()]*len(shape))

        self.hitLocs = []
        self.hitIntensity = []
        self.hitID = []

    def forward(self, rays, surf_idx):
        # A. Geometric Intersection (Detailed & Differentiable)
        # shape.forward returns: t, global_hit_point, global_normal
        _, new_pos_global, normal_global, hit_local = self.shape(rays, surf_idx)

        # B. Surface Physics
        # Get the physics function assigned to this surface
        phys_func = self.surface_functions[surf_idx]

        # Apply physics
        # Returns: new_direction, intensity_modulation
        new_dir_global, intensity_mult = phys_func(hit_local, rays.dir, normal_global)

        self.hitLocs.append(hit_local)
        self.hitIntensity.append(rays.intensity)
        self.hitID.append(rays.id)

        return new_pos_global, new_dir_global, intensity_mult

    def getHitsTensors(self):

        return torch.cat(self.hits, dim=0), torch.cat(self.hitLocs, dim=0), torch.cat(self.hitIntensity, dim=0)

    def getSpot_xy(self, id, target_xy = None):

        hit_xyz, hit_intensity, hit_id = self.getHitsTensors()

        id_mask = (hit_id == id)

        hit_xyz_id, hit_intensity_id = hit_xyz[id_mask], hit_intensity[id_mask]

        # if target unspecified use centroid
        if target_xy is None:
            target_xy = torch.sum(hit_xyz_id[:, 0:2] * hit_intensity_id[:, None]) / torch.sum(hit_intensity_id)



        centroid =





