import torch
import torch.nn as nn
from typing import List, Union

from ..geom.shape import Shape
from ..geom.primitives import Surface, Plane
from ..phys.phys_std import SurfaceFunction, Reflect, Linear

class Element(nn.Module):
    """
    Base class for all optical elements (e.g., Lenses, Mirrors).

    Responsibilities:
    1. Holds the geometric definition (Shape).
    2. Maps each surface in the Shape to a physical interaction model (SurfaceFunction).
    3. Orchestrates the interaction: Geometry -> Hit -> Physics -> New Ray State.
    """

    def __init__(self):
        """
        Args:
            shape: The 3D geometric volume/surfaces of the element.
            surface_functions: A single SurfaceFunction (applied to all surfaces)
                               or a list of functions matching the order of shape.surfaces.
            device: Computation device.
        """
        super().__init__()

        self.shape = None
        self.surface_functions = nn.ModuleList()
        self.Nsurfaces = 0

    def intersectTest(self, rays):
        """
        Checks for intersections between the rays and the element's shape.

        Args:
            rays (Rays): A batch of rays.

        Returns:
            t_matrix (Tensor): [N, K] Matrix of intersection distances (t) for all K surfaces.
                               Misses are marked with float('inf').
        """
        # Delegate to the geometric shape engine
        return self.shape.intersectTest(rays)

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

        return new_pos_global, new_dir_global, intensity_mult


class SingleSurfFunc(Element):

    def __init__(self, shape, surfaceFunc):

        super().__init__()

        self.shape = shape
        if hasattr(shape, surfaces):
            self.Nsurfaces = len(surfaceFunc.surfaces)
        else:
            self.Nsurfaces = 1

        for _ in range(self.Nsurfaces):
            self.surface_functions.append(surfaceFunc)


