import torch
import torch.nn as nn
from typing import List, Union

from ..geom.shape import Shape
from ..phys.base import SurfaceFunction


class Element(nn.Module):
    """
    Base class for all optical elements (e.g., Lenses, Mirrors).

    Responsibilities:
    1. Holds the geometric definition (Shape).
    2. Maps each surface in the Shape to a physical interaction model (SurfaceFunction).
    3. Orchestrates the interaction: Geometry -> Hit -> Physics -> New Ray State.
    """

    def __init__(self, shape: Shape, surface_functions: Union[SurfaceFunction, List[SurfaceFunction]], device='cpu'):
        """
        Args:
            shape: The 3D geometric volume/surfaces of the element.
            surface_functions: A single SurfaceFunction (applied to all surfaces)
                               or a list of functions matching the order of shape.surfaces.
            device: Computation device.
        """
        super().__init__()
        self.device = device
        self.shape = shape.to(device)

        # Validate and Store Surface Functions
        num_surfaces = len(self.shape.surfaces)

        if isinstance(surface_functions, list):
            if len(surface_functions) != num_surfaces:
                raise ValueError(
                    f"Element Shape has {num_surfaces} surfaces, but {len(surface_functions)} physics functions were provided.")
            self.surface_functions = surface_functions
        else:
            # Broadcast single function to all surfaces (e.g., all reflective)
            self.surface_functions = [surface_functions] * num_surfaces

        # Note: Since SurfaceFunctions are not strictly nn.Modules in the base implementation,
        # we store them in a standard list. If they were Modules, we would use nn.ModuleList.

    def to(self, device):
        """Moves internal geometry and physics components to the specified device."""
        super().to(device)
        self.device = device
        self.shape.to(device)
        for sf in self.surface_functions:
            sf.to(device)
        return self