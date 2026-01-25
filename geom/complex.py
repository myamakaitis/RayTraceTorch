import torch

from ..geom import RayTransform
from .primitives import Surface

class Aspheric(Surface):
    def __init__(self, Coefficients: torch.Tensor, transform: RayTransform = None):

        super().__init__(transform=transform)
        raise NotImplementedError("Aspheric Not Implemented")

