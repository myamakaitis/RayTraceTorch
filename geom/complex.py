import torch
from typing import Optional, Union
from ..geom import RayTransform
from .primitives import Surface

class Aspheric(Surface):
    def __init__(self, Coefficients: torch.Tensor, transform: Optional[Union[RayTransform, None]] = None):

        super().__init__(transform=transform)
        raise NotImplementedError("Aspheric Not Implemented")

