import torch

from .primitives import Surface

class Aspheric(Surface):
    def __init__(self, Coefficients: torch.Tensor, device = "cpu", transform = None):

        super().__init__(transform=transform, device=device)
        raise NotImplementedError("Aspheric Not Implemented")

