import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from ..geom.transform import RayTransform

from .std import Transmit


class ApertureFilter(Transmit):
    """
    Surface function for aperture stops.
    Transmits rays whose local intersection point falls inside the aperture bounds,
    and blocks (zeroes intensity and direction) rays outside the bounds.

    The bounds check is delegated to the inBounds callable supplied at construction —
    typically the `inBounds` method of the aperture element's shape.
    """

    def __init__(self, inBounds: Callable):
        super().__init__()
        self._inBounds = inBounds

    def forward(self, local_intersect, ray_dir, normal, **kwargs):
        out_dir, _ = super().forward(local_intersect, ray_dir, normal)

        mask = self._inBounds(local_intersect)           # [N] bool
        mask_f = mask.to(ray_dir.dtype)                  # [N] float

        intensity_mod = mask_f
        out_dir = out_dir * mask_f[:, None]

        return out_dir, intensity_mod


class Fuzzy(Transmit):

    def __init__(self, intensity_function : callable):

        super().__init__()

        self.block_function = intensity_function


    def forward(self, local_intersect, ray_dir, normal, **kwargs):

        out_dir, _ = super().forward(local_intersect, ray_dir, normal)

        intensity_mod = self.block_function(local_intersect)

        return out_dir, intensity_mod