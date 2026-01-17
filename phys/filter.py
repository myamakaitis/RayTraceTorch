import torch
import torch.nn as nn
import torch.nn.functional as F
from ..geom.transform import RayTransform

from .std import Transmit



class Fuzzy(Transmit):

    def __init__(self, intensity_function : callable):

        super().__init__()

        self.block_function = intensity_function


    def forward(self, local_intersect, ray_dir, normal, **kwargs):

        out_dir, _ = super().forward(local_intersect, ray_dir, normal)

        intensity_mod = self.block_function(local_intersect)

        return out_dir, intensity_mod