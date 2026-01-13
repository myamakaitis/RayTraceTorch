import torch
import torch.nn as nn

from .parent import Element
from ..phys import RefractSnell, RefractFresnel, Block


class Lens(Element):

    def __init__(self, refract = 'snell', inked=false, transform=None)

        super().__init__()

        raise NotImplementedError

    @property
    def f(self):
        raise NotImplementedError

    @property
    def R1(self):
        return 1 / self.shape.C1

    @property
    def R2(self):
        return 1 / self.shape.C2

    @property
    def T(self):
        return self.shape.T

    @property
    def T_edge(self):
        return self.shape.T_edge

