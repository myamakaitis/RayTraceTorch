import torch
import torch.nn as nn

from geom import RayTransform
from .parent import Element
from ..phys import Reflect
from ..geom import Singlet, CylSinglet, Doublet, Triplet
from .ideal import ParaxialRefractMat

class Mirror(Element):

    def __init__(self):

        super().__init__()
        self.surface_functions.append(Reflect)

class SphericalMirror(Mirror):

    def __init__(self, c1: float, d: float,
                 c1_grad: bool = False, d_grad: bool = False,
                 transform: RayTransform=None):
        super().__init__()
        raise NotImplementedError

def ParabolicMirror(Mirror):

    def __init__(self, c1: float, d: float,
                 c1_grad: bool = False, d_grad: bool = False,
                 transform: RayTransform=None):
        super().__init__()
        raise NotImplementedError

def ParabolicMirrorOffAxis(Mirror):

    def __init__(self):
        super().__init__()

        raise NotImplementedError