import torch
import torch.nn as nn

from geom import RayTransform
from .parent import Element
from ..phys import RefractSnell, RefractFresnel, Block
from ..geom import Singlet, CylSinglet, Doublet, Triplet
from .ideal import ParaxialRefractMat

class EllipticAperture(Element):
    pass

class CircularAperture(Element):
    pass

class RectangularAperture(Element):
    pass

