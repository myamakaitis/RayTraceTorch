import torch
import torch.nn as nn
import math
from typing import Optional, Union

from geom import RayTransform
from .parent import Element
from ..phys import Reflect
from ..geom import Quadric, QuadricZY, RayTransform
from ..geom.bounded import HalfSphere, HalfCyl
from .ideal import ParaxialMirrorMat


class Mirror(Element):

    def __init__(self):

        super().__init__()
        self.surface_functions.append(Reflect())


class SphericalMirror(Mirror):

    def __init__(self, c1: float, d: float,
                 c1_grad: bool = False, d_grad: bool = False,
                 transform: Optional[Union[RayTransform, None]] = None):

        super().__init__()

        self.shape = HalfSphere(curvature=c1, curvature_grad=c1_grad, transform=transform)
        self.d = nn.Parameter(torch.as_tensor(d), requires_grad=d_grad)

    @property
    def c1(self):
        return self.shape.c

    @property
    def R(self):
        return 1.0 / self.shape.c

    @property
    def f(self):
        return 1.0 / (2.0 * self.shape.c)

    def getParaxial(self):

        T = self.shape.transform.paraxial()
        T_inv = self.shape.transform.paraxial_inv()

        Mat = ParaxialMirrorMat(self.shape.c, self.shape.c)

        return [self.shape.z], [T_inv @ Mat @ T]


class CylindricalMirror(Mirror):

    def __init__(self, c1: float, d: float,
                 c1_grad: bool = False, d_grad: bool = False,
                 transform: Optional[Union[RayTransform, None]] = None):

        super().__init__()

        # HalfCyl is a QuadricZY (k=0) — curves in Y, invariant in X
        self.shape = HalfCyl(curvature=c1, curvature_grad=c1_grad, transform=transform)
        self.d = nn.Parameter(torch.as_tensor(d), requires_grad=d_grad)

    @property
    def c1(self):
        return self.shape.c

    @property
    def R(self):
        return 1.0 / self.shape.c

    @property
    def f(self):
        return 1.0 / (2.0 * self.shape.c)

    def getParaxial(self):

        T = self.shape.transform.paraxial()
        T_inv = self.shape.transform.paraxial_inv()

        # Curves in Y only — no focusing power in X
        zero = torch.zeros_like(self.shape.c)
        Mat = ParaxialMirrorMat(zero, self.shape.c)

        return [self.shape.z], [T_inv @ Mat @ T]


class ParabolicMirror(Mirror):

    def __init__(self, c1: float, d: float,
                 c1_grad: bool = False, d_grad: bool = False,
                 transform: Optional[Union[RayTransform, None]] = None):

        super().__init__()

        # Quadric with k=-1 is a paraboloid of revolution
        self.shape = Quadric(c=c1, k=-1.0, c_grad=c1_grad, transform=transform)
        self.d = nn.Parameter(torch.as_tensor(d), requires_grad=d_grad)

    @property
    def c1(self):
        return self.shape.c

    @property
    def R(self):
        return 1.0 / self.shape.c

    @property
    def f(self):
        return 1.0 / (2.0 * self.shape.c)

    def getParaxial(self):

        T = self.shape.transform.paraxial()
        T_inv = self.shape.transform.paraxial_inv()

        # Paraxial approximation at vertex is identical to the sphere of the same curvature
        Mat = ParaxialMirrorMat(self.shape.c, self.shape.c)

        return [self.shape.z], [T_inv @ Mat @ T]


class ParabolicMirrorXZ(Mirror):

    def __init__(self, c1: float, d: float,
                 c1_grad: bool = False, d_grad: bool = False,
                 transform: Optional[Union[RayTransform, None]] = None):

        super().__init__()

        # QuadricZY (k=-1) curves in Y. A 90° rotation around Z maps Y→X so
        # the parabola focuses in X and is invariant in Y (XZ plane curvature).
        translation = transform.trans.detach().tolist() if transform is not None else None
        xz_transform = RayTransform(rotation=[0.0, 0.0, math.pi / 2.0],
                                    translation=translation)

        self.shape = QuadricZY(c=c1, k=-1.0, c_grad=c1_grad, transform=xz_transform)
        self.d = nn.Parameter(torch.as_tensor(d), requires_grad=d_grad)

    @property
    def c1(self):
        return self.shape.c

    @property
    def R(self):
        return 1.0 / self.shape.c

    @property
    def f(self):
        return 1.0 / (2.0 * self.shape.c)

    def getParaxial(self):

        T = self.shape.transform.paraxial()
        T_inv = self.shape.transform.paraxial_inv()

        # Curves in X only (after the 90° rotation)
        zero = torch.zeros_like(self.shape.c)
        Mat = ParaxialMirrorMat(self.shape.c, zero)

        return [self.shape.z], [T_inv @ Mat @ T]


class ParabolicMirrorOffAxis(Mirror):

    def __init__(self):
        super().__init__()

        raise NotImplementedError
