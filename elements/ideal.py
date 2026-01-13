import torch
import torch.nn as nn
from typing import List, Union

from .parent import Element
from ..geom.primitives import Plane
from ..geom.bounded import Disk
from ..phys.phys_std import Linear

class LinearElement(Element):

    def __init__(self, shape : Plane, linSurfFunc : Linear):

        super().__init__()

        self.shape = shape

        self.linSurfFunc = linSurfFunc
        self.linSurfFunc.transform = shape.transform
        self.Nsurfaces = 1


class IdealThinLens(LinearElement):

    def __init__(self, focal, focal_grad = False, diameter = float("inf"), transform = None):

        if diameter == float("inf"):
            plane = Plane(transform=transform)
        else:
            plane = Disk(radius=diameter/2, transform=transform)

        self.P = nn.Parameter(torch.as_tensor(1 / focal), requires_grad=focal_grad)
        linSurfFunc = Linear()

        linSurfFunc.Cx = self.P
        linSurfFunc.Cy = self.P

        super().__init__(shape = plane, linSurfFunc=linSurfFunc)

    @property
    def f(self):

        return 1 / self.P


class IdealCylThinLens(LinearElement):

    def __init__(self, focal_x, focal_y,
                 focal_x_grad = False, focal_y_grad =False,
                 diameter = float("inf"), transform = None):

        if diameter == float("inf"):
            plane = Plane(transform=transform)
        else:
            plane = Disk(radius=diameter/2, transform=transform)

        self.Px = nn.Parameter(torch.as_tensor(1 / focal_x), requires_grad=focal_x_grad)
        self.Py = nn.Parameter(torch.as_tensor(1 / focal_y), requires_grad=focal_y_grad)

        linSurfFunc = Linear()

        linSurfFunc.Cx = self.Px
        linSurfFunc.Cy = self.Py

        super().__init__(shape = plane, linSurfFunc=linSurfFunc)

    @property
    def fx(self):

        return 1 / self.Px

    @property
    def fy(self):

        return 1 / self.Py