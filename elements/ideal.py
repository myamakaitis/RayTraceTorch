import torch
import torch.nn as nn
from typing import List, Union

from .parent import Element
from ..geom import Plane, Disk
from ..phys import Linear

class LinearElement(Element):

    def __init__(self, shape : Plane, linSurfFunc : Linear):

        super().__init__()

        self.shape = shape


        linSurfFunc.transform = shape.transform

        self.surface_functions.append(linSurfFunc)
        self.Nsurfaces = 1


class IdealThinLens(LinearElement):

    def __init__(self, focal, focal_grad = False, diameter = float("inf"), transform = None):

        if diameter == float("inf"):
            plane = Plane(transform=transform)
        else:
            plane = Disk(radius=diameter/2, transform=transform)


        linSurfFunc = Linear()

        super().__init__(shape = plane, linSurfFunc=linSurfFunc)

        self.P = nn.Parameter(torch.as_tensor(-1 / focal), requires_grad=focal_grad)

        self.surface_functions[0].Cx = self.P
        self.surface_functions[0].Cy = self.P

    @property
    def f(self):
        return -1 / self.P


class IdealCylThinLens(LinearElement):

    def __init__(self, focal_x, focal_y,
                 focal_x_grad = False, focal_y_grad =False,
                 diameter = float("inf"), transform = None):

        if diameter == float("inf"):
            plane = Plane(transform=transform)
        else:
            plane = Disk(radius=diameter/2, transform=transform)

        linSurfFunc = Linear()

        super().__init__(shape = plane, linSurfFunc=linSurfFunc)


        self.Px = nn.Parameter(torch.as_tensor(-1 / focal_x), requires_grad=focal_x_grad)
        self.Py = nn.Parameter(torch.as_tensor(-1 / focal_y), requires_grad=focal_y_grad)

        self.surface_functions[0].Cx = self.Px
        self.surface_functions[1].Cy = self.Py

    @property
    def fx(self):
        return -1 / self.Px

    @property
    def fy(self):
        return -1 / self.Py