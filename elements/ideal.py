import torch
import torch.nn as nn
from typing import List, Union

from .parent import Element
from ..geom import Plane, Disk
from ..phys import Linear

def ParaxialLensMat(lens_power_x, lens_power_y):

    Mat = torch.eye(5, device=lens_power_x.device, dtype=lens_power_x.dtype)
    Mat = Mat.index_put((torch.tensor([1, 3]), torch.tensor([0, 2])),
                        torch.stack([-lens_power_x, -lens_power_y]))

    return Mat

def ParaxialDistMat(dist):

    Mat = torch.eye(5, device=dist.device, dtype=dist.dtype)
    Mat = Mat.index_put((torch.tensor([0, 2]), torch.tensor([1, 3])),
                        torch.stack([dist, dist]))

    return Mat

def ParaxialRefractMat(Cx, Cy, ior_1, ior_2):

    vals = torch.stack([
        Cx * (ior_1 - ior_2) / ior_2,
        Cy * (ior_1 - ior_2) / ior_2,
        ior_1/ior_2,
        ior_1/ior_2])

    Mat = torch.eye(5, device=ior_1.device, dtype=ior_1.dtype)
    Mat = Mat.index_put((torch.tensor([1, 3, 1, 3]), torch.tensor([0, 2, 1, 3])),
                        vals)

    return Mat

class LinearElement(Element):

    def __init__(self, shape : Plane, linSurfFunc : Linear):

        super().__init__()

        self.shape = shape


        linSurfFunc.transform = shape.transform

        self.surface_functions.append(linSurfFunc)

    def _paraxial(self):

        Cx, Cy = self.surface_functions[0].Cx, self.surface_functions[0].Cy

        return ParaxialLensMat(-Cx, -Cy)


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
