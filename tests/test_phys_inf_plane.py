import torch
import torch.nn as nn
import math
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from RayTraceTorch.elements import Element
from RayTraceTorch.rays import fanSource
from RayTraceTorch.geom import Plane
# Assuming these imports exist in your project structure
# from my_library import SingletLens, Rays, collimatedSource
from RayTraceTorch.phys import Reflect, RefractSnell, Transmit, RefractFresnel

def VisualizePhys(surfFunc, test_name, nrays = 15):

    rays = fanSource(origin = [0, 0, -100], ray_direction = [0, 0, 1],
                     fan_angle = 1.5913, fan_direction = [0, 1, 0],
                     N_rays = nrays)

    inf_plane = Element()

    plane = Plane()
    inf_plane.shape = plane

    inf_plane.surface_functions.append(surfFunc)

    p0 = rays.pos.clone()
    p1, d1, _ = inf_plane(rays, 0)

    p2 = p1 + d1*100

    fig, ax = plt.subplots()

    path = torch.stack([p0, p1, p2], dim=0)

    x, y, z = path[:, :, 0], path[:, :, 1], path[:, :, 2]

    ax.plot(z, y, alpha=0.3, color="red")
    ax.set_aspect('equal')

    fig.savefig(os.path.join("plot_inf_plane", test_name + ".png"))


def test_reflect():

    reflect = Reflect()

    VisualizePhys(reflect, "reflect")

    assert True

def test_snell():

    refract1 = RefractSnell(1.4, 1.0)
    VisualizePhys(refract1, "RefractSnell_1.0to1.4")

    refract2 = RefractSnell(1.8, 1.0)
    VisualizePhys(refract2, "RefractSnell_1.0to1.8")

    refract3 = RefractSnell(1.0, 1.4)
    VisualizePhys(refract3, "RefractSnell_1.4to1.0")

    assert True


def test_snell():

    refract1 = RefractSnell(1.4, 1.0)
    VisualizePhys(refract1, "RefractSnell_1.0to1.4")

    refract2 = RefractSnell(1.8, 1.0)
    VisualizePhys(refract2, "RefractSnell_1.0to1.8")

    refract3 = RefractSnell(1.0, 1.4)
    VisualizePhys(refract3, "RefractSnell_1.4to1.0")

    assert True


def test_fresnel():

    refract1 = RefractFresnel(1.4, 1.0)
    VisualizePhys(refract1, "RefractFresnel_1.0to1.4", nrays =181)

    refract2 = RefractFresnel(1.8, 1.0)
    VisualizePhys(refract2, "RefractFresnel_1.0to1.8", nrays=181)

    refract3 = RefractFresnel(1.0, 1.4)
    VisualizePhys(refract3, "RefractFresnel_1.4to1.0", nrays=181)

    assert True


if __name__ == "__main__":
    torch.manual_seed(678967896)

    test_reflect()

    test_snell()

    test_fresnel()




