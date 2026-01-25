import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import RayTraceTorch as rtt


Element = rtt.elements.Element
Fan = rtt.rays.Fan
Plane = rtt.geom.Plane
RayTransform = rtt.geom.RayTransform
# Assuming these imports exist in your project structure
# from my_library import SingletLens, Rays, collimatedSource

def VisualizePhys(surfFunc, test_name, nrays = 15):


    rays = Fan(1.5913, 1).sample(nrays)

    inf_plane = Element()

    plane = Plane(transform = RayTransform(translation=[0, 0, 10]))
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

    reflect = rtt.phys.Reflect()

    VisualizePhys(reflect, "reflect")

    assert True

def test_snell():

    refract1 = rtt.phys.RefractSnell(1.4, 1.0)
    VisualizePhys(refract1, "RefractSnell_1.0to1.4")

    refract2 = rtt.phys.RefractSnell(1.8, 1.0)
    VisualizePhys(refract2, "RefractSnell_1.0to1.8")

    refract3 = rtt.phys.RefractSnell(1.0, 1.4)
    VisualizePhys(refract3, "RefractSnell_1.4to1.0")

    assert True


def test_snell():

    refract1 = rtt.phys.RefractSnell(1.4, 1.0)
    VisualizePhys(refract1, "RefractSnell_1.0to1.4")

    refract2 = rtt.phys.RefractSnell(1.8, 1.0)
    VisualizePhys(refract2, "RefractSnell_1.0to1.8")

    refract3 = rtt.phys.RefractSnell(1.0, 1.4)
    VisualizePhys(refract3, "RefractSnell_1.4to1.0")

    assert True


def test_fresnel():

    refract1 = rtt.phys.RefractFresnel(1.4, 1.0)
    VisualizePhys(refract1, "RefractFresnel_1.0to1.4", nrays =181)

    refract2 = rtt.phys.RefractFresnel(1.8, 1.0)
    VisualizePhys(refract2, "RefractFresnel_1.0to1.8", nrays=181)

    refract3 = rtt.phys.RefractFresnel(1.0, 1.4)
    VisualizePhys(refract3, "RefractFresnel_1.4to1.0", nrays=181)

    assert True




