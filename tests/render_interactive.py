import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import RayTraceTorch as rtt

def render():

    shape1 = rtt.geom.Box(height=10, width=10, length=10)

    element1 = rtt.elements.Element()
    element2 = rtt.elements.SingletLens(0.02, -0.1, 12.0, 3.0, 1.5, transform=rtt.geom.RayTransform(translation=[0, 0, 10]))

    element3 = rtt.elements.Element()
    element3.shape = rtt.geom.Sphere(2.0, transform = rtt.geom.RayTransform(translation=[10, 0, 0]))
    element3.surface_functions.append(rtt.phys.Reflect())

    element1.shape = shape1
    for _ in range(len(element1.shape)):
        element1.surface_functions.append(rtt.phys.Block())

    scene1 = rtt.scene.Scene()
    scene1.elements.extend([element1, element2, element3])
    scene1._build_index_maps()
    scene1.to('cuda')

    rtt.gui.run_gui(scene1)

if __name__ == '__main__':
    render()