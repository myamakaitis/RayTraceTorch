import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import RayTraceTorch as rtt

def render():

    shape1 = rtt.geom.Box(height=10, width=10, length=10)

    element1 = rtt.elements.Element()

    element1.shape = shape1
    for _ in range(len(element1.shape)):
        element1.surface_functions.append(rtt.phys.Block())

    scene1 = rtt.scene.Scene()
    scene1.elements.extend([element1])
    scene1._build_index_maps()
    scene1.to('cuda')

    rtt.render.run_gui(scene1)

if __name__ == '__main__':
    render()