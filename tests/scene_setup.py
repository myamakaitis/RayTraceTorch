import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import RayTraceTorch as rtt

if __name__ == "__main__":
    rtt.gui.run()
