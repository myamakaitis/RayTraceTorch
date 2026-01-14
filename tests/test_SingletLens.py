import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from RayTraceTorch.elements import SingletLens

def test_vals():

    lens1 = SingletLens(c1 = 1/50.0, c2 = 1/-40.0, d=10.0, t=3.0, ior_glass=1.4, ior_media=1.0)
    lens1.status()
    """
    Effective Focal Length, EFL (mm): 56.0897
    Back Focal Length, BFL (mm): 55.1282
    Front Focal Length, FFL (mm): -54.8878
    """

    lens1 = SingletLens(c1 = 1/150.0, c2 = 1/-20.0, d=10.0, t=3.0, ior_glass=1.2, ior_media=1.0)
    lens1.status()

    for name, param in lens1.named_parameters():
        print(f"Name: {name}, Shape: {param.shape}")