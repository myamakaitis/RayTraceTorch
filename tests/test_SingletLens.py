import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from RayTraceTorch.elements import SingletLens, ParaxialDistMat

def test_vals():

    lens1 = SingletLens(c1 = 1/50.0, c2 = -1/50.0, d=10.0, t=30.0, ior_glass=1.4, ior_media=1.0)
    lens1.status()
    """
    Effective Focal Length, EFL (mm): 56.0897
    Back Focal Length, BFL (mm): 55.1282
    Front Focal Length, FFL (mm): -54.8878
    """
    Zs, mats = lens1.getParaxial()

    dZ = Zs[1] - Zs[0]
    mat_dist = ParaxialDistMat(torch.tensor(dZ))

    LensMat = mats[1] @ mat_dist @ mats[0]

    print(LensMat.numpy())

    lens1 = SingletLens(c1 = 1/150.0, c2 = 1/-20.0, d=10.0, t=3.0, ior_glass=1.2, ior_media=1.0)
    # lens1.status()

    # for name, param in lens1.named_parameters():
    #     print(f"Name: {name}, Shape: {param.shape}")