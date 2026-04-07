import torch
import torch.nn as nn
from typing import List, Optional

from ..scene import SequentialScene
from ..rays.ray import Rays
from ..rays.bundle import Bundle
from ..elements import Sensor

class FocalLengthLoss(nn.Module):
    """
    MSE loss on the paraxial system power (P = 1/f) relative to a target.

    Uses power space rather than focal length to avoid the 1/f singularity
    for near-collimating systems, yielding smoother optimization gradients.

    The system power is extracted from the [1, 0] element of the 5x5 paraxial
    matrix returned by scene.getParaxial():  M[1, 0] = -P_sys.

    Args:
        scene:    SequentialScene with a getParaxial() method.
        f_target: Target effective focal length (positive = converging).
    """

    def __init__(self, f_target: float):
        super().__init__()
        self.P_target = nn.Parameter(
            torch.as_tensor(1.0 / f_target), requires_grad=False
        )

    def forward(self, scene: SequentialScene) -> torch.Tensor:
        M = self.scene.getParaxial()
        P_actual = -M[1, 0]
        return (P_actual - self.P_target) ** 2

class SpotTargetLoss(nn.Module):

    def __init(self, sensor, target_rays):


class SpotSizeLoss(nn.Module):
    """
    Mean RMS spot-size loss across a set of field bundles.

    Usage example::

        bundles = [
            CollimatedDisk(radius=0.010, ray_id=0,
                           transform=RayTransformBundle(translation=[0,0,-0.005])),
            CollimatedDisk(radius=0.010, ray_id=1,
                           transform=RayTransformBundle(rotation=[-0.087,0,0],
                                                        translation=[0,0,-0.005])),
        ]
        loss_fn = SpotSizeLoss(scene, sensor, bundles, N_rays=64)
        loss = loss_fn()   # differentiable scalar

    Args:
        scene:     SequentialScene (must contain sensor in its elements list).
        sensor:    Sensor element that accumulates hits during simulate().
        bundles:   Ordered list of Bundle instances.  Each must have a unique
                   integer ray_id in [0, 127] (ids stored as int8).
        N_rays:    Rays sampled per bundle per forward call.
        target_xy: (K, 2) per-bundle target positions.  None → centroid.
    """

    def __init__(self, ray_id: int):
        super().__init__()

        self.target_id = target_rays


    def forward(self, sensor):

        

        return
