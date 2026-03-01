import torch
import torch.nn as nn
from typing import List, Optional

from ..rays.ray import Rays
from ..rays.bundle import Bundle


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

    def __init__(self, scene, f_target: float):
        super().__init__()
        self.scene = scene
        self.P_target = nn.Parameter(
            torch.as_tensor(1.0 / f_target), requires_grad=False
        )

    def forward(self):
        M = self.scene.getParaxial()
        P_actual = -M[1, 0]
        return (P_actual - self.P_target) ** 2


class SpotSizeLoss(nn.Module):
    """
    Mean RMS spot-size loss across a set of field bundles.

    Each bundle (a CollimatedDisk or similar) is sampled to produce N_rays
    rays that are simulated through the scene. The sensor accumulates hit
    positions tagged by ray ID (one unique ID per bundle). Spot sizes are
    then computed per-ID using getSpotSizeParallel_xy.

    The sensor is reset at the start of every forward call so that previous
    simulation runs do not accumulate.

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

    def __init__(self, scene, sensor,
                 bundles: List[Bundle],
                 N_rays: int,
                 target_xy=None):
        super().__init__()
        self.scene     = scene
        self.sensor    = sensor
        self.bundles   = bundles
        self.N_rays    = N_rays
        self.target_xy = target_xy

    def _build_rays(self) -> Rays:
        batches = [b.sample(self.N_rays) for b in self.bundles]

        all_pos = torch.cat([r.pos       for r in batches], dim=0)
        all_dir = torch.cat([r.dir       for r in batches], dim=0)
        all_int = torch.cat([r.intensity for r in batches], dim=0)
        all_id  = torch.cat([r.id        for r in batches], dim=0)
        all_wl  = torch.cat([r.wavelength for r in batches], dim=0)

        N_total = all_pos.shape[0]
        return Rays(pos=all_pos, dir=all_dir, intensity=all_int,
                    id=all_id, wavelength=all_wl, batch_size=[N_total])

    def forward(self):
        self.sensor.reset()
        rays = self._build_rays()
        self.scene.simulate(rays)

        # query_ids matches the ray_id assigned to each bundle
        device = rays.pos.device
        query_ids = torch.tensor(
            [b.ray_id for b in self.bundles], device=device, dtype=torch.int32
        )
        spot_sizes, _ = self.sensor.getSpotSizeParallel_xy(
            query_ids, target_xy=self.target_xy
        )
        return spot_sizes.mean()
