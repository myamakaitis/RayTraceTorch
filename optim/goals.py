import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from ..scene import SequentialScene
from ..rays.ray import Rays
from ..rays.bundle import Bundle
from ..elements import Sensor


class FocalLengthLoss(nn.Module):
    """
    MSE loss on paraxial system power (P = 1/f) relative to a target.

    Uses power space rather than focal length to avoid the 1/f singularity
    for near-collimating systems, yielding smoother optimization gradients.

    The system power is extracted from the [1, 0] element of the 5×5 paraxial
    matrix returned by ``scene.getParaxial()``:  M[1, 0] = −P_sys.

    Args:
        f_target: Target effective focal length (positive = converging, metres).
    """

    def __init__(self, f_target: float):
        super().__init__()
        self.P_target = nn.Parameter(
            torch.as_tensor(1.0 / f_target), requires_grad=False
        )

    def forward(self, scene: SequentialScene) -> torch.Tensor:
        M = scene.getParaxial()
        P_actual = -M[1, 0]
        return (P_actual - self.P_target) ** 2


class SpotTargetLoss(nn.Module):
    """
    Weighted Euclidean distance between each bundle's intensity centroid and a
    fixed target position on the sensor plane.

    Args:
        sensor:     Sensor element that accumulates hits during simulate().
        target_xy:  (2,) or (K, 2) tensor of target XY coordinates.  If a
                    single (2,) target is given it is broadcast across all K
                    bundles.
    """

    def __init__(self, sensor: Sensor,
                 target_xy: torch.Tensor):
        super().__init__()
        self.sensor    = sensor
        target_xy = torch.as_tensor(target_xy, dtype=torch.float32)
        if target_xy.ndim == 1:
            target_xy = target_xy.unsqueeze(0)   # (1, 2)
        self.register_buffer('target_xy', target_xy)

    def forward(self, scene, bundles: List[Bundle], N_rays: int = 128) -> torch.Tensor:
        losses = []
        for i, bundle in enumerate(bundles):
            self.sensor.reset()

            rays = bundle.sample(N_rays)
            if hasattr(scene, 'parameters'):
                dev = next(iter(scene.parameters()), torch.zeros(1)).device
                rays = rays.to(dev)

            scene.rays = rays
            scene.simulate()

            if self.sensor.hitLocs:
                locs, w, _ = self.sensor.getHitsTensors()
                xy = locs[:, :2]
            else:
                xy = scene.rays.pos[:, :2]
                w  = scene.rays.intensity

            if len(w) == 0:
                continue

            w_sum = w.sum().clamp(min=1e-12)
            cx = (xy[:, 0] * w).sum() / w_sum
            cy = (xy[:, 1] * w).sum() / w_sum

            tidx = min(i, self.target_xy.shape[0] - 1)
            tx, ty = self.target_xy[tidx, 0], self.target_xy[tidx, 1]
            losses.append((cx - tx) ** 2 + (cy - ty) ** 2)

        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()


class SpotSizeLoss(nn.Module):
    """
    Mean intensity-weighted RMS spot radius across a set of field bundles.

    For each bundle the scene is simulated fresh (allowing gradient flow),
    the sensor hits are collected, and the weighted RMS radius about a
    reference centroid is computed.  The final loss is the mean over all
    bundles.

    Usage example::

        bundles = [
            CollimatedDisk(radius=0.010, ray_id=0,
                           transform=RayTransformBundle(translation=[0, 0, -0.1])),
            CollimatedDisk(radius=0.010, ray_id=1,
                           transform=RayTransformBundle(rotation=[-0.087, 0, 0],
                                                        translation=[0, 0, -0.1])),
        ]
        loss_fn = SpotSizeLoss(sensor, bundles, N_rays=64)
        loss = loss_fn(scene)   # differentiable scalar

    Args:
        sensor:    Sensor element that accumulates hits during simulate().
        bundles:   List of Bundle instances.
        N_rays:    Rays sampled per bundle per forward call.
        target_xy: (K, 2) per-bundle reference centroid positions.
                   ``None`` → use the intensity-weighted centroid of each bundle
                   (minimises spot size rather than spot position).
    """

    def __init__(self, sensor: Sensor,
                 bundles: List[Bundle],
                 N_rays: int = 128,
                 target_xy: Optional[torch.Tensor] = None):
        super().__init__()
        self.sensor   = sensor
        self.bundles  = bundles
        self.N_rays   = N_rays

        if target_xy is not None:
            target_xy = torch.as_tensor(target_xy, dtype=torch.float32)
            if target_xy.ndim == 1:
                target_xy = target_xy.unsqueeze(0).expand(len(bundles), 2)
        self._target_xy = target_xy   # (K, 2) or None

    def forward(self, scene) -> torch.Tensor:
        losses = []

        for i, bundle in enumerate(self.bundles):
            self.sensor.reset()

            rays = bundle.sample(self.N_rays)
            if hasattr(scene, 'parameters'):
                dev = next(iter(scene.parameters()), torch.zeros(1)).device
                rays = rays.to(dev)

            scene.rays = rays
            scene.simulate()

            if self.sensor.hitLocs:
                locs, w, _ = self.sensor.getHitsTensors()
                xy = locs[:, :2]
            else:
                xy = scene.rays.pos[:, :2]
                w  = scene.rays.intensity

            active = w > 0
            if not active.any():
                continue
            xy, w = xy[active], w[active]

            w_sum = w.sum().clamp(min=1e-12)

            if self._target_xy is not None:
                tidx = min(i, self._target_xy.shape[0] - 1)
                cx = self._target_xy[tidx, 0].to(xy.device)
                cy = self._target_xy[tidx, 1].to(xy.device)
            else:
                cx = (xy[:, 0] * w).sum() / w_sum
                cy = (xy[:, 1] * w).sum() / w_sum

            rms = torch.sqrt(
                ((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2) * (w / w_sum)
            ).sum()
            losses.append(rms)

        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()
