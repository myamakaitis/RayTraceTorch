import torch
from torch.distributions import Uniform
from typing import Optional, Union

from ..geom import RayTransformBundle
from .bundle import DiskSample, Bundle


class LambertianSample:
    """
    Samples directions from a cosine-weighted (Lambertian) hemisphere around +Z.

    Uses the standard square-root mapping:
        x = sqrt(u1) * cos(2π u2)
        y = sqrt(u1) * sin(2π u2)
        z = sqrt(1 − u1)
    which exactly matches the pdf p(θ) = cos(θ)/π.
    """

    def sample(self, N: int,
               device: Union[str, torch.device] = 'cpu',
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
        u1 = torch.rand(N, device=device, dtype=dtype)
        u2 = torch.rand(N, device=device, dtype=dtype)
        r   = torch.sqrt(u1)
        phi = 2.0 * torch.pi * u2
        x   = r * torch.cos(phi)
        y   = r * torch.sin(phi)
        z   = torch.sqrt((1.0 - u1).clamp(min=0.0))
        return torch.stack([x, y, z], dim=1)


class PanelSource(Bundle):
    """
    Base class for flat area light sources.

    Positions are sampled over the emitter surface (defined by subclasses);
    directions follow a Lambertian (cosine-weighted) distribution around the
    local +Z normal so the bundle obeys Lambert's cosine law.
    """

    def __init__(self, ray_id: int,
                 device: Union[str, torch.device] = 'cpu',
                 dtype: torch.dtype = torch.float32,
                 transform: Optional[Union[RayTransformBundle, None]] = None):
        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)
        self._lambertian = LambertianSample()

    def sample_dir(self, N: int) -> torch.Tensor:
        return self._lambertian.sample(N, device=self.device, dtype=self.dtype)


class RectangularPanel(PanelSource):
    """
    Rectangular area light source with uniform spatial sampling and
    Lambertian angular emission.

    Positions are sampled uniformly over [−width/2, width/2] × [−height/2, height/2]
    in the XY plane (z = 0 before the bundle transform is applied).

    Args:
        width:  Extent along the local X axis (metres).
        height: Extent along the local Y axis (metres).
    """

    def __init__(self, width: float, height: float,
                 ray_id: int,
                 device: Union[str, torch.device] = 'cpu',
                 dtype: torch.dtype = torch.float32,
                 transform: Optional[Union[RayTransformBundle, None]] = None):
        super().__init__(ray_id=ray_id, device=device, dtype=dtype, transform=transform)
        w2, h2 = width / 2.0, height / 2.0
        self._x_dist = Uniform(
            torch.tensor(-w2, dtype=dtype),
            torch.tensor( w2, dtype=dtype),
        )
        self._y_dist = Uniform(
            torch.tensor(-h2, dtype=dtype),
            torch.tensor( h2, dtype=dtype),
        )

    def sample_pos(self, N: int) -> torch.Tensor:
        x = self._x_dist.sample((N,)).squeeze().to(device=self.device, dtype=self.dtype)
        y = self._y_dist.sample((N,)).squeeze().to(device=self.device, dtype=self.dtype)
        z = torch.zeros(N, device=self.device, dtype=self.dtype)
        return torch.stack([x, y, z], dim=1)


class RingSource(PanelSource):
    """
    Annular (ring-shaped) area light source with uniform spatial sampling and
    Lambertian angular emission.

    Positions are sampled uniformly over the annular region defined by
    radius_inner ≤ r ≤ radius_outer in the XY plane.  Set radius_inner = 0
    for a filled-disk source.

    Args:
        radius_inner: Inner radius of the ring (metres).
        radius_outer: Outer radius of the ring (metres).
    """

    def __init__(self, radius_inner: float, radius_outer: float,
                 ray_id: int,
                 device: Union[str, torch.device] = 'cpu',
                 dtype: torch.dtype = torch.float32,
                 transform: Optional[Union[RayTransformBundle, None]] = None):
        if radius_inner > radius_outer:
            raise ValueError(
                f"radius_inner ({radius_inner}) must be <= radius_outer ({radius_outer})"
            )
        super().__init__(ray_id=ray_id, device=device, dtype=dtype, transform=transform)
        zero   = torch.tensor([0.0],                 dtype=dtype)
        twopi  = torch.tensor([2.0 * torch.pi],      dtype=dtype)
        r_in2  = torch.tensor([radius_inner ** 2],   dtype=dtype)
        r_out2 = torch.tensor([radius_outer ** 2],   dtype=dtype)
        self._disk = DiskSample(r_in2, r_out2, zero, twopi)

    def sample_pos(self, N: int) -> torch.Tensor:
        return self._disk.sample(N).to(device=self.device, dtype=self.dtype)
