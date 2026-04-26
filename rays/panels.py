import torch
from torch.distributions import Uniform
from typing import Optional, Union

import math
from ..geom import RayTransformBundle
from .bundle import DiskSample, Bundle, SolidAngleSample


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
                 transform: Optional[Union[RayTransformBundle, None]] = None,
                 sampling_type: str = 'lambertian',
                 cone_angle: float = math.pi / 4.0):
        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)
        
        self.sampling_type = sampling_type
        
        if sampling_type == 'lambertian':
            self._lambertian = LambertianSample()
        elif sampling_type == 'solid_angle':
            zero = torch.tensor([0.0], device=device, dtype=dtype)
            twopi = torch.tensor([2*math.pi], device=device, dtype=dtype)
            rad = torch.tensor([cone_angle], device=device, dtype=dtype)
            F_phi_max = SolidAngleSample.CDF_phi(rad)
            self._solid_angle = SolidAngleSample(zero, F_phi_max, zero, twopi)
        else:
            raise ValueError(f"Unknown sampling_type: {sampling_type}")

    def sample_dir(self, N: int) -> torch.Tensor:
        if self.sampling_type == 'lambertian':
            return self._lambertian.sample(N, device=self.device, dtype=self.dtype)
        elif self.sampling_type == 'solid_angle':
            phi, theta = self._solid_angle.sample(N)
            dz = torch.cos(phi)
            dr = torch.sin(phi)
            dx, dy = torch.cos(theta)*dr, torch.sin(theta)*dr
            # Move to device and ensure correct dtype
            res = torch.stack([dx, dy, dz], dim=1)
            return res.to(device=self.device, dtype=self.dtype)


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
                 transform: Optional[Union[RayTransformBundle, None]] = None,
                 sampling_type: str = 'lambertian',
                 cone_angle: float = math.pi / 4.0):
        super().__init__(ray_id=ray_id, device=device, dtype=dtype, transform=transform, sampling_type=sampling_type, cone_angle=cone_angle)
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
                 transform: Optional[Union[RayTransformBundle, None]] = None,
                 sampling_type: str = 'lambertian',
                 cone_angle: float = math.pi / 4.0):
        if radius_inner > radius_outer:
            raise ValueError(
                f"radius_inner ({radius_inner}) must be <= radius_outer ({radius_outer})"
            )
        super().__init__(ray_id=ray_id, device=device, dtype=dtype, transform=transform, sampling_type=sampling_type, cone_angle=cone_angle)
        zero   = torch.tensor([0.0],                 dtype=dtype)
        twopi  = torch.tensor([2.0 * torch.pi],      dtype=dtype)
        r_in2  = torch.tensor([radius_inner ** 2],   dtype=dtype)
        r_out2 = torch.tensor([radius_outer ** 2],   dtype=dtype)
        self._disk = DiskSample(r_in2, r_out2, zero, twopi)

    def sample_pos(self, N: int) -> torch.Tensor:
        return self._disk.sample(N).to(device=self.device, dtype=self.dtype)
