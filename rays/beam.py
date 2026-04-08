import torch
from torch.distributions import Normal
from typing import Optional, Union

from ..geom import RayTransformBundle
from .bundle import Bundle


class GaussianBeam(Bundle):
    """
    Collimated Gaussian beam propagating along +Z.

    Ray positions are drawn from independent normal distributions in X and Y,
    matching the 1/e² intensity diameter convention (beam waist at z = 0).

    Args:
        diameter_1e2_x: 1/e² beam diameter along X (metres).
        diameter_1e2_y: 1/e² beam diameter along Y (metres).
    """

    def __init__(self, diameter_1e2_x: float, diameter_1e2_y: float,
                 ray_id: int,
                 device: Union[str, torch.device] = 'cpu',
                 dtype: torch.dtype = torch.float32,
                 transform: Optional[Union[RayTransformBundle, None]] = None):
        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)

        # σ = diameter / 4  (so that ±2σ spans the 1/e² diameter)
        sigma_x = torch.tensor(diameter_1e2_x / 4.0, device=device, dtype=dtype)
        sigma_y = torch.tensor(diameter_1e2_y / 4.0, device=device, dtype=dtype)
        zero_x  = torch.zeros(1, device=device, dtype=dtype).squeeze()
        zero_y  = torch.zeros(1, device=device, dtype=dtype).squeeze()

        self._x_dist = Normal(zero_x, sigma_x)
        self._y_dist = Normal(zero_y, sigma_y)

    def sample_pos(self, N: int) -> torch.Tensor:
        x = self._x_dist.sample((N,)).squeeze()
        y = self._y_dist.sample((N,)).squeeze()
        z = torch.zeros(N, device=self.device, dtype=self.dtype)
        return torch.stack([x, y, z], dim=1)
