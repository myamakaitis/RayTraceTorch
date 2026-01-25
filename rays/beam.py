import torch

from geom import RayTransform
from .bundle import Bundle
from torch.distributions import Normal

class GaussianBeam(Bundle):

    def __init__(self, diameter_1e2_x, diameter_1e2_y,
            ray_id: int, device: str = 'cpu', dtype: torch.dtype = torch.float32, transform: RayTransform = None):

        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)

        self.x_dist = Normal(0, diameter_1e2_x/4)
        self.y_dist = Normal(0, diameter_1e2_y/4)


    def sample_pos(self, N: int):

        return torch.stack([
                            self.x_dist.sample((N,)),
                            self.y_dist.sample((N,)),
                            torch.zeros(N, device=self.device, dtype=self.dtype)], dim=1)