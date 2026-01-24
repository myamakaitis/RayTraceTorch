import torch

from geom import RayTransform
from .bundle import Collimated
from torch.distributions import Normal

class GaussianBeam(Collimated):

    def __init__(self, radius_1e2_x, radius_1e2_y, correlation, transform : RayTransform, ray_id : int, device : str, dtype: torch.dtype):

        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)

        self.x_dist =Normal(0, radius_1e2_x/4)
        self.y_dist =Normal(0, radius_1e2_y/4)


    def sample_pos(self, N):

        return torch.stack([
                            self.x_dist.sample((N,)),
                            self.y_dist.sample((N,)),
                            torch.zeros((N,), device=self.device, dtype=self.dtype)], dim=1)