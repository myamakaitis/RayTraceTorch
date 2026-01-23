import torch

from geom import RayTransform
from .bundle import Bundle
from torch.distributions import Normal


class Collimated(Bundle):

    def __init__(self, transform : RayTransform, ray_id : int, device : str, dtype: torch.dtype):
        super().__init__(ray_id=ray_id, device=device, dtype=dtype)
        self.transform = transform

    def sample_dir(self, N):

        return torch.as_tensor([[0, 0, 1]], device=self.device, dtype=self.dtype).repeat(N, 1)

class GaussianBeam(Beam):

    def __init__(self, radius_1e2_x, radius_1e2_y, correlation, transform : RayTransform, ray_id : int, device : str, dtype: torch.dtype):

        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)

        self.x_dist =Normal(0, radius_1e2_x/4)
        self.y_dist =Normal(0, radius_1e2_y/4)


