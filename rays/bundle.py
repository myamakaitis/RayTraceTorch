import torch

from ..geom import RayTransformBundle
from .ray import Rays
import torch.nn.functional as F
from torch.distributions import Uniform

class Bundle:

    def __init__(self, ray_id: int, device: str = 'cpu', dtype: torch.dtype = torch.float32, transform: RayTransformBundle = None):

        self.ray_id = ray_id
        self.device = device
        self.dtype = dtype

        if transform is None:
            self.transform = RayTransformBundle()
        else:
            self.transform = transform

    def sample_dir(self, N: int):
        return torch.tensor([[0, 0, 1]], device=self.device, dtype=self.dtype).repeat(N, 1)

    def sample_pos(self, N: int):
        return torch.zeros((N, 3), device=self.device, dtype=self.dtype)

    def sample(self, N: int):

        _pos = self.sample_pos(N)
        _dir = self.sample_dir(N)

        pos_global, dir_global = self.transform.transform_(_pos, _dir)

        return Rays.initialize(pos_global, dir_global, ray_id=self.ray_id, device=self.device, dtype=self.dtype)


class DiskSample:

    def __init__(self, radius_inner_2: torch.Tensor, radius_outer_2: torch.Tensor, theta_min: torch.Tensor, theta_max: torch.Tensor):

        self.r_distribution = Uniform(radius_inner_2, radius_outer_2)
        self.t_distribution = Uniform(theta_min, theta_max)

    def sample(self, N: int):

        theta = self.t_distribution.sample((N,)).squeeze()
        r = torch.sqrt(self.r_distribution.sample((N,))).squeeze()

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        z = torch.zeros_like(x)

        return torch.stack([x, y, z], dim=1)

class SolidAngleSample:

    def __init__(self, F_phi_min: torch.Tensor, F_phi_max: torch.Tensor, theta_min: torch.Tensor, theta_max: torch.Tensor):

        self.phi_distribution = Uniform(F_phi_min, F_phi_max)
        self.t_distribution = Uniform(theta_min, theta_max)

    def sample(self, N: int):

        phi = self.invCDF_phi(self.phi_distribution.sample((N,))).squeeze()
        theta = self.t_distribution.sample((N,)).squeeze()

        return phi, theta

    @classmethod
    def invCDF_phi(cls, F : torch.Tensor):

        return torch.acos(-2*(F)  + 1)

    @classmethod
    def CDF_phi(cls, phi : torch.Tensor):

        return (1 - torch.cos(phi)) / torch.pi


class CollimatedDisk(Bundle):

    def __init__(self, radius: float,
                 ray_id: int, device: str = 'cpu', dtype: torch.dtype = torch.float32, transform: RayTransformBundle = None):

        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)

        self.radius2 = torch.as_tensor(radius*radius, device=device, dtype=dtype)
        self.zero = torch.tensor([0.0], device=device, dtype=dtype)
        self.tmax = torch.tensor([2*torch.pi], device=device, dtype=dtype)

        self.disk = DiskSample(self.zero, self.radius2, self.zero, self.tmax)

    def sample_pos(self, N: int):

        return self.disk.sample(N)


class CollimatedLine(Bundle):

    def __init__(self, length: float,
                 ray_id: int, device: str = 'cpu', dtype: torch.dtype = torch.float32, transform: RayTransformBundle = None):

        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)

        self.length_2 = torch.tensor([length], device=device, dtype=dtype)
        self.x_sampler = Uniform(-self.length_2, self.length_2)

    def sample_pos(self, N: int):

        pos = torch.cat([
            self.x_sampler.sample((N,)),
            torch.zeros((N, 2), device=self.device, dtype=self.dtype)
        ], dim=1)

        return pos


class Fan(Bundle):

    """
    2D fan source,
    default fan spreads in y-direction
    """

    def __init__(self, angle: float,
                 ray_id: int, device: str = 'cpu', dtype: torch.dtype = torch.float32, transform: RayTransformBundle = None):

        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)

        self.angle_2 = torch.tensor([angle/2], device=device, dtype=dtype)
        self.theta_dist = Uniform(-self.angle_2, self.angle_2)

    def sample_dir(self, N):

        theta = self.theta_dist.sample((N,)).squeeze()

        return torch.stack([torch.zeros_like(theta), torch.sin(theta), torch.cos(theta)], dim=1)


class PointSource(Bundle):
    """
    Creates a diverging cone of rays from a point source.
    """

    def __init__(self, NA: float,
                 ray_id: int, device: str = 'cpu', dtype: torch.dtype = torch.float32, transform: RayTransformBundle = None):

        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)

        self.zero = torch.tensor([0.0], device=device, dtype=dtype)
        self.twopi = torch.tensor([2*torch.pi], device=device, dtype=dtype)

        rad = torch.arcsin(torch.tensor(NA, device=device, dtype=dtype))

        self.F_phi_max = SolidAngleSample.CDF_phi(rad)

        self.angle_dist = SolidAngleSample(self.zero, self.F_phi_max, self.zero, self.twopi)

    def sample_dir(self, N):

        phi, theta = self.angle_dist.sample(N)

        dz = torch.cos(phi)
        dr = torch.sin(phi)

        dx, dy = torch.cos(theta)*dr, torch.sin(theta)*dr

        return torch.stack([dx, dy, dz], dim=1)