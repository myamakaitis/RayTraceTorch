import torch

from geom import RayTransform
from .ray import Rays
import torch.nn.functional as F
from torch.distributions import Uniform

class Bundle:

    def __init__(self, ray_id, device='cpu', dtype=torch.float32, transform: RayTransform = None):

        self.ray_id = ray_id
        self.device = device
        self.dtype = dtype
        self.transform = transform

    def sample_dir(self, N):
        raise NotImplementedError()

    def sample_pos(self, N):
        raise NotImplementedError()

    def sample(self, N):

        pos = self.sample_pos(N)
        dir = self.sample_dir(N)

        raise NotImplementedError() ### NEED TRANSFORM LOGIC

        return Rays.initialize(pos, dir, ray_id=self.ray_id, device=self.device, dtype=self.dtype)


class Collimated(Bundle):

    def __init__(self, transform: RayTransform, ray_id: int, device: str, dtype: torch.dtype):
        super().__init__(ray_id=ray_id, device=device, dtype=dtype)
        self.transform = transform

    def sample_dir(self, N):
        return torch.tensor([[0, 0, 1]], device=self.device, dtype=self.dtype).repeat(N, 1)


class Point(Bundle):

    def __init__(self, transform: RayTransform, ray_id: int, device: str, dtype: torch.dtype):
        super().__init__(ray_id=ray_id, device=device, dtype=dtype)
        self.transform = transform

    def sample_pos(self, N):

        return torch.zeros((N, 3), device=self.device, dtype=self.dtype)


class DiskSample:

    def __init__(self, radius_inner_2: torch.Tensor, radius_outer_2: torch.Tensor, theta_min: torch.Tensor, theta_max: torch.Tensor):

        self.r_distribution = Uniform(radius_inner_2, radius_outer_2)
        self.t_distribution = Uniform(theta_min, theta_max)

    def sample(self, N):

        theta = self.t_distribution.sample((N,))
        r = torch.sqrt(self.r_distribution.sample((N,)))

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        z = torch.zeros_like(x)

        return torch.stack([x, y, z], dim=1)

class SolidAngleSample:

    def __init__(self, F_phi_min: torch.Tensor, F_phi_max: torch.Tensor, theta_min: torch.Tensor, theta_max: torch.Tensor):

        self.phi_distribution = Uniform(F_phi_min, F_phi_max)
        self.t_distribution = Uniform(theta_min, theta_max)

    def sample(self, N):

        theta = self.t_distribution.sample((N,))
        phi = self.invCDF_phi(self.phi_distribution.sample((N,)))

        return phi, theta

    @classmethod
    def invCDF_phi(cls, phi : torch.Tensor):

        return torch.acos(-2*(phi)  + 1)

    @classmethod
    def CDF_phi(cls, phi : torch.Tensor):

        return 0.5 * (1 - torch.cos(phi))


class CollimatedDisk(Collimated):

    def __init__(self, radius: float, transform: RayTransform, ray_id: int, device: str, dtype: torch.dtype):

        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)

        self.radius2 = torch.as_tensor(radius*radius, device=device, dtype=dtype)
        self.zero = torch.tensor([0.0], device=device, dtype=dtype)
        self.tmax = torch.tensor([torch.pi], device=device, dtype=dtype)

        self.disk = DiskSample(self.zero, self.radius2, self.zero, self.tmax)

    def sample_pos(self, N):

        return self.disk.sample((N,))


class CollimatedLine(Collimated):

    def __init__(self, length: float, transform: RayTransform, ray_id: int, device: str, dtype: torch.dtype):

        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)

        self.length_2 = torch.tensor([length], device=device, dtype=dtype)
        self.x_sampler = Uniform(-self.length_2, self.length_2)

    def sample_pos(self, N):

        pos = torch.cat([
            self.x_sampler.sample((N,1)),
            torch.zeros((N, 2), device=self.device, dtype=self.dtype)
        ], dim=1)

        return pos


class Fan(Collimated):

    def __init__(self, angle: float, transform: RayTransform, ray_id: int, device: str, dtype: torch.dtype):

        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)

        self.angle_2 = torch.tensor([angle/2], device=device, dtype=dtype)


    def sample_dir(self, N):

        raise NotImplementedError()

class PointSource(Point):
    """
    Creates a diverging cone of rays from a point source.
    """

    def __init__(self, NA: float, transform: RayTransform, ray_id: int, device: str, dtype: torch.dtype):

        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)

        self.zero = torch.tensor([0.0], device=device, dtype=dtype)
        self.twopi = torch.tensor([2*torch.pi], device=device, dtype=dtype)

        rad = np.arcsin(NA)

        self.F_phi_max = SolidAngleSample.invCDF_phi(rad)

        self.angleSample = SolidAngleSample(self.zero, self.F_phi_max, self.zero, self.twopi)

    def sample_dir(self, N):


        collimated = torch.tensor([[0, 0, 1]], device=self.device, dtype=self.dtype).repeat(N, 1)

        phi, theta = self.phiSample.sample(N)

        # apply phi (rotation about x-axis)
        # apply theta (rotation about z-axis)

        return direction