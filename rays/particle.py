from geom import RayTransform
from .bundle import Bundle
import torch

class LambertianSphere:

    def __init__(self, radius,
                 ray_id: int, device: str = 'cpu', dtype: torch.dtype = torch.float32, transform: RayTransform = None):
        raise NotImplementedError

class MieScatter(Bundle):

    def __init__(self, particle_size_nm, wavelength_nm, particle_ior, enviornment_ior,
                 ray_id: int, device: str = 'cpu', dtype: torch.dtype = torch.float32, transform: RayTransform = None):

        raise NotImplementedError

class RayleightScatter(Bundle):

    def __init__(self, particle_size_nm, wavelength_nm, particle_ior, enviornment_ior,
                 ray_id: int, device: str = 'cpu', dtype: torch.dtype = torch.float32, transform: RayTransform = None):
        raise NotImplementedError