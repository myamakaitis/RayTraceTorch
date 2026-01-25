import torch
import torch.nn.functional as F
from .bundle import SolidAngleSample, DiskSample, Bundle, PointSource
from torch.distributions import Uniform
from ..geom import RayTransformBundle

def LambertianSample():
    pass

class PanelSource(Bundle):

    def __init__(self, ray_id: int, device: str = 'cpu', dtype: torch.dtype = torch.float32,
                 transform: RayTransformBundle = None):
        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)

class RectangularPanel(PanelSource):
    def __init__(self, Width: float, Height: float):
        raise NotImplementedError()

class RingSource(PanelSource):
    def __init__(self, radius_inner: float, radius_outer: float, ):

        if radius_inner > radius_outer:
            raise ValueError()


def DiskLightPanel(height, width, center, light_direction, numerical_aperture, ):
    raise NotImplementedError()


def RingLightPanel():
    raise NotImplementedError()


