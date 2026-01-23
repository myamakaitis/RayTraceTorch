import torch
import torch.nn.functional as F
from .bundle import Bundle

class PointSource(Bundle):

    def __init__(self):
        raise NotImplementedError()


class PanelSource(PointSource):

    def __init__(self):
        raise NotImplementedError()


class RectangularPanel(PanelSource):
    def __init__(self, Width: float, Height: float):
        raise NotImplementedError()

class RingSource(PanelSource):
    def __init__(self, radius_inner: float, radius_outer: float, ):

        if radius_inner > radius_outer:
            raise ValueError()

def panelSource(height, width, center, light_direction, numerical_aperture, ):
    raise NotImplementedError()

def ringLight():
    raise NotImplementedError()


