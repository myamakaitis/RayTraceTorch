import torch
import torch.nn.functional as F
from .rays import

class Illuminator():

    def __init__(self, batch_shape, event_shape):

        super().__init__(self, batch_shape=batch_shape, event_shape=event_shape)

        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()


class RandomPointSource(Illuminator):

    def __init__(self):
        raise NotImplementedError()

def panelSource(height, width, center, light_direction, numerical_aperture, ):
    raise NotImplementedError()

def ringLight():
    raise NotImplementedError()


