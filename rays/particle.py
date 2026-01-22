from geom import RayTransform
from .illuminator import Illuminator

class MieScatter(Illuminator):

    def __init__(self, particle_size_nm, wavelength_nm, particle_ior, enviornment_ior, transform: RayTransform):

        raise NotImplementedError

class RayleightScatter(Illuminator):

    def __init__(self, particle_size_nm, wavelength_nm, particle_ior, enviornment_ior, transform: RayTransform):
        raise NotImplementedError