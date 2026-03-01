from typing import Optional, Union

from .parent import Element
from ..geom import Disk, Rectangle, Ellipse, RayTransform
from ..phys import ApertureFilter


class CircularAperture(Element):

    def __init__(self, radius: float,
                 invert: bool = False,
                 transform: Optional[Union[RayTransform, None]] = None):

        super().__init__()

        self.shape = Disk(radius=radius, invert=invert, transform=transform)
        self.surface_functions.append(ApertureFilter(self.shape.inBounds))

    @property
    def radius(self):
        return self.shape.radius


class RectangularAperture(Element):

    def __init__(self, half_x: float, half_y: float,
                 invert: bool = False,
                 transform: Optional[Union[RayTransform, None]] = None):

        super().__init__()

        self.shape = Rectangle(half_x=half_x, half_y=half_y, invert=invert, transform=transform)
        self.surface_functions.append(ApertureFilter(self.shape.inBounds))

    @property
    def half_x(self):
        return self.shape.hx

    @property
    def half_y(self):
        return self.shape.hy


class EllipticAperture(Element):

    def __init__(self, r_major: float, r_minor: float, rot: float = 0.0,
                 invert: bool = False,
                 transform: Optional[Union[RayTransform, None]] = None):

        super().__init__()

        self.shape = Ellipse(r_major=r_major, r_minor=r_minor, rot=rot,
                             invert=invert, transform=transform)
        self.surface_functions.append(ApertureFilter(self.shape.inBounds))

    @property
    def r_major(self):
        return self.shape.r_major

    @property
    def r_minor(self):
        return self.shape.r_minor
