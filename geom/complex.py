from .primitives import Surface

class Aspheric(Surface):
    def __init__(self, Coefficients, device = "cpu", transform = None):

        super().__init__(transform, device)
        raise NotImplementedError("Aspheric Not Implemented")

