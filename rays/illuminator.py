
from torch.distriubtions import Distribution, constraints

from geom import RayTransform


class Illuminator():

    def __init__(self, ray_id, device):

        raise NotImplementedError()

    def sample(self, Nsamples):
        raise NotImplementedError()


class RandPointSource(Illuminator):

    def __init__(self, center, light_direction, numerical_aperture, ray_id, device):

        super().__init__(ray_id, device)

        raise NotImplementedError()

class RandCollimatedSource(Illuminator):

    def __init__(self, center, sample_shape, light_direction, ray_id, device):

        super().__init__(ray_id, device)

        raise NotImplementedError()

class PanelSource(RandPointSource):

    def __init__(self, sample_shape, light_direction, transform: RayTransform, ray_id, device):

        super().__init_([0, 0, 0], [0, 0, 1], )

class LambertianPanelSource(Illuminator):

    def __init__(self, center, light_direction, transform: RayTransform, ray_id, device):

def gaussianBeam(origin, direction, waist_radius, N_rays, ray_id=0, device='cpu'):
    """
    Creates a Gaussian beam distribution (Ray density follows Gaussian intensity).

    Args:
        origin (list/tensor): [x, y, z] Center of the beam waist.
        direction (list/tensor): [x, y, z] Propagation direction.
        waist_radius (float): The 1/e^2 intensity radius (w0).
        N_rays (int): Exact number of rays to generate.
    """
    origin = torch.as_tensor(origin, dtype=torch.float32, device=device)

    # 1. Sample Gaussian Distribution in Local XY
    # For intensity I(r) ~ exp(-2r^2/w0^2), the position standard deviation sigma = w0 / 2.
    sigma = waist_radius / 2.0

    local_x = torch.randn(N_rays, device=device) * sigma
    local_y = torch.randn(N_rays, device=device) * sigma
    local_z = torch.zeros(N_rays, device=device)

    local_points = torch.stack([local_x, local_y, local_z], dim=1)

    # 2. Rotate and Translate
    R = _generate_basis_from_direction(direction, device)
    points_rotated = local_points @ R
    final_origins = points_rotated + origin

    # 3. Directions (Parallel at the waist)
    final_dirs = torch.as_tensor(direction, dtype=torch.float32, device=device)
    final_dirs = final_dirs.unsqueeze(0).repeat(N_rays, 1)

    return Rays(final_origins, final_dirs, ray_id=ray_id, device=device)






