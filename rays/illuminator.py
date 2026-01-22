
from torch.distriubtions import Distribution, constraints

class Illuminator(Distribution):

    def __init__(self, batch_shape, event_shape):

        super().__init__(self, batch_shape=batch_shape, event_shape=event_shape)

        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()


class RandomPointSource(Illuminator):

    def __init__(self):
        raise NotImplementedError()



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

def panelSource(height, width, center, light_direction, numerical_aperture, ):
    raise NotImplementedError()

def ringLight()


