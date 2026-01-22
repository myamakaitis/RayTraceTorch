import torch
import math
from .ray import Rays
import torch.nn.functional as F

def _generate_basis_from_direction(direction, device):
    """
    Constructs an orthonormal basis [u, v, w] where w is aligned with 'direction'.
    This allows us to create 2D shapes (disks/cones) and rotate them to point anywhere.
    """
    w = torch.as_tensor(direction, dtype=torch.float32, device=device)
    w = F.normalize(w, p=2, dim=0)  # Normalize primary axis

    # Arbitrary 'up' vector to compute orthogonal axes
    # If w is close to global Y [0,1,0], use X [1,0,0] as temp up to avoid singularity
    if torch.abs(w[1]) > 0.99:
        temp_up = torch.tensor([1.0, 0.0, 0.0], device=device)
    else:
        temp_up = torch.tensor([0.0, 1.0, 0.0], device=device)

    u = torch.cross(temp_up, w)
    u = F.normalize(u, p=2, dim=0)

    v = torch.cross(w, u)
    v = F.normalize(v, p=2, dim=0)

    # Return shape [3, 3] where rows are u, v, w
    # We can use this to rotate local vectors: v_global = v_local @ basis
    return torch.stack([u, v, w])


def collimatedSource(origin, direction, radius, N_rays, ray_id=0, device='cpu'):
    """
    Creates a cylindrical beam of parallel rays centered at 'origin' and pointing in 'direction'.

    Args:
        origin (list/tensor): [x, y, z] center of the beam.
        direction (list/tensor): [x, y, z] propagation vector.
        radius (float): Beam radius.
        N_rays (int): Approximate number of rays (creates a square grid masked to a circle).
        ray_id (int): Bundle ID.
        device (str): Computation device.
    """
    origin = torch.as_tensor(origin, dtype=torch.float32, device=device)

    # 1. Create a 2D Grid on the local XY plane (z=0)
    grid_side = int(math.sqrt(N_rays))
    if grid_side < 1: grid_side = 1

    t = torch.linspace(-radius, radius, grid_side, device=device)
    grid_y, grid_x = torch.meshgrid(t, t, indexing='ij')

    flat_x = grid_x.flatten()
    flat_y = grid_y.flatten()

    # 2. Mask to circle
    mask = (flat_x ** 2 + flat_y ** 2) <= radius ** 2
    local_x = flat_x[mask]
    local_y = flat_y[mask]
    local_z = torch.zeros_like(local_x)

    num_valid = local_x.shape[0]

    # Stack to create local points [N, 3] centered at (0,0,0) facing +Z
    local_points = torch.stack([local_x, local_y, local_z], dim=1)

    # 3. Rotate and Translate
    # Get rotation basis matrix R [3,3]
    R = _generate_basis_from_direction(direction, device)

    # Rotate: points_global = points_local @ R
    # (Using matmul logic: [N,3] x [3,3] -> [N,3])
    # Note: Our basis rows are u,v,w.
    # local point (x,y,0) -> x*u + y*v + 0*w.
    points_rotated = local_points @ R

    # Translate
    final_origins = points_rotated + origin

    # 4. Directions
    # All rays point in the bundle direction
    final_dirs = torch.as_tensor(direction, dtype=torch.float32, device=device)
    final_dirs = final_dirs.unsqueeze(0).expand(num_valid, 3)  # Broadcast

    return Rays.initialize(final_origins, final_dirs, ray_id=ray_id, device=device)


def collimatedLineSource(center, ray_direction, line_direction, length, N_rays, ray_id=1, 
                         device='cpu', dtype=torch.float32):

    direction = torch.as_tensor(ray_direction).repeat(N_rays, 1)
    
    line_direction = F.normalize(torch.as_tensor(line_direction), 2)

    origins = (length * torch.linspace(-0.5, 0.5, N_rays)[:, None] * line_direction[None, :]
               + torch.as_tensor(center)[None, :])
    
    return Rays.initialize(origins, direction, ray_id=ray_id, device=device, dtype=dtype)


def fanSource(origin, ray_direction, fan_angle, fan_direction, N_rays, ray_id=1, device='cpu', dtype=torch.float32):
    
    origin = torch.as_tensor(origin).repeat(N_rays, 1)

    thetas = torch.linspace(-fan_angle/2, fan_angle/2, N_rays)
    directions = (torch.as_tensor(ray_direction)[None, :] * torch.cos(thetas)[:, None]
                  + torch.as_tensor(fan_direction)[None, :] * torch.sin(thetas)[:, None])
    directions = F.normalize(directions)


    return Rays.initialize(origin, directions, ray_id=ray_id, device=device, dtype = dtype)


def pointSource(origin, direction, half_angle_rad, N_rays, ray_id=1, device='cpu'):
    """
    Creates a diverging cone of rays from a point source.

    Args:
        origin (list/tensor): [x, y, z] point source location.
        direction (list/tensor): [x, y, z] center axis of the cone.
        half_angle_rad (float): The half-angle of the cone in Radians.
                                (Solid Angle Omega = 2*pi*(1 - cos(theta)))
        N_rays (int): Number of rays.
        ray_id (int): Bundle ID.
    """
    origin = torch.as_tensor(origin, dtype=torch.float32, device=device)

    # 1. Sample directions in Local Space (around +Z axis)
    # Uniform sampling on a spherical cap
    phi = torch.rand(N_rays, device=device) * 2 * math.pi

    # cos(theta) ranges from 1.0 (center) to cos(half_angle)
    max_cos = math.cos(half_angle_rad)
    cos_theta = torch.rand(N_rays, device=device) * (1 - max_cos) + max_cos
    sin_theta = torch.sqrt(1 - cos_theta ** 2)

    # Local cartesian directions (z is primary axis)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = cos_theta

    local_dirs = torch.stack([x, y, z], dim=1)  # [N, 3]

    # 2. Rotate Directions to match target vector
    R = _generate_basis_from_direction(direction, device)

    final_dirs = local_dirs @ R

    # 3. Origins (all at the source point)
    final_origins = origin.repeat(N_rays, 1)

    return Rays.initialize(final_origins, final_dirs, ray_id=ray_id, device=device)


