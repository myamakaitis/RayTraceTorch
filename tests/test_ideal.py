import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from RayTraceTorch.elements import IdealThinLens
from RayTraceTorch.rays import Rays, pointSource
from RayTraceTorch.geom import RayTransform


def find_point_of_least_confusion_svd(rays_pos, rays_dir):
    """
    Finds the 3D point X that minimizes the sum of squared distances to a set of rays.
    Method: Solves the linear system A * X = b using SVD (via pseudo-inverse).

    Cost Function: sum || (I - d_i * d_i^T) * (X - p_i) ||^2
    Solution: (sum(I - d_i * d_i^T)) * X = sum((I - d_i * d_i^T) * p_i)

    Args:
        rays_pos (Tensor): [N, 3] Ray origins
        rays_dir (Tensor): [N, 3] Ray directions (normalized)

    Returns:
        intersection (Tensor): [3] The optimal intersection point.
    """
    # 1. Construct Projector Matrices: M_i = I - d_i @ d_i.T
    # Shape: [N, 3, 3]
    I = torch.eye(3, device=rays_pos.device).unsqueeze(0)  # [1, 3, 3]

    # Outer product of directions: [N, 3, 1] @ [N, 1, 3] -> [N, 3, 3]
    d_outer = torch.matmul(rays_dir.unsqueeze(2), rays_dir.unsqueeze(1))

    M = I - d_outer  # [N, 3, 3]

    # 2. Formulate Linear System A*x = b
    # A = sum(M_i) -> [3, 3]
    A = torch.sum(M, dim=0)

    # b = sum(M_i @ p_i) -> [3]
    # M [N, 3, 3] @ pos [N, 3, 1] -> [N, 3, 1] -> sum -> [3, 1]
    b = torch.sum(torch.matmul(M, rays_pos.unsqueeze(2)), dim=0).squeeze(1)

    # 3. Solve using SVD (Pseudo-Inverse) for robustness against singularities
    # x = pinv(A) @ b
    # torch.linalg.pinv uses SVD internally.
    A_pinv = torch.linalg.pinv(A)
    intersection = torch.matmul(A_pinv, b)

    return intersection


def test_thin_lens_conjugate_points():
    """
    Verifies that a point source at 2f is imaged to 2f.
    Formula: 1/so + 1/si = 1/f
    If f=100, so=200, then si=200.
    """
    print("\n--- Testing Thin Lens Conjugate Points (2f -> 2f) ---")

    # Setup
    # Note: Using negative focal length to get converging behavior due to Linear implementation
    f_physical = 100.0
    lens = IdealThinLens(focal=f_physical)



    # 1. Define Point Source at Z = -200 (so = 200)
    so = 2 * f_physical
    origin = [0.0, 0.0, -so]
    direction = [0.0, 0.0, 1.0]  # Pointing +Z

    # Create a cone of rays
    rays = pointSource(origin, direction, half_angle_rad=0.05, N_rays=100)

    # 2. Intersect
    # IdealThinLens has only 1 surface at index 0
    new_pos, new_dir, _ = lens(rays, surf_idx=0)

    # 3. Calculate Image Position
    # The rays leave the lens at z=0 with new_dir.
    # We find where they cross the optical axis (x=0, y=0).
    # Using the X-component: x(t) = pos_x + t * dir_x = 0  => t = -pos_x / dir_x
    # We filter out rays close to the axis (dir_x ~ 0) to avoid division by zero.

    mask = torch.abs(new_dir[:, 0]) > 1e-5
    valid_pos = new_pos[mask]
    valid_dir = new_dir[mask]

    t_vals = -valid_pos[:, 0] / valid_dir[:, 0]

    # Z_image = Z_lens + t * dir_z (Z_lens is approx 0)
    z_image_vals = valid_pos[:, 2] + t_vals * valid_dir[:, 2]

    mean_si = torch.mean(z_image_vals)
    std_si = torch.std(z_image_vals)

    print(f"Object Z: {-so}")
    print(f"Target Image Z: {so}")
    print(f"Calculated Image Z: {mean_si.item():.4f} (Std Dev: {std_si.item():.4f})")

    assert torch.allclose(mean_si, torch.tensor(so), atol=1e-1), \
        f"Image not formed at expected distance. Got {mean_si}, expected {so}"

    # Verify stigmatic imaging (low standard deviation)
    assert std_si < 1e-3, "Rays did not converge to a sharp point"
    print("Conjugate Point Test Passed.")

    print(f"Paraxial Lens matrix")
    print(lens.getParaxial()[1][0].numpy())

    print(f"Paraxial lens matrix (shifted)")
    lens.shape.transform.trans[0] = 4
    print(lens.getParaxial()[1][0].numpy())


def test_magnification_and_gradients():
    """
    Verifies lateral magnification and axial gradients.

    1. Axial Magnification / Sensitivity:
       d(si)/d(so). Formula: si = (f*so)/(so-f).
       Deriv: d(si)/d(so) = -(si/so)^2  (Note: dZi/dZo = (Zi/Zo)^2)

    2. Lateral Magnification:
       M = hi/ho = -si/so.
    """
    print("\n--- Testing Magnification and Gradients ---")

    f_physical = 100.0
    lens = IdealThinLens(focal=f_physical)

    # --- Setup Differentiable Object Position ---
    # Object at Z = -300 (3f). Expected Image at 1.5f = 150.
    z_obj_val = -300.0

    # Create single off-axis ray to test lateral height as well
    # Height ho = 10.0

    # We manually construct the ray to ensure graph connectivity to z_obj
    # Origin: [10, 0, z_obj]
    origin = torch.tensor([0, 0, z_obj_val], requires_grad=True)
    # New setup for Gradient Test:


    with torch.no_grad():

        thetas = torch.linspace(-torch.pi/6, torch.pi/6, 100)

        dirs = torch.stack([torch.sin(thetas), torch.zeros_like(thetas), torch.cos(thetas)], dim=-1)
        start = torch.zeros_like(dirs)

        rays_grad = Rays(start, dirs)


    rays_grad.pos = rays_grad.pos + origin[None, :]
    # Trace
    # We hit the lens at exactly h_hit (if we ignore the small z-shift of the plane? Plane is at 0).
    out_pos, out_dir, _ = lens(rays_grad, surf_idx=0)

    # Calculate Image Z (Intersection with axis)
    # Ray is converging. Hits axis at Z_img.
    # x(t) = out_pos_x + t * out_dir_x = 0  => t = -out_pos_x / out_dir_x

    img = find_point_of_least_confusion_svd(out_pos, out_dir)

    img_sum = torch.sum(img)
    img_sum.backward()


    # 2. Verify Axial Gradient
    # Derivative d(si)/d(so)
    # Theoretical: dZi/dZo = (Zi / Zo)^2 = (150 / -300)^2 = (-0.5)^2 = 0.25

    Mz_theoretical = (150.0 / -300.0) ** 2
    Mxy_theoretical = (150.0 / -300.0)

    M = origin.grad.detach()
    Mx, My, Mz = M[0], M[1], M[2]

    print(f"Gradient d(Zi)/d(Zo): Pytorch={Mz:.4f}, Theoretical={Mz_theoretical:.4f}")
    assert abs(Mz - Mz_theoretical) < 1e-3, "Axial gradient mismatch"

    print(f"Lateral Magnification: Pytorch={Mx:.4f}, Pytorch={My:.4f}, Theoretical={Mxy_theoretical:.4f}")
    assert (abs(Mx - Mxy_theoretical) < 1e-3), "Lateral Magnification mismatch"
    assert (abs(My - Mxy_theoretical) < 1e-3), "Lateral Magnification mismatch"
    print("Magnification and Gradient Test Passed.")