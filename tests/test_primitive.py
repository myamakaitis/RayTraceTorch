
import torch
import math
import sys
import os

from geom import Plane, Sphere, Quadric, RayTransform
from rays import Rays


def create_rays(positions, directions):
    """Helper to create a Rays object batch."""
    return Rays(positions, directions)

def check_ray_equation(rays, t, hit_points):
    """
    Validates: Hit_Point == Origin + t * Direction
    """
    # We only check rays that actually hit (t < infinity)
    mask_hit = t < float('inf')

    if not mask_hit.any():
        return

    expected_pos = rays.pos[mask_hit] + t[mask_hit].unsqueeze(1) * rays.dir[mask_hit]
    actual_pos = hit_points[mask_hit]

    # Check distance between expected and actual
    error = torch.norm(expected_pos - actual_pos, dim=1)
    assert(torch.all(error < 1e-5), f"Ray equation failed. Max error: {error.max()}")

def test_Plane():
    print("\n--- Testing Plane (z=0) ---")

    planeTranslation = torch.zeros(3)
    plane = Plane(RayTransform(translation=planeTranslation))

    # 1. Rays pointing directly at plane from -Z
    pos = [[0, 0, -10], [1, 1, -5], [-2, -2, -10]]
    dir = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
    rays = create_rays(pos, dir)

    t, hits, norms, _ = plane(rays)

    # A. Validate Ray Equation
    check_ray_equation(rays, t, hits)

    # B. Validate Surface Equation (z should be 0)
    z_coords = hits[:, 2]
    assert(torch.allclose(z_coords, torch.zeros_like(z_coords), atol=1e-6),
                    f"Plane intersection z-coordinates not zero: {z_coords}")

    # C. Validate Normals (Should be 0,0,1)
    expected_norm = torch.tensor([0., 0., 1.])
    assert(torch.allclose(norms, expected_norm.expand_as(norms), atol=1e-6),
                    "Plane normals incorrect")

    print("Plane Test Passed.")


def test_Sphere():
    print("\n--- Testing Sphere (R=10) ---")
    R = torch.tensor(10.0)
    sphere = Sphere(radius=R)

    # Rays starting at origin (inside) and outside pointing in
    pos = [[0, 0, 0], [0, 0, -20], [0, 15, 0]]  # 3rd ray is outside R=10 pointing away (miss)
    dir = [[0, 0, 1], [0, 0, 1], [0, 1, 0]]
    rays = create_rays(pos, dir)

    t, hits, norms, _ = sphere(rays)

    # A. Validate Ray Equation
    check_ray_equation(rays, t, hits)

    # B. Validate Surface Equation (|P|^2 = R^2)
    # Only check valid hits (indices 0 and 1)
    valid_mask = t < float('inf')
    valid_hits = hits[valid_mask]

    dist_from_center = torch.norm(valid_hits, dim=1)
    assert(torch.allclose(dist_from_center, torch.tensor(R), atol=1e-4),
                    f"Points not on sphere surface. Radii: {dist_from_center}")

    # C. Check the Missed Ray (Index 2)
    assert(torch.isinf(t[2]), "Ray 2 should have missed sphere but returned finite t")

    print("Sphere Test Passed.")


def test_Quadric():
    print("\n--- Testing Quadric (Parabola) ---")
    # Parabola: c=0.1 (R=10), k=-1
    # Equation: z = (c * r^2) / (1 + sqrt(1 - (1+k)c^2 r^2))
    # For k=-1, term under sqrt becomes 1. Denom becomes 2.
    # z = c * r^2 / 2  =>  z = 0.05 * (x^2 + y^2)

    c = torch.tensor(0.1)
    k = torch.tensor(-1.0)
    quadric = Quadric(c=c, k=k)

    # Ray parallel to axis at y=2
    # Expected intersection:
    # z = 0.05 * (0^2 + 2^2) = 0.05 * 4 = 0.2
    pos = [[0, 2, -10], [0, 5, -10]]
    dir = [[0, 0, 1], [0, 0, 1]]
    rays = create_rays(pos, dir)

    t, hits, norms, _ = quadric(rays)

    check_ray_equation(rays, t, hits)

    # Check analytic solution for Ray 1 (y=2)
    # Expected z = 0.2
    assert(torch.abs(hits[0, 2] - 0.2) < 1e-5, f"Parabola z-height incorrect. Got {hits[0, 2]}, expected 0.2")

    # Check analytic solution for Ray 2 (y=5)
    # Expected z = 0.05 * 25 = 1.25
    assert(torch.abs(hits[1, 2] - 1.25) < 1e-5,
                    f"Parabola z-height incorrect. Got {hits[1, 2]}, expected 1.25")

    print("Quadric Parabola Test Passed.")

    print("\n--- Testing Quadric (General Implicit Check) ---")
    # c=0.1, k=0 (Sphere, effectively)
    # Implicit check: F(x,y,z) = c(x^2 + y^2) + c(1+k)z^2 - 2z = 0

    c = torch.tensor(0.1)
    k = torch.tensor(0.0)
    quadric = Quadric(c=c, k=k)

    # Random bundle of rays
    # Using a grid of rays to hit various parts
    rays = create_rays(
        positions=[[0, 0, -20], [1, 1, -20], [5, 5, -20], [0.1, 8, -20]],
        directions=[[0, 0, 1], [0, 0, 1], [0, 0.1, 1], [0, 0.1, 1]]
    )

    t, hits, norms, _ = quadric(rays)
    check_ray_equation(rays, t, hits)

    # Validate against Implicit Equation
    # c(x^2 + y^2) + c(1+k)z^2 - 2z = 0
    x, y, z = hits[:, 0], hits[:, 1], hits[:, 2]

    lhs = c * (x ** 2 + y ** 2) + c * (1 + k) * z ** 2 - 2 * z

    # Should be close to 0
    # Note: Tolerance slightly higher for quadrics due to iterative/float precision
    max_error = torch.max(torch.abs(lhs))
    print(f"Max Implicit Equation Error: {max_error:.2e}")
    assert(torch.allclose(lhs, torch.zeros_like(lhs), atol=1e-4),
                    "Quadric intersection points do not satisfy implicit surface equation.")

    print("Quadric Implicit Check Passed.")


def test_plane_translation_grad():
    """
    Verifies that gradients propagate correctly from the intersection point
    back to the Plane's translation parameters.
    """
    device = 'cpu'

    # 1. Setup the Parameter to Optimize (The Plane's Position)
    # We initialize it at (0, 0, 5) so the plane is at Z=5.
    # requires_grad=True is the modern equivalent of autograd.Variable
    plane_translation = torch.tensor([0.0, 0.0, 5.0], device=device, requires_grad=True)

    # 2. Setup the Geometry
    # We pass the tracked tensor into the transform
    transform = RayTransform(translation=plane_translation, trans_grad=True, rot_grad=False)
    plane = Plane(transform=transform)

    # 3. Define an Incoming Ray
    # Origin at (0,0,0), pointing at angle (Slope=1 in YZ plane)
    # Direction: (0, 0.707, 0.707)
    # This ensures the ray hits the plane at an angle, so Z-translation affects Y-position.
    origins = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    directions = torch.tensor([[0.0, 1.0, 1.0]], device=device)
    # Normalize manually or rely on Rays class
    directions = directions / torch.norm(directions, dim=1, keepdim=True)

    rays = Rays(origins, directions, device=device)

    # 4. Forward Pass: Intersect
    t, hit_point, normals, _ = plane(rays)

    # --- ANALYTIC CHECK (Forward) ---
    # Ray: P = t * [0, 1/sqrt(2), 1/sqrt(2)]
    # Plane: Z = 5
    # Intersection: t * 1/sqrt(2) = 5  =>  t = 5 * sqrt(2)
    # Hit Point: (0, 5, 5)
    print(f"\nHit Point: {hit_point.detach().cpu().numpy()}")
    assert torch.allclose(hit_point[0, 2], torch.tensor(5.0)), "Forward pass Z-intersection incorrect"
    assert torch.allclose(hit_point[0, 1], torch.tensor(5.0)), "Forward pass Y-intersection incorrect"

    # 5. Define Loss and Backward Pass
    # We want to see how the hit point changes if we move the plane.
    # Loss = Sum of all coordinates of the hit point (H_x + H_y + H_z)
    loss = hit_point.sum()

    # Clear previous grads (good practice, though fresh tensor here)
    if plane_translation.grad is not None:
        plane_translation.grad.zero_()

    loss.backward()

    for name, param in plane.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name}: {param.grad.data}")

    # 6. Verify Gradients (The Core Test)
    grads = transform.trans.grad
    # print(f"Gradients w.r.t Plane Translation (Tx, Ty, Tz): {grads.tolist()}")

    # A. Gradient w.r.t X and Y Translation (Tx, Ty)
    # An infinite plane at Z=5 is invariant to shifts in X and Y.
    # Moving the plane sideways should NOT change where the ray hits it.
    assert torch.isclose(grads[0], torch.tensor(0.0), atol=1e-5), \
        "Gradient w.r.t Tx should be 0 (Infinite plane invariance)"
    assert torch.isclose(grads[1], torch.tensor(0.0), atol=1e-5), \
        "Gradient w.r.t Ty should be 0 (Infinite plane invariance)"

    # B. Gradient w.r.t Z Translation (Tz)
    # Analytic derivation:
    # Ray: Y = Z (since dy=dz=0.707)
    # Plane: Z = Tz
    # Hit Point: H = (0, Tz, Tz)
    # Loss = H_x + H_y + H_z = 0 + Tz + Tz = 2 * Tz
    # d(Loss) / d(Tz) should be 2.0
    expected_grad_z = 2.0
    assert torch.isclose(grads[2], torch.tensor(expected_grad_z), atol=1e-5), \
        f"Gradient w.r.t Tz incorrect. Got {grads[2]}, expected {expected_grad_z}"

def test_quadratic_translation_grad():
    """
    Verifies that gradients propagate correctly from the intersection point
    back to the Plane's translation parameters.
    """
    device = 'cpu'

    # 1. Setup the Parameter to Optimize (The Plane's Position)
    # We initialize it at (0, 0, 5) so the plane is at Z=5.
    # requires_grad=True is the modern equivalent of autograd.Variable
    plane_translation = torch.tensor([0.0, 0.0, 5.0], device=device, requires_grad=True)

    # 2. Setup the Geometry
    # We pass the tracked tensor into the transform
    transform = RayTransform(translation=plane_translation, trans_grad=True, rot_grad=False)
    quadric = Quadric(transform=transform, c=0.01, k=0.0, c_grad=True)

    # 3. Define an Incoming Ray
    # Origin at (0,0,0), pointing at angle (Slope=1 in YZ plane)
    # Direction: (0, 0.707, 0.707)
    # This ensures the ray hits the plane at an angle, so Z-translation affects Y-position.
    origins = torch.tensor([[5.0, 0.0, 0.0]], device=device)
    directions = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    # Normalize manually or rely on Rays class
    directions = directions / torch.norm(directions, dim=1, keepdim=True)

    rays = Rays(origins, directions, device=device)

    # 4. Forward Pass: Intersect
    t, hit_point, normals, _ = quadric(rays)

    # --- ANALYTIC CHECK (Forward) ---
    # Ray: P = t * [0, 1/sqrt(2), 1/sqrt(2)]
    # Plane: Z = 5
    # Intersection: t * 1/sqrt(2) = 5  =>  t = 5 * sqrt(2)
    # Hit Point: (0, 5, 5)
    print(f"\nHit Point: {hit_point.detach().cpu().numpy()}")

    # 5. Define Loss and Backward Pass
    # We want to see how the hit point changes if we move the plane.
    # Loss = Sum of all coordinates of the hit point (H_x + H_y + H_z)
    loss = hit_point.sum()

    loss.backward()

    for name, param in quadric.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name}: {param.grad.data}")

    # 6. Verify Gradients (The Core Test)
    grads = transform.trans.grad
    # print(f"Gradients w.r.t Plane Translation (Tx, Ty, Tz): {grads.tolist()}")


    # B. Gradient w.r.t Z Translation (Tz)
    # Analytic derivation:
    # Ray: Y = Z (since dy=dz=0.707)
    # Plane: Z = Tz
    # Hit Point: H = (0, Tz, Tz)
    # Loss = H_x + H_y + H_z = 0 + Tz + Tz = 2 * Tz
    # d(Loss) / d(Tz) should be 2.0
    expected_grad_z = 1.0
    assert torch.isclose(grads[2], torch.tensor(expected_grad_z), atol=1e-5), \
        f"Gradient w.r.t Tz incorrect. Got {grads[2]}, expected {expected_grad_z}"