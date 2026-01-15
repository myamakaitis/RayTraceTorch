import pytest
import torch
import os
import sys
import matplotlib.pyplot as plt

# Add src to path so we can import modules without installing
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rays.bundle import collimatedSource, pointSource, gaussianBeam

# Output directory for visual checks
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'test_plots')
os.makedirs(PLOT_DIR, exist_ok=True)


def save_plot(rays, filename, title):
    """
    Helper to save projections of the ray bundle for manual visual verification.
    """
    pos = rays.pos.detach().cpu().numpy()
    d = rays.dir.detach().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    # Scale arrow length
    s = 0.5

    # XY
    axs[0].scatter(pos[:, 0], pos[:, 1], s=2, alpha=0.5)
    axs[0].quiver(pos[:, 0], pos[:, 1], d[:, 0], d[:, 1], angles='xy', scale_units='xy', scale=1 / s, color='r',
                  alpha=0.3)
    axs[0].set_title("XY (Front)")
    axs[0].axis('equal')

    # XZ
    axs[1].scatter(pos[:, 0], pos[:, 2], s=2, alpha=0.5)
    axs[1].quiver(pos[:, 0], pos[:, 2], d[:, 0], d[:, 2], angles='xy', scale_units='xy', scale=1 / s, color='r',
                  alpha=0.3)
    axs[1].set_title("XZ (Top)")
    axs[1].axis('equal')

    # YZ
    axs[2].scatter(pos[:, 1], pos[:, 2], s=2, alpha=0.5)
    axs[2].quiver(pos[:, 1], pos[:, 2], d[:, 1], d[:, 2], angles='xy', scale_units='xy', scale=1 / s, color='r',
                  alpha=0.3)
    axs[2].set_title("YZ (Side)")
    axs[2].axis('equal')

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path)
    plt.close(fig)
    print(f"Saved plot to: {path}")


def test_collimatedSource():
    """
    Verifies a collimated beam creates parallel rays within a circular footprint.
    """
    origin = [0.0, 10.0, 0.0]
    direction = [0.0, 1.0, 1.0]  # Angled 45 deg up-right
    radius = 2.0
    N_approx = 100

    rays = collimatedSource(origin, direction, radius, N_approx)

    # --- Assertions ---
    # 1. Check Normalization
    norms = torch.norm(rays.dir, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Directions not normalized"

    # 2. Check Direction Consistency (all rays should be parallel)
    # The dot product of every ray direction with the target direction should be ~1.0
    target_dir = torch.tensor(direction, dtype=torch.float32)
    target_dir = target_dir / torch.norm(target_dir)
    dot_products = (rays.dir @ target_dir)
    assert torch.allclose(dot_products, torch.ones_like(dot_products), atol=1e-5), "Rays are not parallel"

    # 3. Check Radius (Origins should fall within the radius relative to center)
    # We project origins onto the plane perpendicular to direction to check radius
    center = torch.tensor(origin, dtype=torch.float32)
    rel_pos = rays.pos - center
    # Distance from the central axis line
    # dist = ||rel_pos - (rel_pos . dir) * dir||
    proj_on_axis = (rel_pos @ target_dir).unsqueeze(1) * target_dir
    perp_dist = torch.norm(rel_pos - proj_on_axis, dim=1)

    assert torch.all(perp_dist <= radius + 1e-4), "Rays generated outside specified radius"

    # --- Visualization ---
    save_plot(rays, "collimated_angled.png", "Collimated Bundle (Angled)")


def test_pointSource():
    """
    Verifies a point source originates from a single location and diverges.
    """
    origin = [0.0, 0.0, -5.0]
    direction = [0.0, 0.0, 1.0]
    angle_rad = 0.5  # ~28 degrees
    N = 200

    rays = pointSource(origin, direction, angle_rad, N)

    # --- Assertions ---
    # 1. Check Origins (All should be exactly at the source)
    center = torch.tensor(origin, dtype=torch.float32).expand(N, 3)
    assert torch.allclose(rays.pos, center, atol=1e-5), "Point source rays do not originate from same point"

    # 2. Check Cone Angle
    # Dot product with axis should be >= cos(angle)
    target_dir = torch.tensor(direction, dtype=torch.float32)
    target_dir = target_dir / torch.norm(target_dir)
    cos_vals = (rays.dir @ target_dir)
    min_cos = torch.cos(torch.tensor(angle_rad))

    assert torch.all(cos_vals >= min_cos - 1e-4), "Rays generated outside cone angle"

    # --- Visualization ---
    save_plot(rays, "point_source_cone.png", "Point Source (Cone)")


def test_gaussianBeam():
    """
    Verifies Gaussian bundle properties.
    """
    origin = [5.0, 5.0, 0.0]
    direction = [0.0, 0.0, -1.0]
    waist = 1.0
    N = 500

    rays = gaussianBeam(origin, direction, waist, N)

    # --- Assertions ---
    # 1. Check Directions (Parallel)
    target_dir = torch.tensor(direction, dtype=torch.float32)
    target_dir = target_dir / torch.norm(target_dir)
    dot_products = (rays.dir @ target_dir)
    assert torch.allclose(dot_products, torch.ones_like(dot_products), atol=1e-5), "Gaussian rays not parallel"

    # 2. Check Statistical Distribution (Loose check)
    # The standard deviation of positions should be roughly waist/2
    center = torch.tensor(origin, dtype=torch.float32)
    rel_pos = rays.pos - center
    # Project to 2D plane orthogonal to Z (since dir is -Z, we check X and Y)
    std_x = torch.std(rel_pos[:, 0])
    std_y = torch.std(rel_pos[:, 1])
    expected_std = waist / 2.0

    # Allow 15% variance due to random sampling
    assert abs(std_x - expected_std) < 0.15 * expected_std
    assert abs(std_y - expected_std) < 0.15 * expected_std

    # --- Visualization ---
    save_plot(rays, "gaussian_bundle.png", "Gaussian Bundle (Waist=1.0)")
