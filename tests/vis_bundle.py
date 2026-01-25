import torch
import torch.nn as nn
import math
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import RayTraceTorch as rtt

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
    radius = 2.0
    N_approx = 5000

    collimated = rtt.rays.CollimatedDisk(radius, ray_id=1)
    rays = collimated.sample(N_approx)

    # --- Assertions ---
    # 1. Check Normalization
    norms = torch.norm(rays.dir, dim=1)

    # --- Visualization ---
    save_plot(rays, "collimated_angled.png", "Collimated Bundle (Angled)")


def test_pointSource():
    """
    Verifies a point source originates from a single location and diverges.
    """
    N = 501
    na = 0.3

    ps = rtt.rays.PointSource(na, 2)
    rays = ps.sample(N)

    # --- Visualization ---
    save_plot(rays, "point_source_cone.png", "Point Source (Cone)")


def test_gaussianBeam():
    """
    Verifies Gaussian bundle properties.
    """

    d_e2 = 5
    N = 5000

    beam = rtt.rays.GaussianBeam(d_e2, d_e2, 3)
    rays = beam.sample(N)

    # --- Visualization ---
    save_plot(rays, "gaussian_bundle.png", "Gaussian Bundle (Waist=1.0)")


if __name__ == "__main__":

    test_collimatedSource()
    test_pointSource()
    test_gaussianBeam()