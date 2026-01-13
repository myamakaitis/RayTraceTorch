import torch
import matplotlib.pyplot as plt
import sys
import os
import numpy as np


# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from geom import Singlet, RayTransform
from geom import Doublet, Triplet, CylSinglet
from geom import Box
from rays import Rays


def scan_lens_profile(lens, axis='x', num_points=200):
    """
    Fires a row of parallel rays through the lens to map its profile.

    Args:
        lens: The shape object.
        axis: 'x' for XZ profile (y=0), 'y' for YZ profile (x=0).

    Returns:
        profiles: List of (h, z) arrays, one for each surface found.
    """
    # Determine scan width based on available attributes
    if hasattr(lens, 'radius'):
        width = 2*lens.radius.item()
    else:
        width = 3  # Default fallback

    margin = 1.2  # Scan slightly beyond bounds to catch edges
    extent = width * margin

    # Create scan coordinates
    coords = torch.linspace(-extent / 2, extent / 2, num_points)
    zeros = torch.zeros_like(coords)

    # Rays start at Z = -100, pointing +Z
    if axis == 'x':
        origins = torch.stack([coords, zeros, torch.full_like(coords, -100.0)], dim=1)
        directions = torch.tensor([[0.0, 0.0, 1.0]]).expand(num_points, 3)
    else:
        origins = torch.stack([zeros, coords, torch.full_like(coords, -100.0)], dim=1)
        directions = torch.tensor([[0.0, 0.0, 1.0]]).expand(num_points, 3)

    # Create Rays
    rays = Rays(origins, directions)

    # Intersect
    # t_matrix: [N, NumSurfaces]
    t_matrix = lens.intersectTest(rays)

    # Convert t to z: z = origin_z + t * dir_z (where dir_z=1)
    # z = -100 + t
    profiles = []

    # Iterate over columns (surfaces)
    num_surfaces = t_matrix.shape[1]

    for i in range(num_surfaces):
        t = t_matrix[:, i]
        mask = t < float('inf')

        if mask.any():
            h_valid = coords[mask].cpu().detach().numpy()
            z_valid = (-100.0 + t[mask]).cpu().detach().numpy()
            profiles.append((h_valid, z_valid))
        else:
            profiles.append(([], []))

    return profiles


def plot_lens(lens, title, filename):
    """Generates XZ and YZ projection plots."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- XZ Projection ---
    profiles_x = scan_lens_profile(lens, axis='x')
    ax = axes[0]
    for i, (h, z) in enumerate(profiles_x):
        if len(h) == 0: continue

        # For lenses: Last surface is usually the Edge Cylinder.
        # For Box: All surfaces are Planes.
        label = f"Surf {i}"

        ax.plot(h, z, '.', markersize=2, label=label)

    ax.set_title(f"{title} - XZ Projection")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    # ax.legend()

    # --- YZ Projection ---
    profiles_y = scan_lens_profile(lens, axis='y')
    ax = axes[1]
    for i, (h, z) in enumerate(profiles_y):
        if len(h) == 0: continue
        ax.plot(h, z, '.', markersize=2, label=f"Surf {i}")

    ax.set_title(f"{title} - YZ Projection")
    ax.set_xlabel("Y (mm)")
    ax.set_ylabel("Z (mm)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Save
    save_path = os.path.join(os.path.dirname(__file__), 'test_plots', filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()


def test_lenses():
    print("--- Testing Lens Geometry Visualization ---")

    # 1. Bi-Convex Singlet
    # R1=50 (Center Right), R2=-50 (Center Left), D=25, T=10
    s1 = Singlet(C1=1/torch.tensor(50.0), C2=1/torch.tensor(-50.0), D=torch.tensor(25.0), T=torch.tensor(10.0))
    plot_lens(s1, "Bi-Convex Singlet", "singlet_biconvex.png")

    # 2. Meniscus Singlet
    # R1=30 (Convex), R2=100 (Concave, following), D=20, T=5
    s2 = Singlet(C1=1/torch.tensor(30.0), C2=1/torch.tensor(100.0), D=torch.tensor(20.0), T=torch.tensor(5.0))
    plot_lens(s2, "Meniscus Singlet", "singlet_meniscus.png")

    # 3. Plano-Convex Singlet
    # R1=Inf, R2=-40, D=25, T=8
    s3 = Singlet(C1=torch.tensor(0.0), C2=1/torch.tensor(-40.0), D=torch.tensor(25.0), T=torch.tensor(8.0))
    plot_lens(s3, "Plano-Convex Singlet", "singlet_plano.png")

    s4 = Singlet(C1=torch.tensor(0.0), C2=1/torch.tensor(-12.6), D=torch.tensor(25.0), T=torch.tensor(12.7))
    plot_lens(s4, "Half-Sphere Singlet", "half_sphere.png")

    # 4. Doublet (Achromat Style)
    d1 = Doublet(C1=1/torch.tensor(50.0), C2=1/torch.tensor(-50.0), C3=1/torch.tensor(-200.0),
                 T1=torch.tensor(10.0), T2=torch.tensor(5.0), D=torch.tensor(25.0))
    plot_lens(d1, "Cemented Doublet", "doublet.png")

    # 5. Triplet (Symmetric / Hastings Style)
    t1 = Triplet(C1=1/torch.tensor(40.0), C2=1/torch.tensor(30.0), C3=1/torch.tensor(30.0), C4=1/torch.tensor(-40.0),
                 T1=torch.tensor(8.0), T2=torch.tensor(4.0), T3=torch.tensor(8.0), D=torch.tensor(20.0))
    plot_lens(t1, "Symmetric Triplet", "triplet.png")

    # 6. Rectangular Box
    # Center=(0,0,0)
    boxTransform = RayTransform(rotation=(torch.tensor([0, np.pi/4, 0])))

    b1 = Box(length = torch.tensor(1.0), width=torch.tensor(1.0), height=torch.tensor(1.0),
             transform=boxTransform)
    plot_lens(b1, "Rectangular Box (20x30x10)", "box.png")


