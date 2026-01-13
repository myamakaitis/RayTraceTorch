import torch
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Assuming the new module is accessible via geom.cylindrics or added to geom/__init__.py
# Adjust the import based on where you saved the file.
from geom.cylindrics import CylSinglet
from geom import RayTransform
from rays import Rays


def scan_lens_profile(lens, axis='x', num_points=200):
    """
    Fires a row of parallel rays through the lens to map its profile.
    Adapted for CylSinglet to respect rectangular bounds.

    Args:
        lens: The shape object.
        axis: 'x' for XZ profile (y=0), 'y' for YZ profile (x=0).
    """
    # Determine scan width based on available attributes
    # CylSinglet has explicit width/height properties
    if hasattr(lens, 'width') and axis == 'x':
        extent = lens.width.item() * 1.5
    elif hasattr(lens, 'height') and axis == 'y':
        extent = lens.height.item() * 1.5
    else:
        extent = 20.0  # Fallback

    # Create scan coordinates
    coords = torch.linspace(-extent / 2, extent / 2, num_points)
    zeros = torch.zeros_like(coords)

    # Rays start at Z = -100, pointing +Z
    if axis == 'x':
        # Scan along X, Y=0
        origins = torch.stack([coords, zeros, torch.full_like(coords, -100.0)], dim=1)
        directions = torch.tensor([[0.0, 0.0, 1.0]]).expand(num_points, 3)
    else:
        # Scan along Y, X=0
        origins = torch.stack([zeros, coords, torch.full_like(coords, -100.0)], dim=1)
        directions = torch.tensor([[0.0, 0.0, 1.0]]).expand(num_points, 3)

    # Create Rays
    rays = Rays(origins, directions)

    # Intersect
    # t_matrix: [N, NumSurfaces]
    t_matrix = lens.intersectTest(rays)

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # --- XZ Projection (Should look rectangular/flat for CylSinglet) ---
    profiles_x = scan_lens_profile(lens, axis='x')
    ax = axes[0]
    for i, (h, z) in enumerate(profiles_x):
        if len(h) == 0: continue

        # Color coding: 0=Front(Blue), 1=Back(Orange), >1=Edges(Green/Red)
        ax.plot(h, z, '.', markersize=2, label=f"Surf {i}")

    ax.set_title(f"{title} - XZ Projection (Top View)\nShould be flat/rectangular")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # --- YZ Projection (Should look curved for CylSinglet) ---
    profiles_y = scan_lens_profile(lens, axis='y')
    ax = axes[1]
    for i, (h, z) in enumerate(profiles_y):
        if len(h) == 0: continue
        ax.plot(h, z, '.', markersize=2, label=f"Surf {i}")

    ax.set_title(f"{title} - YZ Projection (Side View)\nShould show curvature")
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


def test_cylinders():
    print("--- Testing Cylindrical Lens Geometry ---")

    # 1. Bi-Convex Cylindrical Singlet (Power in Y)
    # R_y = 50, Width=30, Height=20, T=10
    print("Generating Bi-Convex Cylinder...")
    c1 = CylSinglet(C1=1 / torch.tensor(50.0), C2=1 / torch.tensor(-50.0),
                    width=torch.tensor(30.0), height=torch.tensor(20.0), T=torch.tensor(10.0))
    plot_lens(c1, "Bi-Convex CylSinglet", "cyl_biconvex.png")

    # 2. Plano-Convex Cylindrical Singlet (Power in Y)
    # R_y = 30 (Front), Flat (Back), Width=20, Height=25, T=5
    print("Generating Plano-Convex Cylinder...")
    c2 = CylSinglet(C1=1 / torch.tensor(30.0), C2=torch.tensor(0.0),
                    width=torch.tensor(20.0), height=torch.tensor(25.0), T=torch.tensor(8.0))
    plot_lens(c2, "Plano-Convex CylSinglet", "cyl_plano.png")

    # 3. Anamorphic-Style Check (Different Height/Width)
    # Check if the edges are correctly placed at +/- Width/2 and +/- Height/2
    # We use a very thick lens to see the edges clearly
    print("Generating Thick Block for Edge Check...")
    c3 = CylSinglet(C1=torch.tensor(1/-50.0), C2=torch.tensor(1/50.0),
                    width=torch.tensor(40.0), height=torch.tensor(20.0), T=torch.tensor(2.0))
    plot_lens(c3, "Bi-Concave CylSinglet", "cyl_biconcave.png")