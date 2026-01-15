import torch
import torch.nn as nn
import math
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from RayTraceTorch.elements import SingletLens
from RayTraceTorch.rays import Rays, collimatedSource
# Assuming these imports exist in your project structure
# from my_library import SingletLens, Rays, collimatedSource

def test_singlet_optimization():
    # ---------------------------------------------------------
    # 1. Setup Configuration
    # ---------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # Target focal plane
    target_z = 100.0

    # Initial Lens Parameters
    # We start with a roughly biconvex lens, but allow it to evolve
    # C = 1/R. 0.02 is R=50.
    init_c1 = 0.00283 #-0.016667 # goal = 0.016667
    init_c2 = -0.016667 #0.00283 # goal = -0.00283
    thickness = 4.0 # goal = 4.0
    diameter = 25.4
    ior = 1.5168

    # Instantiate the Lens with gradients enabled for shape params
    lens = SingletLens(
        c1=init_c1,
        c2=init_c2,
        d=diameter,
        t=thickness,
        ior_glass=ior,
        ior_media=1.0,
        c1_grad=True,
        c2_grad=True,
        t_grad=False,  # Optimizing thickness as well
        d_grad=False,  # Fixed diameter
        fresnel=False,
        inked=False
    ).to(device)

    # Optimizer
    optimizer = torch.optim.LBFGS(lens.parameters(), lr=1e-1)

    print(f"--- Starting Optimization on {device} ---")
    print(f"--- Initial Focal Length: {lens.f:.2f} ---")
    print(f"Initial State: C1={lens.shape.surfaces[0].c.item():.4f}, C2={lens.shape.surfaces[1].c.item():.4f}")

    # ---------------------------------------------------------
    # 2. Optimization Loop
    # ---------------------------------------------------------
    steps = 10000


    def closure():
        optimizer.zero_grad()

        # A. Generate Rays
        # Create bundle at z= -20 (to give space before the lens) pointing +Z
        # We regenerate rays inside the loop if we wanted to randomize them,
        # but for static optics, generating once is usually fine.
        # Doing it here ensures we have a fresh batch.
        rays = collimatedSource(
            origin=[0, 0, -20],
            direction=[0, 0, 1],
            radius=5,
            N_rays=5000,
            device=device
        )

        p0 = rays.pos.clone()

        # B. Propagate through Surface 1 (Front)
        # lens(rays, 0) -> geometric hit + refraction
        p1, d1, mult = lens(rays, surf_idx=0)

        # Update Ray state for next intersection
        # We assume Rays class allows updating or we create a new wrapper
        rays.update(p1, d1, mult)

        # C. Propagate through Surface 2 (Back)
        p2, d2, _ = lens(rays, surf_idx=1)

        # D. Project to Target Plane (Detector) at z = 100
        # Formula: P_target = P_current + t * D_current
        # We know P_target.z = 100. Solve for t:
        # 100 = p2.z + t * d2.z  =>  t = (100 - p2.z) / d2.z

        delta_z = target_z - p2[:, 2]
        t_dist = delta_z / (d2[:, 2] + 1e-6)  # Avoid div/0

        # Calculate x, y hit positions on the sensor
        sensor_x = p2[:, 0] + t_dist * d2[:, 0]
        sensor_y = p2[:, 1] + t_dist * d2[:, 1]

        p3 = torch.stack([sensor_x, sensor_y, torch.full_like(sensor_x, target_z)], dim=1)

        # E. Loss Calculation (RMS Spot Size)
        # We want all rays to hit (0, 0) on the sensor plane.
        mse_loss = torch.mean(sensor_x ** 2 + sensor_y ** 2)
        rms_spot = torch.sqrt(mse_loss)

        # F. Backprop
        mse_loss.backward()

        # # Monitoring
        # if i % 100 == 0:
        print(f" | Loss (MSE): {mse_loss.item():.6f} | Spot RMS: {rms_spot.item():.4f} | focal: {lens.f:.2f}")

        return mse_loss


    for i in range(steps):
        optimizer.step(closure)


    # with torch.no_grad():
    #
    #     fig, ax = plt.subplots(2)
    #     # p [N x 3]
    #     Pall = torch.stack([p0, p1, p2, p3], dim=0).cpu().detach().numpy() # [K x N x 3]
    #
    #     ax[0].plot(Pall[:, :, 2], Pall[:, :, 0])
    #     ax[1].plot(Pall[:, :, 2], Pall[:, :, 1])
    #
    # fig.show()

    final_c1 = lens.shape.surfaces[0].c.item()
    final_c2 = lens.shape.surfaces[1].c.item()
    print(f"Goal Spot Size: ")
    print("-" * 30)
    print("Optimization Complete")
    print(f"Final T: {lens.shape.T.item():.6f}")
    print(f"Final C1: {final_c1:.6f}, goal =  0.016667")
    print(f"Final C2: {final_c2:.6f}, goal = -0.00283")

    print(f"Focal Length = {lens.f:.2f}")

    # Check the ratio
    # Avoid division by zero if C2 converges to 0 (unlikely for focused lens)
    if abs(final_c2) > 1e-9:
        ratio = final_c1 / final_c2
        print(f"Ratio C1/C2: {ratio:.4f}")

        # Check against the theoretical expectation mentioned (C1 ~ 6*C2)
        # Note: The sign depends on your coordinate convention for curvature.
        # Typically for min spherical aberration (n=1.5), the convex side faces the collimated beam.
        if -5.0 < abs(ratio) < -7.0:
            print("SUCCESS: Ratio is approximately 6.")
        else:
            print(f"NOTE: Ratio converged to {ratio:.2f}. "
                  "Theoretical optimum depends on exact conjugate ratio (object at infinity vs finite).")
    else:
        print("C2 is near zero (Plano-convex).")


if __name__ == "__main__":

    test_singlet_optimization()