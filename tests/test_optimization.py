"""
test_optimization.py
====================
Gradient-based optimization of a cemented achromatic doublet lens.

Starting design: an approximate 50 mm EFL achromat (BK7/SF2), close to a
real Edmund Optics-style design.  We perturb the curvatures slightly, then
run Adam to minimise spot size across three field angles while keeping the
focal length at 50 mm and satisfying thickness / spacing / length constraints.

All distances are in metres.

Expected output: focal length should converge back toward 50 mm and spot sizes
should decrease over the 300 optimisation steps.
"""

import math
import torch

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Library imports
from RayTraceTorch.elements import DoubletLens, Sensor
from RayTraceTorch.geom import RayTransform, Disk, RayTransformBundle
from RayTraceTorch.rays import CollimatedDisk
from RayTraceTorch.scene import SequentialScene
from RayTraceTorch.optim import (
    FocalLengthLoss, SpotSizeLoss,
    ThicknessConstraint, SpacingConstraint, SystemLengthConstraint,
)


# ---------------------------------------------------------------------------
# 1.  Lens prescription  (cemented achromat, EFL ≈ 50 mm)
# ---------------------------------------------------------------------------
# Crown (BK7):  n = 1.517,  Abbe V ≈ 64
# Flint (SF2):  n = 1.648,  Abbe V ≈ 34
#
# Surface radii (metres):
#   R1 =  31.0 mm  →  C1 =  32.26 m⁻¹
#   R2 = -18.5 mm  →  C2 = -54.05 m⁻¹   (cemented interface)
#   R3 = -140 mm   →  C3 =  -7.14 m⁻¹
#
# Perturb C1 by +2 m⁻¹ to give the optimiser something to fix.

C1 =  0.05   # perturbed (true ≈ 32.26)
C2 = -0.015
C3 =  -.0014

D  = 25    # 25 mm clear aperture
T1 = 6    # 6 mm crown element
T2 = 2.5   # 2.5 mm flint element
n1 = 1.517    # BK7
n2 = 1.648    # SF2

lens = DoubletLens(
    c1=C1, c2=C2, c3=C3, d=D, t1=T1, t2=T2,
    ior_glass1=n1, ior_glass2=n2,
    c1_grad=True, c2_grad=True, c3_grad=True,   # optimise curvatures
    t1_grad=True,                                 # optimise crown thickness
    inked=True,                                   # block edge rays
)

# ---------------------------------------------------------------------------
# 2.  Sensor  (placed at approximate BFL ≈ 52 mm past lens centre)
# ---------------------------------------------------------------------------
SENSOR_Z = 50   # metres

sensor = Sensor(
    Disk(radius=0.005,
         transform=RayTransform(translation=[0.0, 0.0, SENSOR_Z]))
)

# ---------------------------------------------------------------------------
# 3.  Sequential scene
# ---------------------------------------------------------------------------
scene = SequentialScene([lens, sensor])

# ---------------------------------------------------------------------------
# 4.  Ray bundles  (CollimatedDisk, one per field angle)
# ---------------------------------------------------------------------------
# Three field angles in the YZ plane: 0°, 5°, 10°
# RayTransformBundle.rotation = [-angle, 0, 0] tilts the bundle direction to
#   [0, sin(angle), cos(angle)]  in global coordinates.

BEAM_RADIUS = 0.010   # 10 mm entrance pupil
Z_START     = -0.005  # 5 mm before the lens front vertex

field_angles_deg = [0.0, 5.0, 10.0]
field_angles_rad = [math.radians(a) for a in field_angles_deg]

bundles = [
    CollimatedDisk(
        radius=BEAM_RADIUS,
        ray_id=i,
        transform=RayTransformBundle(
            rotation=[-a, 0.0, 0.0],
            translation=[0.0, 0.0, Z_START],
        ),
    )
    for i, a in enumerate(field_angles_rad)
]

# ---------------------------------------------------------------------------
# 5.  Loss functions and constraints
# ---------------------------------------------------------------------------
focal_loss = FocalLengthLoss(scene, f_target=50)  # target 50 mm
spot_loss  = SpotSizeLoss(scene, sensor, bundles, N_rays=64)

# Intra-element: 0.5 mm < thickness < 15 mm
thick_c = ThicknessConstraint(
    [lens], t_min=0.5, t_max=15, weight=0.05
)
# Inter-element: sensor must be at least 5 mm past the lens
space_c = SpacingConstraint(
    [lens, sensor], d_min=5, weight=0.05
)
# Total system: first surface to sensor < 120 mm
length_c = SystemLengthConstraint(
    [lens, sensor], L_max=120, weight=0.05
)

# ---------------------------------------------------------------------------
# 6.  Optimiser
# ---------------------------------------------------------------------------
optimizer = torch.optim.Adam(scene.parameters(), lr=2e-4)

print(f"{'Step':>5}  {'Total loss':>12}  {'Spot (mm)':>10}  {'EFL (mm)':>9}")
print("-" * 45)

N_STEPS = 300

for step in range(N_STEPS):
    optimizer.zero_grad()

    s_loss = spot_loss()
    f_loss = focal_loss()
    c_loss = thick_c() + space_c() + length_c()

    total = s_loss + 0.5 * f_loss + c_loss
    total.backward()
    optimizer.step()

    if step % 20 == 0 or step == N_STEPS - 1:
        with torch.no_grad():
            M = scene.getParaxial()
            P = -M[1, 0].item()
            efl_mm = 1000.0 / P if abs(P) > 1e-9 else float('inf')
            spot_mm = math.sqrt(s_loss.item()) * 1000.0  # RMS radius in mm

        print(f"{step:5d}  {total.item():12.4e}  {spot_mm:10.4f}  {efl_mm:9.2f}")

# ---------------------------------------------------------------------------
# 7.  Final prescription
# ---------------------------------------------------------------------------
print("\nFinal curvatures (m⁻¹):")
surfs = lens.shape.surfaces
print(f"  C1 = {surfs[0].c.item():+.4f}   (started at {C1:+.4f})")
print(f"  C2 = {surfs[1].c.item():+.4f}   (started at {C2:+.4f})")
print(f"  C3 = {surfs[2].c.item():+.4f}   (started at {C3:+.4f})")

T1_opt = (surfs[1].transform.trans[2] - surfs[0].transform.trans[2]).item()
print(f"\nCrown thickness T1 = {T1_opt*1000:.2f} mm  (started at {T1*1000:.1f} mm)")
