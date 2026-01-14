import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from RayTraceTorch.phys import RefractSnell, RefractFresnel, Reflect, Transmit, Block

def test_tir_logic():
    device = 'cpu'
    # Glass (1.5) -> Air (1.0)
    # Critical angle is approx 41.8 degrees.
    refract_op = RefractSnell(ior_in=1.5, ior_out=1.0)

    # Normal points +Y (0, 1, 0)
    normal = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

    # Ray 1: Shallow angle (30 deg to normal) -> Should Refract
    # Ray 2: Steep angle (60 deg to normal) -> Should Reflect (TIR)

    # Setup Rays (pointing UP towards the surface from inside)
    # 30 deg: x = sin(30)=0.5, y = cos(30)=0.866
    # 60 deg: x = sin(60)=0.866, y = cos(60)=0.5
    ray_dir = torch.tensor([
        [0.5, 0.866025, 0.0],  # Ray 1
        [0.866025, 0.5, 0.0]  # Ray 2
    ])

    out, _ = refract_op(None, ray_dir, normal)

    # Check Ray 1 (Refraction)
    # Snell: 1.5 * sin(30) = 1.0 * sin(theta2)
    # 1.5 * 0.5 = 0.75 => theta2 = arcsin(0.75) approx 48.6 deg
    # Output X should be 0.75
    assert torch.isclose(out[0, 0], torch.tensor(0.75), atol=1e-4), "Ray 1 should refract"

    # Check Ray 2 (Reflection - TIR)
    # Input was (0.866, 0.5). Reflection across Y normal -> (0.866, -0.5)
    # Ray keeps x, flips y.
    assert torch.isclose(out[1, 0], ray_dir[1, 0], atol=1e-5), "Ray 2 X component mismatch (TIR)"
    assert torch.isclose(out[1, 1], -ray_dir[1, 1], atol=1e-5), "Ray 2 Y component not flipped (TIR)"

    print("TIR Logic Passed")


def test_fresnel_probability():
    device = 'cpu'
    # Air -> Glass (n=1.5)
    # Normal incidence R = ((1-1.5)/(1+1.5))^2 = (-0.5/2.5)^2 = 0.2^2 = 0.04 (4%)
    refract_op = RefractFresnel(ior_in=1.0, ior_out=1.5)

    # 1. Create a large batch of identical rays for statistics
    N = 100000
    ray_dir = torch.tensor([[0.0, 0.0, 1.0]]).repeat(N, 1)  # +Z
    normal = torch.tensor([[0.0, 0.0, -1.0]]).repeat(N, 1)  # -Z (opposing)

    # 2. Run Physics
    out_dir, _ = refract_op(None, ray_dir, normal)

    # 3. Analyze Results
    # Reflected rays will flip direction to -Z (0, 0, -1)
    # Refracted rays will continue to +Z (0, 0, 1)

    # Dot product with original direction
    # +1 = Refract, -1 = Reflect
    direction_check = out_dir[:, 2]

    reflected_count = (direction_check < 0).sum().item()
    measured_R = reflected_count / N

    print(f"\nMeasured Reflectance: {measured_R:.4f} (Expected ~0.04)")

    # Allow some Monte Carlo noise (e.g., +/- 1%)
    assert 0.03 < measured_R < 0.05, f"Monte Carlo Fresnel failed. Got {measured_R}"


def test_tir_deterministic():
    """Ensure TIR is still 100% reflective despite random sampling."""
    device = 'cpu'
    # Glass (1.5) -> Air (1.0)
    # Critical Angle ~41.8 deg.
    refract_op = RefractFresnel(ior_in=1.5, ior_out=1.0)

    # Incidence at 60 deg (Steep) -> 100% TIR
    # Ray (sin60, cos60) against Normal (0,1)
    N = 100
    ray_dir = torch.tensor([[0.866, 0.5, 0.0]]).repeat(N, 1)
    normal = torch.tensor([[0.0, 1.0, 0.0]]).repeat(N, 1)

    out_dir, _ = refract_op(None, ray_dir, normal)

    # Reflected rays should have Y component flipped (-0.5)
    # Refracted rays (impossible) would have Y > 0

    # Check Y component is negative
    assert torch.all(out_dir[:, 1] < 0), "TIR allowed some refraction!"
    print("Probabilistic TIR Test Passed")


def test_transmit_block():
    # 1. Setup
    N = 5
    ray_dir = torch.randn(N, 3)
    normal = torch.randn(N, 3)  # Irrelevant for these physics

    # 2. Test Transmit
    transmit_op = Transmit()
    out_dir, intensity = transmit_op(None, ray_dir, normal)

    assert torch.allclose(out_dir, ray_dir), "Transmit changed ray direction"
    assert torch.allclose(intensity, torch.ones(N)), "Transmit altered intensity"

    # 3. Test Block
    block_op = Block()
    out_dir, intensity = block_op(None, ray_dir, normal)

    # Check Intensity is 0
    assert torch.allclose(intensity, torch.zeros(N)), "Block did not zero intensity"

    # Check Direction (implementation specific, currently zeros)
    assert torch.allclose(out_dir, torch.zeros_like(ray_dir)), "Block did not zero direction"

    print("Transmit/Block Tests Passed")