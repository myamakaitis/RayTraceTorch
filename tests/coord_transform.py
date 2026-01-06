import time
import torch
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rays import Rays


def test_transform_overhead():
    print("--- Benchmarking Ray Object Creation Overhead ---")

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    # 1. Setup: Create a large batch of rays
    N = 100_000
    print(f"Batch Size: {N} rays")

    origins = torch.randn(N, 3, device=device)
    directions = torch.randn(N, 3, device=device)

    # Create the source bundle
    source_rays = Rays(origins, directions, device=device)

    # Pre-calculate transformed coordinates to isolate object creation time
    # (We don't want to measure the math of adding 1, just the object overhead)
    new_pos = origins + 1.0
    new_dir = directions  # directions usually change, but for allocation test irrelevant

    # --- Test 1: Classic Method (Full Init) ---
    # This simulates creating a new Rays object and manually copying metadata
    # which implies __init__ allocates defaults that we waste.

    start_time = time.perf_counter()
    iterations = 1000

    for _ in range(iterations):
        # A. Trigger __init__ (allocates default n, active, intensity, etc.)
        r_new = Rays(new_pos, new_dir, device=device)

        # B. Manually copy metadata (Reference copy is cheap, but Init overhead occurred)
        r_new.n = source_rays.n
        r_new.intensity = source_rays.intensity
        r_new.id = source_rays.id

    if device == 'cuda': torch.cuda.synchronize()
    end_time = time.perf_counter()
    avg_classic = (end_time - start_time) / iterations
    print(f"\nClassic Approach (New Object + Copy): {avg_classic * 1000:.4f} ms per call")

    # --- Test 2: Optimized Method (with_coords) ---
    # Skips __init__, shares metadata references directly.

    start_time = time.perf_counter()

    for _ in range(iterations):
        r_opt = source_rays.with_coords(new_pos, new_dir)

    if device == 'cuda': torch.cuda.synchronize()
    end_time = time.perf_counter()
    avg_opt = (end_time - start_time) / iterations
    print(f"Optimized Approach (with_coords):     {avg_opt * 1000:.4f} ms per call")

    # --- Results ---
    speedup = avg_classic / avg_opt
    print(f"\nSpeedup Factor: {speedup:.2f}x")

    # Correctness Check
    print("\n--- Verifying Correctness ---")
    r_check = source_rays.with_coords(new_pos, new_dir)

    # Check Geometry
    assert torch.allclose(r_check.pos, new_pos), "Positions did not update"

    # Check Metadata Preservation
    # We change a value in the source and ensure the 'view' sees it (Shared Memory)
    source_rays.intensity[0] = 999.0
    assert r_check.intensity[0] == 999.0, "Metadata is not sharing memory correctly"
    print("Correctness Checks Passed.")