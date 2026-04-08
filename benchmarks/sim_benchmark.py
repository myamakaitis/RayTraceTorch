"""
benchmarks/sim_benchmark.py
----------------------------
Measures wall-time speedup from torch.compile on a realistic optical scene.

Scene: collimated disk beam → biconvex singlet lens → sensor plane.

Run from the repository root:
    python -m RayTraceTorch.benchmarks.sim_benchmark

Optional env vars:
    BENCH_DEVICE   cuda | cpu           (default: auto-detect)
    BENCH_REPEATS  <int>                (default: 20)
    BENCH_WARMUP   <int>                (default: 3)
"""

import os
import sys
import copy
import time
import statistics
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Allow running as a standalone script as well as a module
# ---------------------------------------------------------------------------
if __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from RayTraceTorch.scene.base import Scene
from RayTraceTorch.elements.lens import SingletLens
from RayTraceTorch.elements.sensor import Sensor
from RayTraceTorch.elements.aperture import CircularAperture
from RayTraceTorch.rays.bundle import CollimatedDisk
from RayTraceTorch.geom.transform import RayTransform
from RayTraceTorch.geom.bounded import Disk


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE  = torch.device(os.environ.get('BENCH_DEVICE',
                       'cuda' if torch.cuda.is_available() else 'cpu'))
REPEATS = int(os.environ.get('BENCH_REPEATS', 20))
WARMUP  = int(os.environ.get('BENCH_WARMUP', 3))
NBOUNCES = 20   # more than enough for a 2-element scene

RAY_COUNTS = [4096, 16_384, 64_000, 128_000]


# ---------------------------------------------------------------------------
# Scene factory
# ---------------------------------------------------------------------------

def build_scene(device: torch.device) -> tuple[Scene, CollimatedDisk, int]:
    """Build a singlet-lens scene and return (scene, bundle, default_N_rays)."""
    scene = Scene()

    # Biconvex singlet at z=0
    # Focal length ≈ 1/((n-1)*(c1-c2)) = 1/(0.5*0.1) = 20 units
    lens = SingletLens(
        c1=0.05, c2=-0.05,
        d=10.0,  t=3.0,
        ior_glass=1.5, ior_media=1.0,
    )
    scene.add_element(lens)

    # Circular aperture matched to lens diameter
    aperture = CircularAperture(
        radius=5.0,
        transform=RayTransform(translation=[0.0, 0.0, 0.0]),
    )
    scene.add_element(aperture)

    # Sensor near back focal plane (BFL ≈ 19 for this lens)
    sensor = Sensor(shape=Disk(
        radius=6.0,
        transform=RayTransform(translation=[0.0, 0.0, 19.0]),
    ))
    scene.add_element(sensor)

    scene.to(device)
    scene._build_index_maps()

    # Collimated beam from z = -30, propagating +Z
    bundle = CollimatedDisk(
        radius=4.0,
        ray_id=0,
        device=device,
        dtype=torch.float32,
        transform=None,   # default: identity, beam at z=0 facing +Z
    ).to(device)

    return scene, bundle, 1024


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def sync():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def run_simulation(scene: Scene, bundle: CollimatedDisk, N_rays: int) -> float:
    """Run one full simulation; return wall time in seconds."""
    # Sample fresh rays each run so we measure the full pipeline
    scene._build_rays()
    if scene.rays is None:
        raise RuntimeError("No rays — bundle not registered")

    sync()
    t0 = time.perf_counter()

    scene._build_index_maps()
    for _ in range(NBOUNCES):
        if not (scene.rays.intensity > 0).any():
            break
        scene.step()

    sync()
    return time.perf_counter() - t0


def bench(scene: Scene, bundle: CollimatedDisk, N_rays: int,
          label: str) -> float:
    """
    Warm up then time REPEATS simulations.
    Returns mean wall time in milliseconds.
    """
    # Register the bundle each time (fresh ray count)
    scene.clear_bundles()
    scene.add_bundle(bundle, N_rays)

    # Warmup (tracing happens here for compiled scenes)
    for _ in range(WARMUP):
        scene._build_rays()
        run_simulation(scene, bundle, N_rays)

    # Timed runs
    times = []
    for _ in range(REPEATS):
        scene._build_rays()
        times.append(run_simulation(scene, bundle, N_rays))

    mean_ms  = statistics.mean(times)  * 1e3
    stdev_ms = statistics.stdev(times) * 1e3 if len(times) > 1 else 0.0
    print(f"    {label:<28s}  {mean_ms:8.2f} ± {stdev_ms:5.2f} ms")
    return mean_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print(f"  RayTraceTorch simulation benchmark")
    print(f"  device  : {DEVICE}")
    print(f"  repeats : {REPEATS}  warmup : {WARMUP}  bounces : {NBOUNCES}")
    print("=" * 65)

    for N_rays in RAY_COUNTS:
        print(f"\n  N_rays = {N_rays:,}")

        # --- Uncompiled ---
        scene_plain, bundle, _ = build_scene(DEVICE)
        t_plain = bench(scene_plain, bundle, N_rays, "uncompiled")

        # # --- Compiled ---
        # # Build a fresh scene (same structure, independent weights) and compile
        # scene_compiled, bundle_c, _ = build_scene(DEVICE)
        # scene_compiled.compile_elements()
        # t_compiled = bench(scene_compiled, bundle_c, N_rays, "compiled (reduce-overhead)")

        # speedup = t_plain / t_compiled if t_compiled > 0 else float('nan')
        print(f"\n  t_plain = {t_plain:8.2f} ms")
        # print(f"    {'speedup':<28s}  {speedup:8.2f}x")

    print()


if __name__ == '__main__':
    main()
