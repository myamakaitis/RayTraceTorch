import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from RayTraceTorch.elements import Sensor

def sensor_spot_size():

    print("--- Starting Sensor Spot Size Test ---")

    configs = [
        {'id': 0, 'mean': [0.0, 0.0], 'std': 1.0, 'samples' : 100000},
        {'id': 1, 'mean': [10.0, 10.0], 'std': 2.0, 'samples' : 200000},
        {'id': 2, 'mean': [-5.0, 5.0], 'std': 0.5, 'samples' : 1000},
        {'id': 3, 'mean': [-5.0, 80.0], 'std': 4.5, 'samples' : 100000},
        {'id': 4, 'mean': [-15.0, 15.0], 'std': 6.5, 'samples' : 300000},
        {'id': 8, 'mean': [2.0, 2.0], 'std': 0.00001, 'samples' : 1000},
        {'id': 10, 'mean': [1.0, 1.0], 'std': 1.3, 'samples' : 5000},
    ]

    sensor = Sensor((None, None))

    # We will store expected values to verify against later
    expected_results = []

    for config in configs:
        # Generate random XY points from normal distribution
        # Shape (N, 3) - Z is just 0.0 for this test
        locs = torch.randn(config['samples'], 3)

        # Scale by std and shift by mean
        locs[:, 0] = locs[:, 0] * config['std'] + config['mean'][0]
        locs[:, 1] = locs[:, 1] * config['std'] + config['mean'][1]

        # Create uniform intensity of 1.0 (so spot size == statistical variance)
        # If intensity varies, the spot size is the intensity-weighted variance.
        intensities = torch.abs(torch.randn(config['samples']))

        # Create ID tensor
        ids = torch.full((config['samples'],), config['id'], dtype=torch.int32)

        # Add to Sensor buffers (simulating how data is loaded)
        sensor.hitLocs.append(locs)
        sensor.hitIntensity.append(intensities)
        sensor.hitID.append(ids)

        # Store expected variance (Std^2)
        expected_results.append(config['std'] ** 2)

    # 3. Run Calculation
    # We calculate for range [0, 3)
    # We expect output shape (3, 2) -> [Variance X, Variance Y] per ID
    calculated_spot_sizes, total_intensities = sensor.getSpotSizeParallel_xy([config['id'] for config in configs])

    # 4. Verify Results
    print(f"ID | {'Expected (Std^2)':<18} | {'Calculated':<15} | {'Error':<10}")
    print("-" * 75)

    for i, expected in enumerate(expected_results):
        calc = calculated_spot_sizes[i].item()

        err = abs(calc - expected)

        print(f"{i:<5} | {expected:<18.4f} | {calc:<15.4f} | {err:<10.4f}")

    print("-" * 75)

if __name__ == "__main__":
    sensor_spot_size()