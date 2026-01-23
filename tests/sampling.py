import torch
import time
import math


def benchmark_sampling(device_name='cpu'):
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test.")
        return

    device = torch.device(device_name)
    print(f"--- Benchmarking on {device_name.upper()} ---")

    # 1. The Rejection Method (Your current "Risky" implementation)
    # Note: We use 2.0x oversampling.
    # To be statistically 'safe' (6 sigma), you'd actually need ~2.55x, making this even slower.
    def rejection_sampling(N):
        # Oversample by 2x to attempt to get enough points
        x = torch.rand(2 * N, device=device) * 2 - 1
        y = torch.rand(2 * N, device=device) * 2 - 1

        r2 = x ** 2 + y ** 2
        mask = r2 <= 1.0

        x, y = x[mask], y[mask]

        # This is the failure mode you mentioned:
        if x.shape[0] < N:
            return None  # Failed to get enough points

        x, y = x[:N], y[:N]
        return torch.stack([x, y], dim=1)

    # 2. The Polar Method (Exact)
    def polar_sampling(N):
        theta = torch.rand(N, device=device) * 2 * math.pi
        r = torch.sqrt(torch.rand(N, device=device))

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        return torch.stack([x, y], dim=1)

    # Sizes to test
    batch_sizes = [10_000, 100_000, 1_000_000, 10_000_000]

    # Header
    print(f"{'N':<12} | {'Rejection (ms)':<15} | {'Polar (ms)':<15} | {'Speedup'}")
    print("-" * 60)

    for N in batch_sizes:
        # Warmup
        if device_name == 'cuda':
            rejection_sampling(1000)
            polar_sampling(1000)
            torch.cuda.synchronize()

        # Test Rejection
        start = time.perf_counter()
        for _ in range(10):  # Run 10 times to average
            rejection_sampling(N)
        if device_name == 'cuda': torch.cuda.synchronize()
        rej_time = (time.perf_counter() - start) / 10 * 1000

        # Test Polar
        start = time.perf_counter()
        for _ in range(10):
            polar_sampling(N)
        if device_name == 'cuda': torch.cuda.synchronize()
        polar_time = (time.perf_counter() - start) / 10 * 1000

        ratio = rej_time / polar_time
        winner = "Polar" if ratio > 1.0 else "Rejection"

        print(f"{N:<12} | {rej_time:<15.4f} | {polar_time:<15.4f} | {ratio:.2f}x ({winner})")


# Run on CPU
benchmark_sampling('cpu')

# Run on GPU (if you have one)
benchmark_sampling('cuda')