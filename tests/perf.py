import torch
import torch.utils.benchmark as benchmark

# Setup: 1 million elements, float32, on GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
discriminant = torch.randn(1000000, device=device, dtype=torch.float32)

# Define the two functions
def method_1():
    hit_mask = discriminant >= 0
    # Includes zeros_like allocation overhead
    return torch.sqrt(torch.where(hit_mask, discriminant, torch.zeros_like(discriminant)))

def method_2():
    hit_mask = discriminant >= 0
    return torch.sqrt(torch.abs(discriminant))

# Define a third "idiomatic" option (clamp/ReLU)
def method_3():
    hit_mask = discriminant >= 0
    # Conceptually identical to Method 1 (zeros out negatives) but optimized
    return torch.sqrt(torch.clamp(discriminant, min=0))

if __name__ == "__main__":
    t1 = benchmark.Timer(stmt='method_1()', globals={'method_1': method_1})
    t2 = benchmark.Timer(stmt='method_2()', globals={'method_2': method_2})
    t3 = benchmark.Timer(stmt='method_3()', globals={'method_3': method_3})

    print(f"Device: {device.upper()}")
    print(f"Method 1 (Where): {t1.timeit(100).mean * 1000:.4f} ms")
    print(f"Method 2 (Abs):   {t2.timeit(100).mean * 1000:.4f} ms")
    print(f"Method 3 (Clamp): {t3.timeit(100).mean * 1000:.4f} ms")