import torch as tr
from time import perf_counter
import numpy as np

n = 14
size = ((2**n, 2**n))

h = tr.rand(size, dtype=tr.float32)
g = tr.rand(size, dtype=tr.float32)

g_np = g.numpy()

t0 = perf_counter()
y_np = np.fft.fft2(g)
tf = perf_counter()
print(f"NP Mat Mul Time: {tf- t0:.4f}s")

# t0 = perf_counter()
# y_cpu = h*g
# tf = perf_counter()
# print(f"CPU Elementwise {tf- t0}")

t0 = perf_counter()
y_cpu_mm = tr.fft.fft2(g)
tf = perf_counter()
print(f"CPU Mat Mul Time: {tf- t0:.4f}s")

cuda = tr.device("cuda")

h = h.to(device=cuda)
g = g.to(device=cuda)


# t0 = perf_counter()
# y_gpu = h*g
# tf = perf_counter()
# print(f"GPU Elementwise {tf- t0}")


n_repeat = 10
t0 = perf_counter()

for i in range(n_repeat):
    y_gpu_mm = tr.fft.fft2(g)

tf = perf_counter()
print(f"GPU Mat Mul Time: {(tf- t0)/n_repeat:.4f}s")

y_gpu_mm = y_gpu_mm.cpu()

print(tr.mean(tr.abs(y_gpu_mm - y_cpu_mm)))