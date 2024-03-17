import time

from micrograd_cuda.tensor import Tensor
from micrograd_cuda.operations import to_device
from micrograd_cuda.operations_cuda import matrix_mul_cuda, free_gpu_memory
from micrograd_cuda.operations_cpu import matrix_mul_cpu

# Define large matrices A and B
A = [[1 for _ in range(200)] for _ in range(200)]
B = [[1 for _ in range(200)] for _ in range(200)]

### CUDA

# Move matrices A and B to the GPU
d_A, _ = to_device(A, device='cuda')
d_B, _ = to_device(B, device='cuda')

# Perform matrix multiplication on the GPU
start = time.time()
d_C = matrix_mul_cuda(d_A, d_B, (200, 200), (200, 200))
print(f"Elapsed: {time.time() - start:.2f} sec")

# Move the result matrix C back to the CPU and reshape it to nested list
C_gpu = to_device(d_C, device='cpu', original_shape=(200, 200))

# Free GPU memory
free_gpu_memory(d_A)
free_gpu_memory(d_B)
free_gpu_memory(d_C)

### CPU

# Perform matrix multiplication on the CPU
start = time.time()
C = matrix_mul_cpu(A, B)
print(f"Elapsed: {time.time() - start:.2f} sec")

print(C_gpu == C)
