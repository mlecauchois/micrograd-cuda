# micrograd CUDA

![](front.jpg)

Teaching myself basic CUDA by building GPU-accelerated tensor-based autodiff from the ground up, inspired by [Andrej's micrograd](https://github.com/karpathy/micrograd/tree/master).

No dependencies other than Python's standard library and CUDA.

## Compiling

To compile the CUDA kernels:

```bash
nvcc -shared -o liboperations.so micrograd_cuda/operations.cu -Xcompiler -fPIC
```

## Usage

```python
import random
import time

from micrograd_cuda.mlp import MLP
from micrograd_cuda.tensor import Tensor
from micrograd_cuda.operations import Operations

# Model
model = MLP(300, [300, 300, 1])
epochs = 1
device = "cuda"

# Data
xs_batch = Tensor([[random.random() for _ in range(300)] for _ in range(100)]).T
ys_batch = Tensor([[random.random()] for _ in range(100)]).T

# Move to device
model.to(device)
xs_batch.to(device)
ys_batch.to(device)

start = time.time()

for k in range(epochs):

    # Forward pass
    ypred = model(xs_batch)
    diff = ypred - ys_batch
    loss = (diff**2).sum()

    # Backward pass
    for p in model.parameters():
        p.grad.data, _ = Operations(p.device).zeros_matrix_like(shape=p.shape)
    loss.backward()

    # Update
    for p in model.parameters():
            p.data, _ = Operations(p.device).matrix_add(Operations(p.device).matrix_scalar_mul(-0.1, p.grad.data, shape=p.shape), p.data, shape=p.shape)

print(f"Elapsed: {time.time() - start:.2f} sec")
    
loss.to("cpu")
print(loss.data)
```

This code yields up to x1000 speedup on T4 GPU compared to CPU.

## Roadmap

- [x] Micrograd extension with basic 2D tensors and naïve matrix multiplication for MLP
- [x] Batching
- [x] CUDA kernel for matrix multiplication
- [ ] Less verbose code and error handling
- [ ] CUDA optimizations for matrix multiplication
- [ ] >2D tensors, indexing and better tensor logic
- [ ] MNIST with MLP
- [ ] Blogpost
- [ ] ConvNet implementation and CUDA kernel
- [ ] MNIST with ConvNet
- [ ] Transformers? Efficient attention CUDA kernel?

## Running tests

```bash
python -m pytest
```

## License

MIT
