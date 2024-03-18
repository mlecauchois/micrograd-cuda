# micrograd CUDA

![](front.jpg)

Teaching myself basic CUDA by building GPU-accelerated tensor-based autodiff from the ground up, inspired by [Andrej's micrograd](https://github.com/karpathy/micrograd/tree/master).

## Compiling

To compile the CUDA kernels:

```bash
nvcc -shared -o liboperations.so micrograd_cuda/operations.cu -Xcompiler -fPIC
```

## Usage

```python
from micrograd_cuda.mlp import MLP
from micrograd_cuda.tensor import Tensor
from micrograd_cuda.tensor import matrix_add, matrix_scalar_mul, zeros_matrix_like

# Model
model = MLP(3, [4, 4, 1])
epochs = 20

# Data
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [[1.0], [-1.0], [-1.0], [1.0]]
xs_batch = Tensor(xs).T
ys_batch = Tensor(ys).T

for k in range(epochs):

    # Forward pass
    ypred = model(xs_batch)
    diff = ypred - ys_batch
    loss = (diff**2).sum()

    # Backward pass
    for p in model.parameters():
        p.grad = zeros_matrix_like(p.data)
    loss.backward()

    # Update
    for p in model.parameters():
        p.data = matrix_add(matrix_scalar_mul(-0.1, p.grad), p.data)

    print(k, loss.data)
```

## Roadmap

- [x] Micrograd extension with basic 2D tensors and naïve matrix multiplication for MLP
- [x] Batching
- [ ] CUDA kernel for matrix multiplication
- [ ] Error handling
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
