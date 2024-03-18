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

# Move to device
model.to("cuda")
xs_batch.to("cuda")
ys_batch.to("cuda")

for k in range(epochs):

    # Forward pass
    ypred = model(xs_batch)
    diff = ypred - ys_batch
    loss = (diff**2).sum()

    # Backward pass
    for p in model.parameters():
            p.grad.data = zeros_matrix_like(device=device, shape=p.shape)

    loss.backward()

    # Update
    for p in model.parameters():
            p.data = matrix_add(matrix_scalar_mul(-0.1, p.grad.data, device=p.device, shape=p.shape), p.data, device=p.device, shape=p.shape)
    
loss.to("cpu")
print(loss.data)
```

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
