# micrograd CUDA

![](front.jpg)

Teaching myself basic CUDA by building tensor-based autodiff from the ground up, inspired by [Andrej's micrograd](https://github.com/karpathy/micrograd/tree/master).

## Roadmap

- [x] Micrograd extension with basic 2D tensors and naïve matrix multiplication for MLP
- [x] Batching
- [ ] CUDA kernel for matrix multiplication
- [ ] CUDA optimizations for matrix multiplication
- [ ] >2D tensors, indexing and better tensor logic
- [ ] MNIST with MLP
- [ ] ConvNet implementation and CUDA kernel
- [ ] MNIST with ConvNet
- [ ] Transformers? Efficient attention CUDA kernel?

## Running tests

```bash
python -m pytest
```

## License

MIT
