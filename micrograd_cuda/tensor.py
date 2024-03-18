from micrograd_cuda.operations import (
    tanh,
    tanh_prime,
    zeros_matrix_like,
    ones_matrix_like,
    matrix_mul,
    matrix_scalar_mul,
    matrix_add,
    element_wise_mul,
    matrix_transpose,
    power,
    power_prime,
    tanh,
    tanh_prime,
    matrix_concat,
    summation,
    to_device,
    calculate_list_shape,
    free_gpu_memory,
)


class Tensor:

    def __init__(self, data, _children=(), _op="", label="", requires_grad=True, shape=None, device="cpu"):
        self.data = data  # Nested list on cpu, pointer to flat array on gpu
        self.requires_grad = requires_grad
        # TODO: grad should be init on gpu if data is on gpu
        self.grad = Tensor(zeros_matrix_like(data), requires_grad=False) if self.requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.device = device
        self.shape = calculate_list_shape(data) if shape is None else shape
    
    def to(self, device):
        if device not in ["cpu", "cuda"]:
            raise ValueError("Unsupported device. Choose 'cpu' or 'cuda'.")

        if self.device == device:
            return self

        self.data = to_device(self.data, device=device, original_shape=self.shape)
        self.device = device
        
        if self.grad is not None:
            self.grad.data = to_device(self.grad.data, device=device)
            self.grad.device = device

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(matrix_mul(self.data, other.data, self.shape, other.shape, device=self.device), (self, other), "@", shape=(self.shape[0], other.shape[1]), device=self.device, requires_grad=self.requires_grad)

        # TODO: fix grad operations
        def _backward():
            self.grad.data = matrix_add(
                self.grad.data, matrix_mul(out.grad.data, matrix_transpose(other.data, device=self.device), device=self.device), device=self.device
            )
            other.grad.data = matrix_add(
                other.grad.data, matrix_mul(matrix_transpose(self.data, device=self.device), out.grad.data, device=self.device), device=self.device
            )

        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        return self @ other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(element_wise_mul(self.data, other.data, device=self.device, shape=self.shape), (self, other), "*", device=self.device, requires_grad=self.requires_grad, shape=self.shape)

        # TODO: fix grad operations
        def _backward():
            self.grad.data = matrix_add(self.grad.data, element_wise_mul(out.grad.data, other.data, device=self.device), device=self.device)
            other.grad.data = matrix_add(other.grad.data, element_wise_mul(self.data, out.grad.data, device=self.device), device=self.device)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(matrix_add(self.data, other.data, device=self.device, shape=self.shape), (self, other), "+", device=self.device, requires_grad=self.requires_grad, shape=self.shape)

        # TODO: fix grad operations
        def _backward():
            self.grad.data = matrix_add(self.grad.data, out.grad.data, device=self.device)
            other.grad.data = matrix_add(other.grad.data, out.grad.data, device=self.device)

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        neg_one = matrix_scalar_mul(-1, ones_matrix_like(self.data), device=self.device, shape=self.shape)
        return self * neg_one

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __pow__(self, exponent):
        result_data = power(self.data, exponent, device=self.device, shape=self.shape)
        out = Tensor(result_data, (self,), f"**{exponent}", device=self.device, requires_grad=self.requires_grad, shape=self.shape)

        # TODO: fix grad operations
        def _backward():
            self.grad.data = matrix_add(
                self.grad.data, element_wise_mul(power_prime(self.data, exponent, device=self.device), out.grad.data, device=self.device), device=self.device
            )

        out._backward = _backward

        return out

    def __truediv__(self, other):  # self / other
        return self * other**-1

    @property
    def T(self):
        transposed_data = matrix_transpose(self.data, device=self.device, shape=self.shape)
        return Tensor(transposed_data, _children=(self,), _op="transpose", device=self.device, requires_grad=self.requires_grad, shape=(self.shape[1], self.shape[0]))

    def tanh(self):
        result_data = tanh(self.data, device=self.device, shape=self.shape)
        out = Tensor(result_data, (self,), "tanh", device=self.device, requires_grad=self.requires_grad, shape=self.shape)

        # TODO: fix grad operations
        def _backward():
            self.grad.data = matrix_add(
                self.grad.data, element_wise_mul(tanh_prime(self.data, device=self.device), out.grad.data, device=self.device), device=self.device
            )

        out._backward = _backward
        return out

    def sum(self):
        out_data = summation(self.data, device=self.device, shape=self.shape)
        out = Tensor(out_data, _children=(self,), _op="sum", device=self.device, requires_grad=self.requires_grad, shape=(1, 1))

        # TODO: fix grad operations
        def _backward():
            # TODO: proper kernel for this since can't use indexing for cuda tensors
            grad_contribution = [[out.grad.data[0][0] for _ in row] for row in self.data]
            self.grad.data = matrix_add(self.grad.data, grad_contribution, device=self.device)

        out._backward = _backward
        return out

    def concat(self, other, axis=0):
        # This implementation assumes a simple case where tensors are 2D
        # and concatenated along the first dimension (axis=0)
        # TODO: fix this to handle any axis and any dimension
        if axis != 0:
            raise NotImplementedError(
                "Concatenation along this axis is not implemented."
            )

        out = Tensor(matrix_concat(self.data, other.data, device=self.device, matrix_a_shape=self.shape, matrix_b_shape=other.shape), _children=(self, other), _op="concat", device=self.device, requires_grad=self.requires_grad, shape=(self.shape[0] + other.shape[0], self.shape[1]))

        # TODO: fix grad operations
        def _backward():
            # Since concatenation is along axis=0, we split the gradient back to the original tensors
            # based on their data length
            # TODO: proper kernel for this since can't use indexing for cuda tensors
            self_len = len(self.data)
            self.grad.data = matrix_add(self.grad.data, out.grad.data[:self_len], device=self.device)
            other.grad.data = matrix_add(other.grad.data, out.grad.data[self_len:], device=self.device)

        out._backward = _backward
        return out
    
    def abs(self):
        if self.device != "cpu":
            raise NotImplementedError("abs is not implemented for CUDA tensors yet.")
        if self.requires_grad:
            raise NotImplementedError("abs is not implemented for tensors with requires_grad=True yet.")
        out = Tensor([[abs(cell) for cell in row] for row in self.data], _children=(self,), _op="abs")
        return out


    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = Tensor(ones_matrix_like(self.data), requires_grad=False)
        for node in reversed(topo):
            node._backward()

    def __del__(self):
        if self.device == "cuda":
            free_gpu_memory(self.data)
            free_gpu_memory(self.grad.data)
