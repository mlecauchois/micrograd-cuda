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
        if device == "cuda" and shape is None:
            raise ValueError("Shape must be provided for CUDA tensors.")
        self.data = data  # Nested list on cpu, pointer to flat array on gpu
        self._requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.device = device
        self.shape = calculate_list_shape(data) if shape is None else shape
        self.grad = Tensor(zeros_matrix_like(shape=self.shape, device=self.device), requires_grad=False, device=self.device, shape=self.shape) if self.requires_grad else None
    
    def to(self, device):
        if device not in ["cpu", "cuda"]:
            raise ValueError("Unsupported device. Choose 'cpu' or 'cuda'.")

        if self.device == device:
            return self

        self.data = to_device(self.data, device=device, original_shape=self.shape)
        self.device = device
        
        if self.grad is not None:
            self.grad.data = to_device(self.grad.data, device=device, original_shape=self.grad.shape)
            self.grad.device = device

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(matrix_mul(self.data, other.data, self.shape, other.shape, device=self.device), (self, other), "@", shape=(self.shape[0], other.shape[1]), device=self.device, requires_grad=self.requires_grad)

        def _backward():
            self.grad.data = matrix_add(
                self.grad.data, matrix_mul(out.grad.data, matrix_transpose(other.data, device=self.device, shape=other.shape), matrix_a_shape=out.grad.shape, matrix_b_shape=(other.shape[1], other.shape[0]), device=self.device), device=self.device, shape=self.shape
            )
            other.grad.data = matrix_add(
                other.grad.data, matrix_mul(matrix_transpose(self.data, device=self.device, shape=self.shape), out.grad.data, matrix_a_shape=(self.shape[1], self.shape[0]), matrix_b_shape=out.grad.shape, device=self.device), device=self.device, shape=other.shape
            )
        out._backward = _backward
        return out

        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        return self @ other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(element_wise_mul(self.data, other.data, device=self.device, shape=self.shape), (self, other), "*", device=self.device, requires_grad=self.requires_grad, shape=self.shape)

        def _backward():
            self.grad.data = matrix_add(self.grad.data, element_wise_mul(out.grad.data, other.data, device=self.device, shape=self.shape), device=self.device, shape=self.shape)
            other.grad.data = matrix_add(other.grad.data, element_wise_mul(self.data, out.grad.data, device=self.device, shape=self.shape), device=self.device, shape=other.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(matrix_add(self.data, other.data, device=self.device, shape=self.shape), (self, other), "+", device=self.device, requires_grad=self.requires_grad, shape=self.shape)

        def _backward():
            self.grad.data = matrix_add(self.grad.data, out.grad.data, device=self.device, shape=self.shape)
            other.grad.data = matrix_add(other.grad.data, out.grad.data, device=self.device, shape=other.shape)

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        neg_one = Tensor(matrix_scalar_mul(-1, ones_matrix_like(shape=self.shape, device=self.device), device=self.device, shape=self.shape), device=self.device, shape=self.shape)
        return self * neg_one

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return self + (-other)

    def __pow__(self, exponent):
        result_data = power(self.data, exponent, device=self.device, shape=self.shape)
        out = Tensor(result_data, (self,), f"**{exponent}", device=self.device, requires_grad=self.requires_grad, shape=self.shape)

        def _backward():
            self.grad.data = matrix_add(
                self.grad.data, element_wise_mul(power_prime(self.data, exponent, device=self.device, shape=self.shape), out.grad.data, device=self.device, shape=self.shape), device=self.device, shape=self.shape
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

        def _backward():
            self.grad.data = matrix_add(
                self.grad.data, element_wise_mul(tanh_prime(self.data, device=self.device, shape=self.shape), out.grad.data, device=self.device, shape=self.shape), device=self.device, shape=self.shape
            )

        out._backward = _backward
        return out

    def sum(self):
        out_data = summation(self.data, device=self.device, shape=self.shape)
        out = Tensor(out_data, _children=(self,), _op="sum", device=self.device, requires_grad=self.requires_grad, shape=(1, 1))

        def _backward():
            # TODO: make this way more efficient, implement kernel for that
            if self.device == "cuda":
                grad_scalar = to_device(data=out.grad.data, device="cpu", original_shape=(1,1))[0][0]
            else:
                grad_scalar = out.grad.data[0][0]
            grad_contribution = matrix_scalar_mul(scalar=grad_scalar, matrix=ones_matrix_like(shape=self.shape, device=self.device), device=self.device, shape=self.shape)
            self.grad.data = matrix_add(self.grad.data, grad_contribution, device=self.device, shape=self.grad.shape)

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
            # TODO: make this way more efficient, implement kernel for that
            if self.device == "cuda":
                out_grad_data = to_device(data=out.grad.data, device="cpu", original_shape=out.shape)
            else:
                out_grad_data = out.grad.data
            self_len = self.shape[0]
            out_grad_data_self = out_grad_data[:self_len]
            out_grad_data_other = out_grad_data[self_len:]
            other_len = len(out_grad_data_other)
            # Send to gpu
            if self.device == "cuda":
                out_grad_data_self = to_device(data=out_grad_data_self, device=self.device, original_shape=(self_len, self.shape[1]))
                out_grad_data_other = to_device(data=out_grad_data_other, device=self.device, original_shape=(other_len, other.shape[1]))
            self.grad.data = matrix_add(self.grad.data, out_grad_data_self, device=self.device, shape=self.shape)
            other.grad.data = matrix_add(other.grad.data, out_grad_data_other, device=self.device, shape=other.shape)

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

        self.grad = Tensor(ones_matrix_like(shape=self.shape, device=self.device), requires_grad=False, device=self.device, shape=self.shape)
        for node in reversed(topo):
            node._backward()

    @property   
    def requires_grad(self):
        return self._requires_grad
    
    @requires_grad.setter
    def requires_grad(self, value: bool):
        # If requires_grad is set to False, we should also delete the grad tensor to cleanup
        if value == False:
            self.grad = None
        self._requires_grad = value
    
    def __del__(self):
        if self.device == "cuda":
            if self.data is not None:
                free_gpu_memory(self.data)
            if self.grad is not None:
                free_gpu_memory(self.grad.data)
