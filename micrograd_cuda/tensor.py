import math

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
)


class Tensor:

    def __init__(self, data, _children=(), _op="", label="", requires_grad=True):
        self.data = data  # Nested list
        self.requires_grad = requires_grad
        self.grad = Tensor(zeros_matrix_like(data), requires_grad=False) if self.requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.device = "cpu"

    @property
    def shape(self):
        shape = []
        element = self.data
        while isinstance(element, list):
            shape.append(len(element))
            element = element[0] if element else []

        return tuple(shape)

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(matrix_mul(self.data, other.data, device=self.device), (self, other), "@")

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
        out = Tensor(element_wise_mul(self.data, other.data, device=self.device), (self, other), "*")

        def _backward():
            self.grad.data = matrix_add(self.grad.data, element_wise_mul(out.grad.data, other.data, device=self.device), device=self.device)
            other.grad.data = matrix_add(other.grad.data, element_wise_mul(self.data, out.grad.data, device=self.device), device=self.device)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(matrix_add(self.data, other.data, device=self.device), (self, other), "+")

        def _backward():
            self.grad.data = matrix_add(self.grad.data, out.grad.data, device=self.device)
            other.grad.data = matrix_add(other.grad.data, out.grad.data, device=self.device)

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        neg_one = matrix_scalar_mul(-1, ones_matrix_like(self.data), device=self.device)
        return self * neg_one

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __pow__(self, exponent):
        result_data = power(self.data, exponent, device=self.device)
        out = Tensor(result_data, (self,), f"**{exponent}")

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
        transposed_data = matrix_transpose(self.data, device=self.device)
        return Tensor(transposed_data, _children=(self,), _op="transpose")

    def tanh(self):
        result_data = tanh(self.data, device=self.device)
        out = Tensor(result_data, (self,), "tanh")

        def _backward():
            self.grad.data = matrix_add(
                self.grad.data, element_wise_mul(tanh_prime(self.data, device=self.device), out.grad.data, device=self.device), device=self.device
            )

        out._backward = _backward
        return out

    def sum(self):
        # Sum over all elements to produce a single scalar value within a 2D list for consistency
        total_sum = sum(sum(row) for row in self.data)
        out_data = [[total_sum]]  # Output is a 1x1 tensor

        out = Tensor(out_data, _children=(self,), _op="sum")

        def _backward():
            # TODO: implement proper cpu and cuda kernel for this
            # The gradient of the sum operation is 1 for each element in the original tensor,
            # since each element contributes equally to the total sum.
            grad_contribution = [[out.grad.data[0][0] for _ in row] for row in self.data]
            self.grad.data = matrix_add(self.grad.data, grad_contribution, device=self.device)

        out._backward = _backward
        return out

    def concat(self, other, axis=0):
        # This implementation assumes a simple case where tensors are 2D
        # and concatenated along the first dimension (axis=0)
        if axis != 0:
            raise NotImplementedError(
                "Concatenation along this axis is not implemented."
            )

        concatenated_data = self.data + other.data

        out = Tensor(concatenated_data, _children=(self, other), _op="concat")

        def _backward():
            # Since concatenation is along axis=0, we split the gradient back to the original tensors
            # based on their data length
            self_len = len(self.data)
            self.grad.data = matrix_add(self.grad.data, out.grad.data[:self_len], device=self.device)
            other.grad.data = matrix_add(other.grad.data, out.grad.data[self_len:], device=self.device)

        out._backward = _backward
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
