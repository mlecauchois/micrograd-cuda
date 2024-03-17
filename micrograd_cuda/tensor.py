import math

from micrograd_cuda.operations import (
    tanh,
    tanh_prime,
    exp,
    zeros_matrix_like,
    ones_matrix_like,
    matrix_mul,
    matrix_scalar_mul,
    matrix_add,
    element_wise_mul,
    matrix_transpose,
)


class Tensor:

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data  # Nested list
        self.grad = zeros_matrix_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    @property
    def shape(self):
        # Initialize the shape list
        shape = []

        # Check the first element to determine if it's a list
        # and recursively find the dimensions
        element = self.data
        while isinstance(element, list):
            shape.append(len(element))
            element = element[0] if element else []

        return tuple(shape)

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(matrix_mul(self.data, other.data), (self, other), "@")

        def _backward():
            self.grad = matrix_add(
                self.grad, matrix_mul(out.grad, matrix_transpose(other.data))
            )
            other.grad = matrix_add(
                other.grad, matrix_mul(matrix_transpose(self.data), out.grad)
            )

        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        return self @ other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(element_wise_mul(self.data, other.data), (self, other), "*")

        def _backward():
            self.grad = matrix_add(self.grad, element_wise_mul(out.grad, other.data))
            other.grad = matrix_add(other.grad, element_wise_mul(self.data, out.grad))

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(matrix_add(self.data, other.data), (self, other), "+")

        def _backward():
            self.grad = matrix_add(self.grad, out.grad)
            other.grad = matrix_add(other.grad, out.grad)

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        neg_one = matrix_scalar_mul(-1, ones_matrix_like(self.data))
        return self * neg_one

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __pow__(self, power):

        # Apply power function element-wise
        result_data = [[x_ij**power for x_ij in row] for row in self.data]
        out = Tensor(result_data, (self,), f"**{power}")

        def _backward():
            # Compute gradient with respect to the input tensor
            for i, row in enumerate(self.data):
                for j, val in enumerate(row):
                    # Derivative of val**power is power * val**(power-1)
                    self.grad[i][j] += power * (val ** (power - 1)) * out.grad[i][j]

        out._backward = _backward

        return out

    def __truediv__(self, other):  # self / other
        return self * other**-1

    @property
    def T(self):
        transposed_data = matrix_transpose(self.data)
        return Tensor(transposed_data, _children=(self,), _op="transpose")

    def apply_func(self, func, func_prime):
        result_data = [[func(x_ij) for x_ij in row] for row in self.data]
        out = Tensor(result_data, (self,), func.__name__)

        def _backward():
            for i, row in enumerate(self.data):
                for j, _ in enumerate(row):
                    self.grad[i][j] += func_prime(self.data[i][j]) * out.grad[i][j]

        out._backward = _backward
        return out

    def sum(self):
        # Sum over all elements to produce a single scalar value within a 2D list for consistency
        total_sum = sum(sum(row) for row in self.data)
        out_data = [[total_sum]]  # Output is a 1x1 tensor

        out = Tensor(out_data, _children=(self,), _op="sum")

        def _backward():
            # The gradient of the sum operation is 1 for each element in the original tensor,
            # since each element contributes equally to the total sum.
            grad_contribution = [[out.grad[0][0] for _ in row] for row in self.data]
            self.grad = matrix_add(self.grad, grad_contribution)

        out._backward = _backward
        return out

    def tanh(self):
        return self.apply_func(tanh, tanh_prime)

    def exp(self):
        return self.apply_func(exp, exp)

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
            self.grad = matrix_add(self.grad, out.grad[:self_len])
            other.grad = matrix_add(other.grad, out.grad[self_len:])

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

        self.grad = ones_matrix_like(self.data)
        for node in reversed(topo):
            node._backward()
