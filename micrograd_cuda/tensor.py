from micrograd_cuda.operations import Operations


class Tensor:

    def __init__(self, data, children=(), label=None, requires_grad=True, shape=None, device="cpu"):
        if device == "cuda" and shape is None:
            raise ValueError("Shape must be provided for CUDA tensors.")
        self.data = data  # Nested list on cpu, pointer to flat array on gpu
        self._requires_grad = requires_grad
        self._backward = lambda: None
        self._children = set(children)
        self.label = label
        self.device = device
        self.shape = Operations().calculate_list_shape(data) if shape is None else shape
        self.grad = None
        if self.requires_grad:
            zeros_data = Operations(self.device).zeros_matrix_like(shape=self.shape)
            self.grad = Tensor(data=zeros_data, requires_grad=False, device=self.device, shape=self.shape)
    
    def to(self, device):
        if device not in ["cpu", "cuda"]:
            raise ValueError("Unsupported device. Choose 'cpu' or 'cuda'.")

        if self.device == device:
            return self

        self.data = Operations(device).to_device(self.data, original_shape=self.shape)
        self.device = device
        
        if self.grad is not None:
            self.grad.data = Operations(device).to_device(self.grad.data, original_shape=self.grad.shape)
            self.grad.device = device

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(data=other, device=self.device)
        out_data = Operations(self.device).matrix_mul(self.data, other.data, self.shape, other.shape)
        out_shape = (self.shape[0], other.shape[1])
        out = Tensor(data=out_data, children=(self, other), shape=out_shape, device=self.device, requires_grad=self.requires_grad)

        def _backward():
            other_transpose_data = Operations(self.device).matrix_transpose(other.data, shape=other.shape)
            new_self_grad_data = Operations(self.device).matrix_mul(out.grad.data, other_transpose_data, shape_a=out.grad.shape, shape_b=(other.shape[1], other.shape[0]))
            self.grad += Tensor(data=new_self_grad_data, device=self.device, shape=self.grad.shape)
            self_transpose_data = Operations(self.device).matrix_transpose(self.data, shape=self.shape)
            new_other_grad_data = Operations(self.device).matrix_mul(self_transpose_data, out.grad.data, shape_a=(self.shape[1], self.shape[0]), shape_b=out.grad.shape)
            other.grad += Tensor(data=new_other_grad_data, device=self.device, shape=other.grad.shape)
        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        return self @ other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(data=other, device=self.device)
        out_data = Operations(self.device).element_wise_mul(self.data, other.data, shape=self.shape)
        out = Tensor(data=out_data, children=(self, other), device=self.device, requires_grad=self.requires_grad, shape=self.shape)

        def _backward():
            new_self_grad_data = Operations(self.device).element_wise_mul(other.data, out.grad.data, shape=self.shape)
            self.grad += Tensor(data=new_self_grad_data, device=self.device, shape=self.grad.shape)
            new_other_grad_data = Operations(self.device).element_wise_mul(self.data, out.grad.data, shape=other.shape)
            other.grad += Tensor(data=new_other_grad_data, device=self.device, shape=other.grad.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(data=other, device=self.device)
        out_data = Operations(self.device).matrix_add(self.data, other.data, shape=self.shape)
        out = Tensor(data=out_data, children=(self, other), device=self.device, requires_grad=self.requires_grad, shape=self.shape)

        def _backward():
            new_self_grad_data = Operations(self.device).matrix_add(self.grad.data, out.grad.data, shape=self.shape)
            self.grad += Tensor(data=new_self_grad_data, device=self.device, shape=self.grad.shape)
            new_other_grad_data = Operations(self.device).matrix_add(other.grad.data, out.grad.data, shape=other.shape)
            other.grad += Tensor(data=new_other_grad_data, device=self.device, shape=other.grad.shape)

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        ones_data = Operations(self.device).ones_matrix_like(shape=self.shape)
        neg_ones_data = Operations(self.device).matrix_scalar_mul(-1, ones_data, shape=self.shape)
        return self * Tensor(data=neg_ones_data, device=self.device, shape=self.shape)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(data=other, device=self.device)
        return self + (-other)

    def __pow__(self, exponent):
        out_data = Operations(self.device).power(self.data, exponent, shape=self.shape)
        out = Tensor(data=out_data, children=(self,), device=self.device, requires_grad=self.requires_grad, shape=self.shape)

        def _backward():
            power_prime_data = Operations(self.device).power_prime(self.data, exponent, shape=self.shape)
            new_self_grad_data = Operations(self.device).element_wise_mul(power_prime_data, out.grad.data, shape=self.shape)
            self.grad += Tensor(data=new_self_grad_data, device=self.device, shape=self.grad.shape)

        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1

    @property
    def T(self):
        transposed_data = Operations(self.device).matrix_transpose(self.data, shape=self.shape)
        return Tensor(data=transposed_data, children=(self,), device=self.device, requires_grad=self.requires_grad, shape=(self.shape[1], self.shape[0]))

    def tanh(self):
        out_data = Operations(self.device).tanh(self.data, shape=self.shape)
        out = Tensor(data=out_data, children=(self,), device=self.device, requires_grad=self.requires_grad, shape=self.shape)

        def _backward():
            tanh_prime_data = Operations(self.device).tanh_prime(self.data, shape=self.shape)
            new_self_grad_data = Operations(self.device).element_wise_mul(tanh_prime_data, out.grad.data, shape=self.shape)
            self.grad += Tensor(data=new_self_grad_data, device=self.device, shape=self.grad.shape)

        out._backward = _backward
        return out

    def sum(self):
        out_data = Operations(self.device).summation(self.data, shape=self.shape)
        out = Tensor(data=out_data, children=(self,), device=self.device, requires_grad=self.requires_grad, shape=(1, 1))

        def _backward():
            # TODO: make this way more efficient, implement kernel for that
            if self.device == "cuda":
                grad_scalar = Operations("cpu").to_device(data=out.grad.data, original_shape=(1,1))[0][0]
            else:
                grad_scalar = out.grad.data[0][0]
            grad_contribution = Operations(self.device).matrix_scalar_mul(scalar=grad_scalar, matrix=Operations(self.device).ones_matrix_like(shape=self.shape), shape=self.shape)
            self.grad.data = Operations(self.device).matrix_add(self.grad.data, grad_contribution, shape=self.grad.shape)

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

        out = Tensor(data=Operations(self.device).matrix_concat(self.data, other.data, shape_a=self.shape, shape_b=other.shape), children=(self, other), device=self.device, requires_grad=self.requires_grad, shape=(self.shape[0] + other.shape[0], self.shape[1]))

        # TODO: fix grad operations
        def _backward():
            # Since concatenation is along axis=0, we split the gradient back to the original tensors
            # based on their data length
            # TODO: make this way more efficient, implement kernel for that
            if self.device == "cuda":
                out_grad_data = Operations("cpu").to_device(data=out.grad.data, original_shape=out.shape)
            else:
                out_grad_data = out.grad.data
            self_len = self.shape[0]
            out_grad_data_self = out_grad_data[:self_len]
            out_grad_data_other = out_grad_data[self_len:]
            other_len = len(out_grad_data_other)
            # Send to gpu
            if self.device == "cuda":
                out_grad_data_self = Operations(self.device).to_device(data=out_grad_data_self, original_shape=(self_len, self.shape[1]))
                out_grad_data_other = Operations(self.device).to_device(data=out_grad_data_other, original_shape=(other_len, other.shape[1]))
            self.grad.data = Operations(self.device).matrix_add(self.grad.data, out_grad_data_self, shape=self.shape)
            other.grad.data = Operations(self.device).matrix_add(other.grad.data, out_grad_data_other, shape=other.shape)

        out._backward = _backward
        return out
    
    def abs(self):
        if self.device != "cpu":
            raise NotImplementedError("abs is not implemented for CUDA tensors yet.")
        if self.requires_grad:
            raise NotImplementedError("abs is not implemented for tensors with requires_grad=True yet.")
        out = Tensor(data=[[abs(cell) for cell in row] for row in self.data], children=(self,))
        return out


    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        grad_data = Operations(self.device).ones_matrix_like(shape=self.shape)
        self.grad = Tensor(data=grad_data, requires_grad=False, device=self.device, shape=self.shape)
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
                Operations().free_gpu_memory(self.data)
            if self.grad is not None:
                Operations().free_gpu_memory(self.grad.data)
