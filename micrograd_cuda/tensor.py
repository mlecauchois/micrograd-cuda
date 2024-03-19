from micrograd_cuda.operations import Operations, calculate_list_shape, free_gpu_memory


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
        self.shape = calculate_list_shape(data) if shape is None else shape
        self.grad = None
        if self.requires_grad:
            zeros_data, zeros_shape = Operations(self.device).zeros_matrix_like(shape=self.shape)
            self.grad = Tensor(data=zeros_data, requires_grad=False, device=self.device, shape=zeros_shape)
    
    def to(self, device):
        if device not in ["cpu", "cuda"]:
            raise ValueError("Unsupported device. Choose 'cpu' or 'cuda'.")

        if self.device == device:
            return self

        self.data, _ = Operations(device).to_device(self.data, shape=self.shape)
        self.device = device
        
        if self.grad is not None:
            self.grad.data, _ = Operations(device).to_device(self.grad.data, shape=self.grad.shape)
            self.grad.device = device

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(data=other, device=self.device)
        out_data, out_shape = Operations(self.device).matrix_mul(self.data, other.data, self.shape, other.shape)
        out = Tensor(data=out_data, children=(self, other), shape=out_shape, device=self.device, requires_grad=self.requires_grad)

        def _backward():
            other_transpose_data, other_transpose_shape = Operations(self.device).matrix_transpose(other.data, shape=other.shape)
            new_self_grad_data, new_self_grad_shape = Operations(self.device).matrix_mul(out.grad.data, other_transpose_data, shape_a=out.grad.shape, shape_b=other_transpose_shape)
            self.grad += Tensor(data=new_self_grad_data, device=self.device, shape=new_self_grad_shape)
            self_transpose_data, self_transpose_shape = Operations(self.device).matrix_transpose(self.data, shape=self.shape)
            new_other_grad_data, new_other_grad_shape = Operations(self.device).matrix_mul(self_transpose_data, out.grad.data, shape_a=self_transpose_shape, shape_b=out.grad.shape)
            other.grad += Tensor(data=new_other_grad_data, device=self.device, shape=new_other_grad_shape)
        out._backward = _backward
        return out

    def scalar_mul(self, scalar):
        out_data, out_shape = Operations(self.device).matrix_scalar_mul(scalar, self.data, shape=self.shape)
        out = Tensor(data=out_data, children=(self,), device=self.device, requires_grad=self.requires_grad, shape=out_shape)

        def _backward():
            new_self_grad_data, new_self_grad_shape = Operations(self.device).matrix_scalar_mul(scalar, out.grad.data, shape=out.shape)
            self.grad += Tensor(data=new_self_grad_data, device=self.device, shape=new_self_grad_shape)

        out._backward = _backward
        return out

    def __mul__(self, other):

        if isinstance(other, (int, float)):
            return self.scalar_mul(other)

        other = other if isinstance(other, Tensor) else Tensor(data=other, device=self.device)
        out_data, out_shape = Operations(self.device).element_wise_mul(self.data, other.data, shape=self.shape)
        out = Tensor(data=out_data, children=(self, other), device=self.device, requires_grad=self.requires_grad, shape=out_shape)

        def _backward():
            new_self_grad_data, new_self_grad_shape = Operations(self.device).element_wise_mul(other.data, out.grad.data, shape=self.shape)
            self.grad += Tensor(data=new_self_grad_data, device=self.device, shape=new_self_grad_shape)
            new_other_grad_data, new_other_grad_shape = Operations(self.device).element_wise_mul(self.data, out.grad.data, shape=other.shape)
            other.grad += Tensor(data=new_other_grad_data, device=self.device, shape=new_other_grad_shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(data=other, device=self.device)
        out_data, out_shape = Operations(self.device).matrix_add(self.data, other.data, shape_a=self.shape, shape_b=other.shape)
        out = Tensor(data=out_data, children=(self, other), device=self.device, requires_grad=self.requires_grad, shape=out_shape)

        def _backward():
            new_self_grad_data, new_self_grad_shape = Operations(self.device).matrix_add(self.grad.data, out.grad.data, shape_a=self.shape, shape_b=out.shape)
            self.grad += Tensor(data=new_self_grad_data, device=self.device, shape=new_self_grad_shape)
            new_other_grad_data, new_other_grad_shape = Operations(self.device).matrix_add(other.grad.data, out.grad.data, shape_a=other.shape, shape_b=out.shape)
            other.grad += Tensor(data=new_other_grad_data, device=self.device, shape=new_other_grad_shape)

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        ones_data, ones_shape = Operations(self.device).ones_matrix_like(shape=self.shape)
        neg_ones_data, neg_ones_shape = Operations(self.device).matrix_scalar_mul(-1, ones_data, shape=ones_shape)
        return self * Tensor(data=neg_ones_data, device=self.device, shape=neg_ones_shape)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(data=other, device=self.device)
        return self + (-other)

    def __pow__(self, exponent):
        out_data, out_shape = Operations(self.device).power(self.data, exponent, shape=self.shape)
        out = Tensor(data=out_data, children=(self,), device=self.device, requires_grad=self.requires_grad, shape=out_shape)

        def _backward():
            power_prime_data, power_prime_shape = Operations(self.device).power_prime(self.data, exponent, shape=self.shape)
            new_self_grad_data, new_self_grad_shape = Operations(self.device).element_wise_mul(power_prime_data, out.grad.data, shape=power_prime_shape)
            self.grad += Tensor(data=new_self_grad_data, device=self.device, shape=new_self_grad_shape)

        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1

    @property
    def T(self):
        transposed_data, transposed_shape = Operations(self.device).matrix_transpose(self.data, shape=self.shape)
        return Tensor(data=transposed_data, children=(self,), device=self.device, requires_grad=self.requires_grad, shape=transposed_shape)

    def tanh(self):
        out_data, out_shape = Operations(self.device).tanh(self.data, shape=self.shape)
        out = Tensor(data=out_data, children=(self,), device=self.device, requires_grad=self.requires_grad, shape=out_shape)

        def _backward():
            tanh_prime_data, tanh_prime_shape = Operations(self.device).tanh_prime(self.data, shape=self.shape)
            new_self_grad_data, new_self_grad_shape = Operations(self.device).element_wise_mul(tanh_prime_data, out.grad.data, shape=tanh_prime_shape)
            self.grad += Tensor(data=new_self_grad_data, device=self.device, shape=new_self_grad_shape)

        out._backward = _backward
        return out
                     
    def sum(self):
        out_data, out_shape = Operations(self.device).summation(self.data, shape=self.shape)
        out = Tensor(data=out_data, children=(self,), device=self.device, requires_grad=self.requires_grad, shape=out_shape)

        def _backward():
            new_self_grad_data, new_self_grad_shape = Operations(self.device).matrix_add(self.grad.data, out.grad.data, shape_a=self.grad.shape, shape_b=out.grad.shape)
            self.grad += Tensor(data=new_self_grad_data, device=self.device, shape=new_self_grad_shape)

        out._backward = _backward
        return out
    
    def __getitem__(self, index):

        row_slice, col_slice = index 
        if isinstance(row_slice, int):
            row_slice = slice(row_slice, row_slice+1)
        if isinstance(col_slice, int):
            col_slice = slice(col_slice, col_slice+1)
        row_slice_stop = row_slice.stop if row_slice.stop is not None else self.shape[0]
        col_slice_stop = col_slice.stop if col_slice.stop is not None else self.shape[1]
        row_slice_start = row_slice.start if row_slice.start is not None else 0
        col_slice_start = col_slice.start if col_slice.start is not None else 0
        output_rows = row_slice_stop - row_slice_start
        output_cols = col_slice_stop - col_slice_start

        out_data, out_shape = Operations(self.device).indexing_2d(self.data, output_rows, output_cols, row_slice_start, col_slice_start, self.shape)
        out = Tensor(data=out_data, device=self.device, requires_grad=self.requires_grad, shape=out_shape)
        
        def _backward():
            raise NotImplementedError("Backward for indexing is not implemented yet.")
        
        out._backward = _backward
        return out

    def concat(self, other, axis=0):
        # This implementation assumes a simple case where tensors are 2D
        # and concatenated along the first dimension (axis=0)
        if axis != 0:
            raise NotImplementedError(
                "Concatenation along this axis is not implemented."
            )

        out_data, out_shape = Operations(self.device).matrix_concat(self.data, other.data, shape_a=self.shape, shape_b=other.shape)
        out = Tensor(data=out_data, children=(self, other), device=self.device, requires_grad=self.requires_grad, shape=out_shape)

        def _backward():
            # Since concatenation is along axis=0, we split the gradient back to the original tensors
            # based on their data length
            self.grad += out.grad[:self.shape[0], :]
            other.grad += out.grad[self.shape[0]:, :]

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

        grad_data, grad_shape = Operations(self.device).ones_matrix_like(shape=self.shape)
        self.grad = Tensor(data=grad_data, requires_grad=False, device=self.device, shape=grad_shape)
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

    def zero_grad(self):
        if self.grad is not None:
            self.grad.data, _ = Operations(self.device).zeros_matrix_like(shape=self.shape)
    
    def __del__(self):
        if self.device == "cuda":
            if self.data is not None:
                free_gpu_memory(self.data)
            if self.grad is not None:
                free_gpu_memory(self.grad.data)
