import time
import random

from micrograd_cuda.tensor import Tensor
from micrograd_cuda.operations import tanh_prime, power_prime, zeros_matrix_like, ones_matrix_like

def test_zeros_matrix_like():
    shape = (1000, 1000)

    # CUDA

    start = time.time()
    x = Tensor(zeros_matrix_like(shape, device="cuda"), requires_grad=False, device="cuda", shape=shape)
    print(f"Elapsed: {time.time() - start:.5f} sec")
    x.to("cpu")
    h_x_gpu = x

    # CPU

    start = time.time()
    x = Tensor(zeros_matrix_like(shape, device="cpu"), requires_grad=False, device="cpu", shape=shape)
    print(f"Elapsed: {time.time() - start:.5f} sec")

    difference = (x - h_x_gpu).abs().sum().data[0][0]/(shape[0]*shape[1])
    assert difference < 1e-5
    print(f"Difference: {difference}")

def test_ones_matrix_like():
    shape = (1000, 1000)

    # CUDA

    start = time.time()
    x = Tensor(ones_matrix_like(shape, device="cuda"), requires_grad=False, device="cuda", shape=shape)
    print(f"Elapsed: {time.time() - start:.5f} sec")
    x.to("cpu")
    h_x_gpu = x

    # CPU

    start = time.time()
    x = Tensor(ones_matrix_like(shape, device="cpu"), requires_grad=False, device="cpu", shape=shape)
    print(f"Elapsed: {time.time() - start:.5f} sec")

    difference = (x - h_x_gpu).abs().sum().data[0][0]/(shape[0]*shape[1])
    assert difference < 1e-5
    print(f"Difference: {difference}")

def test_power_prime():
    shape = (1000, 1000)

    x = Tensor([[random.random() for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False)

    ### CUDA

    x.to("cuda")
    start = time.time()
    y = power_prime(x.data, 2, device="cuda", shape=shape)
    print(f"Elapsed: {time.time() - start:.5f} sec")
    y = Tensor(y, requires_grad=False, device="cuda", shape=shape)
    y.to("cpu")
    h_y_gpu = y

    ### CPU

    x.to("cpu")
    start = time.time()
    y = power_prime(x.data, 2, device="cpu", shape=shape)
    print(f"Elapsed: {time.time() - start:.5f} sec")
    y = Tensor(y, requires_grad=False, device="cpu", shape=shape)

    difference = (y - h_y_gpu).abs().sum().data[0][0]/(shape[0]*shape[1])
    assert difference < 1e-5
    print(f"Difference: {difference}")

def test_tanh_prime():
    shape = (1000, 1000)

    x = Tensor([[random.random() for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False)

    ### CUDA

    x.to("cuda")
    start = time.time()
    y = tanh_prime(x.data, device="cuda", shape=shape)
    print(f"Elapsed: {time.time() - start:.5f} sec")
    y = Tensor(y, requires_grad=False, device="cuda", shape=shape)
    y.to("cpu")
    h_y_gpu = y

    ### CPU

    x.to("cpu")
    start = time.time()
    y = tanh_prime(x.data, device="cpu", shape=shape)
    print(f"Elapsed: {time.time() - start:.5f} sec")
    y = Tensor(y, requires_grad=False, device="cpu", shape=shape)

    difference = (y - h_y_gpu).abs().sum().data[0][0]/(shape[0]*shape[1])
    assert difference < 1e-5
    print(f"Difference: {difference}")

def test_matrix_concat():
    shape = (1000, 1000)

    # Random matrices
    x = Tensor([[random.random() for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False)
    y = Tensor([[random.random() for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False)

    ### CUDA

    x.to("cuda")
    y.to("cuda")
    start = time.time()
    z = x.concat(y)
    print(f"Elapsed: {time.time() - start:.5f} sec")
    z.to("cpu")
    h_z_gpu = z

    ### CPU

    x.to("cpu")
    y.to("cpu")
    start = time.time()
    z = x.concat(y)
    print(f"Elapsed: {time.time() - start:.5f} sec")

    difference = (z - h_z_gpu).abs().sum().data[0][0]/(z.shape[0]*z.shape[1])
    assert difference < 1e-5
    print(f"Difference: {difference}")

def test_summation():
    shape = (1000, 1)

    # Random matrices
    x = Tensor([[random.random() for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False)

    ### CUDA

    x.to("cuda")
    start = time.time()
    y = x.sum()
    print(f"Elapsed: {time.time() - start:.5f} sec")
    y.to("cpu")
    h_y_gpu = y

    ### CPU

    x.to("cpu")
    start = time.time()
    y = x.sum()
    print(f"Elapsed: {time.time() - start:.5f} sec")

    difference = (y - h_y_gpu).abs().sum().data[0][0]
    assert difference < 1e-3
    print(f"Difference: {difference}")


def test_matrix_transpose():
    shape = (1000, 1000)

    # Random matrices
    x = Tensor([[random.random() for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False)

    ### CUDA

    x.to("cuda")
    start = time.time()
    y = x.T
    print(f"Elapsed: {time.time() - start:.5f} sec")
    y.to("cpu")
    h_y_gpu = y

    ### CPU

    x.to("cpu")
    start = time.time()
    y = x.T
    print(f"Elapsed: {time.time() - start:.5f} sec")

    difference = (y - h_y_gpu).abs().sum().data[0][0]/(shape[0]*shape[1])
    assert difference < 1e-5
    print(f"Difference: {difference}")

def test_power():
    shape = (1000, 1000)

    # Random matrices
    x = Tensor([[random.random() for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False)

    ### CUDA

    x.to("cuda")
    start = time.time()
    y = x**2
    print(f"Elapsed: {time.time() - start:.5f} sec")
    y.to("cpu")
    h_y_gpu = y

    ### CPU

    x.to("cpu")
    start = time.time()
    y = x**2
    print(f"Elapsed: {time.time() - start:.5f} sec")

    difference = (y - h_y_gpu).abs().sum().data[0][0]/(shape[0]*shape[1])
    assert difference < 1e-5
    print(f"Difference: {difference}")

def test_matrix_add():
    shape = (1000, 1000)

    # Random matrices
    x = Tensor([[random.random() for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False)
    y = Tensor([[random.random() for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False)

    ### CUDA

    x.to("cuda")
    y.to("cuda")
    start = time.time()
    z = x + y
    print(f"Elapsed: {time.time() - start:.5f} sec")
    z.to("cpu")
    h_z_gpu = z

    ### CPU

    x.to("cpu")
    y.to("cpu")
    start = time.time()
    z = x + y
    print(f"Elapsed: {time.time() - start:.5f} sec")

    difference = (z - h_z_gpu).abs().sum().data[0][0]/(shape[0]*shape[1])
    assert difference < 1e-5
    print(f"Difference: {difference}")

def test_element_wise_mul():
    shape = (1000, 1000)

    # Random matrices
    x = Tensor([[random.random() for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False)
    y = Tensor([[random.random() for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False)

    ### CUDA

    x.to("cuda")
    y.to("cuda")
    start = time.time()
    z = x * y
    print(f"Elapsed: {time.time() - start:.5f} sec")
    z.to("cpu")
    h_z_gpu = z

    ### CPU

    x.to("cpu")
    y.to("cpu")
    start = time.time()
    z = x * y
    print(f"Elapsed: {time.time() - start:.5f} sec")

    difference = (z - h_z_gpu).abs().sum().data[0][0]/(shape[0]*shape[1])
    assert difference < 1e-5
    print(f"Difference: {difference}")

def test_tanh():
    shape = (1000, 1000)

    x = Tensor([[1 for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False)

    ### CUDA

    x.to("cuda")
    start = time.time()
    y = x.tanh()
    print(f"Elapsed: {time.time() - start:.5f} sec")
    y.to("cpu")
    h_y_gpu = y

    ### CPU

    x.to("cpu")
    start = time.time()
    y = x.tanh()
    print(f"Elapsed: {time.time() - start:.5f} sec")
    
    difference = (y - h_y_gpu).abs().sum().data[0][0]/(shape[0]*shape[1])
    assert difference < 1e-5
    print(f"Difference: {difference}")

def test_matrix_mul():

    shape = (100, 100)

    # Define large matrices A and B
    A = Tensor([[1 for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False)
    B = Tensor([[1 for _ in range(shape[1])] for _ in range(shape[0])], requires_grad=False)

    ### CUDA

    # Move matrices A and B to the GPU
    A.to("cuda")
    B.to("cuda")

    # Perform matrix multiplication on the GPU
    start = time.time()
    C = A @ B
    print(f"Elapsed: {time.time() - start:.5f} sec")

    # Move the result matrix C back to the CPU and reshape it to nested list
    C.to("cpu")
    h_C_gpu = C

    ### CPU

    # Move matrices A and B to the CPU
    A.to("cpu")
    B.to("cpu")

    # Perform matrix multiplication on the CPU
    start = time.time()
    C = A @ B
    print(f"Elapsed: {time.time() - start:.5f} sec")    

    difference = (C - h_C_gpu).abs().sum().data[0][0]/(shape[0]*shape[1])
    assert difference < 1e-5
    print(f"Difference: {difference}")
