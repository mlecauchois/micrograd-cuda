import time
import random

from micrograd_cuda.mlp import MLP
from micrograd_cuda.tensor import Tensor
from micrograd_cuda.tensor import matrix_add, matrix_scalar_mul, zeros_matrix_like

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [[1.0], [-1.0], [-1.0], [1.0]]
x = Tensor([xs[0]]).T
x_no_grad = Tensor([xs[0]], requires_grad=False).T
xs_list = [Tensor([x]).T for x in xs]
xs_batch = Tensor(xs).T
ys_list = [Tensor([y]).T for y in ys]
ys_batch = Tensor(ys).T
first_loss = 4.82390
last_loss = 0.00900
first_value = -0.5952

def test_mlp_inference():

    model = MLP.load("tests/data/mlp")

    for p in model.parameters():
        p.requires_grad = False

    # GPU
    x_no_grad.to("cuda")
    model.to("cuda")
    start = time.time()
    out = model(x_no_grad)
    print(f"Elapsed: {time.time() - start:.5f} sec")
    out.to("cpu")
    assert round(out.data[0][0], 5) == first_value

    # CPU
    x_no_grad.to("cpu")
    model.to("cpu")
    start = time.time()
    out = model(x_no_grad)
    print(f"Elapsed: {time.time() - start:.5f} sec")
    assert round(out.data[0][0], 5) == first_value

def test_mlp_inference_large():

    x_no_grad_large = Tensor([[2.0 for _ in range(1000)]], requires_grad=False).T

    # Create large model
    model = MLP(1000, [1000, 1000, 1000, 1000])

    for p in model.parameters():
        p.requires_grad = False

    # GPU
    x_no_grad_large.to("cuda")
    model.to("cuda")
    start = time.time()
    out = model(x_no_grad_large)
    print(f"Elapsed: {time.time() - start:.5f} sec")
    out.to("cpu")
    out_gpu = out

    # CPU
    x_no_grad_large.to("cpu")
    model.to("cpu")
    start = time.time()
    out = model(x_no_grad_large)
    print(f"Elapsed: {time.time() - start:.5f} sec")
    out_cpu = out

    difference = (out_cpu - out_gpu).abs().sum().data[0][0]/(1000)
    assert difference < 1e-5

def test_backward():

    model = MLP(1000, [1000, 1000, 1000, 1])
    x_test_backward = Tensor([[2.0 for _ in range(1000)]]).T

    # GPU
    x_test_backward.to("cuda")
    model.to("cuda")

    for p in model.parameters():
        p.grad.data = zeros_matrix_like(device="cuda", shape=p.shape)

    out = model(x_test_backward)
    y = Tensor([[1.0]])
    y.to("cuda")
    loss = (out - y).sum()

    start = time.time()
    loss.backward()
    print(f"Elapsed: {time.time() - start:.5f} sec")
    loss.to("cpu")
    loss_gpu = loss

    # CPU
    x_test_backward.to("cpu")
    model.to("cpu")

    for p in model.parameters():
        p.grad.data = zeros_matrix_like(device="cpu", shape=p.shape)

    out = model(x_test_backward)
    y = Tensor([[1.0]])
    loss = (out - y).sum()

    start = time.time()
    loss.backward()
    print(f"Elapsed: {time.time() - start:.5f} sec")

    loss_gpu.requires_grad = False
    loss.requires_grad = False
    difference = (loss - loss_gpu).abs().sum().data[0][0]
    assert difference < 1e-5

def mlp_train_batch(device: str):
    model = MLP.load("tests/data/mlp")
    model.to(device)
    xs_batch.to(device)
    ys_batch.to(device)

    start = time.time()

    for k in range(20):

        # forward pass
        ypred = model(xs_batch)
        diff = ypred - ys_batch
        loss = (diff**2).sum()

        # backward pass
        for p in model.parameters():
            p.grad.data = zeros_matrix_like(device=device, shape=p.shape)

        loss.backward()

        # update
        for p in model.parameters():
            p.data = matrix_add(matrix_scalar_mul(-0.1, p.grad.data, device=p.device, shape=p.shape), p.data, device=p.device, shape=p.shape)
    print(f"Elapsed: {time.time() - start:.2f} sec")
    loss.to("cpu")
    assert round(loss.data[0][0], 5) == last_loss

def test_mlp_train_batch():
    mlp_train_batch("cuda")
    mlp_train_batch("cpu")

def mlp_train_batch_large(device: str, model, xs_batch_large, ys_batch_large):

    model.to(device)
    xs_batch_large.to(device)
    ys_batch_large.to(device)
    
    start = time.time()

    for k in range(1):

        # forward pass
        ypred = model(xs_batch_large)
        diff = ypred - ys_batch_large
        loss = (diff**2).sum()

        # backward pass
        for p in model.parameters():
            p.grad.data = zeros_matrix_like(device=device, shape=p.shape)

        loss.backward()

        # update
        for p in model.parameters():
            p.data = matrix_add(matrix_scalar_mul(-0.1, p.grad.data, device=p.device, shape=p.shape), p.data, device=p.device, shape=p.shape)

    print(f"Elapsed: {time.time() - start:.2f} sec")

    loss.to("cpu")
    
    return loss.data[0][0]

def test_mlp_train_batch_large():
    model_1 = MLP(300, [300, 300, 1])
    model_2 = model_1.copy()
    xs_batch_large = Tensor([[random.random() for _ in range(300)] for _ in range(100)]).T
    ys_batch_large = Tensor([[random.random()] for _ in range(100)]).T

    loss_gpu = mlp_train_batch_large("cuda", model_1, xs_batch_large, ys_batch_large)
    loss_cpu = mlp_train_batch_large("cpu", model_2, xs_batch_large, ys_batch_large)
    assert round(loss_cpu, 2) == round(loss_gpu, 2)
    print(loss_cpu, loss_gpu)
