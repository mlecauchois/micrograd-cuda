import time

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

# def test_mlp_train():
#     model = MLP.load("tests/data/mlp")

#     start = time.time()

#     for k in range(20):

#         # forward pass
#         ypred = [model(x) for x in xs_list]
#         loss = sum(
#             [(yout - ygt) ** 2 for ygt, yout in zip(ys_list, ypred)], Tensor([[0.0]])
#         )

#         # backward pass
#         for p in model.parameters():
#             p.grad.data = zeros_matrix_like(p.data)
#         loss.backward()

#         # update
#         for p in model.parameters():
#             p.data = matrix_add(matrix_scalar_mul(-0.1, p.grad.data, device=p.device), p.data, device=p.device)

#         print(k, loss.data)

#         if k == 0:
#             assert round(loss.data[0][0], 5) == first_loss

#     assert round(loss.data[0][0], 5) == last_loss

#     print(f"Elapsed: {time.time() - start:.2f} sec")


# def test_mlp_train_batch():
#     model = MLP.load("tests/data/mlp")

#     start = time.time()

#     for k in range(20):

#         # forward pass
#         ypred = model(xs_batch)
#         diff = ypred - ys_batch
#         loss = (diff**2).sum()

#         # backward pass
#         for p in model.parameters():
#             p.grad.data = zeros_matrix_like(p.data)
#         loss.backward()

#         # update
#         for p in model.parameters():
#             p.data = matrix_add(matrix_scalar_mul(-0.1, p.grad.data, device=p.device), p.data, device=p.device)

#         print(k, loss.data)

#         if k == 0:
#             assert round(loss.data[0][0], 5) == first_loss

#     assert round(loss.data[0][0], 5) == last_loss

#     print(f"Elapsed: {time.time() - start:.2f} sec")

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
