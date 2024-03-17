from micrograd_cuda.operations import matrix_mul, matrix_mul_cuda

a = [
    [1.0, 2.0],
    [3.0, 4.0]
]

b = [
    [5.0, 6.0, 2.0],
    [7.0, 8.0, 7.0]
]

c = matrix_mul_cuda(a, b)
print(c)

c = matrix_mul(a, b)
print(c)
