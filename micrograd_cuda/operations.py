import math


def tanh(x):
    return (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)


def tanh_prime(x):
    t = tanh(x)
    return 1 - t**2


def exp(x):
    return math.exp(x)


def zeros_matrix_like(matrix):
    return [[0 for _ in row] for row in matrix]


def ones_matrix_like(matrix):
    return [[1 for _ in row] for row in matrix]


def matrix_mul(matrix_a, matrix_b):
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0])

    if cols_a != rows_b:
        raise ValueError(
            f"The number of columns in the first matrix must be equal to the number of rows in the second matrix. Got cols_a {cols_a} and rows_b {rows_b}."
        )

    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def matrix_scalar_mul(scalar, matrix):
    return [[scalar * element for element in row] for row in matrix]


def matrix_add(matrix_a, matrix_b):
    if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
        raise ValueError("Matrices are not of the same dimensions.")

    result = [
        [matrix_a[i][j] + matrix_b[i][j] for j in range(len(matrix_a[0]))]
        for i in range(len(matrix_a))
    ]
    return result


def matrix_transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


def element_wise_mul(matrix_a, matrix_b):
    if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
        raise ValueError(
            "Matrices must be of the same size for element-wise multiplication."
        )
    return [
        [matrix_a[i][j] * matrix_b[i][j] for j in range(len(matrix_a[0]))]
        for i in range(len(matrix_a))
    ]
