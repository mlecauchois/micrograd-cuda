import math

def tanh_cpu(x):
    return [[math.tanh(x_ij) for x_ij in row] for row in x]

def tanh_prime_cpu(x):
    t = tanh_cpu(x)
    return [[1 - t_ij**2 for t_ij in row] for row in t]

def matrix_mul_cpu(matrix_a, matrix_b):
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

def matrix_scalar_mul_cpu(scalar, matrix):
    return [[scalar * element for element in row] for row in matrix]


def matrix_add_cpu(matrix_a, matrix_b):
    if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
        raise ValueError("Matrices are not of the same dimensions.")

    result = [
        [matrix_a[i][j] + matrix_b[i][j] for j in range(len(matrix_a[0]))]
        for i in range(len(matrix_a))
    ]
    return result


def matrix_transpose_cpu(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


def element_wise_mul_cpu(matrix_a, matrix_b):
    if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
        raise ValueError(
            "Matrices must be of the same size for element-wise multiplication."
        )
    return [
        [matrix_a[i][j] * matrix_b[i][j] for j in range(len(matrix_a[0]))]
        for i in range(len(matrix_a))
    ]

def power_cpu(matrix, exponent):
    return [[x_ij**exponent for x_ij in row] for row in matrix]

def power_prime_cpu(matrix, exponent):
    return [[exponent * (x_ij**(exponent - 1)) for x_ij in row] for row in matrix]

def matrix_concat_cpu(matrix_a, matrix_b):
    return matrix_a + matrix_b

def summation_cpu(matrix):
    return [[sum([sum(row) for row in matrix])]]
