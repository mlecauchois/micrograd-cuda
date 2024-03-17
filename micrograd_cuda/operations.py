import math

from micrograd_cuda.operations_cpu import matrix_mul_cpu, matrix_scalar_mul_cpu, matrix_add_cpu, matrix_transpose_cpu, element_wise_mul_cpu, power_cpu, power_prime_cpu, tanh_cpu, tanh_prime_cpu

def zeros_matrix_like(matrix):
    return [[0 for _ in row] for row in matrix]

def ones_matrix_like(matrix):
    return [[1 for _ in row] for row in matrix]

def matrix_mul(matrix_a, matrix_b, device: str):
    if device == "cpu":
        return matrix_mul_cpu(matrix_a, matrix_b)

def matrix_scalar_mul(scalar, matrix, device: str):
    if device == "cpu":
        return matrix_scalar_mul_cpu(scalar, matrix)
    
def matrix_add(matrix_a, matrix_b, device: str):
    if device == "cpu":
        return matrix_add_cpu(matrix_a, matrix_b)
    
def matrix_transpose(matrix, device: str):
    if device == "cpu":
        return matrix_transpose_cpu(matrix)
    
def element_wise_mul(matrix_a, matrix_b, device: str):
    if device == "cpu":
        return element_wise_mul_cpu(matrix_a, matrix_b)

def power(matrix, exponent, device: str):
    if device == "cpu":
        return power_cpu(matrix, exponent)
    
def power_prime(matrix, exponent, device: str):
    if device == "cpu":
        return power_prime_cpu(matrix, exponent)
    
def tanh(matrix, device: str):
    if device == "cpu":
        return tanh_cpu(matrix)
    
def tanh_prime(matrix, device: str):
    if device == "cpu":
        return tanh_prime_cpu(matrix)
    