import ctypes
from micrograd_cuda.operations_cpu import matrix_mul_cpu, matrix_scalar_mul_cpu, matrix_add_cpu, matrix_transpose_cpu, element_wise_mul_cpu, power_cpu, power_prime_cpu, tanh_cpu, tanh_prime_cpu, matrix_concat_cpu, summation_cpu
from micrograd_cuda.operations_cuda import matrix_mul_cuda

# Load the shared library
lib = ctypes.CDLL('./liboperations.so')

# Specify the argument and return types of the functions
lib.move_to_gpu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.move_to_gpu.restype = ctypes.POINTER(ctypes.c_float)
lib.move_to_cpu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def to_device(data, device, original_shape=None):
    if device == 'cuda':
        # Flatten the nested list and convert to ctypes array
        flat_data = flatten_list(data)
        array_type = ctypes.c_float * len(flat_data)
        data_ctypes = array_type(*flat_data)
        d_data = lib.move_to_gpu(data_ctypes, len(flat_data))
        return d_data, data_ctypes._length_
    elif device == 'cpu':
        # Convert data wich is int pointer to ctypes
        data = ctypes.cast(data, ctypes.POINTER(ctypes.c_float))
        size = original_shape[0] * original_shape[1]
        # Allocate a ctypes array for data on CPU
        c_data_ctypes = (ctypes.c_float * size)()
        # Move data from GPU to CPU
        lib.move_to_cpu(c_data_ctypes, data, size)
        # Convert ctypes array back to nested lists if original shape is provided
        if original_shape:
            c_data = [list(c_data_ctypes[i:i+original_shape[1]]) for i in range(0, len(c_data_ctypes), original_shape[1])]
        else:
            c_data = list(c_data_ctypes)
        return c_data
    else:
        raise ValueError("Unsupported device or data type")

def zeros_matrix_like(matrix):
    return [[0 for _ in row] for row in matrix]

def ones_matrix_like(matrix):
    return [[1 for _ in row] for row in matrix]

def matrix_mul(matrix_a, matrix_b, device: str):
    if device == "cpu":
        return matrix_mul_cpu(matrix_a, matrix_b)
    elif device == "cuda":
        return matrix_mul_cuda(matrix_a, matrix_b)

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
    
def matrix_concat(matrix_a, matrix_b, device: str):
    if device == "cpu":
        return matrix_concat_cpu(matrix_a, matrix_b)

def summation(matrix, device: str):
    if device == "cpu":
        return summation_cpu(matrix)
