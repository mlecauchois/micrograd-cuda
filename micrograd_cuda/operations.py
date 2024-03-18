import ctypes
from micrograd_cuda.operations_cpu import matrix_mul_cpu, matrix_scalar_mul_cpu, matrix_add_cpu, matrix_transpose_cpu, element_wise_mul_cpu, power_cpu, power_prime_cpu, tanh_cpu, tanh_prime_cpu, matrix_concat_cpu, summation_cpu, zeros_matrix_like_cpu, ones_matrix_like_cpu
from micrograd_cuda.operations_cuda import matrix_mul_cuda, tanh_cuda, tanh_prime_cuda, matrix_transpose_cuda, matrix_add_cuda, matrix_scalar_mul_cuda, element_wise_mul_cuda, power_cuda, power_prime_cuda, matrix_concat_cuda, summation_cuda, zeros_matrix_like_cuda, ones_matrix_like_cuda

# Load the shared library
lib = ctypes.CDLL('./liboperations.so')

# Specify the argument and return types of the functions
lib.move_to_gpu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.move_to_gpu.restype = ctypes.POINTER(ctypes.c_float)
lib.move_to_cpu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]

def calculate_list_shape(data):
    shape = []
    while isinstance(data, list):
        shape.append(len(data))
        data = data[0] if data else []
    return tuple(shape)

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def reshape_to_nested_list(flat_array, original_shape):
    num_columns = original_shape[1]
    return [list(flat_array[i:i + num_columns]) for i in range(0, len(flat_array), num_columns)]

def free_gpu_memory(d_ptr):
    if not isinstance(d_ptr, ctypes.POINTER(ctypes.c_float)):
        d_ptr = ctypes.cast(d_ptr, ctypes.POINTER(ctypes.c_float))
    lib.free_gpu_memory(d_ptr)

def to_device(data, device, original_shape=None):
    if device == 'cuda':
        flat_data = flatten_list(data)
        array_type = ctypes.c_float * len(flat_data) # Create array type
        data_ctypes = array_type(*flat_data) # Instantiate array with data
        d_data = lib.move_to_gpu(data_ctypes, data_ctypes._length_)
        return d_data
    elif device == 'cpu':
        data = ctypes.cast(data, ctypes.POINTER(ctypes.c_float))
        size = original_shape[0] * original_shape[1]
        h_data = (ctypes.c_float * size)() # Create array type and instantiate array with empty data
        lib.move_to_cpu(h_data, data, size) # Move size elements starting from data on gpu to h_data on cpu
        h_data_reshaped = reshape_to_nested_list(h_data, original_shape)
        free_gpu_memory(data)
        return h_data_reshaped
    else:
        raise ValueError("Unsupported device or data type")

def zeros_matrix_like(shape: tuple, device: str):
    if device == "cpu":
        return zeros_matrix_like_cpu(shape)
    elif device == "cuda":
        return zeros_matrix_like_cuda(shape)

def ones_matrix_like(shape: tuple, device: str):
    if device == "cpu":
        return ones_matrix_like_cpu(shape)
    elif device == "cuda":
        return ones_matrix_like_cuda(shape)

def matrix_mul(matrix_a, matrix_b, matrix_a_shape, matrix_b_shape, device: str):
    if device == "cpu":
        return matrix_mul_cpu(matrix_a, matrix_b)
    elif device == "cuda":
        return matrix_mul_cuda(matrix_a, matrix_b, matrix_a_shape, matrix_b_shape)

def matrix_scalar_mul(scalar, matrix, device: str, shape: tuple):
    if device == "cpu":
        return matrix_scalar_mul_cpu(scalar, matrix)
    elif device == "cuda":
        return matrix_scalar_mul_cuda(scalar, matrix, shape)
    
def matrix_add(matrix_a, matrix_b, device: str, shape: tuple):
    if device == "cpu":
        return matrix_add_cpu(matrix_a, matrix_b)
    elif device == "cuda":
        return matrix_add_cuda(matrix_a, matrix_b, shape)
    
def matrix_transpose(matrix, device: str, shape: tuple):
    if device == "cpu":
        return matrix_transpose_cpu(matrix)
    elif device == "cuda":
        return matrix_transpose_cuda(matrix, shape)
    
def element_wise_mul(matrix_a, matrix_b, device: str, shape: tuple):
    if device == "cpu":
        return element_wise_mul_cpu(matrix_a, matrix_b)
    elif device == "cuda":
        return element_wise_mul_cuda(matrix_a, matrix_b, shape)

def power(matrix, exponent, device: str, shape: tuple):
    if device == "cpu":
        return power_cpu(matrix, exponent)
    elif device == "cuda":
        return power_cuda(matrix, exponent, shape)
    
def power_prime(matrix, exponent, device: str, shape: tuple):
    if device == "cpu":
        return power_prime_cpu(matrix, exponent)
    elif device == "cuda":
        return power_prime_cuda(matrix, exponent, shape)
    
def tanh(matrix, device: str, shape: tuple):
    if device == "cpu":
        return tanh_cpu(matrix)
    elif device == "cuda":
        return tanh_cuda(matrix, shape)
    
def tanh_prime(matrix, device: str, shape: tuple):
    if device == "cpu":
        return tanh_prime_cpu(matrix)
    elif device == "cuda":
        return tanh_prime_cuda(matrix, shape)
    
def matrix_concat(matrix_a, matrix_b, device: str, matrix_a_shape, matrix_b_shape):
    if device == "cpu":
        return matrix_concat_cpu(matrix_a, matrix_b)
    elif device == "cuda":
        return matrix_concat_cuda(matrix_a, matrix_b, matrix_a_shape, matrix_b_shape)

def summation(matrix, device: str, shape: tuple):
    if device == "cpu":
        return summation_cpu(matrix)
    elif device == "cuda":
        return summation_cuda(matrix, shape)
