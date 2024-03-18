import ctypes

# Load the shared library
lib = ctypes.CDLL('./liboperations.so')

lib.matmul_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.allocate_on_gpu.restype = ctypes.c_void_p
lib.tanh_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.tanh_prime_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.transpose_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.add_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.scalar_mul_on_gpu.argtypes = [ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.element_wise_mul_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.power_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
lib.power_prime_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
lib.matrix_concat_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.summation_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.add_scalar_on_gpu.argtypes = [ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def matrix_mul_cuda(d_a, d_b, a_shape, b_shape):
    c_size = a_shape[0] * b_shape[1]
    d_c = lib.allocate_on_gpu(c_size)
    lib.matmul_on_gpu(d_a, d_b, d_c, a_shape[0], a_shape[1], b_shape[1])
    return d_c

def tanh_cuda(d_a, a_shape):
    d_c = lib.allocate_on_gpu(a_shape[0] * a_shape[1])
    lib.tanh_on_gpu(d_a, d_c, a_shape[0] * a_shape[1])
    return d_c

def tanh_prime_cuda(d_a, a_shape):
    d_c = lib.allocate_on_gpu(a_shape[0] * a_shape[1])
    lib.tanh_prime_on_gpu(d_a, d_c, a_shape[0] * a_shape[1])
    return d_c

def matrix_transpose_cuda(d_a, a_shape):
    d_c = lib.allocate_on_gpu(a_shape[0] * a_shape[1])
    lib.transpose_on_gpu(d_a, d_c, a_shape[0], a_shape[1])
    return d_c

def matrix_add_cuda(d_a, d_b, a_shape):
    d_c = lib.allocate_on_gpu(a_shape[0] * a_shape[1])
    lib.add_on_gpu(d_a, d_b, d_c, a_shape[0] * a_shape[1])
    return d_c

def matrix_scalar_mul_cuda(scalar, d_a, a_shape):
    d_c = lib.allocate_on_gpu(a_shape[0] * a_shape[1])
    lib.scalar_mul_on_gpu(scalar, d_a, d_c, a_shape[0] * a_shape[1])
    return d_c

def element_wise_mul_cuda(d_a, d_b, a_shape):
    d_c = lib.allocate_on_gpu(a_shape[0] * a_shape[1])
    lib.element_wise_mul_on_gpu(d_a, d_b, d_c, a_shape[0] * a_shape[1])
    return d_c

def power_cuda(d_a, exponent, a_shape):
    d_c = lib.allocate_on_gpu(a_shape[0] * a_shape[1])
    lib.power_on_gpu(d_a, exponent, d_c, a_shape[0] * a_shape[1])
    return d_c

def power_prime_cuda(d_a, exponent, a_shape):
    d_c = lib.allocate_on_gpu(a_shape[0] * a_shape[1])
    lib.power_prime_on_gpu(d_a, exponent, d_c, a_shape[0] * a_shape[1])
    return d_c

def matrix_concat_cuda(d_a, d_b, a_shape, b_shape):
    # TODO: any axis concat
    d_c = lib.allocate_on_gpu((a_shape[0] + b_shape[0]) * a_shape[1])
    lib.matrix_concat_on_gpu(d_a, d_b, d_c, a_shape[0], a_shape[1], b_shape[0], b_shape[1])
    return d_c

def summation_cuda(d_a, a_shape):
    c_size = 1
    d_c = lib.allocate_on_gpu(c_size)
    lib.summation_on_gpu(d_a, d_c, a_shape[0] * a_shape[1])
    return d_c

def zeros_matrix_like_cuda(shape):
    # TODO: accelerate this
    d_c = lib.allocate_on_gpu(shape[0] * shape[1])
    lib.scalar_mul_on_gpu(0.0, d_c, d_c, shape[0] * shape[1])
    return d_c

def ones_matrix_like_cuda(shape):
    # TODO: accelerate this
    d_c = lib.allocate_on_gpu(shape[0] * shape[1])
    lib.scalar_mul_on_gpu(0.0, d_c, d_c, shape[0] * shape[1])
    lib.add_scalar_on_gpu(1.0, d_c, d_c, shape[0] * shape[1])
    return d_c
