import ctypes

# Load the shared library
lib = ctypes.CDLL('./liboperations.so')

lib.matmul_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.allocate_on_gpu.restype = ctypes.c_void_p


def matrix_mul_cuda(d_a, d_b, a_shape, b_shape):
    c_size = a_shape[0] * b_shape[1]
    d_c = lib.allocate_on_gpu(c_size)
    lib.matmul_on_gpu(d_a, d_b, d_c, a_shape[0], a_shape[1], b_shape[1])
    return d_c

def free_gpu_memory(d_ptr):
    lib.free_gpu_memory(d_ptr)

