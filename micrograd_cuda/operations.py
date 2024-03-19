import ctypes
import math

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
lib.move_to_gpu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.move_to_gpu.restype = ctypes.POINTER(ctypes.c_float)
lib.move_to_cpu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]

class OperationsBase:
    
    @staticmethod
    def calculate_list_shape(data):
        shape = []
        while isinstance(data, list):
            shape.append(len(data))
            data = data[0] if data else []
        return tuple(shape)

    @staticmethod
    def flatten_list(nested_list):
        return [item for sublist in nested_list for item in sublist]

    @staticmethod
    def reshape_to_nested_list(flat_array, original_shape):
        num_columns = original_shape[1]
        return [list(flat_array[i:i + num_columns]) for i in range(0, len(flat_array), num_columns)]

    @staticmethod
    def free_gpu_memory(d_ptr):
        if not isinstance(d_ptr, ctypes.POINTER(ctypes.c_float)):
            d_ptr = ctypes.cast(d_ptr, ctypes.POINTER(ctypes.c_float))
        lib.free_gpu_memory(d_ptr)

class OperationsCuda(OperationsBase):
    
    @staticmethod
    def matrix_mul(d_a, d_b, a_shape, b_shape):
        c_size = a_shape[0] * b_shape[1]
        d_c = lib.allocate_on_gpu(c_size)
        lib.matmul_on_gpu(d_a, d_b, d_c, a_shape[0], a_shape[1], b_shape[1])
        return d_c

    @staticmethod
    def tanh(matrix, shape):
        d_c = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.tanh_on_gpu(matrix, d_c, shape[0] * shape[1])
        return d_c

    @staticmethod
    def tanh_prime(matrix, shape):
        d_c = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.tanh_prime_on_gpu(matrix, d_c, shape[0] * shape[1])
        return d_c

    @staticmethod
    def matrix_transpose(matrix, shape):
        d_c = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.transpose_on_gpu(matrix, d_c, shape[0], shape[1])
        return d_c

    @staticmethod
    def matrix_add(d_a, d_b, shape):
        d_c = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.add_on_gpu(d_a, d_b, d_c, shape[0] * shape[1])
        return d_c

    @staticmethod
    def matrix_scalar_mul(scalar, matrix, shape):
        d_c = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.scalar_mul_on_gpu(scalar, matrix, d_c, shape[0] * shape[1])
        return d_c

    @staticmethod
    def element_wise_mul(d_a, d_b, shape):
        d_c = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.element_wise_mul_on_gpu(d_a, d_b, d_c, shape[0] * shape[1])
        return d_c

    @staticmethod
    def power(matrix, exponent, shape):
        d_c = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.power_on_gpu(matrix, exponent, d_c, shape[0] * shape[1])
        return d_c

    @staticmethod
    def power_prime(matrix, exponent, shape):
        d_c = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.power_prime_on_gpu(matrix, exponent, d_c, shape[0] * shape[1])
        return d_c

    @staticmethod
    def matrix_concat(d_a, d_b, a_shape, b_shape):
        # TODO: any axis concat
        d_c = lib.allocate_on_gpu((a_shape[0] + b_shape[0]) * a_shape[1])
        lib.matrix_concat_on_gpu(d_a, d_b, d_c, a_shape[0], a_shape[1], b_shape[0], b_shape[1])
        return d_c

    @staticmethod
    def summation(matrix, shape):
        c_size = 1
        d_c = lib.allocate_on_gpu(c_size)
        lib.summation_on_gpu(matrix, d_c, shape[0] * shape[1])
        return d_c

    @staticmethod
    def zeros_matrix_like(shape):
        # TODO: accelerate this
        d_c = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.scalar_mul_on_gpu(0.0, d_c, d_c, shape[0] * shape[1])
        return d_c

    @staticmethod
    def ones_matrix_like(shape):
        # TODO: accelerate this
        d_c = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.scalar_mul_on_gpu(0.0, d_c, d_c, shape[0] * shape[1])
        lib.add_scalar_on_gpu(1.0, d_c, d_c, shape[0] * shape[1])
        return d_c
    
    @staticmethod
    def to_device(data, original_shape=None):
        flat_data = OperationsCuda.flatten_list(data)
        array_type = ctypes.c_float * len(flat_data) # Create array type
        data_ctypes = array_type(*flat_data) # Instantiate array with data
        d_data = lib.move_to_gpu(data_ctypes, data_ctypes._length_)
        return d_data

class OperationsCpu(OperationsBase):

    @staticmethod
    def tanh(x, shape):
        return [[math.tanh(x_ij) for x_ij in row] for row in x]

    @staticmethod
    def tanh_prime(x, shape):
        t = OperationsCpu.tanh(x, shape)
        return [[1 - t_ij**2 for t_ij in row] for row in t]

    @staticmethod
    def matrix_mul(matrix_a, matrix_b, a_shape, b_shape):
        rows_a = a_shape[0]
        cols_a = a_shape[1]
        rows_b = b_shape[0]
        cols_b = b_shape[1]

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

    @staticmethod
    def matrix_scalar_mul(scalar, matrix, shape):
        return [[scalar * element for element in row] for row in matrix]


    @staticmethod
    def matrix_add(matrix_a, matrix_b, shape):
        if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
            raise ValueError("Matrices are not of the same dimensions.")

        result = [
            [matrix_a[i][j] + matrix_b[i][j] for j in range(len(matrix_a[0]))]
            for i in range(len(matrix_a))
        ]
        return result

    @staticmethod
    def matrix_transpose(matrix, shape):
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

    @staticmethod
    def element_wise_mul(matrix_a, matrix_b, shape):
        if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
            raise ValueError(
                "Matrices must be of the same size for element-wise multiplication."
            )
        return [
            [matrix_a[i][j] * matrix_b[i][j] for j in range(len(matrix_a[0]))]
            for i in range(len(matrix_a))
        ]

    @staticmethod
    def power(matrix, exponent, shape):
        return [[x_ij**exponent for x_ij in row] for row in matrix]

    @staticmethod
    def power_prime(matrix, exponent, shape):
        return [[exponent * (x_ij**(exponent - 1)) for x_ij in row] for row in matrix]

    @staticmethod
    def matrix_concat(matrix_a, matrix_b, a_shape, b_shape):
        return matrix_a + matrix_b

    @staticmethod
    def summation(matrix, shape):
        return [[sum([sum(row) for row in matrix])]]

    @staticmethod
    def zeros_matrix_like(shape):
        return [[0 for _ in range(shape[1])] for _ in range(shape[0])]

    @staticmethod
    def ones_matrix_like(shape):
        return [[1 for _ in range(shape[1])] for _ in range(shape[0])]

    @staticmethod
    def to_device(data, original_shape=None):
        data = ctypes.cast(data, ctypes.POINTER(ctypes.c_float))
        size = original_shape[0] * original_shape[1]
        h_data = (ctypes.c_float * size)() # Create array type and instantiate array with empty data
        lib.move_to_cpu(h_data, data, size) # Move size elements starting from data on gpu to h_data on cpu
        h_data_reshaped = OperationsCpu.reshape_to_nested_list(h_data, original_shape)
        OperationsCpu.free_gpu_memory(data)
        return h_data_reshaped

class Operations:

    def __new__(cls, device=None):
        if device == 'cuda':
            return OperationsCuda
        elif device == 'cpu':
            return OperationsCpu
        else:
            return OperationsBase
