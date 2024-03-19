import ctypes
import math

# Load the shared library
lib = ctypes.CDLL('./liboperations.so')
lib.matmul_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.allocate_on_gpu.restype = ctypes.c_void_p
lib.tanh_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.tanh_prime_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.transpose_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.add_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
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
lib.indexing_2d_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.copy_on_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def calculate_list_shape(data):
    shape = []
    while isinstance(data, list):
        shape.append(len(data))
        data = data[0] if data else []
    return tuple(shape)

def flatten_list(data):
    return [item for sublist in data for item in sublist]

def reshape_to_nested_list(flat_array, shape):
    num_columns = shape[1]
    return [list(flat_array[i:i + num_columns]) for i in range(0, len(flat_array), num_columns)]

def free_gpu_memory(data):
    if not isinstance(data, ctypes.POINTER(ctypes.c_float)):
        data = ctypes.cast(data, ctypes.POINTER(ctypes.c_float))
    lib.free_gpu_memory(data)

class OperationsBase:
    pass

class OperationsCuda(OperationsBase):
    
    @staticmethod
    def matrix_mul(matrix_a, matrix_b, shape_a, shape_b):
        size = shape_a[0] * shape_b[1]
        out = lib.allocate_on_gpu(size)
        lib.matmul_on_gpu(matrix_a, matrix_b, out, shape_a[0], shape_a[1], shape_b[1])
        return out, (shape_a[0], shape_b[1])

    @staticmethod
    def tanh(matrix, shape):
        out = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.tanh_on_gpu(matrix, out, shape[0] * shape[1])
        return out, shape

    @staticmethod
    def tanh_prime(matrix, shape):
        out = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.tanh_prime_on_gpu(matrix, out, shape[0] * shape[1])
        return out, shape

    @staticmethod
    def matrix_transpose(matrix, shape):
        out = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.transpose_on_gpu(matrix, out, shape[0], shape[1])
        return out, (shape[1], shape[0])

    @staticmethod
    def matrix_add(matrix_a_gpu, matrix_b_gpu, shape_a, shape_b):
        c_rows = max(shape_a[0], shape_b[0])
        c_cols = max(shape_a[1], shape_b[1])
        out_gpu = lib.allocate_on_gpu(c_rows * c_cols)
        lib.add_on_gpu(matrix_a_gpu, matrix_b_gpu, out_gpu, shape_a[0], shape_a[1], shape_b[0], shape_b[1])
        return out_gpu, (c_rows, c_cols)

    @staticmethod
    def matrix_scalar_mul(scalar, matrix, shape):
        out = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.scalar_mul_on_gpu(scalar, matrix, out, shape[0] * shape[1])
        return out, shape

    @staticmethod
    def element_wise_mul(matrix_a, matrix_b, shape):
        out = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.element_wise_mul_on_gpu(matrix_a, matrix_b, out, shape[0] * shape[1])
        return out, shape

    @staticmethod
    def power(matrix, exponent, shape):
        out = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.power_on_gpu(matrix, exponent, out, shape[0] * shape[1])
        return out, shape

    @staticmethod
    def power_prime(matrix, exponent, shape):
        out = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.power_prime_on_gpu(matrix, exponent, out, shape[0] * shape[1])
        return out, shape

    @staticmethod
    def matrix_concat(matrix_a, matrix_b, shape_a, shape_b):
        # TODO: any axis concat
        out = lib.allocate_on_gpu((shape_a[0] + shape_b[0]) * shape_a[1])
        lib.matrix_concat_on_gpu(matrix_a, matrix_b, out, shape_a[0], shape_a[1], shape_b[0], shape_b[1])
        return out, (shape_a[0] + shape_b[0], shape_a[1])

    @staticmethod
    def summation(matrix, shape):
        out = lib.allocate_on_gpu(1)
        lib.summation_on_gpu(matrix, out, shape[0] * shape[1])
        return out, (1, 1)

    @staticmethod
    def zeros_matrix_like(shape):
        # TODO: accelerate this
        out = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.scalar_mul_on_gpu(0.0, out, out, shape[0] * shape[1])
        return out, shape

    @staticmethod
    def ones_matrix_like(shape):
        # TODO: accelerate this
        out = lib.allocate_on_gpu(shape[0] * shape[1])
        lib.scalar_mul_on_gpu(0.0, out, out, shape[0] * shape[1])
        lib.add_scalar_on_gpu(1.0, out, out, shape[0] * shape[1])
        return out, shape
    
    @staticmethod
    def to_device(data, shape=None):
        flat_data = flatten_list(data)
        array_type = ctypes.c_float * len(flat_data) # Create array type
        data_ctypes = array_type(*flat_data) # Instantiate array with data
        out = lib.move_to_gpu(data_ctypes, data_ctypes._length_)
        return out, shape
    
    @staticmethod
    def indexing_2d(matrix, output_rows, output_cols, row_slice_start, col_slice_start, shape):
        size = output_rows * output_cols
        out = lib.allocate_on_gpu(size * ctypes.sizeof(ctypes.c_float))
        lib.indexing_2d_on_gpu(matrix, out, shape[0], shape[1], output_rows, output_cols, row_slice_start, col_slice_start)
        return out, (output_rows, output_cols)
    
    @staticmethod
    def copy(data, shape):
        data = ctypes.cast(data, ctypes.POINTER(ctypes.c_float))
        size = shape[0] * shape[1]
        out = lib.allocate_on_gpu(size)
        lib.copy_on_gpu(out, data, size)
        return out, shape

class OperationsCpu(OperationsBase):

    @staticmethod
    def tanh(matrix, shape):
        return [[math.tanh(x_ij) for x_ij in row] for row in matrix], shape

    @staticmethod
    def tanh_prime(matrix, shape):
        t, _ = OperationsCpu.tanh(matrix, shape)
        return [[1 - t_ij**2 for t_ij in row] for row in t], shape

    @staticmethod
    def matrix_mul(matrix_a, matrix_b, shape_a, shape_b):
        rows_a = shape_a[0]
        cols_a = shape_a[1]
        rows_b = shape_b[0]
        cols_b = shape_b[1]

        if cols_a != rows_b:
            raise ValueError(
                f"The number of columns in the first matrix must be equal to the number of rows in the second matrix. Got cols_a {cols_a} and rows_b {rows_b}."
            )

        result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += matrix_a[i][k] * matrix_b[k][j]

        return result, (rows_a, cols_b)

    @staticmethod
    def matrix_scalar_mul(scalar, matrix, shape):
        return [[scalar * element for element in row] for row in matrix], shape


    @staticmethod
    def matrix_add(matrix_a, matrix_b, shape_a, shape_b):
        rows_a = shape_a[0]
        cols_a = shape_a[1]
        rows_b = shape_b[0]
        cols_b = shape_b[1]

        # Determine output dimensions
        out_rows = max(rows_a, rows_b)
        out_cols = max(cols_a, cols_b)

        # Initialize the result matrix with zeros
        result = [[0 for _ in range(out_cols)] for _ in range(out_rows)]

        # Perform the addition with broadcasting
        for i in range(out_rows):
            for j in range(out_cols):
                a_val = matrix_a[i % rows_a][j % cols_a] if rows_a > 1 else matrix_a[0][j % cols_a]
                b_val = matrix_b[i % rows_b][j % cols_b] if rows_b > 1 else matrix_b[0][j % cols_b]
                result[i][j] = a_val + b_val

        return result, (out_rows, out_cols)

    @staticmethod
    def matrix_transpose(matrix, shape):
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))], (shape[1], shape[0])

    @staticmethod
    def element_wise_mul(matrix_a, matrix_b, shape):
        if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
            raise ValueError(
                "Matrices must be of the same size for element-wise multiplication."
            )
        return [
            [matrix_a[i][j] * matrix_b[i][j] for j in range(len(matrix_a[0]))]
            for i in range(len(matrix_a))
        ], shape

    @staticmethod
    def power(matrix, exponent, shape):
        return [[x_ij**exponent for x_ij in row] for row in matrix], shape

    @staticmethod
    def power_prime(matrix, exponent, shape):
        return [[exponent * (x_ij**(exponent - 1)) for x_ij in row] for row in matrix], shape

    @staticmethod
    def matrix_concat(matrix_a, matrix_b, shape_a, shape_b):
        return matrix_a + matrix_b, (shape_a[0] + shape_b[0], shape_a[1])

    @staticmethod
    def summation(matrix, shape):
        return [[sum([sum(row) for row in matrix])]], (1, 1)

    @staticmethod
    def zeros_matrix_like(shape):
        return [[0 for _ in range(shape[1])] for _ in range(shape[0])], shape

    @staticmethod
    def ones_matrix_like(shape):
        return [[1 for _ in range(shape[1])] for _ in range(shape[0])], shape

    @staticmethod
    def to_device(data, shape=None):
        data = ctypes.cast(data, ctypes.POINTER(ctypes.c_float))
        size = shape[0] * shape[1]
        h_data = (ctypes.c_float * size)() # Create array type and instantiate array with empty data
        lib.move_to_cpu(h_data, data, size) # Move size elements starting from data on gpu to h_data on cpu
        h_data_reshaped = reshape_to_nested_list(h_data, shape)
        free_gpu_memory(data)
        return h_data_reshaped, shape
    
    @staticmethod
    def indexing_2d(matrix, output_rows, output_cols, row_slice_start, col_slice_start, shape):
        return [row[col_slice_start:col_slice_start + output_cols] for row in matrix[row_slice_start:row_slice_start + output_rows]], (output_rows, output_cols)

    @staticmethod
    def copy(data, shape):
        return data, shape

class Operations:

    def __new__(cls, device=None):
        if device == 'cuda':
            return OperationsCuda
        elif device == 'cpu':
            return OperationsCpu
        else:
            return OperationsBase
