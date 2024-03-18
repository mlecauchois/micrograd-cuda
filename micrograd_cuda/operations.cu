#include <cuda_runtime.h>
#include <stdio.h>

extern "C" float* move_to_gpu(float *a, int size) {
    float *d_a;
    cudaMalloc(&d_a, size * sizeof(float));
    cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    return d_a;
}

extern "C" void move_to_cpu(float *a, float *d_a, int size) {
    cudaMemcpy(a, d_a, size * sizeof(float), cudaMemcpyDeviceToHost);
}

extern "C" float* allocate_on_gpu(int size) {
    float* d_ptr;
    cudaMalloc((void**)&d_ptr, size * sizeof(float));
    return d_ptr;
}

extern "C" void free_gpu_memory(float* d_ptr) {
    cudaFree(d_ptr);
}

// Matrix multiplication

__global__ void matmul_kernel(float *a, float *b, float *c, int a_rows, int a_cols, int b_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < a_rows && col < b_cols) {
        float sum = 0.0;
        for(int i = 0; i < a_cols; i++) {
            sum += a[row * a_cols + i] * b[i * b_cols + col];
        }
        c[row * b_cols + col] = sum;
    }
}

extern "C" void matmul_on_gpu(float *d_a, float *d_b, float *d_c, int a_rows, int a_cols, int b_cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((b_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (a_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, a_rows, a_cols, b_cols);

    cudaDeviceSynchronize();
}

// Tanh and its derivative

__global__ void tanh_kernel(float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = tanhf(x[idx]);
    }
}

__global__ void tanh_prime_kernel(float *tanh_x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = 1.0f - tanh_x[idx] * tanh_x[idx];
    }
}

extern "C" void tanh_on_gpu(float *d_x, float *d_y, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    tanh_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, n);
    cudaDeviceSynchronize();
}

extern "C" void tanh_prime_on_gpu(float *d_x, float *d_y, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
     // First compute tanh
    tanh_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, n);
    cudaDeviceSynchronize();
    // Then compute tanh prime
    tanh_prime_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_y, d_y, n);
    cudaDeviceSynchronize();

}

// Transpose

__global__ void transpose_kernel(float *in, float *out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        out[idx * rows + idy] = in[idy * cols + idx];
    }
}

extern "C" void transpose_on_gpu(float *d_in, float *d_out, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();
}

// Matrix addition

__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" void add_on_gpu(float *d_a, float *d_b, float *d_c, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
}

// Element wise multiplication

__global__ void element_wise_mul_kernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

extern "C" void element_wise_mul_on_gpu(float *d_a, float *d_b, float *d_c, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    element_wise_mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
}

// Power

__global__ void power_kernel(float *in, float power, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = powf(in[idx], power);
    }
}

extern "C" void power_on_gpu(float *d_in, float power, float *d_out, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    power_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, power, d_out, n);
    cudaDeviceSynchronize();
}

// Power prime

__global__ void power_prime_kernel(float *in, float power, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (in[idx] == 0 && power <= 1) {
            out[idx] = 0;
        } else {
            out[idx] = power * powf(in[idx], power - 1);
        }
    }
}

extern "C" void power_prime_on_gpu(float *d_in, float power, float *d_out, int n) {
    int threadsPerBlock = 256; // This value is often a good starting point, but you might need to optimize it for your specific GPU.
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    power_prime_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, power, d_out, n);

    cudaDeviceSynchronize(); // Ensure the kernel execution completes before returning.
}


// Concatenation

__global__ void matrix_concat_kernel(const float *A, const float *B, float *C, int a_rows, int b_rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < cols) {
        if (row < a_rows) {
            // Copy from matrix A
            C[row * cols + col] = A[row * cols + col];
        } else if (row < a_rows + b_rows) {
            // Copy from matrix B
            C[row * cols + col] = B[(row - a_rows) * cols + col];
        }
    }
}

extern "C" void matrix_concat_on_gpu(const float *d_A, const float *d_B, float *d_C, int a_rows, int a_cols, int b_rows, int b_cols) {
    // Assuming a_cols and b_cols are equal since we are concatenating along rows
    int cols = a_cols;

    dim3 threadsPerBlock(16, 16); // 16x16 is a common choice; adjust based on your GPU's architecture
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (a_rows + b_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matrix_concat_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, a_rows, b_rows, cols);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();
}


// Scalar multiplication

__global__ void scalar_mul_kernel(float scalar, float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = scalar * in[idx];
    }
}

extern "C" void scalar_mul_on_gpu(float scalar, float *d_in, float *d_out, int n) {
    int threadsPerBlock = 256;  // You can adjust this value depending on your device's capability
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the scalar multiplication kernel
    scalar_mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(scalar, d_in, d_out, n);

    // Wait for the GPU to finish before proceeding
    cudaDeviceSynchronize();
}

// Summation kernel
// Dumb version that launches single thread and does for loop
// TODO: use smart CUDA reduction

__global__ void summation_kernel(float *in, float *out, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += in[i];
    }
    *out = sum;
}

extern "C" void summation_on_gpu(float *d_in, float *d_out, int n) {
    summation_kernel<<<1, 1>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
}