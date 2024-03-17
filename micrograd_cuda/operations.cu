// matmul_kernel.cu
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

extern "C" __global__ void matmul_kernel(float *a, float *b, float *c, int a_rows, int a_cols, int b_cols) {
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
    // Assuming d_a, d_b are already on the device and d_c is allocated
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((b_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (a_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, a_rows, a_cols, b_cols);

    cudaDeviceSynchronize();
}

