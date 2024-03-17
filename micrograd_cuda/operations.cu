#include <cuda_runtime.h>

__global__ void matmulKernel(float *a, float *b, float *c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < width && col < width) {
        float sum = 0.0;
        for(int i = 0; i < width; ++i) {
            sum += a[row * width + i] * b[i * width + col];
        }
        c[row * width + col] = sum;
    }
}

extern "C" void matmul(float *a, float *b, float *c, int width) {
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, width * width * sizeof(float));
    cudaMalloc(&d_b, width * width * sizeof(float));
    cudaMalloc(&d_c, width * width * sizeof(float));

    cudaMemcpy(d_a, a, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, width * width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (width + dimBlock.y - 1) / dimBlock.y);
    matmulKernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);

    cudaMemcpy(c, d_c, width * width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
