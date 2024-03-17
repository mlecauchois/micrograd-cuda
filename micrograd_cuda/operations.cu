// matmul_kernel.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmulKernel(float *a, float *b, float *c, int widthA, int heightA, int widthB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < heightA && col < widthB) {
        float sum = 0.0;
        for(int i = 0; i < widthA; ++i) {
            sum += a[row * widthA + i] * b[i * widthB + col];
        }
        c[row * widthB + col] = sum;
    }
}

extern "C" void matmul(float *a, float *b, float *c, int widthA, int heightA, int widthB) {
    float *d_a, *d_b, *d_c;

    int heightB = widthA;

    cudaMalloc(&d_a, widthA * heightA * sizeof(float));
    cudaMalloc(&d_b, widthB * heightB * sizeof(float));
    cudaMalloc(&d_c, widthB * heightA * sizeof(float));

    cudaMemcpy(d_a, a, widthA * heightA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, widthB * heightB * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((widthB + dimBlock.x - 1) / dimBlock.x, (heightA + dimBlock.y - 1) / dimBlock.y);
    matmulKernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, widthA, heightA, widthB);

    cudaMemcpy(c, d_c, widthB * heightA * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
