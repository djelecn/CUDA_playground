#ifndef NAIVE_GMEM 
#define NAIVE_GMEM

#include <cuda_runtime.h>
namespace myKernels{

    // Naive CUDA kernel for matrix multiplication
    __global__ void naive(float *A, float *B, float *C, int m, int k, int n) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < m && col < n) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[row * k + l] * B[l * n + col];
            }
            C[row * n + col] = sum;
        }
    }

}
#endif