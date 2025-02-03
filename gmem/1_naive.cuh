#ifndef NAIVE_SGMEM 
#define NAIVE_SGMEM

#include <cuda_runtime.h>
namespace myKernels{

  __global__ void sgemm_naive(int M, int K, int N, float alpha, const float *A,
                              const float *B, float beta, float *C) {

    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
      float tmp = 0.0;
      for (int i = 0; i < K; ++i) {
        tmp += A[x * K + i] * B[i * N + y];
      }
      // C = α*(A@B)+β*C
      C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
  }


}
#endif