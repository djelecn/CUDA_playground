#ifndef GM_COALESCING
#define GM_COALESCING

#include <cuda_runtime.h>

namespace myKernels{


    __global__ void gm_coalescing_sgmem(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C){

        const int x = blockIdx.x * 32 + (threadIdx.x / 32);
        const int y = blockIdx.y * 32 + (threadIdx.x % 32);

        if ((x < M && y < N)){

            float tmp = 0.0;
            for (uint i = 0; i < K; i++){
                tmp += A[x * K + y] * B[y * N + i];

            }
            C[x * N + y] = alpha * tmp + beta * C[x * N + y];
        }

        
    }
}
#endif