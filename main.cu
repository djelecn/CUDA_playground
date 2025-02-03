#include <stdio.h>
#include <cuda_runtime.h>
#include "gmem/1_naive.cuh"
#include "gmem/2_global_memory_coalescing.cuh"


#define M 256  // Number of rows in A and C
#define K 512   // Number of columns in A and rows in B
#define N 256  // Number of columns in B and C
#define BLOCK_SIZE 32

// Example 3x2 @ 2x4 = 3x4 -> (M x K) @ (K x N) = (M x N)
// A = [[1, 2], 
//      [3, 4], 
//      [5, 6]]

// B = [[7, 8, 9, 10],
//      [11, 12, 13, 14]]

// C = A * B = [[1*7 + 2*11, 1*8 + 2*12, 1*9 + 2*13, 1*10 + 2*14],
//              [3*7 + 4*11, 3*8 + 4*12, 3*9 + 4*13, 3*10 + 4*14],
//              [5*7 + 6*11, 5*8 + 6*12, 5*9 + 6*13, 5*10 + 6*14]]

// C = [[29, 32, 35, 38],
//      [65, 72, 79, 86],
//      [101, 112, 123, 134]]



// CPU matrix multiplication
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}



int main() {

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    float alpha = 0.5;
    float beta = 0.3;


    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(h_A, d_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(h_B, d_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        matmul_cpu(h_A, h_B, h_C, M, K, N);
        myKernels::sgemm_naive<<<gridDim, blockDim>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmul_cpu(h_A, h_B, h_C, M, K, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;


    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        myKernels::sgemm_naive<<<gridDim, blockDim>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Benchmark GM Coalescing implementation
    printf("Benchmarking GM Coalescing implementation...\n");
    double gmc_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        myKernels::gm_coalescing_sgmem<<<gridDim, blockDim>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gmc_total_time += end_time - start_time;
    }
    double gmc_avg_time = gmc_total_time / 20.0;

    // Print results
    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
    printf("GM Coalescing average time: %f microseconds\n", (gmc_avg_time * 1e6f));

    printf("Speedup: %fx\n", cpu_avg_time / gmc_avg_time);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);



    return 0;
}