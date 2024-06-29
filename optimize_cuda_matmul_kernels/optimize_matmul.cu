#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

// create as many blocks as necessary to map all of C
// dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
// 32 * 32 = 1024 thread per block
// dim3 blockDim(32, 32, 1);
// launch the asynchronous execution of the kernel on the device
// The function call returns immediately on the host
// sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

/* 
-- __global__ means this is a CUDA kernel function that runs on the GPU.
-- sgemm_naive is the function name (SGEMM stands for Single-precision General Matrix Multiply).
-- Parameters include matrix dimensions (M, N, K), scaling factors (alpha, beta), and pointers to matrices (A, B, C).

*/
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // compute position in C that this thread is responsible for
  /*
  This calculates the unique 2D position (x, y) for each thread.
    -- blockIdx.x and blockIdx.y identify the block.
    -- threadIdx.x and threadIdx.y identify the thread within the block.
    -- This is like determining which cell in the output matrix C this thread will compute.
  */
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];

    /*
    Key Points to Understand:

    -- Each thread computes one element of the output matrix C.
    -- The kernel parallelizes the computation across many threads.
    -- The matrix multiplication is done in a naive way, with each thread doing a full dot product.
    -- This implementation isn't optimized for memory access patterns or GPU architecture.
    */
  }
}

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

int main() {
    // Matrix dimensions
    int M = 1024; // Example dimension, change as needed
    int N = 1024; // Example dimension, change as needed
    int K = 1024; // Example dimension, change as needed

    // Scaling factors
    float alpha = 1.0f;
    float beta = 0.0f;

    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize host matrices
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M * N; ++i) h_C[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy host memory to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(32, 32, 1);

    // Launch kernel
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("SGEMM completed successfully.\n");
    return 0;
}