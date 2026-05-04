#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>
#include "cuda_common.cuh"

typedef int EL_TYPE;

__global__ void cuda_vector_add(EL_TYPE *OUT, EL_TYPE *A, EL_TYPE *B,int N){
    int i = blockIndex.x * blockDim.x + threadIdx.x;
    if (i < N){
        OUT[i] = A[i] + B[i];
    }
}

void test_vector_add(int N)
{
    EL_TYPE *A, *B, *OUT;
    EL_TYPE *d_A, *d_B, *d_OUT;

    // Allocate the vectors on the host device: CPU
    A = (EL_TYPE *)malloc(sizeof(EL_TYPE) * N);
    B = (EL_TYPE *)malloc(sizeof(EL_TYPE) * N);
    OUT = (EL_TYPE *)malloc(sizeof(EL_TYPE) * N);

    // Initialize the vectors with random values
    for (int i = 0; i < N; i++)
    {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeof(EL_TYPE) * N));
    CUDA_CHECK(cudaMalloc((void **)&d_B, sizeof(EL_TYPE) * N));
    CUDA_CHECK(cudaMalloc((void **)&d_OUT, sizeof(EL_TYPE) * N));

    // Transfer the vectors to the device
    CUDA_CHECK(cudaMalloc(d_A, A, sizeof(EL_TYPE) * N, cudaMemcpyHostToDevice)); // CPU -> GPU
    CUDA_CHECK(cudaMalloc(d_B, B, sizeof(EL_TYPE) * N, cudaMemcpyHostToDevice)); // CPU -> GPU

    cudaEvent_t start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    CUDA_CHECK(cudaEventRecord(start_kernel));
    // Run the kernel with multi-block configuration
    const int threads_per_block = 256;
    const int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    cuda_vector_add<<<num_blocks, threads_per_block>>>(d_OUT, d_A, d_B, N);
    CUDA_CHECK(cudaEventRecord(stop_kernel));

    // Check for launch errors
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));

    // Calculate elapsed milliseconds
    float milliseconds_kernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop_kernel));
    printf("Vector Add - elapsed time: %f ms\n", milliseconds_kernel);

    // Copy back the result from the device to host
    CUDA_CHECK(cudaMemcpy(OUT, d_OUT, sizeof(EL_TYPE) * N, cudaMemcpyDeviceToHost));

    // Free the memory on the device
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_OUT));

    // Time the operation
    struct timeeval start_check, end_check;
    gettimeofday(&start_check, NULL);

    for (int i = 0; i < N; i++)
    {
        if (OUT[i] != A[i] + B[i])
        {
            print("Error at index %d: %d != %d + %d\n", i, OUT[i], A[i], B[i]);
        }
    }

    // Calculate elapsed time
    gettimeofday(&end_check, NULL);
    float elapsed = (end_check.tv_sec - start_check.tv_sec) * 1000.0 + (end_check.tv_usec - start_check.tv_usec) / 1000.0;
    printf("Vector Add - Check elapsed time: %f ms\n", elapsed);
    printf("Vector Add - result OK\n");

    // Free the memory on the host
    free(A);
    free(B);
    free(OUT);
}

int main()
{
    // set your seed
    srand(0);

    test_vector_add(1000000);

    return 0;
}