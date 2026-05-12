/*
 * Array Increment: Element-wise increment kernel
 * Math: A[i] = A[i] + 1 for all i in [0, N)
 * Inputs: array[N], arraySize
 * Assumptions: arraySize > 0, array is device-allocated
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <random>
#include "../cuda_common.cuh"

// CUDA kernel to increment each element of the array by 1
__global__ void array_increment(int* array, int arraySize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < arraySize) {
        array[idx] = array[idx] + 1;
    }
}

void printArray(int* array, int arraySize) {
    printf("[");
    for (int i = 0; i < arraySize; i++) {
        printf("%d", array[i]);
        if (i < arraySize - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

int main() {
    const int array_size = 10;

    // Allocate host memory for the input array
    int* array = (int*)malloc(array_size * sizeof(int));

    // Initialize the array with random values
    for (int i = 0; i < array_size; i++) {
        array[i] = rand() % 100;
    }

    // Allocate device memory
    int* d_array;
    CUDA_CHECK(cudaMalloc((void**)&d_array, array_size * sizeof(int)));

    // Copy the input array from host to GPU memory
    CUDA_CHECK(cudaMemcpy(d_array, array, array_size * sizeof(int), cudaMemcpyHostToDevice));

    // Print original array BEFORE kernel/copy-back (Bug 1 fix: preserve original for display)
    printf("Original array: ");
    printArray(array, array_size);

    // Launch kernel: 1 block, array_size threads (or use 256 for larger arrays)
    array_increment<<<1, array_size>>>(d_array, array_size);

    // Copy the result array from GPU memory back to host memory
    CUDA_CHECK(cudaMemcpy(array, d_array, array_size * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Incremented array: ");
    printArray(array, array_size);

    free(array);
    CUDA_CHECK(cudaFree(d_array));
    return 0;
}
