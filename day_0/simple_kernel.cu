#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void simple_kernel(int *d_array, int size)
{
    int idx = threadIdx.x;
    if (idx < size)
    {
        d_array[idx] = idx;
    }
}

int main(int argc, char **argv)