#include <cuda_runtime.h>
/*

Implement a program that reverses an array of 32-bit floating point numbers in-place. The program should perform an in-place reversal of input.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored back in input
*/

__global__ void reverse_array(float* input, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x<N) {
        int i = x;
        int j = N - x - 1;
        float temp = input[i];
        input[i] = input[j];
        input[j] = temp;
    }
}

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}