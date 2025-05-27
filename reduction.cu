/*
Write a CUDA program that performs parallel reduction on an array of 32-bit floating point numbers to compute their sum. The program should take an input array and produce a single output value containing the sum of all elements.

Implementation Requirements
Use only CUDA native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
*/

#include <cuda_runtime.h>

__global__ void reduce_sum(const float* input, float* output, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // block-level reduction
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, sdata[0]);
}
// input, output are device pointers
void solve(const float* input, float* output, int N) {  
    int total = N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    reduce_sum<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output, N);
    
}



