#include <cuda_runtime.h>

/*
Implement a program that performs R rounds of parallel hashing on an array of 32-bit integers using the provided hash function. The hash should be applied R times iteratively (the output of one round becomes the input to the next).

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in array output
*/


__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    
    unsigned int hash = OFFSET_BASIS;
    
    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }
    
    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < N){
        unsigned int hash = fnv1a_hash(input[x]);
        for(int i = 1; i < R; i++){
            hash = fnv1a_hash(hash);
        }
        output[x] = hash;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    cudaDeviceSynchronize();
}