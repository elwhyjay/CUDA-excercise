#include <cuda_runtime.h>
/*
Implement a program that copies an n by n matrix of 32-bit floating point numbers from input array "A"
 to output array "B" on the GPU. The program should perform a direct element-wise copy so that 
 B_ij = A_ij for all valid indices.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in matrix B
*/  
__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * N + x;
    if(index < N * N){
        B[index] = A[index];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
} 