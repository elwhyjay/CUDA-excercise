#include <cuda_runtime.h>
/*
Implement a program that performs the Rectified Linear Unit (ReLU) activation function on a vector of 32-bit floating point numbers. The ReLU function sets all negative values to zero and leaves positive values unchanged:

ReLU(x) = max(0, x)

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in output

*/
__global__ void relu_kernel(const float* input, float* output, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < N){
        output[x] = input[x] > 0 ? input[x] : 0;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
