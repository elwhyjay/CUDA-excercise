#include <cuda_runtime.h>
/*
Implement a program that performs the leaky ReLU activation function on a vector of floating-point numbers. The leaky ReLU function is defined as:
leakyReLU(x) = max(0, x) + alpha * min(0, x)
 
where alpha is a small positive constant (0.01 in this problem).

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in vector output
Use alpha =0.01 as the leaky coefficient

*/
__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < N){
        output[x] = input[x] > 0 ? input[x] : 0.01 * input[x]; // 0.01 is the leaky coefficient
        
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}