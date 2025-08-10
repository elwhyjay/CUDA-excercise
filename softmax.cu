/*
Write a program that computes the softmax function for an array of 32-bit floating-point numbers on a GPU. The softmax function is defined as follows:

For an input array x of length n , the softmax of x, denoted sig(x), is an array of length n where the i-th element is:

sig(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x))) for j = 0 to n-1
 
Your solution should handle potential overflow issues by using the "max trick". Subtract the maximum value of the input array from each element before exponentiation.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the array output

*/  

#include <cuda_runtime.h>

#define warpSize 32
#define INFINITY __int_as_float(0x7F800000)
__inline__ __device__ float warpShuffleMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}
 
__inline__ __device__ float warpShuffleSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
 
__global__ void reduceMaxKernel(const float* input, float* block_maxes, int N) {
    extern __shared__ float sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
 
    float val = (i < N) ? input[i] : -INFINITY;
 
    val = warpShuffleMax(val);
 
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();
 
    val = (threadIdx.x < blockDim.x / warpSize) ? sdata[threadIdx.x] : -INFINITY;
    if (warp_id == 0) {
        val = warpShuffleMax(val);
    }
 
    if (threadIdx.x == 0) {
        block_maxes[blockIdx.x] = val;
    }
}
 
__global__ void computeSoftmaxKernel(
    const float* input, float* output, int N, 
    const float* max_val, float* sum_exp
) {
    extern __shared__ float sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
 
    float exp_val = 0.0f;
    if (i < N) {
        exp_val = expf(input[i] - *max_val);  
    }
 
    float sum = warpShuffleSum(exp_val);
 
    if (lane_id == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
 
    sum = (threadIdx.x < blockDim.x / warpSize) ? sdata[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        sum = warpShuffleSum(sum);
    }
 
    if (threadIdx.x == 0) {
        atomicAdd(sum_exp, sum);
    }
 
    __syncthreads();
 
    if (i < N) {
        output[i] = exp_val / *sum_exp;
    }
}
 


 
// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    

    const int blockSize = 256;  
    const int gridSize = (N + blockSize - 1) / blockSize; 
    const int sharedMemSize = (blockSize / warpSize) * sizeof(float);  

    float* d_block_maxes;
    cudaMalloc(&d_block_maxes, gridSize * sizeof(float));
 

    reduceMaxKernel<<<gridSize, blockSize, sharedMemSize>>>(input, d_block_maxes, N);
    cudaGetLastError();
    cudaDeviceSynchronize();
 

    int currentGridSize = gridSize;
    while (currentGridSize > 1) {
        int newGridSize = (currentGridSize + blockSize - 1) / blockSize;
        reduceMaxKernel<<<newGridSize, blockSize, sharedMemSize>>>(d_block_maxes, d_block_maxes, currentGridSize);
        cudaGetLastError();
        cudaDeviceSynchronize();
        currentGridSize = newGridSize;
    }
    float* d_max_val = d_block_maxes;  
    float* d_sum_exp;
    cudaMalloc(&d_sum_exp, sizeof(float));
    cudaMemset(d_sum_exp, 0, sizeof(float));
 

    computeSoftmaxKernel<<<gridSize, blockSize, sharedMemSize>>>(input, output, N, d_max_val, d_sum_exp);
    cudaGetLastError();
    cudaDeviceSynchronize();

    cudaFree(d_block_maxes);
    cudaFree(d_sum_exp);
}