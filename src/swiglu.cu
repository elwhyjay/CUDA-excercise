#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    float tmp1=0.0f, tmp2 = 0.0f;
    if(tid < halfN) {
        output[tid] = input[tid]/(1.0f + __expf(-input[tid]));
        output[tid] *= input[tid+halfN];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}