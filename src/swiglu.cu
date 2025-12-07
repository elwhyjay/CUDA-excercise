#include <cuda_runtime.h>


__global__ void swiglu_big(const float* __restrict__ input, float* __restrict__ output, int N) {
    int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    float inp, inp2;
    for(int i = tid; i < N; i += (gridDim.x * blockDim.x)) {
        inp = input[i];
        inp2 = input[i + N];
        inp = inp / (1.0f + __expf(-inp));
        inp *= inp2;
        output[i] = inp;
    }
}

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