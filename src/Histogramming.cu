#include <cuda_runtime.h>

#define COARSE 4
#define BLOCK_THREADS 512
__global__ void his_func(const int* input,int* histogram,int N,int num_bins) {
    
    extern __shared__ int sdata[];
    int x = blockDim.x*blockIdx.x + threadIdx.x;

    for(int i = threadIdx.x ; i<num_bins;i+=blockDim.x){
        sdata[i] = 0;
    }
    __syncthreads();
    for(int i = x;i<N;i+=blockDim.x*gridDim.x){
        atomicAdd(&sdata[input[i]],1);
    }
    __syncthreads();
    for(int i = threadIdx.x; i < num_bins;i+=blockDim.x){
        atomicAdd(&histogram[i],sdata[i]);
    }
    __syncthreads();
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    dim3 blockSize(BLOCK_THREADS);
    dim3 gridSize((N+COARSE*BLOCK_THREADS-1)/(COARSE*BLOCK_THREADS));
    const int shared_memory_size = num_bins*sizeof(int);
    his_func<<<gridSize,blockSize,shared_memory_size>>>(input,histogram,N,num_bins);
    cudaDeviceSynchronize();
}
