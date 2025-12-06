
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define MEM 32
// A, B, and C are device pointers
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta)
{
    __shared__ half mem_a[MEM][MEM];
    __shared__ half mem_b[MEM][MEM];

    int t_x = threadIdx.x;
    int t_y = threadIdx.y;

    int b_x = blockIdx.x;
    int b_y = blockIdx.y;
    
    int row = b_y * MEM + t_y;
    int col = b_x * MEM + t_x;

    float sum = 0.0f;

    for(int stride = 0; stride < (K + MEM - 1)/MEM; stride++)
    {
        if(row < M && (stride * MEM + t_x) < K)
            mem_a[t_y][t_x] = A[row * K + stride * MEM + t_x];
        else
            mem_a[t_y][t_x] = __float2half(0.0f);

        if(col < N && (stride * MEM + t_y) < K)
            mem_b[t_y][t_x] = B[(stride * MEM + t_y) * N + col];
        else
            mem_b[t_y][t_x] = __float2half(0.0f);
        
        __syncthreads();

        for(unsigned int i = 0; i < MEM; i++)
            sum += __half2float(mem_a[t_y][i]) * __half2float(mem_b[i][t_x]);

        __syncthreads();

    }

    if (row < M && col < N)
        C[row * N + col] = __float2half(alpha * sum + beta * __half2float(C[row * N + col]));

}
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 thread(32, 32);
    dim3 block((N + MEM - 1)/MEM, (M + MEM - 1)/MEM);

    gemm<<<block, thread>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
