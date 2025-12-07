#include <cuda_runtime.h>

// Q, K, V, output are device pointers
/*
Constraints
    Matrix Q is of size M×d and matrices K and V are of size N×d
    1 ≤ M, N ≤ 100,000
    1 ≤ d ≤ 1024
*/

const int warp_size = 32;
const int tile_size = 32;
const int coarse_factor = 4;
__global__ void matrix_transpose_naive(const float* input, float* output,int row,int col) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if( r < row && c < col) {
        output[c*row + r] = input[r*col + c];
    }
}

__global__ void divide_sqrt_naive(const float* input,float* output,int row,int col,float d_inv_sqrt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = row*col;
    for(int i =idx;i<total;i+=blockDim.x*gridDim.x) {
        output[i] = input[i] * d_inv_sqrt;
    }
}

__global__ void matrix_multiplication_naive(const float* A, const float*B,float* C,int M,int N,int K) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(r < M && c < N) {
        float acc = 0.0f;
        for(int i =0;i<N;i++) {
            acc += A[r*N + i] * B[i*K+c];
        }
        C[r*K+c] = acc;
    } 
}

__global__ void softmax_naive(float* scores,int M,int N) {
    int r= blockIdx.x;
    if(r>=M) return;
    float maxv = -INFINITY;
    for(int i=0;i<N;i++){
        maxv = fmaxf(maxv,scores[r*N+i]);
    }
    float sum = 0.0f;
    for (int i=0;i<N;i++){
        float e = expf(scores[r*N+i] - maxv);
        scores[r*N+i] = e;
        sum+=e;
    }
    for(int i=0;i<N;i++)
        scores[r*N+i]/=sum;
}


extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    // Q: M x d
    // K: N x d
    // V: N x d
    // output: M x d
    
    // Step 1: Transpose K (N x d) -> K^T (d x N)
    float* K_T;
    cudaMalloc(&K_T, N * d * sizeof(float));
    
    dim3 transpose_block(tile_size, tile_size);
    dim3 transpose_grid((d + tile_size - 1) / tile_size, (N + tile_size - 1) / tile_size);
    matrix_transpose_naive<<<transpose_grid, transpose_block>>>(K, K_T, N, d);
    
    // Step 2: Compute Q @ K^T -> scores (M x N)
    // Q: M x d, K^T: d x N, scores: M x N
    float* scores;
    cudaMalloc(&scores, M * N * sizeof(float));
    
    dim3 matmul_block(tile_size, tile_size);
    dim3 matmul_grid1((N + tile_size - 1) / tile_size, (M + tile_size - 1) / tile_size);
    matrix_multiplication_naive<<<matmul_grid1, matmul_block>>>(Q, K_T, scores, M, d, N);
    
    // Step 3: Divide by sqrt(d)
    float d_inv_sqrt = 1.0f / sqrtf((float)d);
    int total_scores = M * N;
    int threads = 256;
    int blocks = (total_scores + threads - 1) / threads;
    divide_sqrt_naive<<<blocks, threads>>>(scores, scores, M, N, d_inv_sqrt);
    
    // Step 4: Apply softmax row-wise
    softmax_naive<<<M, 1>>>(scores, M, N);
    
    // Step 5: Compute scores @ V -> output (M x d)
    // scores: M x N, V: N x d, output: M x d
    dim3 matmul_grid2((d + tile_size - 1) / tile_size, (M + tile_size - 1) / tile_size);
    matrix_multiplication_naive<<<matmul_grid2, matmul_block>>>(scores, V, output, M, N, d);
    
    // Cleanup
    cudaFree(K_T);
    cudaFree(scores);

    

}
